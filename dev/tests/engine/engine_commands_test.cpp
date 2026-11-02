#include "CostSeeds.hpp"
#include "EngineTestSupport.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;
using splash::test::tickUntil;

namespace {

using splash::test::require;

// A prompt with images runs prefill groups as a text prompt does: an image
// anywhere in a conversation leaves its long prompts' commands groups.
void testImagePromptsRunPrefillGroups() {
  for (bool images : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    Events events;
    engine::Engine engine(test::engineConfig(), cache, model, events);
    guardReleases(storage, engine);
    // Groups run once prefill has a measured cost, as after the start's.
    test::seedAsStartup(engine, model);
    auto value = request(1, std::vector<uint32_t>(8193, 7));
    if (images) {
      value.images = {{4000, 16, 8, 8, 1, 1}};
      value.imagePixels.assign(value.images.front().pixelBytes(), 1);
    }
    engine.submit(std::move(value));
    runUntilIdle(engine);
    require(events.completedCount == 1 && model.prefillChunks.at(1) == 2,
            images ? "a prompt with an image did not run prefill groups"
                   : "a text prompt did not run prefill groups");
  }
}

// A ticket of several commands, as a prefill command whose images are
// encoded ahead of it, runs each on the engine's wakes (advance()): the
// second only once the first has completed, however many wakes come before.
void testTicketAdvancesOnTheEngineWakes() {
  test::TestKvStorage storage(64, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.stagePrefillUntil = std::make_shared<std::atomic<bool>>(false);
  Events events;
  engine::Engine engine(test::engineConfig(), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(100, 7)));
  double now = 1;
  for (uint32_t tick = 0; tick < 8; ++tick)
    static_cast<void>(engine.tick(now++));
  require(engine.commandInFlight() && events.completedCount == 0,
          "a prefill command of two ran its second before its first completed");
  *model.stagePrefillUntil = true;
  tickUntil(engine, now, [&] { return idle(engine); },
            "a prefill command of two did not run its second on the engine's wakes");
  require(events.completedCount == 1 && events.failedCount == 0,
          "a prefill command of two did not complete its request");
}

// Where the model runs a prefill command in passes over its layers, a long
// prompt's commands run in passes, each its own command with the command's
// rows: a decoding lane's steps run between them, its own page table goes
// with each, and only the last commits the rows. A cancel between two
// passes ends the request at once and leaves the rest of its command unrun.
void testPrefillPassesInterleaveDecodeAndEndAtACancel() {
  for (const bool cancel : {false, true}) {
    test::TestKvStorage storage(512, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    model.passes = 48;
    model.decodeFinishes = false;
    // Each command in order: its kind, first lane, pass and passes.
    struct Command final {
      WorkKind kind;
      uint64_t requestId;
      uint32_t pass, passes, rows;
    };
    std::vector<Command> commands;
    model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      commands.push_back({plan.kind, plan.items.front().requestId, plan.pass, plan.passes,
                          plan.items.front().tokenCount});
      return plan.kind == WorkKind::Decode ? test::CommandTimes{.wallMilliseconds = 100.0}
                                           : test::CommandTimes{500.0, 300.0};
    };
    Events events;
    engine::Engine engine(test::engineConfig({.decodeShare = 0.5}), cache, model, events);
    guardReleases(storage, engine);
    test::seedAsStartup(engine, model);
    EngineRequest decoding = request(1, std::vector<uint32_t>(100, 3));
    decoding.maxNewTokens = 10'000;
    engine.submit(std::move(decoding));
    double now = 1;
    tickUntil(engine, now, [&] { return events.outputs.contains(1); }, "the short request did not decode");
    // Five chunks: two groups, of three chunks' rows and two's, each in the
    // passes that bound it beside the decoding lane.
    engine.submit(request(2, std::vector<uint32_t>(8193, 7)));
    const auto prefill = [](const Command &command) {
      return command.kind == WorkKind::Prefill && command.requestId == 2;
    };
    const auto passes = [&] { return std::ranges::count_if(commands, prefill); };
    if (cancel) {
      tickUntil(engine, now, [&] { return passes() == 3 && !engine.commandInFlight(); },
                "the long prompt did not run three passes");
      engine.cancel(2);
      require(engine.snapshot().cancelled == 1, "a cancel between passes did not end the request at once");
      const size_t ran = commands.size();
      for (uint32_t step = 0; step < 8; ++step) static_cast<void>(engine.tick(now++));
      require(passes() == 3 && commands.size() > ran && !model.prefillChunks.contains(2),
              "a cancelled command's passes ran on, or the decoding lane stopped");
    } else {
      tickUntil(engine, now, [&] { return events.outputs.contains(2); }, "the long prompt did not end");
      require(model.prefillChunks.at(2) == 2 && model.prefillRows == 100 + 8193,
              "a command in passes committed its rows other than once, with its last pass");
    }
    std::vector<Command> prefills;
    for (size_t index = 0; index < commands.size(); ++index) {
      const Command &command = commands[index];
      if (!prefill(command)) continue;
      // The command it is a pass of: its first, or once that ended its
      // second.
      const bool second = !prefills.empty() && prefills.front().passes <= prefills.size();
      const uint32_t pass = static_cast<uint32_t>(second ? prefills.size() - prefills.front().passes : prefills.size());
      prefills.push_back(command);
      require(command.passes > 1 && command.rows == (second ? 2049u : 6144u) && command.pass == pass &&
                  command.passes == prefills[second ? prefills.front().passes : 0].passes,
              "a long prompt's command did not run its rows in its passes in order");
      require(!command.pass || (commands[index - 1].kind == WorkKind::Decode && commands[index - 1].requestId == 1),
              "the decoding lane did not step between two passes");
    }
    engine.cancel(1);
    if (!cancel) engine.cancel(2);
    runUntilIdle(engine);
  }
}

// An engine on a target whose prefill commands may run in six passes over
// its layers, a streamed target's (test::passingTarget, its price seeded),
// and the commands it ran. A command takes a streamed target's chunk as
// measured (4.9 s, 3.4 s of it on expert reads) again, over its passes, or
// with targetTimes the target's own time, which the price was seeded with;
// neither trains it. A decode cycle takes 100 ms.
struct PassingEngine {
  struct Command final {
    WorkKind kind;
    std::vector<uint64_t> lanes;
    uint32_t pass, passes;
    // The states each lane's item captures.
    std::vector<size_t> captures;
  };
  PassingEngine() {
    model.decodeFinishes = false;
    model.commandTimes = [this](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      Command command{plan.kind, {}, plan.pass, plan.passes, {}};
      std::vector<model::PrefillLane> lanes;
      for (const ModelBatchItem &item : items) {
        command.lanes.push_back(item.requestId);
        command.captures.push_back(item.stateCaptures.size());
        lanes.push_back({static_cast<uint32_t>(item.logicalPosition), item.tokenCount, 0});
      }
      commands.push_back(std::move(command));
      if (plan.kind == WorkKind::Decode)
        return test::CommandTimes{.wallMilliseconds = 100.0};
      if (targetTimes)
        return test::CommandTimes{
            .wallMilliseconds =
                test::prefillTimes(test::passingTarget(), model.prefillWork(lanes, plan.pass, plan.passes))
                    .wallMilliseconds};
      uint32_t rows = 0;
      for (const model::PrefillLane &lane : lanes)
        rows += lane.rows;
      return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
    };
    guardReleases(storage, engine);
    test::seedAsStartup(engine, model);
  }
  // A request of `tokens` prompt tokens of its id, of `priority`.
  void submit(uint64_t id, uint32_t tokens, RequestPriority priority = RequestPriority::Normal) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    engine.submit(std::move(value));
  }
  // The prefill commands run so far, a pass each.
  [[nodiscard]] size_t prefills() const {
    return std::ranges::count_if(commands, [](const Command &command) { return command.kind == WorkKind::Prefill; });
  }
  void tickUntil(const std::function<bool()> &done, const char *message) {
    ::tickUntil(engine, now, done, message);
  }

  test::TestKvStorage storage{512, 4096, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  // What a command may hold, which the engine takes as it is built.
  bool shaped = (test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6)(model), true);
  Events events;
  engine::Engine engine{test::engineConfig({.decodeShare = 0.5}), cache, model, events};
  std::vector<Command> commands;
  bool targetTimes = false;
  double now = 1;
};

// Where the model runs a prefill command in passes over its layers, short
// prompts that arrive together share one command: every pass runs all their
// items, with the state each captures, and the last commits each lane's rows
// and publishes its replay point. A lane cancelled between its command's
// passes stays in the command to its last pass, which publishes nothing of
// it, and ends then; once every lane of it is cancelled, the command ends at
// once, and so it does after the pass that runs as they are.
void testShortPromptsShareACommandInPasses() {
  for (const auto [cancelled, inFlight] : {std::pair{0u, false}, {1u, false}, {3u, false}, {3u, true}}) {
    PassingEngine passing;
    for (uint64_t id = 1; id <= 3; ++id)
      passing.submit(id, 300);
    if (cancelled) {
      passing.tickUntil([&] { return passing.prefills() == 2 && passing.engine.commandInFlight() == inFlight; },
                        "the short prompts did not run two passes");
      for (uint64_t id = 1; id <= cancelled; ++id)
        passing.engine.cancel(id);
      require(passing.engine.snapshot().cancelled == (cancelled == 3 && !inFlight ? 3u : 0u),
              cancelled == 3 && !inFlight ? "a command whose lanes were all cancelled went on"
                                          : "a lane cancelled during its command's passes left the command");
    }
    passing.tickUntil([&] { return idle(passing.engine); }, "the short prompts did not end");
    require(passing.prefills() == (cancelled == 3 ? 2u : 6u) &&
                std::ranges::all_of(passing.commands,
                                    [](const PassingEngine::Command &command) {
                                      return command.kind != WorkKind::Prefill ||
                                             (command.lanes == std::vector<uint64_t>{1, 2, 3} &&
                                              command.passes == 6 &&
                                              command.captures == std::vector<size_t>{1, 1, 1});
                                    }),
            "short prompts did not share one command in passes, each pass with every lane's capture");
    const EngineSnapshot counts = passing.engine.snapshot();
    require(counts.cancelled == cancelled && passing.events.completedCount == 3 &&
                counts.replayStatePublications == 3 - cancelled,
            "a lane cancelled between its command's passes committed, or a lane that ran did not");
  }
}

// Requests that arrive while a short prompt's command runs its first pass,
// as concurrent agent turns arrive a few milliseconds after the first, start
// at the next command's boundary, and the command is planned again with
// them: one command of all four runs its passes, each prompt's rows
// committed once, rather than the first prompt's command and then theirs.
void testArrivalsDuringPassesShareTheCommand() {
  PassingEngine passing;
  passing.submit(1, 300);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the first prompt did not run a pass");
  for (uint64_t id = 2; id <= 4; ++id)
    passing.submit(id, 300);
  static_cast<void>(passing.engine.tick(passing.now++));
  require(passing.events.startIds == std::vector<uint64_t>({1, 2, 3, 4}),
          "the requests that arrived during a command's pass did not start before its next");
  passing.tickUntil([&] { return idle(passing.engine); }, "the prompts did not end");
  std::vector<PassingEngine::Command> prefills;
  for (const PassingEngine::Command &command : passing.commands)
    if (command.kind == WorkKind::Prefill)
      prefills.push_back(command);
  bool regrouped = prefills.size() == 7 && prefills.front().lanes == std::vector<uint64_t>{1};
  for (uint32_t pass = 0; regrouped && pass < 6; ++pass)
    regrouped = prefills[pass + 1].lanes == std::vector<uint64_t>{1, 2, 3, 4} && prefills[pass + 1].pass == pass;
  require(regrouped,
          "the first prompt's command was not planned again with the requests that arrived in its first pass");
  require(std::ranges::all_of(passing.model.prefillChunks, [](const auto &entry) { return entry.second == 1; }) &&
              passing.engine.snapshot().replayStatePublications == 4 && passing.events.completedCount == 4,
          "a prompt of the command planned again did not commit its rows once");
}

// A command between its passes does not hold back a higher priority: a
// Foreground request that arrives between them starts at once, and its
// prompt runs before the command's next pass, which yields to it,
// abandoned. Its lanes then run their rows again from a first pass.
void testAHigherPriorityIsNotHeldBehindPasses() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.submit(3, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  passing.tickUntil([&] { return passing.events.usage.contains(3); }, "the Foreground request did not end");
  require(passing.events.startIds.back() == 3 && passing.commands.at(arrived).lanes == std::vector<uint64_t>{3},
          "a higher priority's start waited for a command between its passes");
  passing.tickUntil([&] { return idle(passing.engine); }, "the Background requests did not end");
  const auto rerun = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(arrived),
                                          passing.commands.end(), [](const PassingEngine::Command &command) {
                                            return command.lanes == std::vector<uint64_t>{1, 2};
                                          });
  require(rerun != passing.commands.end() && rerun->pass == 0 && passing.events.completedCount == 3 &&
              passing.events.failedCount == 0,
          "an abandoned command's lanes did not run their rows again from a first pass");
}

// A lane suspended between its command's passes takes the command's
// remaining passes with it: four Background prompts share every lane in a
// command in passes, and a Foreground arrival suspends one of them, which
// abandons the command before anything plans its next pass; the other
// three run their rows again from a first pass, and the suspended one once
// it resumes. A lane cancelled between the passes, which stays in the
// command to its last, gives the arrival its lane instead: the command is
// given up for it, which ends that lane, and no lane is suspended.
void testALaneSuspendedBetweenPassesAbandonsItsCommand() {
  for (const bool cancelled : {false, true}) {
    PassingEngine passing;
    // Passes as the price has them: the Foreground start saves the rest of
    // the command, about 2.4 s, more than the four lanes lose, about 0.5 s
    // each (Scheduler::yieldsTo).
    passing.targetTimes = true;
    for (uint64_t id = 1; id <= 4; ++id)
      passing.submit(id, 300, RequestPriority::Background);
    const auto lanes = [](const PassingEngine::Command &command, std::vector<uint64_t> ids) {
      return command.kind == WorkKind::Prefill && command.lanes == ids;
    };
    passing.tickUntil(
        [&] {
          return std::ranges::count_if(passing.commands, [&](const PassingEngine::Command &command) {
                   return lanes(command, {1, 2, 3, 4});
                 }) == 1 &&
                 !passing.engine.commandInFlight();
        },
        "the Background prompts did not run a pass of their command");
    if (cancelled) {
      passing.engine.cancel(4);
      require(passing.engine.snapshot().cancelled == 0, "a lane cancelled between its command's passes left it");
    }
    passing.submit(5, 100, RequestPriority::Foreground);
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    std::vector<PassingEngine::Command> prefills;
    for (const PassingEngine::Command &command : passing.commands)
      if (command.kind == WorkKind::Prefill)
        prefills.push_back(command);
    // The lanes that run again, and those once the suspended one resumes.
    const std::vector<uint64_t> again{1, 2, 3};
    const std::vector<uint64_t> resumed = cancelled ? std::vector<uint64_t>{1, 2, 3} : std::vector<uint64_t>{1, 2, 3, 4};
    require(passing.engine.snapshot().prioritySuspensions == !cancelled &&
                passing.engine.snapshot().cancelled == cancelled && prefills.size() > 2 &&
                prefills[1].lanes == again && prefills[1].pass == 0 &&
                std::ranges::count_if(prefills, [&](const PassingEngine::Command &command) {
                  return lanes(command, resumed) && command.pass;
                }) == 5 &&
                passing.events.completedCount == 5 && passing.events.failedCount == 0,
            "a command whose lane was suspended between its passes ran on, or its lanes did not run again");
  }
}

// A prompt of a higher priority that the engine's limit refuses its KV
// takes what the Background command between its passes holds before any
// lane of it yields: the command, which a lane's suspension would abandon
// anyway, is given up, and the buffers of the states it would capture are
// what the prompt's KV fits in. No lane is suspended, and the Background
// prompts run their rows again from a first pass.
void testAHigherPriorityTakesWhatACommandBetweenPassesHolds() {
  PassingEngine passing;
  passing.model.kvGrowthBlocked = &passing.storage.growthBlocked;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.storage.growthBlocked = true;
  passing.submit(3, 100, RequestPriority::Foreground);
  const size_t arrived = passing.commands.size();
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const auto rerun = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(arrived),
                                          passing.commands.end(), [](const PassingEngine::Command &command) {
                                            return command.lanes == std::vector<uint64_t>{1, 2};
                                          });
  require(passing.engine.snapshot().resourceSuspensions == 0 && passing.model.reclaimedIdleStateBytes != 0 &&
              passing.commands.at(arrived).lanes == std::vector<uint64_t>{3} && rerun != passing.commands.end() &&
              rerun->pass == 0 && passing.events.completedCount == 3 && passing.events.failedCount == 0,
          "a higher priority's refused KV suspended a lane of a command between its passes");
}

// Each later pass of a command in passes runs every lane's page table cut
// anew: a memory-control pass between passes that empties an extent moves
// the second lane's page out of it, and the next pass finds the page where
// it went.
void testLaterPassesFollowEveryLanesMovedPages() {
  test::TestKvStorage storage(16, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.kv = &storage;
  model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  model.passes = 6;
  std::vector<std::pair<std::vector<uint64_t>, uint32_t>> prefills;
  model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    std::vector<uint64_t> lanes;
    uint32_t rows = 0;
    for (const ModelBatchItem &item : items) {
      lanes.push_back(item.requestId);
      rows += item.tokenCount;
    }
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(std::move(lanes), plan.pass);
    return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
  };
  Events events;
  engine::Engine engine(test::engineConfig(), cache, model, events);
  guardReleases(storage, engine);
  test::seedAsStartup(engine, model);
  double now = 1;
  engine.submit(request(1, std::vector<uint32_t>(65, 1)));
  tickUntil(engine, now, [&] { return idle(engine); }, "the first prompt did not end");
  prefills.clear();
  for (const auto &[id, tokens] : {std::pair{uint64_t{2}, 129u}, {uint64_t{3}, 50u}}) {
    EngineRequest prompt = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    prompt.maxNewTokens = 2;
    engine.submit(std::move(prompt));
  }
  tickUntil(engine, now, [&] { return prefills.size() == 1 && !engine.commandInFlight(); },
            "the two prompts did not run a pass of their command");
  require(prefills.back() == std::pair{std::vector<uint64_t>{3, 2}, 0u}, "the two prompts did not share a command");
  const uint64_t revision = cache.pageTable(2).revision;
  static_cast<void>(engine.reclaimMemory({.critical = true}));
  const CacheSnapshot reclaimed = cache.snapshot();
  require(reclaimed.pool.extentCompactions == 1 && reclaimed.pool.extentReleases == 1 &&
              cache.pageTable(2).revision > revision,
          "the memory-control pass did not move the second lane's page");
  tickUntil(engine, now, [&] { return idle(engine); }, "the two prompts did not end");
  require(events.completedCount == 3 && events.failedCount == 0,
          "a later pass did not follow a lane's moved page");
}

// A start of a higher priority that the engine's limit refuses its state
// early in a Background command's passes, where its prompt would supersede
// the command (Scheduler::yieldsTo), takes what the command holds before any
// lane of it is suspended: given up, the command gives back the buffers of
// the states it would capture.
void testAHigherPriorityStartTakesWhatACommandBetweenPassesHolds() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.model.beginGrowthBlocked = [&] {
    return passing.model.lastBeginId == 3 && *passing.model.evictedStateBytes == 0;
  };
  passing.submit(3, 100, RequestPriority::Foreground);
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.prioritySuspensions == 0 && counts.resourceSuspensions == 0 &&
              passing.events.completedCount == 3 && passing.events.failedCount == 0,
          "a higher priority's refused start suspended a lane of a command between its passes");
}

// A lane of several that ends at its deadline during its command's passes
// ends once: while the command's next pass is in flight, the engine has
// nothing more to do for it.
void testALaneThatExpiredBetweenPassesEndsOnce() {
  PassingEngine passing;
  passing.submit(1, 300);
  EngineRequest expiring = request(2, std::vector<uint32_t>(300, 2));
  expiring.deadlineMilliseconds = 5000;
  passing.engine.submit(std::move(expiring));
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the prompts did not run a pass of their command");
  const auto hold = std::make_shared<std::atomic<bool>>(false);
  passing.model.holdPrefillUntil = hold;
  passing.tickUntil([&] { return passing.engine.commandInFlight(); }, "the command's next pass did not start");
  passing.now = 5000;
  require(passing.engine.tick(passing.now++) && !passing.engine.tick(passing.now++),
          "a lane that ended during its command's passes kept the engine busy");
  *hold = true;
  passing.model.holdPrefillUntil.reset();
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  require(passing.events.failedCount == 1 && passing.events.completedCount == 1,
          "the expired lane did not fail once, or the other did not complete");
}

// The ladder's first rung may give the command between its passes up as a
// lane's allocation climbs it, which ends the lanes of it that ended: the lane
// that yields is the one still resident after it. A Foreground prompt
// supersedes two Background prompts' command, one of them cancelled between
// its passes, and the engine's limit refuses the prompt's KV: the command is
// given up, which ends the cancelled lane, and the live one is suspended
// for the prompt, nothing ending twice.
void testTheLaneThatYieldsIsChosenAfterTheLadder() {
  PassingEngine passing;
  for (uint64_t id = 1; id <= 2; ++id)
    passing.submit(id, 300, RequestPriority::Background);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
  passing.engine.cancel(2);
  passing.storage.growthBlocked = true;
  passing.submit(3, 3000, RequestPriority::Foreground);
  passing.tickUntil([&] { return passing.engine.snapshot().resourceSuspensions == 1; },
                    "the Foreground prompt's refused KV did not suspend the live Background lane");
  require(passing.engine.snapshot().cancelled == 1, "the cancelled lane did not end with its command");
  passing.storage.growthBlocked = false;
  passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
  const EngineSnapshot counts = passing.engine.snapshot();
  require(counts.cancelled == 1 && counts.completed == 2 && counts.failed == 0,
          "a lane the ladder ended was chosen to yield");
}

// A lane that ended between its command's passes, cancelled or at its
// deadline, is never suspended as the lane that yields first: where a
// decoding lane's KV the engine's limit refuses there would suspend it, the
// command is given up in its place, which ends that lane and gives back its
// memory and the command's captures, with no lane suspended, and the
// command's live lane runs its rows again from a first pass. So it is where
// the lane that would yield is the live one, the later submission, beside the
// one that ended.
void testALaneThatEndedBetweenPassesIsNotSuspended() {
  for (const uint64_t ending : {uint64_t{3}, uint64_t{2}}) {
    for (const bool cancelled : {true, false}) {
      PassingEngine passing;
      passing.model.decodeTokens = 8;
      EngineRequest decoding = request(1, std::vector<uint32_t>(100, 1));
      decoding.maxNewTokens = 400;
      passing.engine.submit(std::move(decoding));
      passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the decoding lane did not decode");
      for (const uint64_t id : {uint64_t{2}, uint64_t{3}}) {
        EngineRequest prompt = request(id, std::vector<uint32_t>(300, static_cast<uint32_t>(id)));
        if (id == ending)
          prompt.deadlineMilliseconds = 5000;
        passing.engine.submit(std::move(prompt));
      }
      const size_t submitted = passing.commands.size();
      const size_t prefills = passing.prefills();
      // A decoding step runs between the passes.
      passing.tickUntil(
          [&] { return passing.prefills() == prefills + 1 && passing.commands.back().kind == WorkKind::Decode; },
          "the prompts did not run a pass of their command");
      const auto pass = std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(submitted),
                                             passing.commands.end(), [](const PassingEngine::Command &command) {
                                               return command.kind == WorkKind::Prefill;
                                             });
      require(pass->lanes == std::vector<uint64_t>{2, 3} && pass->passes > 1,
              "the prompts did not share a command in passes");
      if (cancelled) {
        passing.engine.cancel(ending);
      } else {
        passing.now = 5000;
        static_cast<void>(passing.engine.tick(passing.now++));
      }
      const EngineSnapshot before = passing.engine.snapshot();
      require(before.cancelled + before.failed == 0, "a lane that ended between its command's passes left it");
      passing.storage.growthBlocked = true;
      const size_t ended = passing.commands.size();
      passing.tickUntil(
          [&] {
            const EngineSnapshot counts = passing.engine.snapshot();
            return counts.cancelled + counts.failed == 1 || counts.resourceSuspensions;
          },
          "the lane that ended did not end");
      passing.storage.growthBlocked = false;
      passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
      // The command never runs its last pass: it is given up, and the live
      // lane's rows run again from a first pass.
      const auto lanes = [&](std::vector<uint64_t> ids) {
        return std::ranges::find_if(passing.commands.begin() + static_cast<std::ptrdiff_t>(ended),
                                    passing.commands.end(), [&](const PassingEngine::Command &command) {
                                      return command.kind == WorkKind::Prefill && command.lanes == ids &&
                                             (ids.size() == 1 || command.pass + 1 == command.passes);
                                    });
      };
      const auto rerun = lanes({ending == 3 ? uint64_t{2} : uint64_t{3}});
      const EngineSnapshot counts = passing.engine.snapshot();
      require(counts.resourceSuspensions == 0 && counts.prioritySuspensions == 0 &&
                  counts.cancelled == cancelled && counts.failed == !cancelled && counts.completed == 2 &&
                  lanes({2, 3}) == passing.commands.end() && rerun != passing.commands.end() && rerun->pass == 0,
              "a lane that ended between its command's passes was suspended, or the command not given up");
    }
  }
}

// A start of a higher priority whose prompt would not supersede the command
// between its passes (Scheduler::yieldsTo) takes nothing the command holds
// and suspends no lane of it: refused memory after four of a two-lane
// Background command's six passes, a Foreground start climbs the rest of the
// ladder, giving back an expert chunk, and waits for the command's end,
// which gives the captures back. Refused at every attempt after the first
// pass of a Background prompt's command, Foreground starts give the command
// up until its lane has waited three times its remaining cost, and let it run
// to its end then.
void testAStartThatWouldNotSupersedeACommandWaitsForItsEnd() {
  // Whether the latest prefill command ran a pass short of its last.
  const auto betweenPasses = [](const PassingEngine &passing) {
    const auto last = std::ranges::find_if(passing.commands.rbegin(), passing.commands.rend(),
                                           [](const PassingEngine::Command &command) {
                                             return command.kind == WorkKind::Prefill;
                                           });
    return last != passing.commands.rend() && last->pass + 1 < last->passes;
  };
  const auto prefillsOf = [](const PassingEngine &passing, std::vector<uint64_t> lanes, uint32_t pass) {
    return std::ranges::count_if(passing.commands, [&](const PassingEngine::Command &command) {
      return command.kind == WorkKind::Prefill && command.lanes == lanes && command.pass == pass;
    });
  };
  {
    PassingEngine passing;
    passing.model.expertChunks = {1000};
    for (uint64_t id = 1; id <= 2; ++id)
      passing.submit(id, 300, RequestPriority::Background);
    passing.tickUntil([&] { return passing.prefills() == 4 && !passing.engine.commandInFlight(); },
                      "the Background prompts did not run four passes");
    require(passing.commands.back().passes == 6, "the Background prompts did not run in six passes");
    // Its memory comes back once the command gives its captures back.
    while (passing.cache.reclaimOneState(StateClass::Ordinary, false)) {
    }
    *passing.model.evictedStateBytes = 0;
    passing.model.beginGrowthBlocked = [&] {
      return passing.model.lastBeginId == 3 && *passing.model.evictedStateBytes == 0 && betweenPasses(passing);
    };
    passing.submit(3, 100, RequestPriority::Foreground);
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    const auto first = std::ranges::find_if(passing.commands, [](const PassingEngine::Command &command) {
      return command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{3};
    });
    const EngineSnapshot counts = passing.engine.snapshot();
    require(prefillsOf(passing, {1, 2}, 0) == 1 && prefillsOf(passing, {1, 2}, 5) == 1 &&
                first != passing.commands.end() && first->pass == 0 &&
                std::ranges::count(passing.model.reclaimSteps, Executor::ReclaimStep::ExpertRelease) == 1 &&
                counts.prioritySuspensions == 0 && counts.resourceSuspensions == 0 && counts.completed == 3,
            "a start whose prompt would not supersede a command took what it holds, or suspended a lane of it");
  }
  {
    PassingEngine passing;
    passing.submit(1, 300, RequestPriority::Background);
    passing.model.beginGrowthBlocked = [&] {
      return passing.model.lastBeginId != 1 && *passing.model.evictedStateBytes == 0 && betweenPasses(passing);
    };
    for (uint64_t foreground = 2; !passing.events.usage.contains(1); ++foreground) {
      require(foreground < 30, "Foreground starts kept giving a Background command up");
      const auto attempts = prefillsOf(passing, {1}, 0);
      passing.tickUntil(
          [&] {
            return (prefillsOf(passing, {1}, 0) > attempts || passing.events.usage.contains(1)) &&
                   !passing.engine.commandInFlight();
          },
          "the Background prompt did not run a first pass");
      if (passing.events.usage.contains(1))
        break;
      // As above; the earlier starts' states are gone.
      while (passing.cache.reclaimOneState(StateClass::Ordinary, false)) {
      }
      *passing.model.evictedStateBytes = 0;
      passing.submit(foreground, 100, RequestPriority::Foreground);
      passing.tickUntil([&] { return passing.events.usage.contains(foreground); },
                        "the Foreground request did not end");
    }
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    // Each abandonment counts the pass it lost as the lane's wait; once that
    // reaches three times its remaining cost (its 2.55 s seeded price, over
    // passes of 0.6 s), it is overdue and its command runs to its end.
    require(prefillsOf(passing, {1}, 0) > 1 && prefillsOf(passing, {1}, 0) <= 14 &&
                passing.engine.snapshot().prioritySuspensions == 0 && passing.events.failedCount == 0,
            "refused Foreground starts gave a Background command up past the bound");
  }
}

// A lane that prepare refuses memory the other lanes of its command fit in
// waits for the command's end, as a lane any command leaves out does: a
// short prompt's command in passes, planned again with a short arrival
// whose KV the engine's limit then refuses, runs its passes to its last
// without it, rather than its first again at every boundary, and the
// arrival runs once memory allows.
void testARefusedLaneWaitsForItsCommandsEnd() {
  PassingEngine passing;
  passing.submit(1, 300);
  passing.tickUntil([&] { return passing.prefills() == 1 && !passing.engine.commandInFlight(); },
                    "the first prompt did not run a pass");
  passing.storage.growthBlocked = true;
  passing.submit(2, 300);
  passing.tickUntil([&] { return passing.events.usage.contains(1); }, "the first prompt's command did not end");
  passing.storage.growthBlocked = false;
  passing.tickUntil([&] { return idle(passing.engine); }, "the refused prompt did not run");
  std::vector<uint32_t> passes;
  for (const PassingEngine::Command &command : passing.commands)
    if (command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{1})
      passes.push_back(command.pass);
  require(passes == std::vector<uint32_t>{0, 0, 1, 2, 3, 4, 5},
          "a command planned again with a lane prepare refused did not run its passes to its last without it");
  require(passing.events.completedCount == 2 && passing.events.failedCount == 0,
          "the refused prompt did not run once memory allowed");
}

// A lane of a command in passes ends at its deadline as it does at a
// cancel: between the passes, while a decoding lane's step runs there, a
// lone lane's command ends at once, its next pass unrun; during a pass, the
// command ends after it; and a lane of two that expires between passes is
// not planned again when an arrival regroups the command.
void testADeadlineEndsALaneOfACommandInPasses() {
  enum class When { BetweenPasses, InAPass, BeforeARegroup };
  for (const When when : {When::BetweenPasses, When::InAPass, When::BeforeARegroup}) {
    PassingEngine passing;
    if (when == When::BetweenPasses) {
      EngineRequest decoding = request(1, std::vector<uint32_t>(100, 1));
      decoding.maxNewTokens = 10'000;
      decoding.deadlineMilliseconds = 1e9;
      passing.engine.submit(std::move(decoding));
      passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the decoding lane did not decode");
    } else if (when == When::BeforeARegroup) {
      passing.submit(1, 300);
    }
    EngineRequest expiring = request(2, std::vector<uint32_t>(when == When::BeforeARegroup ? 300 : 8193, 2));
    expiring.deadlineMilliseconds = 5000;
    passing.engine.submit(std::move(expiring));
    const auto expiringPasses = [&] {
      return std::ranges::count_if(passing.commands, [](const PassingEngine::Command &command) {
        return command.kind == WorkKind::Prefill && std::ranges::count(command.lanes, 2u);
      });
    };
    passing.tickUntil(
        [&] {
          return expiringPasses() == 1 + (when == When::InAPass) &&
                 passing.engine.commandInFlight() == (when != When::BeforeARegroup) &&
                 (when != When::BetweenPasses || passing.commands.back().kind == WorkKind::Decode);
        },
        "the expiring prompt did not run its passes");
    if (when == When::BeforeARegroup)
      passing.submit(3, 300);
    passing.now = 5000;
    static_cast<void>(passing.engine.tick(passing.now++));
    const std::optional<double> wakeup = passing.engine.nextWakeupMilliseconds();
    require(passing.events.failedCount == 1 && (!wakeup || *wakeup >= passing.now),
            "a lane whose deadline passed during its command's passes did not end there");
    if (when == When::BetweenPasses)
      passing.engine.cancel(1);
    passing.tickUntil([&] { return idle(passing.engine); }, "the requests did not end");
    require(expiringPasses() == 1 + (when == When::InAPass) && passing.events.failedCount == 1,
            "a lane whose deadline passed during its command's passes was run again");
  }
}

// During recovery admission waits for no command's passes either: a
// Background lane a Normal start's refusal suspended resumes between them,
// and its replay, of a lower priority, runs once the Normal prompt's command
// has run its last.
void testRecoveryAdmitsBetweenPasses() {
  PassingEngine passing;
  passing.model.beginGrowthBlocked = [&] {
    return passing.model.lastBeginId == 2 && passing.model.requests.at(1).resident;
  };
  EngineRequest background = request(1, std::vector<uint32_t>(65, 1));
  background.priority = RequestPriority::Background;
  background.maxNewTokens = 1000;
  passing.engine.submit(std::move(background));
  passing.tickUntil([&] { return passing.events.outputs.contains(1); }, "the Background lane did not decode");
  passing.submit(2, 300);
  const auto normal = [](const PassingEngine::Command &command) {
    return command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{2};
  };
  passing.tickUntil([&] { return std::ranges::count_if(passing.commands, normal) == 2 && !passing.engine.commandInFlight(); },
                    "the Normal prompt did not run two passes");
  require(passing.engine.snapshot().prioritySuspensions == 1 && passing.engine.snapshot().resourceResumptions == 1,
          "a suspended request did not resume between a command's passes");
  // The first command of the Normal prompt, and the Background lane's replay
  // after it, if it ran.
  const auto replay = [&] {
    const auto first = std::ranges::find_if(passing.commands, normal);
    return std::pair{first, std::find_if(first, passing.commands.end(), [](const PassingEngine::Command &command) {
                       return command.kind == WorkKind::Prefill && command.lanes == std::vector<uint64_t>{1};
                     })};
  };
  passing.tickUntil([&] { return replay().second != passing.commands.end(); }, "the suspended request did not replay");
  const auto [first, replayed] = replay();
  require(std::count_if(first, replayed, normal) == 6, "a lower priority's replay ran between a command's passes");
  passing.engine.cancel(1);
  passing.tickUntil([&] { return idle(passing.engine); }, "engine did not reach idle");
  require(passing.events.failedCount == 0, "a request failed");
}

// A Resident request that continues a Spill request's prefix holds its
// spilled blocks spilled too (Cache::restoreRequest), so its prompt reads
// its history from the SSD cache as a Spill one does, and runs as prefill
// groups, which read it once for their chunks.
void testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups() {
  // Resident KV holds 120 pages.
  Spilling spill(256, false, 120 * 100, 0, {4}, 0,
                 test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget));
  test::seedAsStartup(spill.engine, spill.model, test::heavyHistoryTarget());
  // Eight pages and 4,096 tokens of output project 136 pages: it spills,
  // and decodes below the next request's priority.
  EngineRequest spilling = Spilling::prompt(1, 1, 8, 4'096);
  spilling.priority = RequestPriority::Background;
  spill.engine.submit(std::move(spilling));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                 "the Spill request did not decode");
  // Its eight pages, then 64 pages and a token, two chunks or a group: 73
  // pages in all with its output, which memory holds.
  std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 1);
  prompt.resize(prompt.size() + 64 * KvCache::pageTokens + 1, 2);
  EngineRequest continuing = request(2, std::move(prompt));
  continuing.maxNewTokens = 1;
  spill.engine.submit(std::move(continuing));
  spill.runUntil([&] { return spill.events.usage.contains(2); },
                 "the Resident request did not run");
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.spill == 1 && counts.resident == 1 &&
              spill.events.starts.back() == 8 * KvCache::pageTokens &&
              spill.model.prefillChunks.at(2) == 1,
          "a Resident request holding spilled blocks ran its prompt a chunk a command");
  spill.engine.cancel(1);
  spill.runUntilIdle();
}

// So does a Resident lane that converts to Spill between its prefill
// commands, as a pass the host is short for converts it: the chunks its
// prompt has left run as groups from its next command.
void testALaneThatConvertsMidPromptRunsPrefillGroups() {
  Spilling spill(1024, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget));
  test::seedAsStartup(spill.engine, spill.model, test::heavyHistoryTarget());
  // 128 pages and a token: three chunks, the first of them Resident.
  spill.engine.submit(Spilling::prompt(1, 1, 128, 1));
  spill.runUntil([&] { return spill.model.prefillChunks[1] == 1 && !spill.engine.commandInFlight(); },
                 "the prompt's first chunk did not run");
  require(spill.engine.reclaimMemory({.targetBytes = 1'000'000, .upTo = ReclaimClass::Ordinary,
                                      .hostShort = true})
                  .outcome == ReclaimOutcome::Pending &&
              spill.engine.snapshot().spill.converted == 1,
          "the short pass did not convert the prompt's lane");
  // Its pages spill as their copies land, before its next command: its
  // history, read from the SSD cache, is worth a group.
  spill.tier.complete();
  spill.runUntilIdle();
  // The two chunks left run as one group.
  require(spill.events.completedCount == 1 && spill.model.prefillChunks.at(1) == 2,
          "a lane that converted mid-prompt ran the rest of it a chunk a command");
}

// And so does a Resident lane that comes to hold another request's spilled
// blocks as it prefills: one that shares a Spill request's prefix but no
// state on it starts from nothing, and the blocks its first chunk computes
// take that request's Index pages (KvCache::insert), so the chunks it has
// left read its history from the SSD cache, and run as groups.
void testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups() {
  Spilling spill(1024, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget));
  test::seedAsStartup(spill.engine, spill.model, test::heavyHistoryTarget());
  // Eight pages, four more of its own and a token, its state at their end.
  // It starts Resident and decodes below the next request's priority, and a
  // short pass converts it: its pages but its dense and window pages spill.
  std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 1);
  prompt.resize(prompt.size() + 4 * KvCache::pageTokens + 1, 9);
  EngineRequest spilling = request(1, prompt);
  spilling.maxNewTokens = 4'096;
  spilling.priority = RequestPriority::Background;
  spill.engine.submit(std::move(spilling));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 2 && !spill.engine.commandInFlight(); },
                 "the first request did not decode");
  static_cast<void>(spill.engine.reclaimMemory(
      {.targetBytes = 1'000'000, .upTo = ReclaimClass::Ordinary, .hostShort = true}));
  spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages == 10; },
                 "the converted request's pages did not spill");
  // Its eight pages, then 120 pages and a token: three chunks, the first of
  // which takes the seven spilled pages among them.
  prompt.resize(8 * KvCache::pageTokens);
  prompt.resize(prompt.size() + 120 * KvCache::pageTokens + 1, 2);
  EngineRequest continuing = request(2, std::move(prompt));
  continuing.maxNewTokens = 1;
  spill.engine.submit(std::move(continuing));
  spill.runUntil([&] { return spill.events.usage.contains(2); }, "the Resident request did not run");
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  // The two chunks left run as one group.
  require(counts.resident == 2 && counts.converted == 1 && spill.events.starts.back() == 0 &&
              spill.model.prefillChunks.at(2) == 2,
          "a Resident lane that took spilled blocks as it prefilled ran the rest a chunk a command");
  spill.engine.cancel(1);
  spill.runUntilIdle();
}

// A lane between the passes of its prefill command reads its whole history
// from memory: a pass the host is short for, which converts the Resident
// lane holding the most KV to Spill, leaves it as it is, and its command
// runs its last pass as it began.
void testALaneBetweenItsPassesDoesNotConvert() {
  Spilling spill(64, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4));
  // Each prefill command's lane, pass and passes.
  std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass, plan.passes);
    return test::CommandTimes{};
  };
  // The first turn converts at the memory wall as it decodes and spills its
  // chain; the next restores it, which leaves its copies on the SSD cache.
  spill.storage.budgetPages = 9;
  spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
  spill.runUntilIdle();
  while (spill.cache.reclaimOneState(StateClass::Ordinary, false)) {
  }
  std::vector<uint32_t> history(7 * KvCache::pageTokens + 1, 1);
  history.insert(history.end(), spill.events.outputs.at(1).begin(), spill.events.outputs.at(1).end());
  spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
  spill.engine.submit(request(2, history));
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.converted == 1, "the first turn did not spill its chain");
  // The third turn starts Resident, restores the chain into Full pages, its
  // copies kept, and prefills what follows it in passes.
  test::seedAsStartup(spill.engine, spill.model);
  history.insert(history.end(), 192, 5);
  spill.engine.submit(request(3, history));
  spill.runUntil(
      [&] {
        return !prefills.empty() && prefills.back() == std::tuple{3UL, 0U, 4U} &&
               !spill.engine.commandInFlight();
      },
      "the lane did not run its command's first pass");
  static_cast<void>(spill.engine.reclaimMemory(
      {.targetBytes = 100'000, .upTo = ReclaimClass::Ordinary, .hostShort = true}));
  require(spill.engine.snapshot().spill.converted == 1, "a lane between its passes converted to Spill");
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
          "the lane's command did not run its passes");
}

// As testALaneBetweenItsPassesDoesNotConvert, with the lane that holds the
// most KV second in a command of two: no lane of a command between its
// passes converts, not only the one it leads with.
void testNoLaneOfACommandBetweenItsPassesConverts() {
  Spilling spill(64, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4));
  // Each prefill command's lanes, pass and passes.
  std::vector<std::tuple<std::vector<uint64_t>, uint32_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint64_t> lanes;
      for (const BatchItem &item : plan.items)
        lanes.push_back(item.requestId);
      prefills.emplace_back(std::move(lanes), plan.pass, plan.passes);
    }
    return test::CommandTimes{};
  };
  spill.storage.budgetPages = 9;
  spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
  spill.runUntilIdle();
  while (spill.cache.reclaimOneState(StateClass::Ordinary, false)) {
  }
  std::vector<uint32_t> history(7 * KvCache::pageTokens + 1, 1);
  history.insert(history.end(), spill.events.outputs.at(1).begin(), spill.events.outputs.at(1).end());
  spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
  spill.engine.submit(request(2, history));
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.converted == 1, "the first turn did not spill its chain");
  // The third turn restores the chain; a short fourth joins it ahead, since
  // less of its prompt is left, and the two run in passes.
  test::seedAsStartup(spill.engine, spill.model);
  history.insert(history.end(), 192, 5);
  spill.engine.submit(request(3, history));
  static_cast<void>(spill.engine.tick(spill.now++));
  spill.engine.submit(request(4, std::vector<uint32_t>(100, 9)));
  const std::vector<uint64_t> shared{4, 3};
  spill.runUntil(
      [&] {
        return !prefills.empty() && std::get<0>(prefills.back()) == shared &&
               std::get<1>(prefills.back()) + 1 < std::get<2>(prefills.back()) &&
               !spill.engine.commandInFlight();
      },
      "the two lanes did not share a command in passes");
  static_cast<void>(spill.engine.reclaimMemory(
      {.targetBytes = 100'000, .upTo = ReclaimClass::Ordinary, .hostShort = true}));
  require(spill.engine.snapshot().spill.converted == 1, "a lane between its command's passes converted to Spill");
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 4,
          "the command of two lanes did not run its passes");
  require(std::get<0>(prefills.back()) == shared &&
              std::get<1>(prefills.back()) + 1 == std::get<2>(prefills.back()),
          "the command of two lanes did not run to its last pass");
}

// The rule holds where a lane refused the model's memory for its reads of
// spilled KV takes it from the lane that yields first. Refused a decoding Spill
// lane's floor of the hot block cache, a cancelled lane of the command between
// its passes gives that command up rather than be suspended. A Foreground prompt
// that supersedes a command, refused the staging of the history it streams,
// gives the command up on the ladder, which ends its cancelled lane, and the
// lane that then yields is the live one, chosen after the ladder.
void testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded() {
  struct Lanes final {
    std::vector<uint64_t> ids;
    uint32_t pass, passes;
  };
  const auto setUp = [](Spilling &spill, std::vector<Lanes> &prefills) {
    spill.model.commandTimes = [&prefills](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      if (plan.kind == WorkKind::Decode)
        return test::CommandTimes{.wallMilliseconds = 100.0};
      Lanes command{{}, plan.pass, plan.passes};
      uint32_t rows = 0;
      for (const ModelBatchItem &item : items) {
        command.ids.push_back(item.requestId);
        rows += item.tokenCount;
      }
      prefills.push_back(std::move(command));
      return test::CommandTimes{(3400.0 + rows * 1500.0 / 2048) / plan.passes, 3400.0 / plan.passes};
    };
    test::seedAsStartup(spill.engine, spill.model);
  };
  const auto sharing = [](const Spilling &spill, const std::vector<Lanes> &prefills) {
    return !prefills.empty() && prefills.back().ids.size() == 2 && prefills.back().pass + 1 < prefills.back().passes &&
           !spill.engine.commandInFlight();
  };
  {
    Spilling spill(1024, false, 200 * 100, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6));
    std::vector<Lanes> prefills;
    setUp(spill, prefills);
    spill.engine.submit(Spilling::prompt(1, 1, 80, 4096));
    spill.runUntil([&] { return spill.events.outputs.contains(1) && spill.cache.holdsIndexPages(1); },
                   "the Spill lane did not decode with Index pages");
    spill.engine.submit(Spilling::prompt(2, 2, 9, 1));
    spill.engine.submit(Spilling::prompt(3, 3, 9, 1));
    spill.runUntil([&] { return sharing(spill, prefills); }, "the two prompts did not share a command in passes");
    spill.engine.cancel(3);
    require(spill.engine.snapshot().cancelled == 0, "the cancelled lane left its command");
    spill.model.refusedSpilledReads = {1};
    spill.runUntil([&] { return spill.engine.snapshot().cancelled == 1; }, "the cancelled lane did not end");
    spill.model.refusedSpilledReads.clear();
    require(spill.engine.snapshot().resourceSuspensions == 0, "a lane yielded beside one that gave its command up");
    spill.engine.cancel(1);
    spill.runUntilIdle();
    require(spill.events.completedCount == 3 && spill.events.failedCount == 0, "the requests did not end");
  }
  {
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{64}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6));
    std::vector<Lanes> prefills;
    setUp(spill, prefills);
    const std::vector<uint32_t> earlier(40 * KvCache::pageTokens + 1, 7);
    spill.engine.submit(request(1, earlier));
    spill.runUntilIdle();
    for (uint64_t id = 2; id <= 3; ++id) {
      EngineRequest background = request(id, std::vector<uint32_t>(300, static_cast<uint32_t>(id)));
      background.priority = RequestPriority::Background;
      spill.engine.submit(std::move(background));
    }
    spill.runUntil([&] { return sharing(spill, prefills); }, "the Background prompts did not share a command in passes");
    spill.engine.cancel(3);
    spill.model.refusedStaging = {4};
    std::vector<uint32_t> prompt = earlier;
    prompt.insert(prompt.end(), 300, 9);
    EngineRequest foreground = request(4, prompt);
    foreground.priority = RequestPriority::Foreground;
    spill.engine.submit(std::move(foreground));
    spill.runUntil([&] { return spill.engine.snapshot().resourceSuspensions == 1; },
                   "the refused prompt did not take its memory from the live lane");
    require(spill.engine.snapshot().cancelled == 1, "the cancelled lane did not end with its command");
    spill.model.refusedStaging.clear();
    spill.runUntilIdle();
    const EngineSnapshot counts = spill.engine.snapshot();
    require(counts.cancelled == 1 && spill.events.failedCount == 0 && spill.events.completedCount == 4,
            "the requests did not end once each");
  }
}

// On a target that can spill, a prompt of a higher priority that the
// engine's limit refuses its KV takes what the Background command between its
// passes holds before it converts itself to Spill: giving the command up,
// which that prompt's command does once it runs, is the first rung above the
// cache nothing in use holds, and the buffers of the states the command
// would capture are what its KV fits in.
void testASupersedingPromptTakesACommandsMemoryBeforeSpilling() {
  Spilling spill(64, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6));
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass);
    return test::CommandTimes{};
  };
  spill.model.kvGrowthBlocked = &spill.storage.growthBlocked;
  test::seedAsStartup(spill.engine, spill.model);
  for (uint64_t id = 1; id <= 2; ++id) {
    EngineRequest background = Spilling::prompt(id, static_cast<uint32_t>(id), 4, 1);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
  }
  spill.runUntil([&] { return !prefills.empty() && !spill.engine.commandInFlight(); },
                 "the Background prompts did not run a pass");
  spill.storage.growthBlocked = true;
  EngineRequest foreground = Spilling::prompt(3, 3, 4, 1);
  foreground.priority = RequestPriority::Foreground;
  spill.engine.submit(std::move(foreground));
  spill.runUntilIdle();
  const EngineSnapshot counts = spill.engine.snapshot();
  require(counts.spill.converted == 0 && counts.resourceSuspensions == 0 && spill.events.completedCount == 3 &&
              spill.events.failedCount == 0,
          "a superseding prompt converted itself to Spill before the command between its passes gave its memory back");
}

// A lane that ended between its command's passes, whose KV leaves with that
// command, is no part of a start's projection: a turn that would fit beside
// the lanes that go on starts Resident, not Spill for life.
void testAnEndedLaneLeavesAStartResident() {
  Spilling spill(64, false, 3000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6));
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass);
    return test::CommandTimes{};
  };
  test::seedAsStartup(spill.engine, spill.model);
  spill.engine.submit(Spilling::prompt(1, 1, 20, 1));
  spill.engine.submit(Spilling::prompt(2, 2, 2, 1));
  spill.runUntil([&] { return !prefills.empty() && !spill.engine.commandInFlight(); },
                 "the prompts did not run a pass of their command");
  require(spill.engine.snapshot().spill.spill == 0 && prefills.back() == std::pair{2UL, 0U},
          "the prompts did not share a command");
  spill.engine.cancel(1);
  spill.engine.submit(Spilling::prompt(3, 3, 6, 1));
  spill.runUntilIdle();
  require(spill.engine.snapshot().spill.spill == 0 && spill.events.failedCount == 0 &&
              spill.events.completedCount == 3,
          "a start counted a lane that had ended between its command's passes");
}

// A lane in passes whose replay point the first pass's preparation is
// refused, with nothing but the lane itself to give memory back, converts
// to Spill there: the plan made with it Resident goes back to the scheduler
// (Prepared::Replan), and the one made again runs in passes, the lane
// holding the staging its pages that land between them need (the model
// checks a lane reads no Index page without it).
void testALaneThatConvertsInItsFirstPassReplans() {
  Spilling spill(64, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4));
  std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill)
      prefills.emplace_back(plan.items.front().requestId, plan.pass, plan.passes);
    return test::CommandTimes{};
  };
  // As testALaneBetweenItsPassesDoesNotConvert, a chain whose copies are on
  // the SSD cache, here of 13 pages.
  spill.storage.budgetPages = 9;
  spill.engine.submit(Spilling::prompt(1, 1, 7, 200));
  spill.runUntilIdle();
  std::vector<uint32_t> history(7 * KvCache::pageTokens + 1, 1);
  history.insert(history.end(), spill.events.outputs.at(1).begin(), spill.events.outputs.at(1).end());
  spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
  spill.engine.submit(request(2, history));
  spill.runUntilIdle();
  // Nothing ordinary is left in memory: the chain is on the SSD cache alone.
  for (int step = 0; step < 100; ++step) {
    spill.tier.complete();
    static_cast<void>(spill.cache.pollTransfers());
    if (!spill.cache.reclaimOne(CacheReclaimMode::KeepExtents, ReclaimClass::Ordinary).madeProgress &&
        !spill.cache.transfersInFlight())
      break;
  }
  // The next turn restores it, Resident, its copies kept, and runs in
  // passes up to its replay point, at the chain's end, which the engine's
  // limit refuses until the lane has converted; a state outweighs every
  // extent the pool could give back, as one does.
  test::seedAsStartup(spill.engine, spill.model);
  spill.model.stateBytes = 100'000;
  const auto replay = static_cast<uint32_t>((history.size() - 1) / KvCache::pageTokens * KvCache::pageTokens);
  spill.model.reservationObserver = [&] {
    spill.model.denyReservationAt = spill.engine.snapshot().spill.converted > 1 ? 0 : replay;
  };
  spill.engine.submit(request(3, history));
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 3 &&
              spill.engine.snapshot().spill.converted == 2,
          "the lane did not convert in its first pass and finish");
  using Pass = std::tuple<uint64_t, uint32_t, uint32_t>;
  require(spill.engine.snapshot().replans >= 1 && std::ranges::any_of(prefills,
                                                                      [](const Pass &each) {
                                                                        return std::get<0>(each) == 3 &&
                                                                               std::get<2>(each) > 1;
                                                                      }),
          "the command the lane converted in was not planned again and run in passes");
}

// A command is planned again once a refusal while it takes shape converts
// any of its lanes to Spill, not only its leader: here the rest of a long
// prompt, which shares a command with a short one, and whose replay point
// the engine's limit refuses until the lane has converted. The plan made
// again runs, the converted lane holding the staging its pages that land
// between passes need.
void testACommandAnyLaneConvertsInReplans() {
  Spilling spill(256, false, 100'000, 0, {4}, 0, test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 4));
  // Each prefill command's lanes and passes.
  using Command = std::pair<std::vector<uint64_t>, uint32_t>;
  std::vector<Command> prefills;
  spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint64_t> lanes;
      for (const BatchItem &item : plan.items)
        lanes.push_back(item.requestId);
      prefills.emplace_back(std::move(lanes), plan.passes);
    }
    return test::CommandTimes{};
  };
  test::seedAsStartup(spill.engine, spill.model);
  spill.model.reservationObserver = [&] {
    spill.model.denyReservationAt = spill.engine.snapshot().spill.converted ? 0 : 79 * KvCache::pageTokens;
  };
  // The long prompt's first chunk runs in passes; the short one, which
  // arrives before its last, too late to join it, leads the next command.
  spill.engine.submit(Spilling::prompt(2, 2, 79, 1));
  spill.runUntil([&] { return prefills.size() == 3 && !spill.engine.commandInFlight(); },
                 "the long prompt did not run three passes");
  spill.engine.submit(Spilling::prompt(1, 1, 1, 1));
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 2 &&
              spill.engine.snapshot().spill.converted == 1 && prefills.front() == Command{{2}, 4} &&
              spill.engine.snapshot().replans >= 1,
          "the command a lane converted in was not planned again");
}

// A Resident lane that restored a spilled chain, whose copies landed, and
// converts to Spill when memory refuses its prefill command spills the
// chain's pages at once (Cache::convertToSpill): from that command on it
// streams its history from the SSD cache, which the plan made before did
// not know: the plan goes back to the scheduler (Prepared::Replan), which
// plans again with the lane's Index pages, and the lane holds the staging of
// each command that streams its history. Refused it, the lane leaves the
// command, and the peer planned beside it runs there: nothing yields.
void testALaneThatConvertsAtItsPrefillReplans() {
  // The model's passes; the rows of a peer whose prompt has run a chunk,
  // beside which the lane's command is planned: none, more than the lane's,
  // or fewer; and whether the model refuses the lane its staging there.
  for (const auto &[passes, peerRows, refused] :
       {std::tuple{4u, 0u, false}, {1u, 300u, false}, {1u, 40u, false}, {1u, 300u, true}}) {
    Spilling spill(64, false, 100'000, 0, {4}, 0, test::commandShape(model::ExecutionLimits::prefillTokenBudget, passes));
    // Each prefill command: its lanes, its passes, and the lane whose
    // history it streams, if any (one at most, as the model checks).
    struct Command final {
      std::vector<uint64_t> lanes;
      uint32_t passes;
      uint64_t streamed;
    };
    std::vector<Command> prefills;
    spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      if (plan.kind == WorkKind::Prefill) {
        const auto streamed = std::ranges::find_if(items, [](const ModelBatchItem &item) {
          return item.pageSources != nullptr;
        });
        std::vector<uint64_t> lanes;
        for (const ModelBatchItem &item : items)
          lanes.push_back(item.requestId);
        prefills.push_back(
            {std::move(lanes), plan.passes, streamed == items.end() ? 0 : streamed->requestId});
      }
      return test::CommandTimes{};
    };
    // The first turn converts at the memory wall as it decodes, and its
    // pages spill as their copies land, but its dense page and its window's.
    // It converts before the wall takes any state in use, so it keeps the
    // one at its end; dropped here, the next turn restores the chain from
    // its copies, as from a turn whose state the wall took.
    spill.storage.budgetPages = 9;
    spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
    spill.runUntilIdle();
    while (spill.cache.reclaimOneState(StateClass::Ordinary, false)) {
    }
    std::vector<uint32_t> history(7 * KvCache::pageTokens + 1, 1);
    history.insert(history.end(), spill.events.outputs.at(1).begin(), spill.events.outputs.at(1).end());
    spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
    spill.engine.submit(request(2, history));
    spill.runUntilIdle();
    require(spill.engine.snapshot().spill.converted == 1 && spill.cache.spillSnapshot().spilledPages == 6,
            "the first turn did not spill its chain");
    // The next turn starts Resident and restores the chain into Full pages,
    // its copies kept; 192 tokens more take seven pages past its eight.
    test::seedAsStartup(spill.engine, spill.model);
    history.insert(history.end(), 192, 5);
    spill.engine.submit(request(3, history));
    static_cast<void>(spill.engine.tick(spill.now++));
    if (peerRows) {
      // A peer runs its first chunk while the restore is in flight, which
      // lands before the peer's next command.
      spill.engine.submit(
          request(4, std::vector<uint32_t>(model::ExecutionLimits::prefillTokenBudget + peerRows, 7)));
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
      static_cast<void>(spill.engine.tick(spill.now++));
    }
    // The host refuses growth until the chain spills again: the lane's
    // command converts it, and its pages move into free Index pages.
    spill.storage.growthAllowed = [&](uint32_t) { return spill.cache.spillSnapshot().spilledPages != 0; };
    if (refused) {
      spill.model.refusedStaging = {3};
      spill.runUntil([&] { return spill.engine.snapshot().spill.converted == 2; },
                     "the lane did not convert at its prefill");
      // The command in flight, beside which no lane decodes, is the one the
      // lane converted at: its peer's.
      require(spill.engine.commandInFlight() && prefills.back().lanes == std::vector<uint64_t>{4} &&
                  spill.engine.snapshot().resourceSuspensions == 0,
              "a lane refused its staging held back the peer it leads, or a lane yielded");
      spill.model.refusedStaging.clear();
    }
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == (peerRows ? 4u : 3u) &&
                spill.engine.snapshot().spill.converted == 2 &&
                std::ranges::any_of(prefills, [](const Command &command) { return command.streamed == 3; }),
            "the lane did not convert at its prefill and stream its history");
  }
}

// A lane joins a command with the rows that spare it a command of its own,
// never to fill the leader's chunk, which delays the leader and advances no
// one's completion: beside a prompt of 30 rows, one of 2060 takes 12, and its
// other 2048 run as a chunk of their own.
void testALaneJoinsWithTheRowsThatSpareItACommand() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  // Each prefill command's rows, by lane in submission order.
  std::vector<std::vector<uint32_t>> commands;
  model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    if (plan.kind == WorkKind::Prefill) {
      std::vector<uint32_t> rows(2);
      for (const BatchItem &item : plan.items)
        rows.at(item.requestId - 1) = item.tokenCount;
      commands.push_back(std::move(rows));
    }
    return test::CommandTimes{};
  };
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(2060, 7)));
  engine.submit(request(2, std::vector<uint32_t>(30, 8)));
  runUntilIdle(engine);
  require(events.completedCount == 2 &&
              commands == std::vector<std::vector<uint32_t>>{{12, 30}, {2048, 0}},
          "a lane joined a command with more rows than spare it a command of its own");
}

// The scheduler reports what prices its units: the prefill estimate's seven
// terms by quantity, trained by every command that ran rows, and the decode
// estimate's time a token, trained by the decode commands.
void testTheEstimatesReportWhatPricesTheUnits() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  model.time(test::kM6FlashNextQ2);
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  auto value = request(1, std::vector<uint32_t>(3000, 7));
  value.maxNewTokens = 40;
  engine.submit(std::move(value));
  runUntilIdle(engine);
  const SchedulerSnapshot scheduler = engine.snapshot().scheduler;
  std::vector<std::string> quantities;
  bool priced = false;
  for (const CostTermSnapshot &term : scheduler.prefillEstimate.terms) {
    quantities.push_back(term.quantity);
    priced |= term.millisecondsPerUnit > 0.0;
  }
  require(events.completedCount == 1 &&
              quantities == std::vector<std::string>{"units", "chunk_layers", "row_layers", "attention_pairs",
                                                     "index_pairs", "weight_read_bytes", "spilled_kv_read_bytes"} &&
              priced && scheduler.prefillEstimate.samples == model.prefillChunks.at(1),
          "the prefill estimate did not report its terms as its commands trained them");
  require(scheduler.decodeEstimate.terms.size() == 1 && scheduler.decodeEstimate.terms.front().quantity == "token" &&
              scheduler.decodeEstimate.terms.front().millisecondsPerUnit > 0.0 && scheduler.decodeEstimate.samples > 0,
          "the decode estimate did not report its time a token");
}

// A prefill command of each lane's rows as the model ran it.
using RanRows = std::vector<std::vector<std::pair<uint64_t, uint32_t>>>;

void recordPrefills(Executor &model, RanRows &commands) {
  model.commandTimes = [out = &commands](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
    if (plan.kind == WorkKind::Prefill && !plan.pass) {
      std::vector<std::pair<uint64_t, uint32_t>> rows;
      for (size_t index = 0; index < plan.items.size(); ++index) {
        require(items[index].requestId == plan.items[index].requestId &&
                    items[index].tokenCount == plan.items[index].tokenCount,
                "a command reached the model other than as planned");
        rows.emplace_back(plan.items[index].requestId, plan.items[index].tokenCount);
      }
      out->push_back(std::move(rows));
    }
    return test::CommandTimes{};
  };
}

// A state capture refused at a boundary inside a lane's command cuts the
// command there: prepare sends the plan back with the lane's rows capped at
// the boundary (LaneShape::rowCap), the plan made again reaches the model as
// planned, and the next command goes on from the boundary. (The prompt's
// group of 7000 rows crosses its checkpoint at 4096, which the model
// refuses.)
void testACaptureCutReplansWithARowCap() {
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget)(model);
  RanRows commands;
  recordPrefills(model, commands);
  model.denyReservationAt = engine::kPrefillCheckpointTokens;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  test::seedAsStartup(engine, model);
  engine.submit(request(1, std::vector<uint32_t>(7000, 7)));
  runUntilIdle(engine);
  const EngineSnapshot counts = engine.snapshot();
  require(events.completedCount == 1 && !commands.empty() &&
              commands.front() == std::vector<std::pair<uint64_t, uint32_t>>{{1, engine::kPrefillCheckpointTokens}},
          "the command did not end at the boundary whose capture was refused");
  require(counts.captureCuts == 1 && counts.replans >= 1,
          "the capture cut was not planned again with the lane's rows capped");
  require(commands.size() >= 2 && commands[1].front() == std::pair<uint64_t, uint32_t>{1, 7000 - 4096},
          "the next command did not go on from the boundary");
}

// A capture cut's row cap holds for the tick that cut the command only: the
// next tick's admission plans the lane's command uncut, as that tick's
// planning does, and starts no prompt the command cannot hold. (The first
// group of the prompt, 6144 rows, ends at its checkpoint at 4096, which the
// model refuses; its next is a whole command of 8192 rows. The long prompt
// queued meanwhile would join 4096 rows with its 24 sparing rows at the M6's
// price, but not 8192.)
void testARowCapEndsWithItsTick() {
  test::TestKvStorage storage(2048, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget)(model);
  RanRows commands;
  recordPrefills(model, commands);
  model.denyReservationAt = engine::kPrefillCheckpointTokens;
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  test::seedAsStartup(engine, model, test::kM6FlashNextQ2);
  engine.submit(request(1, std::vector<uint32_t>(18'096, 7)));
  double now = 1;
  static_cast<void>(engine.tick(now++));
  require(commands == RanRows{{{1, engine::kPrefillCheckpointTokens}}} && engine.snapshot().captureCuts == 1,
          "the prompt's first group did not end at the boundary whose capture was refused");
  engine.submit(request(2, std::vector<uint32_t>(24'600, 8)));
  tickUntil(engine, now, [&] { return commands.size() == 2; }, "the prompt's next command did not run");
  require(commands[1] == std::vector<std::pair<uint64_t, uint32_t>>{{1, 8192}} &&
              events.startIds == std::vector<uint64_t>{1},
          "admission started a prompt the next command did not serve, planned beside the cut command's rows");
  tickUntil(engine, now, [&] { return idle(engine); }, "the prompts did not end");
  require(events.completedCount == 2 && events.failedCount == 0, "the prompts did not complete");
}

// A lane refused its KV leaves the plan: prepare sends the plan back
// without it, the others keeping what they were granted, and the plan made
// again runs them; the refused lane runs once memory frees. (The short
// prompt leads and the long one joins with its rows; the pool's budget
// holds the short one's pages but not both.)
void testAPartialAdmissionReplansWithoutTheDeniedLane() {
  test::TestKvStorage storage(512, 4096, 4);
  storage.budgetPages = 52;
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model;
  RanRows commands;
  recordPrefills(model, commands);
  Events events;
  engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
  guardReleases(storage, engine);
  engine.submit(request(1, std::vector<uint32_t>(300, 7)));
  engine.submit(request(2, std::vector<uint32_t>(1500, 8)));
  static_cast<void>(engine.tick(1));
  require(commands.size() == 1 && commands.front() == std::vector<std::pair<uint64_t, uint32_t>>{{1, 300}} &&
              engine.snapshot().replans >= 1,
          "the plan was not made again without the lane refused its KV");
  runUntilIdle(engine);
  require(events.completedCount == 2 && events.failedCount == 0 && model.prefillChunks.at(1) == 1 &&
              model.prefillRows == 1800,
          "the lanes did not each run their rows once");
}

// However prepare's refusals fall, a tick's planning ends: each plan sent
// back leaves a lane more out, caps a lane's rows at an earlier boundary, or
// follows a change of a lane's shape, so a tick plans again at most once a
// lane for refusals and once a lane for each state boundary its command may
// cross (a command of 8192 rows crosses at most two checkpoints of 4096 and
// the replay point), and once for the command between its passes taken.
// Random refusals of states over prompts with checkpoints and a short one.
// (A tick that planned only to wait leaves a later wakeup: EngineHarness,
// V's I7.)
void testReplansPerTickAreBounded() {
  constexpr uint32_t groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  constexpr std::array<uint32_t, 4> prompts{300, 7000, 9500, 13'000};
  constexpr uint64_t most = prompts.size() * (1 + groupRows / engine::kPrefillCheckpointTokens + 1) + 1;
  uint64_t replanned = 0;
  for (uint32_t seed = 1; seed <= 20; ++seed) {
    std::mt19937 random(seed);
    test::TestKvStorage storage(2048, 4096, 4);
    KvPool pool(storage, 0);
    engine::Cache cache(pool, nullptr, nullptr);
    Executor model;
    test::commandShape(groupRows)(model);
    model.reservationRoom = [&] { return random() % 3 != 0; };
    Events events;
    engine::Engine engine(test::engineConfig({.maxContext = 102400}), cache, model, events);
    guardReleases(storage, engine);
    test::seedAsStartup(engine, model);
    for (uint64_t id = 1; id <= prompts.size(); ++id)
      engine.submit(request(id, std::vector<uint32_t>(prompts[id - 1], static_cast<uint32_t>(id))));
    double now = 1;
    for (uint32_t step = 0; step < 2000 && !idle(engine); ++step) {
      const uint64_t before = engine.snapshot().replans;
      static_cast<void>(engine.tick(now++));
      const uint64_t replans = engine.snapshot().replans - before;
      replanned += replans;
      require(replans <= most, "seed " + std::to_string(seed) + ": a tick planned again " +
                                   std::to_string(replans) + " times");
    }
    require(idle(engine) && events.completedCount == prompts.size(),
            "seed " + std::to_string(seed) + ": the prompts did not end");
  }
  require(replanned > 0, "no refusal sent a plan back");
}

} // namespace

std::span<const test::TestCase> commandsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testImagePromptsRunPrefillGroups", testImagePromptsRunPrefillGroups},
      {"testTicketAdvancesOnTheEngineWakes", testTicketAdvancesOnTheEngineWakes},
      {"testPrefillPassesInterleaveDecodeAndEndAtACancel", testPrefillPassesInterleaveDecodeAndEndAtACancel},
      {"testShortPromptsShareACommandInPasses", testShortPromptsShareACommandInPasses},
      {"testArrivalsDuringPassesShareTheCommand", testArrivalsDuringPassesShareTheCommand},
      {"testAHigherPriorityIsNotHeldBehindPasses", testAHigherPriorityIsNotHeldBehindPasses},
      {"testALaneSuspendedBetweenPassesAbandonsItsCommand", testALaneSuspendedBetweenPassesAbandonsItsCommand},
      {"testARefusedLaneWaitsForItsCommandsEnd", testARefusedLaneWaitsForItsCommandsEnd},
      {"testAHigherPriorityTakesWhatACommandBetweenPassesHolds",
       testAHigherPriorityTakesWhatACommandBetweenPassesHolds},
      {"testAHigherPriorityStartTakesWhatACommandBetweenPassesHolds",
       testAHigherPriorityStartTakesWhatACommandBetweenPassesHolds},
      {"testALaneThatExpiredBetweenPassesEndsOnce", testALaneThatExpiredBetweenPassesEndsOnce},
      {"testTheLaneThatYieldsIsChosenAfterTheLadder", testTheLaneThatYieldsIsChosenAfterTheLadder},
      {"testALaneThatEndedBetweenPassesIsNotSuspended", testALaneThatEndedBetweenPassesIsNotSuspended},
      {"testAStartThatWouldNotSupersedeACommandWaitsForItsEnd", testAStartThatWouldNotSupersedeACommandWaitsForItsEnd},
      {"testLaterPassesFollowEveryLanesMovedPages", testLaterPassesFollowEveryLanesMovedPages},
      {"testADeadlineEndsALaneOfACommandInPasses", testADeadlineEndsALaneOfACommandInPasses},
      {"testRecoveryAdmitsBetweenPasses", testRecoveryAdmitsBetweenPasses},
      {"testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups",
       testAResidentLaneHoldingSpilledBlocksRunsPrefillGroups},
      {"testALaneThatConvertsMidPromptRunsPrefillGroups", testALaneThatConvertsMidPromptRunsPrefillGroups},
      {"testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups",
       testALaneThatTakesSpilledBlocksAsItPrefillsRunsPrefillGroups},
      {"testALaneThatConvertsAtItsPrefillReplans", testALaneThatConvertsAtItsPrefillReplans},
      {"testALaneBetweenItsPassesDoesNotConvert", testALaneBetweenItsPassesDoesNotConvert},
      {"testNoLaneOfACommandBetweenItsPassesConverts", testNoLaneOfACommandBetweenItsPassesConverts},
      {"testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded",
       testASpilledReadsRefusalFollowsTheRuleForLanesThatEnded},
      {"testASupersedingPromptTakesACommandsMemoryBeforeSpilling",
       testASupersedingPromptTakesACommandsMemoryBeforeSpilling},
      {"testAnEndedLaneLeavesAStartResident", testAnEndedLaneLeavesAStartResident},
      {"testALaneThatConvertsInItsFirstPassReplans", testALaneThatConvertsInItsFirstPassReplans},
      {"testACommandAnyLaneConvertsInReplans", testACommandAnyLaneConvertsInReplans},
      {"testALaneJoinsWithTheRowsThatSpareItACommand", testALaneJoinsWithTheRowsThatSpareItACommand},
      {"testTheEstimatesReportWhatPricesTheUnits", testTheEstimatesReportWhatPricesTheUnits},
      {"testACaptureCutReplansWithARowCap", testACaptureCutReplansWithARowCap},
      {"testARowCapEndsWithItsTick", testARowCapEndsWithItsTick},
      {"testAPartialAdmissionReplansWithoutTheDeniedLane", testAPartialAdmissionReplansWithoutTheDeniedLane},
      {"testReplansPerTickAreBounded", testReplansPerTickAreBounded},
  };
  return kTests;
}
