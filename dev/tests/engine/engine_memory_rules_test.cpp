// The memory ledger's rules as the engine applies them (DESIGN section 4.4,
// amend/M.md B2-B15): who a claim may take memory from, what a waiting claim
// holds, and what is no claim at all.

#include "CostSeeds.hpp"
#include "EngineHarness.hpp"
#include "EngineTestSupport.hpp"
#include "FakeElasticPool.hpp"
#include "GovernedHost.hpp"
#include "MemoryPasses.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::HarnessOptions;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;

namespace {

using splash::test::require;

// Whether the engine lists the request with its KV in `mode`.
bool listedAs(const Engine &engine, uint64_t id, KvMode mode) {
  for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
    if (active.id == id)
      return active.kvMode == mode;
  return false;
}

// A Foreground request of `prompt` whose two tokens each wait for a token
// mask: once its prompt has run it waits for its first, in service beside
// the lanes of a lower priority, which run meanwhile.
EngineRequest maskedForeground(uint64_t id, std::vector<uint32_t> prompt) {
  EngineRequest result = request(id, prompt);
  result.priority = RequestPriority::Foreground;
  result.constraint = ConstraintMode::TokenMask;
  result.maxNewTokens = 2;
  result.deadlineMilliseconds = 1e9;
  return result;
}

// Answers every mask the request asks for until it ends.
void answerMasks(Events &events, Engine &engine, uint64_t id, const std::function<void()> &tick) {
  const std::array<uint32_t, 1> mask{1};
  size_t answered = 0;
  for (uint32_t step = 0; step < 2000 && !events.usage.contains(id); ++step) {
    for (; answered < events.maskRequests.size(); ++answered)
      if (events.maskRequests[answered].first == id)
        engine.provideMask(id, mask);
    tick();
  }
  require(events.usage.contains(id), "the masked request did not finish");
}

// B7: a claim takes nothing from a lane of a higher priority: no conversion
// of such a lane, and no unit of a pool while one is in service, since the
// pools serve every lane. A Background lane decodes while a Foreground lane
// waits for its token mask; refused growth by the engine's limit, it gives
// way itself: the Foreground lane keeps its KV in memory and the expert
// store its units.
void testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds() {
  {
    // The store holds four units above its floor; beside it the limit
    // leaves the KV eight extents. The two lanes start with thirteen pages
    // each; the Background lane's twentieth would take a ninth extent.
    constexpr uint64_t kUnit = 16 * 4096;
    test::EngineHarness harness(
        {.pages = 128, .memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 5 * kUnit + 32 * 4096 + 4096}});
    test::FakeElasticPool experts("experts", *harness.memory, kUnit, 1, 5, 8, 5, 2.5, 80.0);
    harness.model.experts = &experts;
    static_cast<void>(harness.governor->ledger().addPool(experts));
    harness.model.decodeFinishes = false;
    EngineRequest background = harness.request(1, std::vector<uint32_t>(400, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 400;
    background.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(background));
    harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the Background lane did not decode");
    harness.engine.submit(maskedForeground(2, std::vector<uint32_t>(400, 2)));
    harness.runUntil([&] { return harness.engine.snapshot().resourceSuspensions != 0 || experts.shrinks != 0; },
                     "the Background lane's growth was not refused beside the waiting Foreground lane");
    require(experts.shrinks == 0 && !harness.events.outputs.contains(2) && harness.events.maskRequests.size() == 1,
            "a Background lane's growth took a unit of the expert store beside a Foreground lane in service: "
            "store units " + std::to_string(experts.units()));
    answerMasks(harness.events, harness.engine, 2, [&] {
      if (!harness.tick())
        harness.advanceTo(harness.now() + 1.0);
    });
    harness.runUntilIdle();
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0 &&
                harness.events.outputs[1].size() == 400 && experts.shrinks == 0,
            "the Background request did not finish in the room the Foreground one left: completed " +
                std::to_string(harness.events.completedCount) + ", failed " + std::to_string(harness.events.failedCount) +
                ", outputs " + std::to_string(harness.events.outputs[1].size()) + ", store shrinks " +
                std::to_string(experts.shrinks) +
                (harness.events.failureDetails.empty() ? "" : " (" + harness.events.failureDetails.front().first + ")"));
  }
  {
    // A Background lane of two pages decodes; the Foreground lane of 22
    // pages waits for its mask beside it, and no extent is allocated past
    // what the two hold.
    Spilling spill(1024, false);
    EngineRequest background = request(1, std::vector<uint32_t>(64, 1));
    background.priority = RequestPriority::Background;
    background.maxNewTokens = 400;
    background.deadlineMilliseconds = 1e9;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Background lane did not decode");
    spill.engine.submit(maskedForeground(2, std::vector<uint32_t>(680, 2)));
    spill.runUntil([&] { return spill.events.maskRequests.size() == 1; },
                   "the Foreground lane did not wait for its mask");
    require(listedAs(spill.engine, 2, KvMode::Resident), "the Foreground lane is not Resident");
    spill.storage.budgetPages = static_cast<uint32_t>(spill.storage.allocatedBytes() / 100);
    spill.runUntil(
        [&] {
          return spill.engine.snapshot().resourceSuspensions != 0 || listedAs(spill.engine, 1, KvMode::Spill) ||
                 listedAs(spill.engine, 2, KvMode::Spill);
        },
        "the Background lane's growth was not refused beside the waiting Foreground lane");
    require(listedAs(spill.engine, 2, KvMode::Resident) && spill.events.failedCount == 0,
            "a Background lane's growth converted the Foreground lane beside it to Spill");
    spill.storage.budgetPages = std::numeric_limits<uint32_t>::max();
    answerMasks(spill.events, spill.engine, 2, [&] {
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
    });
    spill.runUntilIdle();
    require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
            "the Background request did not finish after the Foreground one");
  }
}

// An engine whose prompts run in passes (test::commandShape), its `pages`
// KV pages of 16 bytes and the model's states of 64 allocated through
// FakeMemory under a governor of `limitBytes` (GovernedHost); the prefill
// commands' first lane and pass, in order.
struct PassesUnderALimit {
  explicit PassesUnderALimit(uint64_t limitBytes, uint32_t pages = 512)
      : host({.limitBytes = limitBytes}), storage(pages, 16, 4) {
    model.memory = &host.memory;
    model.admission = host.governor.allocationAdmission();
    model.decodeFinishes = false;
    guardReleases(storage, engine);
    test::seedAsStartup(engine, model);
    model.commandTimes = [this](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      if (plan.kind == WorkKind::Prefill)
        prefills.emplace_back(plan.items.front().requestId, plan.pass);
      return test::CommandTimes{};
    };
  }
  void submit(uint64_t id, uint32_t tokens, RequestPriority priority) {
    EngineRequest value = request(id, std::vector<uint32_t>(tokens, static_cast<uint32_t>(id)));
    value.priority = priority;
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
  }
  void runUntil(const std::function<bool()> &done, const char *message) {
    test::tickUntil(engine, now, done, message);
  }

  test::GovernedHost host;
  test::TestKvStorage storage;
  KvPool pool{(host.through(storage), storage), 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  bool shaped = (test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6)(model), true);
  Events events;
  engine::Engine engine{host.connect({}), cache, model, events};
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  double now = 1;
};

// B7 for the Index pages a Spill lane's landed copies move into: their
// claim, the highest-ranked Spill lane's, is of that lane's priority, which
// decides that beside a lane of a higher priority in service it takes no
// unit of a pool (MemoryLedger::dryRun). A Background Spill prompt's moves
// fill the runway and allocate Index extents in that claim.
void testASpillLanesIndexPagesAreClaimedAtItsPriority() {
  Spilling spill(1024);
  std::vector<RequestPriority> claimed;
  spill.storage.growthAllowed = [&](uint32_t) {
    for (const auto &[_, state] : spill.host.governor.ledger().claims())
      if (!state.spec.requestId && state.spec.cls == ClaimClass::InService &&
          state.spec.timing == NeedTiming::Immediate)
        claimed.push_back(state.spec.priority);
    return true;
  };
  EngineRequest background = Spilling::prompt(1, 1, 20, 8);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.cache.spillSnapshot().spilledPages != 0,
          "the Background Spill prompt did not run and spill");
  require(!claimed.empty() && std::ranges::all_of(claimed, [](RequestPriority priority) {
            return priority == RequestPriority::Background;
          }),
          "the Index pages of a Background Spill lane were claimed at another priority: " +
              std::to_string(std::ranges::count(claimed, RequestPriority::Foreground)) + " of " +
              std::to_string(claimed.size()) + " allocations at Foreground");
}

// B15: a prompt whose work supersedes the command between its passes takes
// what that command holds, by the same rule as any unit and without a flag
// of the engine's: the buffers of the states the command would capture,
// which its KV fits in, rather than a lane's suspension. A lane of the taken
// command that ended between its passes is settled once the preparation
// that took it returned (Prepared::Replan), never inside the take.
void testSupersedingClaimTakesThePassCommand() {
  for (const bool ended : {false, true}) {
    // Two Background prompts of five pages share a command in passes: their
    // KV takes three extents (192 bytes) and their replay points two states
    // (128). The Foreground prompt's first extent passes the limit beside
    // them; without the states, its KV and its replay point fit.
    PassesUnderALimit passes(352);
    passes.submit(1, 129, RequestPriority::Background);
    passes.submit(2, 129, RequestPriority::Background);
    passes.runUntil([&] { return !passes.prefills.empty() && !passes.engine.commandInFlight(); },
                    "the Background prompts did not run a pass");
    require(passes.prefills.back().second == 0 && passes.model.heldStateBytes == 128,
            "the Background prompts' command did not hold its states between passes");
    if (ended)
      passes.engine.cancel(2);
    passes.submit(3, 129, RequestPriority::Foreground);
    passes.runUntil([&] { return passes.events.usage.contains(3); }, "the Foreground prompt did not finish");
    passes.runUntil([&] { return idle(passes.engine); }, "the requests did not end");
    const EngineSnapshot counts = passes.engine.snapshot();
    require(counts.resourceSuspensions == 0 && counts.prioritySuspensions == 0 && counts.cancelled == ended &&
                passes.events.completedCount == 3 && passes.events.failedCount == 0,
            std::string("a superseding prompt did not take the memory of the command between its passes") +
                (ended ? " with a lane that ended between them" : "") + ": suspensions " +
                std::to_string(counts.resourceSuspensions) + "+" + std::to_string(counts.prioritySuspensions) +
                ", cancelled " + std::to_string(counts.cancelled) + ", completed " +
                std::to_string(passes.events.completedCount) + ", failed " +
                std::to_string(passes.events.failedCount));
  }
}

// A lane cancelled between the passes of its command ends at once though a
// higher priority's decode holds the command off (smoke a trace; the server
// fails every stream 150 s after a cancel it has no answer for): while a
// constrained Foreground lane waits for its first mask, two Background
// prompts share a command in passes; the mask comes and the Foreground lane
// decodes, every mask answered at once; one Background lane is cancelled
// between the passes, and the command gives way, whole, to the decode.
void testACancelBetweenPassesEndsBesideAHigherPrioritysDecode() {
  PassesUnderALimit passes(uint64_t{1} << 30);
  EngineRequest foreground = request(3, std::vector<uint32_t>(64, 3));
  foreground.priority = RequestPriority::Foreground;
  foreground.constraint = ConstraintMode::TokenMask;
  foreground.maxNewTokens = 1000;
  foreground.deadlineMilliseconds = 1e9;
  passes.engine.submit(std::move(foreground));
  passes.runUntil([&] { return !passes.events.maskRequests.empty(); },
                  "the Foreground lane did not ask for its first mask");
  passes.submit(1, 129, RequestPriority::Background);
  passes.submit(2, 129, RequestPriority::Background);
  passes.runUntil(
      [&] {
        return std::ranges::any_of(passes.prefills, [](const auto &prefill) { return prefill.first != 3; }) &&
               !passes.engine.commandInFlight();
      },
      "the Background prompts did not run a pass");
  require(passes.prefills.back().second == 0 && passes.prefills.back().first != 3,
          "the Background prompts' command is not between its passes");
  const std::array<uint32_t, 1> mask{1};
  size_t answered = 0;
  const auto answer = [&] {
    for (; answered < passes.events.maskRequests.size(); ++answered)
      if (passes.events.maskRequests[answered].first == 3)
        passes.engine.provideMask(3, mask);
  };
  answer();
  passes.engine.cancel(2);
  for (uint32_t step = 0; step < 50 && !passes.events.usage.contains(2); ++step) {
    answer();
    static_cast<void>(passes.engine.tick(passes.now++));
  }
  require(passes.events.usage.contains(2) && !passes.events.usage.contains(3),
          "the cancelled lane did not end while the Foreground lane decoded");
}

// A start the host refused beside a resident lane waits for the host's
// passes, which free what it lacks (earlyfix.md, MemoryLedger.cpp:555 and
// MemoryGovernor.cpp:515; amend/M.md B3): a Foreground lane waits for its
// token mask, the host has the margin and 48 KiB more, and a start needs 64
// KiB of fresh lane state; a control pass frees a unit of a pool at its cap
// (no band), and the start runs.
void testAHostRefusedStartIsFreedForByThePass() {
  test::EngineHarness harness({.pages = 64, .memory = test::FakeMemory::Config{}});
  test::FakeElasticPool pool("experts", *harness.memory, 64 * 1024, 1, 3, 4, 3, 0.25, 1.0);
  static_cast<void>(harness.governor->ledger().addPool(pool));
  harness.model.laneFootprintBytes = 64 * 1024;
  harness.engine.submit(harness.request(1, std::vector<uint32_t>(513, 1)));
  harness.runUntilIdle();
  harness.engine.submit(maskedForeground(2, std::vector<uint32_t>(64, 2)));
  harness.runUntil([&] { return harness.events.maskRequests.size() == 1; },
                   "the Foreground lane did not wait for its mask");
  test::FakeMemory &memory = *harness.memory;
  memory.setHost(memory.hostBytes() - memory.total() - test::GovernorLimits{}.hostSpareReserveBytes -
                     kHostMarginBytes - 48 * 1024,
                 MemoryPressure::Normal);
  harness.engine.submit(harness.request(3, std::vector<uint32_t>(64, 3)));
  double pass = harness.now();
  for (const double end = harness.now() + 3000.0; harness.now() < end && !harness.events.outputs.contains(3);) {
    if (harness.now() >= pass) {
      static_cast<void>(harness.governor->controlPass(harness.now()));
      pass += 500.0;
    }
    if (!harness.tick())
      harness.advanceTo(harness.now() + 1.0);
  }
  const MemoryGovernorSnapshot governor = harness.governor->snapshot();
  require(harness.events.outputs.contains(3),
          "a start the host refused beside a resident lane was never freed for by the host's passes: begins " +
              std::to_string(harness.model.beginAttempts) + ", host headroom " +
              std::to_string(governor.hostHeadroomBytes) + ", waiting claims " +
              std::to_string(governor.claims.waiting) + ", states " + std::to_string(memory.held("states")) +
              ", kv " + std::to_string(memory.held("kv")) + ", host refused " +
              std::to_string(harness.governor->ledger().hostRefusedWaitingBytes()));
}

// A request that has not started ranks by the work it holds, what its start
// restores (smoke a trace): a cold start's claim, at its first attempt beside
// a lane in prefill of its priority, counts none of its prompt as completed,
// so it does not outrank the lane, which has processed fewer rows than that
// prompt holds, nor hold room the lane waits for.
void testAStartRanksByTheWorkItHolds() {
  test::EngineHarness harness({.pages = 256, .memory = test::FakeMemory::Config{}});
  EngineRequest lane = request(1, std::vector<uint32_t>(4000, 1));
  lane.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(lane));
  harness.runUntil(
      [&] {
        const EngineSnapshot counts = harness.engine.snapshot();
        return !counts.activeRequests.empty() && counts.activeRequests.front().promptProcessed != 0 &&
               !harness.engine.commandInFlight();
      },
      "the long prompt did not run its first command");
  const auto rank = [&](uint64_t id) -> std::optional<ClaimRank> {
    for (const auto &[_, state] : harness.governor->ledger().claims())
      if (state.spec.requestId == id)
        return state.spec.rank;
    return std::nullopt;
  };
  std::optional<ClaimRank> started;
  harness.model.beginObserver = [&] {
    if (harness.model.lastBeginId == 2 && !started)
      started = rank(2);
  };
  EngineRequest start = request(2, std::vector<uint32_t>(3000, 2));
  start.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(start));
  harness.runUntil([&] { return started.has_value(); }, "the start was never attempted");
  require(started->completedTokens == 0,
          "a cold start's claim counted " + std::to_string(started->completedTokens) +
              " tokens of its prompt as completed work");
}

// Five cached one-page blocks under states on disk in a pool the budget
// holds to two extents, three pages free: evicting a block starts a
// demotion whose page comes back once its copy lands (tier.complete()).
struct LandingPages {
  LandingPages() {
    storage.budgetPages = 8;
    tier.transferLimit = 8;
    model.decodeFinishes = false;
    guardReleases(storage, engine);
    test::cacheBlocksUnderDiskStates(cache, 5);
    require(pool.freePageCount() == 3 && tier.demotions == 0, "the fixture's pages are off");
  }
  // A decoding lane of two pages whose fourth page waits for a demotion,
  // its claim holding the page that lands.
  void waitingDecoder(uint64_t id) {
    EngineRequest waiter = request(id, std::vector<uint32_t>(64, static_cast<uint32_t>(id)));
    waiter.maxNewTokens = 64;
    waiter.deadlineMilliseconds = 1e9;
    engine.submit(std::move(waiter));
    test::tickUntil(engine, now, [&] { return tier.demotions != 0 && !engine.commandInFlight(); },
                    "the decoding lane's growth did not wait for a demotion");
    require(pool.freePageCount() == 0 && governor().ledger().claimWaiting() && events.outputs[id].size() < 64,
            "the decoding lane does not wait on its claim for the page landing");
  }
  [[nodiscard]] MemoryGovernor &governor() const { return *config.governor; }
  // The KV pages the request holds, none before its lane begins.
  [[nodiscard]] size_t pages(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
      if (active.id == id && active.phase != Phase::Queued && active.phase != Phase::WaitingResources)
        return cache.pageTable(id).pages.size();
    return 0;
  }
  // Whether the request is listed as waiting for resources (suspended).
  [[nodiscard]] bool suspended(uint64_t id) const {
    for (const ActiveRequestSnapshot &active : engine.snapshot().activeRequests)
      if (active.id == id)
        return active.phase == Phase::WaitingResources;
    return false;
  }

  test::TestKvStorage storage{16, 4096, 4};
  KvPool pool{storage, 0};
  test::TestKvTier tier;
  engine::Cache cache{pool, &tier, test::ampleDiskBudget()};
  Executor model;
  Events events;
  const EngineConfig config = test::engineConfig({.maxContext = 102400});
  engine::Engine engine{config, cache, model, events};
  double now = 1;
};

// M-P3: memory on its way to a waiting claim lands for it (DESIGN section
// 4.4 R1, which replaces spill-wait's hold): a claim ranked below it cannot
// take what lands while it waits, and one ranked above it can. The
// starvation spill-wait fixed (DATA.md: decoders waiting 370-506 s while
// prefills took every page that came back) must not return.
void testLandingPaysItsOwnerFirst() {
  {
    // A prefill ranks below a decoding lane of its priority: arriving while
    // the decoder waits for its landing, it takes none of it, and the
    // decoder never gives way.
    LandingPages landing;
    landing.waitingDecoder(1);
    landing.engine.submit(request(2, std::vector<uint32_t>(31, 2)));
    bool decoderSuspended = false;
    for (uint32_t step = 0; step < 400 && !idle(landing.engine); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
      decoderSuspended = decoderSuspended || landing.suspended(1);
      require(landing.pages(2) == 0 || landing.pages(1) >= 4 || landing.pool.freePageCount() != 0,
              "a prefill ranked below the waiting decoder took the page that landed for it");
    }
    require(idle(landing.engine) && !decoderSuspended && landing.events.failedCount == 0 &&
                landing.events.outputs[1].size() == 64 && landing.events.usage.contains(2),
            "the waiting decoder gave way to a prefill ranked below it, or the requests did not end");
  }
  {
    // A Foreground prompt ranks above a Normal decoder that waits: the room
    // the decoder's claim holds does not hold it back, and it runs first.
    LandingPages landing;
    landing.waitingDecoder(1);
    const size_t decoded = landing.events.outputs[1].size();
    EngineRequest foreground = request(2, std::vector<uint32_t>(31, 2));
    foreground.priority = RequestPriority::Foreground;
    foreground.deadlineMilliseconds = 1e9;
    landing.engine.submit(std::move(foreground));
    for (uint32_t step = 0; step < 400 && !landing.events.usage.contains(2); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
    }
    require(landing.events.usage.contains(2) && landing.events.outputs[1].size() == decoded,
            "a Foreground prompt waited behind the room a Normal decoder's claim holds");
    for (uint32_t step = 0; step < 400 && !idle(landing.engine); ++step) {
      static_cast<void>(landing.engine.tick(landing.now++));
      landing.tier.complete();
    }
    require(idle(landing.engine) && landing.events.failedCount == 0 && landing.events.outputs[1].size() == 64,
            "the decoder did not finish after the Foreground prompt");
  }
}

// M-P6: a start that waits for the SSD cache's quota is no memory claim
// (design/M.md L3): while it waits no claim waits for memory, so nothing
// holds room for it and the control pass grows the expert store back.
void testQuotaWaitIsNotAMemoryClaim() {
  Spilling spill(10);
  test::FakeElasticPool experts("experts", spill.host.memory, 4096, 1, 4, 8, 3, 2.5, 80.0);
  spill.model.experts = &experts;
  static_cast<void>(spill.host.governor.ledger().addPool(experts));
  // The first request commits 8 of the 10 slots; the second needs 4 for its
  // prompt and a page of output, and waits for them.
  spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  spill.engine.submit(Spilling::prompt(2, 2, 3, 200));
  spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                 "the second request did not wait for the quota");
  for (int step = 0; step < 8; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    const MemoryGovernorSnapshot memory = spill.host.governor.snapshot();
    require(spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1 &&
                !spill.host.governor.ledger().claimWaiting() && memory.claims.waiting == 0,
            "a start waiting for the SSD cache's quota waits for memory");
    if (!spill.engine.commandInFlight())
      static_cast<void>(spill.host.governor.controlPass(spill.now));
  }
  require(experts.shrinks == 0 && experts.units() == 4,
          "the expert store did not grow back while only the quota kept a start waiting: units " +
              std::to_string(experts.units()));
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
          "the start did not run once the quota it waited for came back");
}

// The host claim's choice between a Resident lane's conversion and a unit of
// the expert store follows their value (amend/M.md B8; DESIGN section 0,
// 19:45), priced on what was measured: the store's ghost hits over the
// control passes, a decode command's hot-block reads. A store whose units
// the decode commands keep missing keeps them, and the lane converts; one
// whose units nothing misses gives one, and the lane keeps its KV in memory.
void testConversionVsStoreFollowsValue() {
  const auto hostTakes = [](uint64_t ghostHitsPerSecond) {
    // As testSpillDecodeTermIsMeasured: memory for KV holds the short
    // request's projection, not the long one's, which spills.
    Spilling spill(1024, false, 30 * 100, 0, {4}, 0, {}, {.limitBytes = 3'000});
    test::seedAsStartup(spill.engine, spill.model);
    spill.model.commandTimes = [](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      test::CommandTimes times{.wallMilliseconds = 10.0};
      if (plan.kind == WorkKind::Decode)
        times.sample.spilledKvReadMilliseconds = 9.0;
      return times;
    };
    // Four units of 100 bytes, its floor one, of the eight that would hold
    // every expert.
    test::FakeElasticPool experts("experts", spill.host.memory, 100, 1, 4, 8, 4, 2.5, 80.0);
    spill.model.experts = &experts;
    static_cast<void>(spill.host.governor.ledger().addPool(experts));
    spill.engine.submit(Spilling::prompt(1, 1, 40, 100'000));
    spill.runUntil([&] { return spill.events.outputs[1].size() >= 20 && spill.cache.holdsIndexPages(1); },
                   "the Spill lane did not decode over Index pages");
    spill.engine.submit(Spilling::prompt(2, 2, 9, 200));
    spill.runUntil([&] { return spill.events.outputs[2].size() >= 20 && !spill.engine.commandInFlight(); },
                   "the Resident lane did not decode");
    require(spill.engine.snapshot().spill.spill == 1 && spill.engine.snapshot().spill.resident == 1,
            "the fixture did not run one Spill and one Resident lane");
    // Thirty control passes a second apart, decode commands between them.
    for (uint32_t pass = 1; pass <= 30; ++pass) {
      experts.served(0, ghostHitsPerSecond, 0, 0);
      const size_t decoded = spill.events.outputs[2].size();
      spill.runUntil([&] { return spill.events.outputs[2].size() > decoded && !spill.engine.commandInFlight(); },
                     "the lanes did not decode between control passes");
      spill.engine.refreshMemoryRates(pass * 1000.0);
    }
    std::string took;
    spill.host.governor.ledger().setDecisionLog([&](std::string_view line) {
      if (took.empty() && line.find(", took") != std::string_view::npos)
        took = line;
    });
    static_cast<void>(test::hostPass(spill.host.governor, 100));
    spill.engine.cancel(1);
    spill.engine.cancel(2);
    spill.runUntilIdle();
    return took;
  };
  const std::string missed = hostTakes(10'000);
  require(missed.find("engine conversion") != std::string::npos && missed.find("experts unit") == std::string::npos,
          "a store whose units decode keeps missing gave one before the Resident lane converted: " + missed);
  const std::string unmissed = hostTakes(0);
  require(unmissed.find("experts unit") != std::string::npos &&
              unmissed.find("engine conversion") == std::string::npos,
          "the Resident lane converted before a store whose units nothing misses gave one: " + unmissed);
}

// Placement is decided once per candidate in each state of memory
// (amend/M.md B13): its two questions to the model, the spilled reads its
// prefill and its decode would add, are asked again only once an admission
// or a refusal changed what memory holds.
void testPlacementOncePerCandidatePerState() {
  // Four prompts of 20 pages beside a limit that holds two of them at once.
  HarnessOptions options{.pages = 256, .memory = test::FakeMemory::Config{}, .limits = {.limitBytes = 48 * 4096}};
  options.engine.spill = {.capable = true,
                          .denseTokens = KvCache::pageTokens,
                          .geometry = {.fullPageBytes = 4096,
                                       .slotBytes = 4096,
                                       .residentTokens = 100'000,
                                       .densePages = 1,
                                       .windowPages = 1,
                                       .groupPages = 4}};
  test::EngineHarness harness(std::move(options));
  harness.model.decodeFinishes = false;
  for (uint64_t id = 1; id <= 4; ++id) {
    EngineRequest prompt = harness.request(id, std::vector<uint32_t>(20 * KvCache::pageTokens, static_cast<uint32_t>(id)));
    prompt.maxNewTokens = 40;
    prompt.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(prompt));
  }
  harness.runUntilIdle();
  require(harness.events.completedCount == 4 && harness.events.failedCount == 0, "the prompts did not all run");
  require(!harness.shortfallQueries().empty(), "no candidate was placed by the spilled reads it would add");
  require(harness.mostShortfallQueries() <= 2,
          "a candidate was placed more than once in one state of memory: " +
              std::to_string(harness.mostShortfallQueries()) + " queries");
}

// A state only on disk at `prompt`'s 64 tokens, over KV the cache holds in
// memory, built in a start's Floor claim as a start that adopts a
// persistent cache builds it.
void cacheDiskState(test::EngineHarness &harness, const std::vector<uint32_t> &prompt) {
  const test::ClaimScope start(*harness.governor, ClaimClass::Floor);
  harness.cache.beginRequest(999);
  require(harness.cache.ensureTokens(999, 64).granted(), "the disk state's KV was not allocated");
  test::demoteState(harness.cache, test::publishBlocks(harness.cache, 999, prompt, 64));
  harness.cache.endRequest(999);
}

// L7: a state restored from disk takes its RAM copy in a claim of its
// request, an Optional one, since the copy only saves the next read. With
// room the copy is admitted and the next request on the prompt reads no
// disk. A restore that lands while a command is in flight waits for the
// command to end (L8: its claim takes units only then), and its copy is
// admitted in the room there is by then.
void testARestoredStateIsPromotedInAClaim() {
  const std::vector<uint32_t> prompt(65, 17);
  {
    test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
    cacheDiskState(harness, prompt);
    harness.model.restoreControl->ready = true;
    harness.engine.submit(harness.request(1, prompt));
    harness.runUntilIdle();
    harness.engine.submit(harness.request(2, prompt));
    harness.runUntilIdle();
    const StateCacheSnapshot states = harness.cache.snapshot().stateCache;
    require(harness.model.diskReads == 1 && states.promotions == 1 && harness.model.heldStateBytes == 64 &&
                harness.events.completedCount == 2 && harness.events.failedCount == 0,
            "a restored state was not promoted into memory its claim was admitted: disk reads " +
                std::to_string(harness.model.diskReads) + ", promotions " + std::to_string(states.promotions) +
                ", state bytes " + std::to_string(harness.model.heldStateBytes));
  }
  {
    test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
    cacheDiskState(harness, prompt);
    const std::shared_ptr<std::atomic<bool>> release = harness.model.holdCommands();
    harness.engine.submit(harness.request(1, prompt));
    harness.engine.submit(harness.request(2, std::vector<uint32_t>(31, 2)));
    harness.runUntil([&] { return harness.engine.commandInFlight() && harness.model.diskReads == 1; },
                     "the restore did not start beside a command in flight", 100);
    harness.memory->hold("limit", harness.governor->snapshot().headroomBytes - 32);
    harness.model.restoreControl->ready = true;
    static_cast<void>(harness.tick());
    const StateCacheSnapshot landed = harness.cache.snapshot().stateCache;
    require(harness.engine.commandInFlight() && landed.promotions == 0 && landed.promotionsSkipped == 0,
            "a restore that landed beside a command in flight gave up its copy before the command ended");
    harness.memory->release("limit", harness.memory->held("limit"));
    *release = true;
    harness.runUntilIdle();
    const StateCacheSnapshot states = harness.cache.snapshot().stateCache;
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && states.promotions == 1 &&
                states.promotionsSkipped == 0,
            "a restore that landed beside a command in flight was not promoted once the command ended");
  }
}

// A unit of `bytes` that leaves memory once it lands, at no cost: a claim
// whose free() takes it waits for it.
class LandingUnit final : public VictimSource {
public:
  explicit LandingUnit(uint64_t bytes) : bytes_(bytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "landing"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    if (after || taken_)
      return std::nullopt;
    return Victim{.key = 1, .bytes = bytes_, .later = true};
  }
  FreeResult take(const VictimQuery &, const Victim &) override {
    taken_ = true;
    return {FreeOutcome::Pending, 0, bytes_};
  }

private:
  uint64_t bytes_;
  bool taken_ = false;
};

// R1, R2: a lane below a claim that waits for bytes lacks, when the host
// refuses its extent, the pages it adds beyond the free ones with those the
// claim holds left free, which its one free() takes in place from the
// cache: its retry is granted, and it keeps its lane. A Foreground claim
// waits for 8 pages' bytes beside 2 free pages; a Normal lane decoding
// beside it needs a page with the host short of its margin (the base asked
// for that page alone, was refused the held pages again, and suspended the
// lane).
void testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages() {
  test::EngineHarness harness({.memory = test::FakeMemory::Config{}});
  constexpr uint32_t page = KvCache::pageTokens;
  harness.engine.submit(harness.request(1, std::vector<uint32_t>(8 * page + 1, 7)));
  harness.runUntilIdle();
  harness.model.decodeFinishes = false;
  EngineRequest lane = harness.request(2, std::vector<uint32_t>(page + 1, 3));
  lane.maxNewTokens = 4 * page;
  harness.engine.submit(std::move(lane));
  harness.runUntil([&] { return harness.events.outputs.contains(2) && !harness.engine.commandInFlight(); },
                   "the lane did not decode");
  // The claim takes the landing alone: the cached chain, which costs
  // nothing at the harness's rates, is set aside meanwhile.
  MemoryLedger &ledger = harness.governor->ledger();
  LandingUnit landing(8 * harness.cache.bytesPerPage(kv::PageForm::Full));
  ledger.removeSource(harness.cache);
  ledger.addSource(landing);
  const ClaimId waiter = ledger.open({.cls = ClaimClass::InService, .rank = {.priority = 2}});
  const ClaimId below = ledger.open({.cls = ClaimClass::InService, .rank = {.priority = 1}});
  require(ledger.free(waiter, {.bytes = 8 * harness.cache.bytesPerPage(kv::PageForm::Full)}, 0, false, false)
                  .outcome == FreeOutcome::Pending &&
              ledger.claim(waiter).waiting && ledger.heldPages(below)[0] == 8,
          "the Foreground claim did not wait for 8 pages' bytes");
  ledger.close(below);
  ledger.removeSource(landing);
  ledger.addSource(harness.cache);
  harness.memory->setHost(harness.memory->hostBytes() - (2ULL << 30) - kHostMarginBytes / 2, MemoryPressure::Normal);
  harness.runUntil(
      [&] {
        return harness.events.outputs.at(2).size() >= 2 * page || harness.engine.snapshot().resourceSuspensions;
      },
      "the lane neither decoded past its next page nor was suspended");
  const uint64_t suspensions = harness.engine.snapshot().resourceSuspensions;
  const uint32_t free = harness.pool.freePageCount(kv::PageForm::Full);
  require(suspensions == 0 && free >= 8,
          "a lane below a waiting claim refused an extent by the host did not take what it lacked beyond the held "
          "pages, or took them: " +
              std::to_string(suspensions) + " suspensions, " + std::to_string(free) + " free pages");
  ledger.close(waiter);
  harness.memory->setHost(test::FakeMemory::Config{}.otherBytes, MemoryPressure::Normal);
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0, "the lane did not finish");
}

// L7: a command that captures a checkpoint and the prompt's replay point
// reserves the replay point's buffers beside the checkpoint's in the replay
// point's claim, and takes the checkpoint's only when refused. A checkpoint
// lies a chunk or more before the replay point (plannedCheckpoints), so
// only a command of more rows holds both: the 8192-row command of a prompt
// of 6145 tokens captures the checkpoint at 4096 and the replay point at
// 6144, the model's state pool empty.
void testAReplayPointBesideACheckpointIsReservedInAClaim() {
  PassesUnderALimit passes(1ULL << 30);
  passes.submit(1, 6145, RequestPriority::Normal);
  passes.runUntil([&] { return idle(passes.engine); }, "the prompt did not finish");
  const EngineSnapshot counts = passes.engine.snapshot();
  require(test::capturedStates(passes.model, 1) ==
                  std::vector<std::pair<uint32_t, uint64_t>>{{4096, 6145}, {6144, 6145}} &&
              counts.checkpointPublications == 1 && counts.replayStatePublications == 1 &&
              passes.model.heldStateBytes == 128 && passes.events.usage.contains(1) &&
              passes.events.failedCount == 0,
          "the command did not capture the checkpoint and the replay point in buffers of their own: "
          "checkpoints " + std::to_string(counts.checkpointPublications) + ", replay points " +
              std::to_string(counts.replayStatePublications) + ", state bytes " +
              std::to_string(passes.model.heldStateBytes));
}

// A checkpoint's claim is worth the rows since the lane's recovery point
// before it, which a recovery would compute again without it (smoke a,
// finding 16): on 8192-row commands a 20K prompt's checkpoint at 8192 the
// rows from 0, not from the checkpoint at 4096 its command passed; a second
// conversation that restores the first's checkpoint at 16384 and shares no
// junction above it, its checkpoint at 20480 the rows from 16384, not from
// 0. So the second, 4096 rows, is worth well under the first, 8192 rows.
void testACheckpointIsWorthTheRowsSinceTheLanesRecoveryPoint() {
  PassesUnderALimit passes(1ULL << 30, 4096);
  std::map<std::pair<uint64_t, uint32_t>, double> benefits;
  passes.model.reservationObserver = [&] {
    for (const auto &[_, state] : passes.host.governor.ledger().claims())
      if (state.spec.requestId == passes.model.lastReservation.first && state.spec.cls == ClaimClass::Optional &&
          std::isfinite(state.spec.benefitSeconds))
        benefits[passes.model.lastReservation] = state.spec.benefitSeconds;
  };
  // Both conversations share their first 17,392 tokens, the second's
  // junction less than a draft window past the first's checkpoint at 16384.
  for (const auto &[id, tokens] : {std::pair<uint64_t, uint32_t>{1, 20000}, {2, 24000}}) {
    std::vector<uint32_t> prompt(tokens, 1);
    std::fill(prompt.begin() + 17392, prompt.end(), static_cast<uint32_t>(id) + 1);
    EngineRequest value = request(id, std::move(prompt));
    value.deadlineMilliseconds = 1e9;
    passes.engine.submit(std::move(value));
    passes.runUntil([&] { return idle(passes.engine); }, "a conversation did not end");
  }
  require(passes.events.starts.back() == 16384, "the second conversation did not restore the checkpoint at 16384");
  const auto worth = [&](uint64_t id, uint32_t tokens) {
    const auto found = benefits.find({id, tokens});
    return found == benefits.end() ? -1.0 : found->second;
  };
  require(worth(1, 8192) > 0.0 && worth(2, 20480) > 0.0 && 1.5 * worth(2, 20480) < worth(1, 8192),
          "the checkpoint at 8192 was worth " + std::to_string(worth(1, 8192)) + " s and the one at 20480 after "
              "the restore at 16384 " + std::to_string(worth(2, 20480)) + " s");
}

// amend/M.md B8 (S-9): the ledger prices a byte the expert store misses in a
// prefill at the scheduler's fitted rate of a byte read, not at what an
// expected byte adds to a prefill's price, which the share of expected
// bytes units read scales. On the M6 profile, its prefills reading half the
// expert bytes they expect (amend/I.md F1: 0.46-0.59), the rate the control
// pass gives the ledger is the fitted rate, not half of it.
void testAMissedExpertByteIsPricedAtTheFittedRate() {
  test::EngineHarness harness({.pages = 1024, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2});
  test::seedAsStartup(harness.engine, harness.model, test::kM6FlashNextQ2);
  harness.model.prefillReadShare = 0.5;
  EngineRequest prompt = harness.request(1, std::vector<uint32_t>(8 * 2048, 1));
  prompt.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(prompt));
  harness.runUntilIdle();
  harness.engine.refreshMemoryRates(harness.now());
  const CostTermSnapshot weights =
      harness.engine.snapshot().scheduler.prefillEstimate.terms[static_cast<size_t>(CostTerm::WeightReadBytes)];
  const double priced = harness.governor->ledger().rates().weightReadSecondsPerByte * 1000.0;
  require(std::abs(weights.realizedPerExpected - 0.5) < 0.01, "the prefills did not read half the expected bytes");
  require(weights.millisecondsPerUnit > 0.0 && std::abs(priced - weights.millisecondsPerUnit) <
                                                   1e-6 * weights.millisecondsPerUnit,
          "a missed expert byte was priced " + std::to_string(priced * 1e9) + " ms per GB against the fitted " +
              std::to_string(weights.millisecondsPerUnit * 1e9) + " ms per GB of bytes read");
}

// Each engine built on a governor gives its ledger the cache once, and its
// own lanes only while it lives: a second engine over the same cache, as
// backend_benchmark builds one beside the bootstrap's, leaves the cache's
// units counted once, and an engine that ended leaves nothing of its own.
void testEnginesOnOneGovernorOfferTheCacheOnce() {
  test::GovernedHost host;
  test::TestKvStorage storage{64, 100, 4};
  KvPool pool{storage, 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  Events events;
  // A cached chain of two extents, one of which the host's claim may take.
  const std::vector<uint32_t> prompt(8 * KvCache::pageTokens, 9);
  cache.beginRequest(900);
  require(cache.ensureTokens(900, prompt.size()).granted(), "the cached chain's KV was not allocated");
  static_cast<void>(test::publishBlocks(cache, 900, prompt, static_cast<uint32_t>(prompt.size())));
  cache.endRequest(900);
  const ClaimSpec spec{.cls = ClaimClass::Host, .holdSeconds = kValueHorizonSeconds};
  const auto found = [&] { return host.governor.available(spec, std::numeric_limits<uint64_t>::max()); };
  engine::Engine first{host.connect({}), cache, model, events};
  const uint64_t once = found();
  require(once != 0, "the cache offered the host nothing");
  {
    engine::Engine second{host.connect({}), cache, model, events};
    require(found() == once, "a second engine over the cache made its units count twice: " + std::to_string(found()) +
                                 " bytes for " + std::to_string(once));
  }
  require(found() == once, "an engine that ended left what it registered with the ledger");
}

// A junction is worth the prefill it saves a later request that branches
// there, the rows since the state boundary before it, as a checkpoint is
// worth what a recovery computes again without it: refused its buffers at
// the engine's limit, its capture takes nothing that costs more than that
// (amend/M.md B2: Optional claims take pool units while they cost less than
// their benefit), neither a unit of the expert store nor a state of a
// conversation whose turns resume from it, worth its rows at their reuse.
void testAJunctionTakesNothingWorthMoreThanItSaves() {
  test::GovernedHost host;
  test::TestKvStorage storage(512, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model(1);
  Events events;
  engine::Engine engine(host.connect({}), cache, model, events);
  guardReleases(storage, engine);
  test::seedAsStartup(engine, model);
  test::FakeElasticPool experts("experts", host.memory, 1ULL << 20, 0, 2, 4, 2, 0.25, 0.0);
  MemoryLedger &ledger = host.governor.ledger();
  std::string decisions;
  ledger.setDecisionLog([&](std::string_view line) { decisions += std::string(line) + "\n"; });
  static_cast<void>(ledger.addPool(experts));
  // Decode misses the store's units would serve, many a second.
  for (double seconds = 0.5; seconds <= 10.0; seconds += 0.5) {
    experts.served(0, 1'000'000, 0, 0);
    ledger.observePools(seconds, true);
  }
  // One prefill chunk every prompt shares, then a conversation of its own:
  // a branch off the cached conversation plans a junction at 2048.
  const auto branch = [](uint32_t token, uint32_t own) {
    std::vector<uint32_t> prompt(model::ExecutionLimits::prefillTokenBudget);
    std::iota(prompt.begin(), prompt.end(), 1);
    prompt.resize(prompt.size() + own, token);
    return prompt;
  };
  // The conversation, then three turns that resume from its state at 6144,
  // the last decoding meanwhile, its state in use: three of the four
  // probes hit, which the control pass's rates price its states by.
  const std::vector<uint32_t> conversation = branch(777, 4097);
  double now = 1;
  engine.refreshMemoryRates(now);
  for (const uint64_t id : {5, 8, 9}) {
    EngineRequest value = request(id, conversation);
    value.deadlineMilliseconds = 1e9;
    engine.submit(std::move(value));
    test::tickUntil(engine, now, [&] { return idle(engine); }, "the conversation did not end");
  }
  model.decodeFinishes = false;
  EngineRequest turn = request(7, conversation);
  turn.maxNewTokens = 200;
  turn.deadlineMilliseconds = 1e9;
  engine.submit(std::move(turn));
  test::tickUntil(engine, now, [&] { return events.outputs.contains(7); }, "the next turn did not decode");
  engine.refreshMemoryRates(now);
  const StateCacheSnapshot before = cache.snapshot().stateCache;
  // The junction's buffers ask the engine's limit for more than the room.
  bool junction = true;
  model.reservationRoom = [&] {
    return !std::exchange(junction, false) || static_cast<bool>(host.reserve(512 * 1024));
  };
  host.leaveRoom(0);
  EngineRequest branched = request(6, branch(888, 65));
  branched.deadlineMilliseconds = 1e9;
  engine.submit(std::move(branched));
  test::tickUntil(engine, now, [&] { return idle(engine); }, "the requests did not end");
  host.leaveRoom(std::nullopt);
  const StateCacheSnapshot after = cache.snapshot().stateCache;
  require(!junction, "the branch planned no junction");
  require(experts.shrinks == 0 && after.evictions == before.evictions && events.failedCount == 0 &&
              events.completedCount == 5,
          "a junction's capture took what is worth more than the prefill it saves: " +
              std::to_string(experts.shrinks) + " store units, " + std::to_string(after.evictions - before.evictions) +
              " states\n" + decisions);
}

// The cache's reuse, which prices every unit not certain to be reused, is the
// probes' hit share over the rates' time constant (design/M.md section 4.3;
// kRateTimeConstantSeconds), not over the server's life: after ten minutes
// of turns that each hit the cache, a minute of prompts that each miss it
// leaves e^-1 of the share, as a workload that moved on would have it.
void testCacheReuseFollowsTheRecentProbes() {
  test::TestKvStorage storage(256, 4096, 4);
  KvPool pool(storage, 0);
  engine::Cache cache(pool, nullptr, nullptr);
  Executor model(1);
  Events events;
  const EngineConfig config = test::engineConfig();
  engine::Engine engine(config, cache, model, events);
  guardReleases(storage, engine);
  double now = 1;
  uint64_t id = 0;
  // One request every 10 s, the control pass's rates refreshed after each.
  const auto serve = [&](uint32_t token) {
    EngineRequest value = request(++id, std::vector<uint32_t>(65, token));
    value.deadlineMilliseconds = 1e12;
    engine.submit(std::move(value));
    test::tickUntil(engine, now, [&] { return idle(engine); }, "a request did not finish");
    now += 10'000;
    engine.refreshMemoryRates(now);
  };
  for (int turn = 0; turn <= 60; ++turn)
    serve(7);
  for (uint32_t miss = 0; miss < 6; ++miss)
    serve(1000 + miss);
  const double reuse = config.governor->ledger().rates().reuse;
  require(engine.snapshot().cacheHits == 60 && std::abs(reuse - std::exp(-1.0)) < 0.02,
          "the cache's reuse did not follow the recent probes: " + std::to_string(reuse) + " after " +
              std::to_string(engine.snapshot().cacheHits) + " hits, where e^-1 is 0.368");
}

// AMENDMENTS Borrowed 7: a request's time to its first token splits into
// what held it, each interval once, by precedence: a prefill command in
// flight with its rows, its claim waiting, a decode command, another
// request's prefill command. On the harness clock with the M6 profile, two
// prompts arrive while a third request decodes; each prompt's parts are the
// time its own prefill commands, the decode commands and the other prompt's
// commands ran between its arrival and its first token, and they leave out
// only the time no command ran (the rest, which the native runtime takes as
// the TTFT less them).
void testTtftPartsSumToTtft() {
  test::EngineHarness harness({.pages = 256, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 64;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the decoder did not decode");
  const double arrived = harness.now();
  for (const uint64_t id : {2, 3}) {
    EngineRequest prompt = harness.request(id, std::vector<uint32_t>(3000, static_cast<uint32_t>(id)));
    prompt.maxNewTokens = 2;
    prompt.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(prompt));
  }
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2) && harness.events.tokenTimes.contains(3); },
                   "the prompts did not answer");
  for (const uint64_t id : {2, 3}) {
    const double first = harness.events.tokenTimes.at(id).front();
    TtftParts expected{};
    for (const test::EngineHarness::Command &command : harness.commands()) {
      const double from = std::max(command.startMilliseconds, arrived);
      const double to = std::min(command.startMilliseconds + command.wallMilliseconds, first);
      if (to <= from)
        continue;
      const bool own = std::ranges::find(command.lanes, id) != command.lanes.end();
      const TtftPart part = command.kind == WorkKind::Decode ? TtftPart::Decode
                            : own                            ? TtftPart::OwnPrefill
                                                             : TtftPart::OtherPrefill;
      expected[static_cast<size_t>(part)] += to - from;
    }
    const TtftParts &parts = harness.events.ttftParts.at(id);
    double counted = 0.0;
    for (size_t part = 0; part < kTtftParts; ++part) {
      require(std::abs(parts[part] - expected[part]) < 1e-6,
              "request " + std::to_string(id) + "'s TTFT part " + std::to_string(part) + " was " +
                  std::to_string(parts[part]) + " ms, not the " + std::to_string(expected[part]) +
                  " ms its commands ran");
      counted += parts[part];
    }
    require(parts[static_cast<size_t>(TtftPart::OwnPrefill)] > 0.0 &&
                parts[static_cast<size_t>(TtftPart::Decode)] > 0.0 && counted <= first - arrived + 1e-6,
            "the TTFT parts did not cover the commands that held the prompt within its TTFT");
  }
}

// AMENDMENTS Borrowed 7: a request that waits to retry a start the host
// refused waits on its claim, as one whose claim waits for memory on its
// way back does: the time counts as its claim's wait, not as the decode
// commands' that ran beside it. With the M6 profile, a short prompt whose
// lane's state takes buffers of its own arrives while a lane decodes, the
// host's headroom short of the margin; other applications give memory back
// 2 s later, and the prompt then starts.
void testAHostRefusedStartWaitsOnItsClaim() {
  test::EngineHarness harness({.pages = 256, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  harness.model.laneFootprintBytes = 64 << 10;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 100'000;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the decoder did not decode");
  const test::GovernorLimits limits;
  const uint64_t other = harness.memory->hostBytes() - limits.hostSpareReserveBytes - harness.memory->total();
  harness.memory->setHost(other - kHostMarginBytes + (1ULL << 20), MemoryPressure::Normal);
  const double arrived = harness.now();
  EngineRequest prompt = harness.request(2, std::vector<uint32_t>(300, 2));
  prompt.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(prompt));
  harness.runUntil([&] { return harness.now() >= arrived + 2000.0; }, "the clock did not move");
  require(!harness.events.tokenTimes.contains(2), "the host admitted a start short of its margin");
  harness.memory->setHost(other - 4 * kHostMarginBytes, MemoryPressure::Normal);
  const double released = harness.now();
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2); }, "the prompt did not answer");
  const TtftParts &parts = harness.events.ttftParts.at(2);
  require(parts[static_cast<size_t>(TtftPart::ClaimWait)] > 0.5 * (released - arrived),
          "a start the host refused waited " + std::to_string(released - arrived) + " ms, of which " +
              std::to_string(parts[static_cast<size_t>(TtftPart::ClaimWait)]) + " ms counted as its claim's wait and " +
              std::to_string(parts[static_cast<size_t>(TtftPart::Decode)]) + " ms as decode commands'");
}

// A start beside a decoding lane may release a cache the model rebuilds on
// use that nothing holds (owner 11-03 00:35, amending amend/M.md B2; smoke
// issue 20): a request beside a text request's decode, under the engine's
// limit, whose start holds none of the idle vision arena (a text request,
// or an image whose encode needs a larger arena than the idle one, which
// Runtime::activate does not keep), starts by giving the arena back, as
// 1.5.0 freed it first, not once the decode ended; the store, whose units
// the decode reads, stays out of a start's reach though a unit of it costs
// no more (B2, A103). The limit leaves the text lane its growth and less
// than a lane's state beside it.
void testAStartReleasesAnIdleArenaBesideADecodingLane() {
  constexpr uint64_t kExtent = 4 * 4096, kState = 64 * 1024, kChunk = 64 * 1024, kArena = 256 * 1024;
  test::EngineHarness harness(
      {.pages = 64,
       .memory = test::FakeMemory::Config{},
       .limits = {.limitBytes = 3 * kChunk + kArena + kState + 2 * kExtent + kExtent}});
  // The store: three units, one its floor, of the four that would hold every
  // expert, which nothing misses. The arena: one unit, which the next image's
  // start builds again.
  test::FakeElasticPool experts("experts", *harness.memory, kChunk, 1, 3, 4, 3, 0.25, 0.0);
  test::FakeElasticPool arena("vision_arena", *harness.memory, kArena, 0, 1, 1, 1, 5.0, 0.0);
  arena.rebuildOnly = true;
  static_cast<void>(harness.governor->ledger().addPool(experts));
  static_cast<void>(harness.governor->ledger().addPool(arena));
  harness.model.laneFootprintBytes = kState;
  harness.model.decodeFinishes = false;
  EngineRequest text = harness.request(1, std::vector<uint32_t>(40, 1));
  text.maxNewTokens = 30;
  text.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(text));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the text request did not decode");
  EngineRequest second = harness.request(2, std::vector<uint32_t>(16, 2));
  second.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(second));
  harness.runUntil([&] { return harness.events.outputs.contains(2); }, "the second request did not answer");
  require(!harness.events.usage.contains(1) && arena.shrinks == 1,
          "the second request started once the text request's decode ended (" +
              std::to_string(harness.events.outputs[1].size()) + " of its 30 tokens), not by giving back the "
              "idle arena (" + std::to_string(arena.shrinks) + " units)");
  require(experts.shrinks == 0, "a start took a unit of the store");
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && experts.shrinks == 0,
          "the text request did not finish beside the second request");
}

// Smoke issue 20's trigger (owner 11-03 00:35): an image request beside a
// text request's decode, under the engine's limit. Its refused activation
// keeps the encoder its image fits (Runtime::activate's Matched, which the
// engine holds while it frees for the start and tries again), so that arena
// is the start's own and stays. The start gives back the embedding rows
// nothing holds, which an encode rebuilds, and answers before the decode
// ends; the store, whose units the decode reads, keeps its units though one
// costs no more (B2, A103). The limit leaves the text lane its growth and
// less than a lane's state beside it.
void testAnImageStartKeepsItsArenaAndGivesBackIdleRows() {
  constexpr uint64_t kExtent = 4 * 4096, kState = 64 * 1024, kChunk = 64 * 1024, kCache = 256 * 1024;
  test::EngineHarness harness(
      {.pages = 64,
       .memory = test::FakeMemory::Config{},
       .limits = {.limitBytes = 3 * kChunk + 2 * kCache + kState + 2 * kExtent + kExtent}});
  // The store as testAStartReleasesAnIdleArenaBesideADecodingLane's. The
  // arena and the rows: one unit each, built again by an image's start and
  // encode.
  test::FakeElasticPool experts("experts", *harness.memory, kChunk, 1, 3, 4, 3, 0.25, 0.0);
  test::FakeElasticPool arena("vision_arena", *harness.memory, kCache, 0, 1, 1, 1, 5.0, 0.0);
  test::FakeElasticPool rows("embeddings", *harness.memory, kCache, 0, 1, 1, 1, 5.0, 0.0);
  arena.rebuildOnly = rows.rebuildOnly = true;
  static_cast<void>(harness.governor->ledger().addPool(experts));
  static_cast<void>(harness.governor->ledger().addPool(arena));
  static_cast<void>(harness.governor->ledger().addPool(rows));
  harness.model.laneFootprintBytes = kState;
  harness.model.decodeFinishes = false;
  // The image request's activation holds the arena from its first attempt.
  harness.model.beginObserver = [&] {
    if (harness.model.lastBeginId == 2) arena.hold(1);
  };
  EngineRequest text = harness.request(1, std::vector<uint32_t>(40, 1));
  text.maxNewTokens = 30;
  text.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(text));
  harness.runUntil([&] { return harness.events.outputs.contains(1); }, "the text request did not decode");
  EngineRequest image = harness.request(2, std::vector<uint32_t>(16, 2));
  image.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(image));
  harness.runUntil([&] { return harness.events.outputs.contains(2); }, "the image request did not answer");
  require(!harness.events.usage.contains(1) && rows.shrinks == 1,
          "the image request started once the text request's decode ended (" +
              std::to_string(harness.events.outputs[1].size()) + " of its 30 tokens), not by giving back the "
              "embedding rows nothing holds (" + std::to_string(rows.shrinks) + " units)");
  require(arena.shrinks == 0 && experts.shrinks == 0, "a start took the arena its image holds or a unit of the store");
  harness.runUntilIdle();
  require(harness.events.completedCount == 2 && harness.events.failedCount == 0 && experts.shrinks == 0,
          "the text request did not finish beside the image request");
}

// AMENDMENTS Borrowed 4: a decode gap past the scheduler's bound counts
// once, under the first cause that held in it, and a gap within the bound
// nowhere. On the harness clock with the M6 profile a lane decodes alone,
// its gaps a decode command each, within the bound; then a long prompt's
// prefill units run between its decode commands, and the gaps they make
// past the bound count under a prefill unit, with the engine's price of
// those units and their time.
void testADecodeGapNamesItsCause() {
  test::EngineHarness harness({.pages = 1024, .timing = &test::kM6FlashNextQ2});
  harness.model.decodeFinishes = false;
  EngineRequest decoder = harness.request(1, std::vector<uint32_t>(40, 1));
  decoder.maxNewTokens = 4000;
  decoder.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(decoder));
  harness.runUntil([&] { return harness.events.outputs[1].size() > 8; }, "the decoder did not decode");
  const auto causes = [&] { return harness.engine.snapshot().decodeGapCauses; };
  require(causes().gaps == std::array<uint64_t, kGapCauses>{} &&
              causes().milliseconds == std::array<double, kGapCauses>{},
          "a decode gap within the bound counted under a cause");
  EngineRequest prompt = harness.request(2, std::vector<uint32_t>(16384, 2));
  prompt.maxNewTokens = 1;
  prompt.deadlineMilliseconds = 1e9;
  harness.engine.submit(std::move(prompt));
  harness.runUntil([&] { return harness.events.tokenTimes.contains(2); }, "the prompt did not answer");
  const DecodeGapCausesSnapshot counted = causes();
  const auto at = [](GapCause cause) { return static_cast<size_t>(cause); };
  const double target = harness.engine.snapshot().scheduler.decodeGapTargetMilliseconds;
  require(counted.gaps[at(GapCause::PrefillUnit)] > 0 &&
              counted.milliseconds[at(GapCause::PrefillUnit)] > target * counted.gaps[at(GapCause::PrefillUnit)] &&
              counted.prefillPredictedMilliseconds > 0.0 && counted.prefillMilliseconds > 0.0,
          "the decode gaps the prompt's prefill units made past the bound did not count under a prefill unit: " +
              std::to_string(counted.gaps[at(GapCause::PrefillUnit)]) + " gaps");
  for (const GapCause cause : {GapCause::ClaimWait, GapCause::SsdRead, GapCause::PoolRelease, GapCause::Replan,
                               GapCause::Other})
    require(!counted.gaps[at(cause)], "a decode gap counted under a cause that did not hold in it");
}

// AMENDMENTS Borrowed 4 (plan R3: an SSD wait is a history-stream or
// expert-read wait in the sample): a decode command that reads experts the
// store missed waits on the SSD, and a gap past the bound it ends counts as
// an SSD wait, for every lane of it, whether or not the lane holds Index
// pages. With the M6 profile and a store that holds a 32nd of the experts,
// four Resident lanes decode in one command past the bound, each command
// reading the experts its rows route that the store lacks.
void testADecodeGapOnExpertReadsIsAnSsdWait() {
  test::EngineHarness harness({.pages = 1024, .memory = test::FakeMemory::Config{}, .timing = &test::kM6FlashNextQ2});
  test::FakeElasticPool experts("experts", *harness.memory, 1ULL << 20, 1, 1, 32, 1, 2.5, 0.0);
  harness.model.experts = &experts;
  test::seedAsStartup(harness.engine, harness.model, test::kM6FlashNextQ2);
  harness.model.decodeFinishes = false;
  for (uint64_t id = 1; id <= 4; ++id) {
    EngineRequest decoder = harness.request(id, std::vector<uint32_t>(40, static_cast<uint32_t>(id)));
    decoder.maxNewTokens = 64;
    decoder.deadlineMilliseconds = 1e9;
    harness.engine.submit(std::move(decoder));
  }
  harness.runUntilIdle();
  const EngineSnapshot snapshot = harness.engine.snapshot();
  const DecodeGapCausesSnapshot &counted = snapshot.decodeGapCauses;
  const auto at = [](GapCause cause) { return static_cast<size_t>(cause); };
  const double bound = std::max(
      snapshot.scheduler.decodeGapTargetMilliseconds,
      snapshot.scheduler.prefillEstimate.terms[static_cast<size_t>(CostTerm::Units)].millisecondsPerUnit /
          kPassOverheadShare);
  require(harness.longestDecodeGap() > bound && snapshot.scheduler.decodeBatchesByWidth[3] > 1,
          "four lanes did not decode in commands longer than the bound, " + std::to_string(bound) + " ms");
  require(counted.gaps[at(GapCause::SsdRead)] > 0 && !counted.gaps[at(GapCause::Other)] &&
              !counted.gaps[at(GapCause::Replan)] && !counted.gaps[at(GapCause::PoolRelease)],
          "decode gaps past the bound that commands reading missed experts ended counted " +
              std::to_string(counted.gaps[at(GapCause::SsdRead)]) + " times as an SSD wait, " +
              std::to_string(counted.gaps[at(GapCause::PoolRelease)]) + " as a pool release, " +
              std::to_string(counted.gaps[at(GapCause::Replan)]) + " as a replan and " +
              std::to_string(counted.gaps[at(GapCause::Other)]) + " under no cause");
}

} // namespace

std::span<const test::TestCase> rulesTests() {
  static constexpr test::TestCase kTests[] = {
      {"testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds",
       testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds},
      {"testASpillLanesIndexPagesAreClaimedAtItsPriority", testASpillLanesIndexPagesAreClaimedAtItsPriority},
      {"testSupersedingClaimTakesThePassCommand", testSupersedingClaimTakesThePassCommand},
      {"testLandingPaysItsOwnerFirst", testLandingPaysItsOwnerFirst},
      {"testQuotaWaitIsNotAMemoryClaim", testQuotaWaitIsNotAMemoryClaim},
      {"testConversionVsStoreFollowsValue", testConversionVsStoreFollowsValue},
      {"testPlacementOncePerCandidatePerState", testPlacementOncePerCandidatePerState},
      {"testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages",
       testALaneBelowAWaitingClaimTakesWhatItLacksBeyondTheHeldPages},
      {"testARestoredStateIsPromotedInAClaim", testARestoredStateIsPromotedInAClaim},
      {"testAReplayPointBesideACheckpointIsReservedInAClaim", testAReplayPointBesideACheckpointIsReservedInAClaim},
      {"testACheckpointIsWorthTheRowsSinceTheLanesRecoveryPoint",
       testACheckpointIsWorthTheRowsSinceTheLanesRecoveryPoint},
      {"testAMissedExpertByteIsPricedAtTheFittedRate", testAMissedExpertByteIsPricedAtTheFittedRate},
      {"testEnginesOnOneGovernorOfferTheCacheOnce", testEnginesOnOneGovernorOfferTheCacheOnce},
      {"testAJunctionTakesNothingWorthMoreThanItSaves", testAJunctionTakesNothingWorthMoreThanItSaves},
      {"testCacheReuseFollowsTheRecentProbes", testCacheReuseFollowsTheRecentProbes},
      {"testTtftPartsSumToTtft", testTtftPartsSumToTtft},
      {"testAHostRefusedStartWaitsOnItsClaim", testAHostRefusedStartWaitsOnItsClaim},
      {"testAStartReleasesAnIdleArenaBesideADecodingLane", testAStartReleasesAnIdleArenaBesideADecodingLane},
      {"testAnImageStartKeepsItsArenaAndGivesBackIdleRows", testAnImageStartKeepsItsArenaAndGivesBackIdleRows},
      {"testADecodeGapNamesItsCause", testADecodeGapNamesItsCause},
      {"testADecodeGapOnExpertReadsIsAnSsdWait", testADecodeGapOnExpertReadsIsAnSsdWait},
      {"testACancelBetweenPassesEndsBesideAHigherPrioritysDecode",
       testACancelBetweenPassesEndsBesideAHigherPrioritysDecode},
      {"testAHostRefusedStartIsFreedForByThePass", testAHostRefusedStartIsFreedForByThePass},
      {"testAStartRanksByTheWorkItHolds", testAStartRanksByTheWorkItHolds},
  };
  return kTests;
}
