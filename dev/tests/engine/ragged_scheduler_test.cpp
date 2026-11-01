#include "TestChecks.hpp"
#include "engine/Scheduler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

engine::RequestSpec
request(uint64_t id, uint32_t prompt,
        RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, prompt};
}

void completePrefill(engine::Scheduler &scheduler,
                     const BatchPlan &plan, double wallMilliseconds = 0.0,
                     bool representativePrefillTiming = true) {
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items) {
    results.push_back(
        {item.requestId, item.tokenCount, false, DecodeStage::Regular});
  }
  scheduler.commit(plan, {});
  scheduler.complete(plan, results, wallMilliseconds,
                      representativePrefillTiming);
}

void completeDecode(engine::Scheduler &scheduler, bool finished = false,
                    double wallMilliseconds = 0.0) {
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode, "expected a decode command");
  scheduler.commit(plan, {});
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, 0, finished, DecodeStage::Regular});
  scheduler.complete(plan, results, wallMilliseconds, true);
}

// Prefills a constrained request's prompt, whose end waits for the mask of
// its first token.
void awaitFirstMask(engine::Scheduler &scheduler, uint64_t id) {
  const BatchPlan prompt = *scheduler.next({});
  require(prompt.kind == WorkKind::Prefill && prompt.width() == 1 &&
              prompt.items[0].requestId == id,
          "expected the constrained prompt");
  scheduler.commit(prompt, {});
  const std::array result{StepResult{id, prompt.items[0].tokenCount, false,
                                     DecodeStage::ApplyInitialMask}};
  scheduler.complete(prompt, result, 0.0, true);
  require(scheduler.phase(id) == engine::Phase::WaitingMask,
          "a constrained prompt did not wait for its first mask");
}

void testAdmissionSharesDispatchOrderAndBudget() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 8193));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 8193));
  scheduler.submit(request(3, 4097));
  scheduler.submit(request(4, 65));
  const std::array candidates{PrefillAdmission{2, 0},
                              PrefillAdmission{3, 4096},
                              PrefillAdmission{4, 0}};
  require(scheduler.prefillAdmissionOrder(candidates) ==
              std::vector<uint64_t>({3, 4}),
          "admission did not take cached and short work ahead of cold work");
  scheduler.resourcesReady(3, 4096);
  scheduler.resourcesReady(4, 0);
  const auto plan = *scheduler.next({});
  require(plan.items.size() == 3 && plan.items[0].requestId == 3 &&
              plan.items[1].requestId == 4 && plan.items[2].requestId == 1 &&
              plan.items[2].tokenCount == 1982,
          "dispatch disagreed with admission work accounting");

  Scheduler shortPrompts(0.0);
  std::vector<PrefillAdmission> many;
  for (uint64_t id = 1; id <= 8; ++id) {
    shortPrompts.submit(request(id, 65));
    many.push_back({id, 0});
  }
  require(shortPrompts.prefillAdmissionOrder(many) ==
              std::vector<uint64_t>({1, 2, 3, 4}),
          "short prefill admission lost batching or exceeded the real width");
}

void testAdmissionRespectsContendedBudgetAndDecodePriority() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 6144.0);
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(request(id, 65));
    candidates.push_back({id, 0});
  }
  require(scheduler.prefillAdmissionOrder(candidates) ==
              std::vector<uint64_t>({1, 2}),
          "admission ignored the contended actual-row budget");
  scheduler.submit(request(5, 1, RequestPriority::Foreground));
  scheduler.resourcesReady(5, 1);
  require(scheduler.prefillAdmissionOrder(candidates).empty(),
          "lower-priority prefill reserved lanes ahead of runnable foreground decode");
  scheduler.cancel(5);
  require(!scheduler.prefillAdmissionOrder(candidates).empty(),
          "prefill admission did not resume after foreground decode left");
}

// The tier admission can select is set by the lanes that prefill or decode,
// not by those waiting for a mask or for admission.
void testHighestRunnablePriority() {
  Scheduler scheduler(0.0);
  require(!scheduler.highestRunnablePriority(), "an idle scheduler had a runnable tier");
  scheduler.submit(request(1, 1, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.submit(request(2, 1, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  scheduler.submit(request(3, 100));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 100, RequestPriority::Foreground));
  require(scheduler.phase(1) == Phase::WaitingMask &&
              scheduler.highestRunnablePriority() == RequestPriority::Normal,
          "a mask wait or a queued request set the runnable tier");
}

void testQueuedPrefillCannotBeOvertakenIndefinitely() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 8193));
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 2048));
    const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{id, 0}};
    require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{id},
            "short work did not overtake queued long work");
    scheduler.resourcesReady(id, 0);
    completePrefill(scheduler, *scheduler.next({}));
    scheduler.cancel(id);
    scheduler.remove(id);
  }
  scheduler.submit(request(5, 65));
  const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{5, 0}};
  require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{1},
          "queued long prefill starved behind short arrivals");
}

// A prefill command's attention stays within a full chunk's at the end of a
// 262,144-token native window: below it every command takes a whole chunk;
// past it, at depths only YaRN serves, a lane's rows halve until they fit
// (512 at a million tokens, after a cached prefix too), and another deep lane
// joins the command only within what is left. Without a bound every command
// takes a whole chunk.
void testDeepPrefillKeepsANativeChunksAttention() {
  const uint64_t native = Scheduler::attentionPairs(262'144 - 2048, 2048);
  const auto rows = [](uint64_t bound, std::initializer_list<uint32_t> positions) {
    Scheduler scheduler(0.0, bound);
    uint64_t id = 0;
    for (const uint32_t position : positions) {
      scheduler.submit(request(++id, 1'010'000));
      scheduler.resourcesReady(id, position);
    }
    const BatchPlan plan = *scheduler.next({});
    std::vector<uint32_t> result;
    for (const BatchItem &item : plan.items)
      result.push_back(item.tokenCount);
    return result;
  };
  using Rows = std::vector<uint32_t>;
  require(rows(native, {0}) == Rows{2048} && rows(native, {260'096}) == Rows{2048} &&
              rows(native, {300'000}) == Rows{1024} && rows(native, {1'000'000}) == Rows{512} &&
              rows(std::numeric_limits<uint64_t>::max(), {1'000'000}) == Rows{2048},
          "a prefill command's attention left a native chunk's");
  require(rows(native, {600'000, 600'000}) == Rows({512, 192}),
          "a second deep lane overran the command's attention");
}

// Every lane but a prefill command's last lies within its first chunk, so
// that at most one lane's rows run past a chunk however the engine cuts or
// drops lanes, and a command holds at most each lane's group rows: a deep
// group leader whose rows halve to more than a chunk runs alone, one halved
// to at most a chunk leaves the rest of its group to a peer, a lane without
// groups (a prompt with images) joins only a command of at most a chunk, and
// no estimate, depth or peer breaks either.
void testPrefillGroupHasOneSpanningLane() {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  const uint64_t native = Scheduler::attentionPairs(262'144 - 2048, 2048);
  // A command whose 2048 rows took wallMilliseconds, hostMilliseconds of
  // it waiting on expert reads; none before the first command.
  struct Sample final {
    double wallMilliseconds = 0.0;
    double hostMilliseconds = 0.0;
  };
  // A lane with `remaining` rows after `position`, of 8192-row groups or,
  // like a prompt with images, of chunks.
  struct Lane final {
    uint32_t position = 0;
    uint32_t remaining = 0;
    uint32_t groupRows = 4 * chunk;
  };
  using Rows = std::vector<uint32_t>;
  // The rows of the first command's lanes, and their group rows.
  struct Planned final {
    Rows rows;
    Rows groupRows;
  };
  const auto plan = [&](Sample sample, std::initializer_list<Lane> lanes) {
    Scheduler scheduler(0.0, native);
    if (sample.wallMilliseconds > 0.0)
      scheduler.observePrefill(chunk, sample.wallMilliseconds, sample.hostMilliseconds);
    uint64_t id = 0;
    for (const Lane &lane : lanes) {
      RequestSpec spec = request(++id, lane.position + lane.remaining);
      spec.prefillGroupRows = lane.groupRows;
      scheduler.submit(spec);
      scheduler.resourcesReady(id, lane.position);
    }
    const BatchPlan command = *scheduler.next({});
    Planned result;
    for (const BatchItem &item : command.items) {
      result.rows.push_back(item.tokenCount);
      result.groupRows.push_back(lanes.begin()[item.requestId - 1].groupRows);
    }
    return result;
  };
  // The lanes whose rows run past a chunk as the runtime packs them.
  const auto spanning = [&](const Rows &rows) {
    uint32_t count = 0, offset = 0;
    for (const uint32_t value : rows) {
      count += value && offset / chunk != (offset + value - 1) / chunk;
      offset += value;
    }
    return count;
  };
  const Sample streamed{4900.0, 3400.0};
  // 70K deep, the leader's group of 8192 rows halves to 4096 for its
  // attention; the long peer would have taken the other 4096.
  require(plan(streamed, {{70'000, 20'000}, {0, 40'000}}).rows == Rows{4096},
          "a leader past a chunk shared its command");
  // From 128.5K deep it halves to a chunk, and the peer takes the rest of
  // the group, past the chunk's end.
  require(plan(streamed, {{130'000, 20'000}, {0, 40'000}}).rows == Rows({2048, 6144}),
          "a peer of a leader of one chunk lost the rest of the group");
  // 300K deep it halves to 1024: the peer takes the rest of the group, and
  // the next lane none.
  require(plan(streamed, {{300'000, 20'000}, {0, 40'000}, {0, 50'000}}).rows == Rows({1024, 7168}),
          "a lane joined after one ran past a chunk");
  // A peer of chunks takes only the chunk's other 1024 rows, and keeps the
  // next lane to them too.
  require(plan(streamed, {{300'000, 20'000}, {0, 40'000, chunk}, {0, 50'000}}).rows == Rows({1024, 1024}),
          "a lane without prefill groups ran in a group");
  require(plan(streamed, {{130'000, 20'000}, {0, 40'000, chunk}, {0, 50'000}}).rows == Rows{2048},
          "a lane without prefill groups ran in a group");
  for (const Sample sample : {Sample{}, Sample{600.0}, streamed, Sample{20'000.0, 19'000.0}})
    for (const uint32_t depth : {0u, 70'000u, 150'000u, 255'000u, 300'000u, 600'000u})
      for (const uint32_t remaining : {300u, 3000u, 9000u, 20'000u})
        for (const uint32_t peer : {0u, 100'000u})
          for (const uint32_t peerGroupRows : {chunk, 4 * chunk})
            for (const uint32_t shortest : {1000u, remaining + 1000}) {
              const auto [rows, groupRows] = plan(sample, {{depth, remaining},
                                                           {peer, shortest, peerGroupRows},
                                                           {peer, remaining + 2000},
                                                           {0, remaining + 3000, peerGroupRows}});
              // The engine runs any of the lanes, in order, each with its rows
              // or fewer: here all of them or half.
              for (uint32_t kept = 1; kept < 1u << rows.size(); ++kept)
                for (uint32_t halved = 0; halved < 1u << rows.size(); ++halved) {
                  Rows run;
                  for (uint32_t lane = 0; lane < rows.size(); ++lane)
                    if (kept >> lane & 1)
                      run.push_back(halved >> lane & 1 ? rows[lane] / 2 : rows[lane]);
                  require(spanning(run) <= 1, "several lanes of a prefill command ran past a chunk");
                }
              uint32_t total = 0;
              for (const uint32_t value : rows)
                total += value;
              require(total <= *std::min_element(groupRows.begin(), groupRows.end()),
                      "a prefill command held more than a lane's group rows");
            }
}

void testWarmupTimingSeedsFirstContendedCommand() {
  for (double sample : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN(), 6144.0}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, sample);
    scheduler.observePrefill(1, 10000.0);
    scheduler.submit(request(1, 8193));
    scheduler.resourcesReady(1, 0);
    require(scheduler.next({})->items[0].tokenCount == 2048,
            "warmup shrank an uncontended prefill");
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const auto first = *scheduler.next({});
    require(first.kind == WorkKind::Prefill &&
                first.items[0].tokenCount == (sample == 6144.0 ? 128 : 2048),
            "first contended prefill ignored warmup or accepted invalid timing");
  }
}

// A command's wait on its host steps (a streamed target's expert reads) is
// fixed whatever its rows: the contended slice fits fixed plus per-row cost,
// and grows to twice the fixed part, so a command's rows take at least as
// long as its fixed part and the slice has no step where the fixed part
// alone fills it.
void testFixedPrefillCostSizesContendedSlice() {
  const auto slice = [](double wallMilliseconds, double hostMilliseconds) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, wallMilliseconds, hostMilliseconds);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 1,
            "expected the contended prefill");
    return plan.items[0].tokenCount;
  };
  // 0.3 ms a row after the fixed part.
  const auto wall = [](double fixed) { return fixed + 2048 * 0.3; };
  require(slice(wall(0.0), 0.0) == 1024, "a prefill without host waits lost its per-row slice");
  require(slice(wall(200.0), 200.0) == 512, "the contended slice ignored the fixed prefill cost");
  require(slice(wall(490.0), 490.0) == 1024 && slice(wall(510.0), 510.0) == 1024,
          "the contended slice jumped where the fixed part fills it");
  require(slice(4900.0, 3400.0) == 2048, "a streamed prefill was cut below its fixed cost");
}

// A lane whose commands may be prefill groups takes one alone when nothing
// contends: its remaining chunks split evenly over the fewest groups that
// hold them. Beside a decoding peer it runs chunks, and a lane without groups
// (a model without them, or a request with images) keeps to chunks.
void testPrefillGroupsRunUncontended() {
  const auto first = [](uint32_t prompt, uint32_t groupRows, bool contended) {
    Scheduler scheduler(0.0);
    // A streamed target's command: 3.4 s of its 4.9 s waiting on expert reads.
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    RequestSpec spec = request(1, prompt);
    spec.prefillGroupRows = groupRows;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    if (contended) {
      scheduler.submit(request(2, 1));
      scheduler.resourcesReady(2, 1);
      completeDecode(scheduler);
    }
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1,
            "expected the long prefill alone");
    return plan.items[0].tokenCount;
  };
  require(first(10'000, 8192, false) == 6144, "five chunks did not split into two even prefill groups");
  require(first(8192, 8192, false) == 8192, "four chunks did not run as one prefill group");
  require(first(3000, 8192, false) == 3000, "a prompt of two chunks did not run as one prefill group");
  require(first(2048, 8192, false) == 2048, "a prompt of one chunk did not run as a chunk");
  require(first(10'000, 2048, false) == 2048, "a lane without prefill groups took more than a chunk");
  require(first(10'000, 8192, true) == 2048, "a prefill group ran beside a decoding peer");
}

// A lane of 100,000 rows whose commands may be prefill groups and run in up
// to `passes` passes over its model's layers.
RequestSpec passingLane(uint64_t id, uint32_t passes) {
  RequestSpec spec = request(id, 100'000);
  spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  spec.prefillPasses = passes;
  return spec;
}

// A lane of `remaining` rows past a cached prefix of 20,000, whose commands
// may be prefill groups and run in up to 48 passes, of `priority`.
RequestSpec shortPassingLane(uint64_t id, uint32_t remaining,
                             RequestPriority priority = RequestPriority::Normal) {
  RequestSpec spec = passingLane(id, 48);
  spec.priority = priority;
  spec.prefillTokens = 20'000 + remaining;
  return spec;
}

// Completes a pass of a prefill command, planned without the excluded
// lanes, which consumes its lanes' rows if it is the last, and else none of
// them.
void completePass(engine::Scheduler &scheduler, const BatchPlan &plan, double wallMilliseconds,
                  double hostMilliseconds = 0.0, std::span<const uint64_t> excluded = {}) {
  require(plan.kind == WorkKind::Prefill, "expected a prefill command's pass");
  scheduler.commit(plan, excluded);
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, plan.pass + 1 == plan.passes ? item.tokenCount : 0, false,
                       DecodeStage::Regular});
  scheduler.complete(plan, results, wallMilliseconds, true, hostMilliseconds);
}

// Where its model runs a prefill command in passes over its layers, a lane
// keeps its group whatever contends, and runs it alone in as many passes as
// keep each within the contended slice by the command estimate, up to its
// model's. Its later passes come before any other prefill of its priority,
// with a decoding peer's commands between them; only the last consumes the
// rows, and the estimate learns from the passes' time together. A lane that
// streams its history runs its commands whole, and so does one whose model
// has no passes or before any estimate.
void testPrefillPassesBoundCommands() {
  Scheduler scheduler(0.5);
  // A streamed target's chunk: 4.9 s, 3.4 s of it on expert reads. A group of
  // four chunks: 3.4 s + 4 x 1.5 s = 9.4 s, 19 passes of the slice.
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(passingLane(1, 48));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  scheduler.submit(request(3, 200'000));
  scheduler.resourcesReady(3, 0);
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == 8192 && plan.pass == 0 && plan.passes == 19,
          "a lane whose model runs passes did not run its group alone in passes within the slice");
  for (uint32_t pass = 0; pass < plan.passes; ++pass) {
    require(plan.items[0].requestId == 1 && plan.items[0].tokenCount == 8192 && plan.pass == pass &&
                plan.passes == 19 && scheduler.promptProcessed(1) == 0,
            "a command's next pass did not go on, or an earlier one consumed its rows");
    completePass(scheduler, plan, 500.0, 300.0);
    // The decoding peer runs what each pass owes it, 250 ms.
    completeDecode(scheduler, false, 250.0);
    plan = *scheduler.next({});
  }
  require(scheduler.promptProcessed(1) == 8192 && scheduler.phase(1) == engine::Phase::Prefill &&
              plan.items[0].requestId == 1 && plan.passes == 19 && plan.pass == 0,
          "the last pass did not consume the command's rows, or the estimate learned from each pass alone");
  require(scheduler.snapshot().prefillBatches == 1 && scheduler.snapshot().prefillRows == 8192,
          "a command in passes counted more than once");
  scheduler.setStreamsHistory(1, true);
  plan = *scheduler.next({});
  require(plan.items[0].requestId == 1 && plan.passes == 1, "a lane that streams its history ran passes");

  for (const uint32_t passes : {1u, 48u}) {
    Scheduler fresh(0.0);
    if (passes == 1)
      fresh.observePrefill(2048, 4900.0, 3400.0);
    fresh.submit(passingLane(1, passes));
    fresh.resourcesReady(1, 0);
    require(fresh.next({})->passes == 1, passes == 1 ? "a model without passes ran passes"
                                                     : "a command ran passes before any estimate");
  }
}

// The time of a pass of `plan`, a command of the estimate of a streamed
// target's chunk (3.4 s and 0.73 ms a row, observed as 4.9 s for 2048 rows).
double passMilliseconds(const BatchPlan &plan) {
  uint32_t rows = 0;
  for (const BatchItem &item : plan.items)
    rows += item.tokenCount;
  return (3400.0 + rows * 1500.0 / 2048) / plan.passes;
}

// Short prompts that arrive together on a target whose commands run in
// passes, each a few hundred rows past its cached prefix as an agent's turns
// are, are admitted together and share one command, which reads the experts
// once for all of them: the last of four ends its prompt with the first,
// one command's time after they arrived (3.4 s of expert reads and their
// rows, in passes of the slice), not four commands'.
void testShortPromptsShareACommandInPasses() {
  Scheduler scheduler(0.5);
  // A streamed target's chunk: 4.9 s, 3.4 s of it on expert reads.
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(shortPassingLane(id, 300));
    candidates.push_back({id, 20'000});
  }
  require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>({1, 2, 3, 4}),
          "short prompts on a target whose commands run in passes were admitted one at a time");
  for (uint64_t id = 1; id <= 4; ++id)
    scheduler.resourcesReady(id, 20'000);
  require(scheduler.next({})->width() == 4 && scheduler.next({})->passes == 9 &&
              std::ranges::all_of(scheduler.next({})->items,
                                  [](const BatchItem &item) { return item.tokenCount == 300; }),
          "short prompts did not share one command in passes");
  double now = 0.0;
  std::array<double, 4> prompted{};
  while (std::ranges::any_of(prompted, [](double at) { return at == 0.0; })) {
    const BatchPlan plan = *scheduler.next({});
    completePass(scheduler, plan, passMilliseconds(plan), 3400.0 / plan.passes);
    now += passMilliseconds(plan);
    for (uint64_t id = 1; id <= 4; ++id)
      if (!prompted[id - 1] && scheduler.phase(id) == engine::Phase::Decode)
        prompted[id - 1] = now;
  }
  require(std::ranges::all_of(prompted, [&](double at) { return at == prompted[0]; }) &&
              prompted[0] < 3400.0 + 1200 * 1500.0 / 2048 + 1.0,
          "the last of four short prompts waited for more than one command");
}

// Requests that arrive during a command's first pass, as concurrent agent
// turns do a few milliseconds after the first, are admitted together before
// its next, and the command is planned again with them: the four end their
// prompts together, a pass and one command after the first arrived, rather
// than the first a command before the others. So they do by the M6's
// estimate, 2.6 s fixed and 1.1 ms a row, which a short Flash-Next command
// of 3.0 s on a 24 GB M6 fits. A lane ready late in a command's passes waits
// for its end instead.
void testArrivalsRegroupACommandInPasses() {
  // A command's measured rows, time, and time on host steps.
  struct Sample final {
    uint32_t rows;
    double wallMilliseconds, hostMilliseconds;
  };
  for (const Sample sample : {Sample{2048, 4900.0, 3400.0}, Sample{300, 2964.0, 2632.0}}) {
    // The time of a pass of `plan` by the sample's estimate.
    const auto passTime = [&](const BatchPlan &plan) {
      uint32_t rows = 0;
      for (const BatchItem &item : plan.items)
        rows += item.tokenCount;
      return (sample.hostMilliseconds +
              rows * (sample.wallMilliseconds - sample.hostMilliseconds) / sample.rows) /
             plan.passes;
    };
    const auto complete = [&](Scheduler &scheduler, const BatchPlan &plan) {
      completePass(scheduler, plan, passTime(plan), sample.hostMilliseconds / plan.passes);
    };
    Scheduler scheduler(0.0);
    scheduler.observePrefill(sample.rows, sample.wallMilliseconds, sample.hostMilliseconds);
    scheduler.submit(shortPassingLane(1, 300));
    scheduler.resourcesReady(1, 20'000);
    BatchPlan plan = *scheduler.next({});
    require(plan.width() == 1 && plan.passes > 1, "a short prompt did not run in passes");
    complete(scheduler, plan);
    double now = passTime(plan);
    std::vector<PrefillAdmission> candidates;
    for (uint64_t id = 2; id <= 4; ++id) {
      scheduler.submit(shortPassingLane(id, 300));
      candidates.push_back({id, 20'000});
    }
    require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>({2, 3, 4}),
            "the requests that arrived during a command's pass were not admitted together before its next");
    for (uint64_t id = 2; id <= 4; ++id)
      scheduler.resourcesReady(id, 20'000);
    plan = *scheduler.next({});
    require(plan.pass == 0 && plan.width() == 4 && plan.items[0].requestId == 1 &&
                scheduler.abandonsPasses(plan),
            "a command did not regroup with the lanes ready early in its passes");
    scheduler.abandonPasses();
    double regrouped = now + passTime(plan) * plan.passes;
    std::array<double, 4> prompted{};
    while (std::ranges::any_of(prompted, [](double at) { return at == 0.0; })) {
      plan = *scheduler.next({});
      complete(scheduler, plan);
      now += passTime(plan);
      for (uint64_t id = 1; id <= 4; ++id)
        if (!prompted[id - 1] && scheduler.phase(id) == engine::Phase::Decode)
          prompted[id - 1] = now;
    }
    require(std::ranges::all_of(prompted, [&](double at) { return std::abs(at - regrouped) < 1.0; }),
            "the four prompts did not end together, one command after the first pass");

    Scheduler late(0.0);
    late.observePrefill(sample.rows, sample.wallMilliseconds, sample.hostMilliseconds);
    late.submit(shortPassingLane(1, 300));
    late.resourcesReady(1, 20'000);
    const BatchPlan command = *late.next({});
    for (uint32_t pass = 0; pass + 2 < command.passes; ++pass)
      complete(late, pass ? *late.next({}) : command);
    late.submit(shortPassingLane(2, 300));
    late.resourcesReady(2, 20'000);
    plan = *late.next({});
    require(plan.pass + 2 == command.passes && plan.items == command.items && !late.abandonsPasses(plan),
            "a lane ready late in a command's passes did not wait for its end");
  }
}

// Past a short leader of a command in passes, a lane joins with the fewest
// rows that spare it a command of its own, or not at all: a 5000-row prompt,
// which takes a command of its own whatever it takes, leaves a 300-row one
// alone in its command, and a 9000-row one joins it with the 808 rows that
// leave it one group of 8192, where a 9941-row one, whose 1749 rows would
// pass the 1748 left in the leader's chunk, waits for its own; two 300-row
// prompts share their command without the 5000-row one.
void testPassesShareTheRowsThatSpareACommand() {
  const auto rows = [](std::initializer_list<uint32_t> remaining) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    uint64_t id = 0;
    for (const uint32_t prompt : remaining) {
      scheduler.submit(shortPassingLane(++id, prompt));
      scheduler.resourcesReady(id, 20'000);
    }
    std::vector<uint32_t> result;
    const BatchPlan plan = *scheduler.next({});
    for (const BatchItem &item : plan.items)
      result.push_back(item.tokenCount);
    return result;
  };
  require(rows({300, 5000}) == std::vector<uint32_t>{300},
          "a long prompt joined a short one's command in passes with rows that spared it no command");
  require(rows({300, 9000}) == std::vector<uint32_t>{300, 808},
          "a long prompt did not join a short one's command in passes with the rows that spare it one");
  require(rows({300, 9941}) == std::vector<uint32_t>{300},
          "a long prompt joined a short one's command in passes with more rows than its chunk had left");
  require(rows({300, 300, 5000}) == std::vector<uint32_t>{300, 300},
          "short prompts did not share a command in passes without a long one");
}

// A command between its passes is planned again only as one that holds
// every lane of it with the same rows, beside lanes its first pass was not
// planned without. A 300-row arrival beside a lane's 8192-row group, which
// the command planned again would cut, waits for its end; so does one that
// a full command has no room for, and a lane its first pass was planned
// without, as prepare refusing it memory the others fit in leaves one,
// which a command planned again would leave out too. A lane the first pass
// was planned without while it waited for memory on its way back (excluded)
// joins, as an arrival does.
void testRegroupsKeepTheCommandsLanes() {
  const auto arrive = [](Scheduler &scheduler, uint64_t id, uint32_t rows) {
    scheduler.submit(shortPassingLane(id, rows));
    scheduler.resourcesReady(id, 20'000);
  };
  const auto goesOn = [](const Scheduler &scheduler, const BatchPlan &command) {
    const BatchPlan plan = *scheduler.next({});
    return plan.pass == command.pass + 1 && plan.items == command.items && !scheduler.abandonsPasses(plan);
  };
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(passingLane(1, 48));
    scheduler.resourcesReady(1, 0);
    const BatchPlan command = *scheduler.next({});
    require(command.items[0].tokenCount == 8192 && command.passes == 19, "a lane did not run its group in passes");
    completePass(scheduler, command, 500.0);
    arrive(scheduler, 2, 300);
    require(goesOn(scheduler, command), "a short arrival cut a lane's group in passes, planned again");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    for (uint64_t id = 1; id <= 4; ++id)
      arrive(scheduler, id, 300);
    const BatchPlan command = *scheduler.next({});
    require(command.width() == 4, "four short prompts did not share a command");
    completePass(scheduler, command, 500.0);
    arrive(scheduler, 5, 100);
    require(goesOn(scheduler, command), "an arrival a full command had no room for left a lane of it out, planned again");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    for (uint64_t id = 1; id <= 2; ++id)
      arrive(scheduler, id, 300);
    BatchPlan command = *scheduler.next({});
    command.items.resize(1);
    completePass(scheduler, command, 500.0);
    require(goesOn(scheduler, command), "a lane a command's first pass was planned without joined it, planned again");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    for (uint64_t id = 1; id <= 2; ++id)
      arrive(scheduler, id, 300);
    const std::array waiting{uint64_t{2}};
    const BatchPlan command = *scheduler.next(waiting);
    completePass(scheduler, command, 500.0, 0.0, waiting);
    const BatchPlan plan = *scheduler.next({});
    require(plan.pass == 0 && plan.width() == 2 && scheduler.abandonsPasses(plan),
            "a lane that waited for memory on its way back at a command's first pass did not join it");
  }
}

// Whether a command of `lanes` prompts of `laneRows` rows takes one of
// `joining` rows that becomes ready after its pass `pass`, on the estimate
// the commands `observed` (rows, wall and host milliseconds) leave.
bool regroupsAfter(std::span<const std::array<double, 3>> observed, uint32_t lanes, uint32_t laneRows,
                   uint32_t joining, uint32_t pass) {
  Scheduler scheduler(0.0);
  for (const auto &[rows, wall, host] : observed)
    scheduler.observePrefill(static_cast<uint32_t>(rows), wall, host);
  for (uint64_t id = 1; id <= lanes; ++id) {
    scheduler.submit(shortPassingLane(id, laneRows));
    scheduler.resourcesReady(id, 20'000);
  }
  BatchPlan plan = *scheduler.next({});
  for (uint32_t ran = 0; ran < pass; ++ran, plan = *scheduler.next({}))
    completePass(scheduler, plan, 500.0);
  scheduler.submit(shortPassingLane(lanes + 1, joining));
  scheduler.resourcesReady(lanes + 1, 20'000);
  return scheduler.abandonsPasses(*scheduler.next({}));
}

// A command planned again with the lanes that join it costs its lanes the
// passes they ran and what the joining rows add to its time, which the
// joining lanes' saving of what is left of its passes must outweigh. On the
// Studio's estimate a short prompt's command of 8 passes takes a 1748-row
// prompt (2.8 passes more), or the 1748 rows that leave a 9940-row one a
// group, up to its second pass, and goes on from its third; a 300-row
// prompt it takes up to its third.
void testRegroupsWeighTheRowsThatJoin() {
  const std::array<std::array<double, 3>, 1> studio{{{2048, 4900.0, 3400.0}}};
  require(regroupsAfter(studio, 1, 300, 1748, 2) && !regroupsAfter(studio, 1, 300, 1748, 3) &&
              !regroupsAfter(studio, 1, 300, 9940, 3) && regroupsAfter(studio, 1, 300, 300, 3),
          "a short prompt's command did not weigh the rows that join it");
}

// The command estimate fits its fixed part and its cost per row to commands
// of different rows, where a streamed target's waits on host steps miss the
// GPU work a command does whatever its rows. On the M6's commands (its
// 2048-row warm-up, two cold 8K prompts' groups, then agent turns of a few
// hundred rows past a cached prefix), it gives a turn of 300 rows 2.8 s, in
// 6 passes, and a group of 8192 12.2 s, in 25, about as the M6 takes them
// (3.0 s and 11.7 s), where the waits' split gave 2.1 s and 22.9 s, in 46
// passes. A command of three
// turns, a pass after which a fourth becomes ready, then takes it, which its
// rows barely lengthen, and goes on from a second pass; so it takes 64 rows
// that make three turns of 260 rows, 7 passes, run into an eighth, for the
// seventh of a pass they add.
void testPrefillEstimateFitsCommandsOfDifferentRows() {
  // Flash-Next Q2_0 on the 24 GB M6: rows, wall and host milliseconds.
  const std::array<std::array<double, 3>, 11> m6{{{2048, 3213, 724},
                                                  {6144, 9172, 653},
                                                  {2431, 5263, 839},
                                                  {6144, 9192, 666},
                                                  {2782, 5536, 684},
                                                  {8188, 11745, 668},
                                                  {6144, 9254, 708},
                                                  {2272, 5270, 1027},
                                                  {305, 2987, 1693},
                                                  {336, 2892, 1620},
                                                  {724, 3588, 1628}}};
  for (const auto &[rows, passes] : {std::pair{300u, 6u}, {8192u, 25u}}) {
    Scheduler scheduler(0.0);
    for (const auto &[observed, wall, host] : m6)
      scheduler.observePrefill(static_cast<uint32_t>(observed), wall, host);
    RequestSpec spec = passingLane(1, 48);
    spec.prefillTokens = rows;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    require(scheduler.next({})->passes == passes, "the command estimate did not fit the M6's commands");
  }
  require(regroupsAfter(m6, 3, 300, 300, 1) && !regroupsAfter(m6, 3, 300, 300, 2) &&
              regroupsAfter(m6, 3, 260, 64, 1),
          "a wider command did not weigh the rows that join it");
}

// Contended commands of one size keep the estimate's fixed part at their
// waits on host steps, none on a target whose commands wait for none, and
// put what the depth of a prompt's rows adds into the cost per row: a long
// prompt beside a decoding lane, whose rows cost twice as much 20,000 rows
// deeper, runs commands that stay about within the slice to 120,000 rows.
void testContendedCommandsStayWithinTheSliceAsTheyDeepen() {
  Scheduler scheduler(0.5);
  scheduler.observePrefill(2048, 1024.0);
  scheduler.submit(request(1, 140'000));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  double longest = 0.0;
  while (scheduler.promptProcessed(1) < 120'000) {
    const BatchPlan plan = *scheduler.next({});
    scheduler.commit(plan, {});
    if (plan.kind == WorkKind::Decode) {
      const std::array decoded{StepResult{2, 0, false, DecodeStage::Regular}};
      scheduler.complete(plan, decoded, 25.0, true);
      continue;
    }
    const BatchItem &item = plan.items[0];
    const double wall = item.tokenCount * 0.5 * (1.0 + item.promptOffset / 20'000.0);
    longest = std::max(longest, wall);
    const std::array result{StepResult{1, item.tokenCount, false, DecodeStage::Regular}};
    scheduler.complete(plan, result, wall, true);
  }
  require(longest <= 600.0, "contended commands grew past the slice as their rows deepened");
}

// The command estimate counts a command at most twice and at least half the
// time it gave it, weighs each command three quarters of the next, and keeps
// both its parts non-negative. Seeded with a streamed target's chunk (3.4 s
// fixed, 0.73 ms a row): a group that stalls ten times as long as the
// estimate gives it counts twice, 18.8 s, 38 passes, not 94 s, 48; one a
// tenth as long counts half, 4.7 s, 10 passes, not 1 s, 2; groups of 9.4,
// 9.4 and 18.8 s weigh to 13.5 s, 27 passes, where counted alike they make
// 12.5 s, 26; commands of 2048 rows in 1 s and 8192 in 9 s (counted 8 s),
// whose line would cross zero 1.1 s below it, keep no fixed part and 0.9 ms
// a row, 15 passes for 8192 rows, not 16; and commands of 64 rows in 8 s
// and 8192 in 0.1 s (counted 4 s), whose line would fall 0.37 ms a row,
// keep no cost per row, 5.7 s whatever their rows, 12 passes, not 9.
void testPrefillEstimateBoundsItsFit() {
  const auto passes = [](std::initializer_list<std::array<double, 3>> observed) {
    Scheduler scheduler(0.0);
    for (const auto &[rows, wall, host] : observed)
      scheduler.observePrefill(static_cast<uint32_t>(rows), wall, host);
    scheduler.submit(passingLane(1, 48));
    scheduler.resourcesReady(1, 0);
    return scheduler.next({})->passes;
  };
  require(passes({{2048, 4900, 3400}, {8192, 94'000, 3400}}) == 38,
          "a stalled command priced the commands after it past twice the estimate");
  require(passes({{2048, 4900, 3400}, {8192, 9400, 3400}, {8192, 9400, 3400}, {8192, 18'800, 3400}}) == 27,
          "the estimate did not weigh each command three quarters of the next");
  require(passes({{2048, 4900, 3400}, {8192, 1000, 1000}}) == 10,
          "a fast command priced the commands after it under half the estimate");
  require(passes({{2048, 1024, 0}, {2048, 1000, 0}, {8192, 9000, 0}}) == 15,
          "the estimate's fixed part went negative");
  require(passes({{2048, 4000, 4000}, {64, 8000, 8000}, {8192, 100, 100}}) == 12,
          "the estimate's cost per row went negative");
}

// Past a short leader of a command in passes, a lane joins with the rows
// that spare it a command of its own only while the fixed part it saves
// outweighs what its rows add for the lanes planned before it: on the M6's
// estimate, 1748 rows of a 9940-row prompt beside one 300-row turn do (2.1 s
// against 2.5), and 1448 rows of a 9640-row one beside two turns do not
// (3.5 s). One whose rows exceed the command's attention bound joins at none
// of them, though fewer would fit.
void testPassesShareOnlyWhatTheFixedPartOutweighs() {
  const std::array<std::array<double, 3>, 11> m6{{{2048, 3213, 724},
                                                  {6144, 9172, 653},
                                                  {2431, 5263, 839},
                                                  {6144, 9192, 666},
                                                  {2782, 5536, 684},
                                                  {8188, 11745, 668},
                                                  {6144, 9254, 708},
                                                  {2272, 5270, 1027},
                                                  {305, 2987, 1693},
                                                  {336, 2892, 1620},
                                                  {724, 3588, 1628}}};
  const auto rows = [&](std::initializer_list<uint32_t> remaining) {
    Scheduler scheduler(0.0);
    for (const auto &[observed, wall, host] : m6)
      scheduler.observePrefill(static_cast<uint32_t>(observed), wall, host);
    uint64_t id = 0;
    for (const uint32_t prompt : remaining) {
      scheduler.submit(shortPassingLane(++id, prompt));
      scheduler.resourcesReady(id, 20'000);
    }
    std::vector<uint32_t> result;
    const BatchPlan plan = *scheduler.next({});
    for (const BatchItem &item : plan.items)
      result.push_back(item.tokenCount);
    return result;
  };
  require(rows({300, 9940}) == std::vector<uint32_t>{300, 1748},
          "a long prompt did not join a short one's command for the fixed part it saves");
  require(rows({300, 300, 9640}) == std::vector<uint32_t>{300, 300},
          "a long prompt's rows joined a command whose lanes they cost more than the fixed part they save");
  Scheduler scheduler(0.0, Scheduler::attentionPairs(262'144 - 2048, 2048));
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(shortPassingLane(1, 300));
  scheduler.resourcesReady(1, 20'000);
  RequestSpec deep = passingLane(2, 48);
  deep.prefillTokens = 401'748;
  scheduler.submit(deep);
  scheduler.resourcesReady(2, 400'000);
  require(scheduler.next({})->width() == 1,
          "a lane joined a command in passes with fewer rows than spare it a command, to fit its attention");
}

// A Foreground prompt's own rows are not held against it when it
// supersedes a Background command: 2000 rows, more than the command's 300,
// still supersede it after 3 of its 8 passes, as one lane each.
void testAHigherPriorityPromptsRowsDoNotCountAgainstIt() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(shortPassingLane(1, 300, RequestPriority::Background));
  scheduler.resourcesReady(1, 20'000);
  BatchPlan plan = *scheduler.next({});
  require(plan.passes == 8, "the Background prompt did not run in 8 passes");
  for (uint32_t pass = 0; pass < 3; ++pass, plan = *scheduler.next({}))
    completePass(scheduler, plan, 500.0);
  scheduler.submit(request(2, 2000, RequestPriority::Foreground));
  scheduler.resourcesReady(2, 0);
  plan = *scheduler.next({});
  require(plan.items[0].requestId == 2 && plan.items[0].tokenCount == 2000 && scheduler.abandonsPasses(plan),
          "a higher priority's rows counted against its superseding a command between its passes");
}

// A Foreground prompt supersedes a Background command between its passes
// while the passes the command's lane ran are fewer than those the prompt
// saves, counted for each Foreground lane: ready after 9 of 19 passes, one is
// planned at once; after 10, it waits for the command's last pass, which
// commits its rows first, where two ready together supersede it (10 passes
// against 2 * 9), and after 13 they wait too (13 against 2 * 6).
void testAHigherPriorityWaitsForACommandPastItsMiddle() {
  for (const auto &[ran, prompts, supersedes] :
       {std::tuple{9u, 1u, true}, {10u, 1u, false}, {10u, 2u, true}, {13u, 2u, false}}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    RequestSpec spec = passingLane(1, 48);
    spec.priority = RequestPriority::Background;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    BatchPlan plan = *scheduler.next({});
    require(plan.passes == 19, "the Background group did not run in 19 passes");
    for (uint32_t pass = 0; pass < ran; ++pass, plan = *scheduler.next({}))
      completePass(scheduler, plan, 500.0);
    for (uint64_t id = 2; id < 2 + prompts; ++id) {
      scheduler.submit(request(id, 100, RequestPriority::Foreground));
      scheduler.resourcesReady(id, 0);
    }
    plan = *scheduler.next({});
    if (supersedes) {
      require(plan.items[0].requestId != 1 && plan.width() == prompts && scheduler.abandonsPasses(plan),
              "Foreground prompts did not supersede a Background command early in its passes");
      continue;
    }
    for (; plan.items[0].requestId == 1; plan = *scheduler.next({}))
      completePass(scheduler, plan, 500.0);
    require(scheduler.promptProcessed(1) == 8192 && plan.items[0].requestId != 1,
            "Foreground prompts superseded a Background command past its middle");
  }
}

// A command between its passes runs at the priority of the most urgent request
// it holds back, which so waits for the rest of its passes and the decode they
// owe, not for every decode above the command's own priority. A Foreground
// lane decodes, and a Background command planned without it, as memory on its
// way back held it, has run past its middle, or is in its fourth attempt after
// three abandons, so that it yields to nothing. Nothing held back, decode goes
// first. A Foreground prompt it does not yield to, or a Foreground start it
// leaves waiting for its end, lets its passes run, three 100 ms decodes owed
// for each of 500 ms: not a prompt that memory on its way back holds, nor a
// start beside a prefill that is no command between its passes.
void testACommandRunsAtThePriorityOfWhatItHoldsBack() {
  for (const bool abandoned : {false, true}) {
    for (const bool ready : {true, false}) {
      Scheduler scheduler(0.5);
      scheduler.observePrefill(2048, 4900.0, 3400.0);
      scheduler.submit(request(2, 100, RequestPriority::Foreground));
      scheduler.resourcesReady(2, 0);
      completePrefill(scheduler, *scheduler.next({}));
      RequestSpec spec = passingLane(1, 48);
      spec.priority = RequestPriority::Background;
      scheduler.submit(spec);
      scheduler.resourcesReady(1, 0);
      const std::array<uint64_t, 1> decoder{2};
      for (uint64_t attempt = 0; abandoned && attempt < 3; ++attempt) {
        completePass(scheduler, *scheduler.next(decoder), 500.0, 0.0, decoder);
        const uint64_t id = 10 + attempt;
        scheduler.submit(request(id, 100, RequestPriority::Foreground));
        scheduler.resourcesReady(id, 0);
        const BatchPlan superseding = *scheduler.next(decoder);
        require(superseding.items[0].requestId == id && scheduler.abandonsPasses(superseding),
                "a Foreground prompt did not supersede a Background command early in its passes");
        scheduler.abandonPasses();
        scheduler.cancel(id);
      }
      const uint32_t ran = abandoned ? 1 : 10;
      uint32_t left = 0;
      for (uint32_t pass = 0; pass < ran; ++pass) {
        const BatchPlan plan = *scheduler.next(decoder);
        require(plan.items[0].requestId == 1 && plan.pass == pass, "the Background command did not run its passes");
        left = plan.passes - pass - 1;
        completePass(scheduler, plan, 500.0, 0.0, decoder);
      }
      const auto decode = [&](const BatchPlan &plan) {
        scheduler.commit(plan, {});
        const std::array decoded{StepResult{2, 0, false, DecodeStage::Regular}};
        scheduler.complete(plan, decoded, 100.0, true);
      };
      BatchPlan plan = *scheduler.next({});
      require(plan.kind == WorkKind::Decode, "a command that held nothing back ran before a higher priority's decode");
      decode(plan);
      scheduler.submit(request(3, 100, RequestPriority::Foreground));
      if (ready) {
        scheduler.resourcesReady(3, 0);
        const std::array<uint64_t, 1> held{3};
        require(scheduler.next(held)->kind == WorkKind::Decode,
                "a command ran at the priority of a prompt that memory on its way back held");
      } else {
        scheduler.waitForResources(3);
      }
      uint32_t passes = 0, decodes = 0;
      while (passes < left && decodes <= 3 * left) {
        plan = *scheduler.next({});
        if (plan.kind == WorkKind::Decode) {
          decode(plan);
          ++decodes;
        } else {
          require(plan.items[0].requestId == 1, "a Foreground prompt superseded a command that does not yield to it");
          completePass(scheduler, plan, 500.0);
          ++passes;
        }
      }
      require(passes == left && decodes <= 3 * left && scheduler.promptProcessed(1) == 8192,
              "what a command held back waited for a higher priority's decode, not for its passes");
      plan = *scheduler.next({});
      if (ready) {
        while (plan.kind == WorkKind::Decode) {
          decode(plan);
          plan = *scheduler.next({});
        }
        require(plan.items[0].requestId == 3, "a prompt a command held back did not run after its last pass");
      }
    }
  }
  Scheduler scheduler(0.5);
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(request(2, 100, RequestPriority::Foreground));
  scheduler.resourcesReady(2, 0);
  completePrefill(scheduler, *scheduler.next({}));
  scheduler.submit(request(1, 4096, RequestPriority::Background));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(3, 100, RequestPriority::Foreground));
  scheduler.waitForResources(3);
  completeDecode(scheduler);
  require(scheduler.next({})->kind == WorkKind::Decode,
          "a prefill that is no command between its passes ran at a waiting start's priority");
}

// The oldest of near-equal prompts leads a command in passes as the shortest
// would. Lane 1 of 34,000 rows leads lane 2 of 30,000, keeps its rows and runs
// them in passes, alone: lane 2 does not start beside the command, nor a command
// of its own, since the next plan leaves it out. A prompt's own command goes on
// when an older near-equal prompt becomes ready between its passes, or a short
// arrival, whose plan keeps none of its rows, and it is not given up for them:
// each leads the command after its last pass, which commits its rows first.
void testANearEqualLeaderKeepsThePassesRules() {
  const auto longLane = [](uint64_t id, uint32_t rows) {
    RequestSpec spec = passingLane(id, 48);
    spec.prefillTokens = rows;
    return spec;
  };
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(longLane(1, 34'000));
    scheduler.submit(longLane(2, 30'000));
    scheduler.resourcesReady(1, 0);
    const std::array waiting{PrefillAdmission{2, 0}};
    require(scheduler.prefillAdmissionOrder(waiting).empty(),
            "a younger near-equal prompt started beside the older one's command");
    const BatchPlan first = *scheduler.next({});
    require(first.items.size() == 1 && first.items[0].requestId == 1 && first.passes > 1,
            "the older of near-equal prompts did not lead a command in passes");
    completePass(scheduler, first, 500.0);
    require(scheduler.prefillAdmissionOrder(waiting).empty(),
            "a younger near-equal prompt started beside a command between its passes");
    scheduler.resourcesReady(2, 0);
    const BatchPlan second = *scheduler.next({});
    require(second.pass == 1 && second.items == first.items && !scheduler.abandonsPasses(second),
            "a near-equal prompt cut the command of the one that leads it");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(longLane(1, 34'000));
    scheduler.submit(longLane(2, 30'000));
    scheduler.resourcesReady(2, 0);
    const BatchPlan first = *scheduler.next({});
    require(first.items.size() == 1 && first.items[0].requestId == 2 && first.passes > 1,
            "the only ready prompt did not lead a command in passes");
    completePass(scheduler, first, 500.0);
    scheduler.resourcesReady(1, 0);
    scheduler.submit(shortPassingLane(3, 300));
    scheduler.resourcesReady(3, 20'000);
    BatchPlan plan = *scheduler.next({});
    require(plan.pass == 1 && plan.items == first.items && !scheduler.abandonsPasses(plan),
            "an older near-equal prompt or a short arrival gave a command up");
    for (; plan.pass + 1 < plan.passes; plan = *scheduler.next({}))
      completePass(scheduler, plan, 500.0);
    const uint32_t rows = plan.items[0].tokenCount;
    completePass(scheduler, plan, 500.0);
    plan = *scheduler.next({});
    require(scheduler.promptProcessed(2) == rows && plan.pass == 0 && plan.items[0].requestId == 3,
            "a command's last pass did not commit its rows before the next command");
  }
}

// However often work that supersedes a command between its passes comes, a
// lane's rows commit. Short arrivals at every pass of a lane's group leave
// it its rows, which commit at its last pass. Once a lane has seen three of
// its commands abandoned, kMaximumOvertakes, its command runs to its end:
// arrivals that join a short prompt's command, which then runs without them
// as their memory comes back (excluded), abandon it three times at most;
// and a Background group that a Foreground prompt arrives during at every
// attempt yields three times, runs to its end before the fourth's prompt,
// and its next group yields again.
void testCommandsInPassesEndUnderArrivals() {
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(passingLane(1, 48));
    scheduler.resourcesReady(1, 0);
    BatchPlan plan = *scheduler.next({});
    for (uint64_t id = 2; scheduler.promptProcessed(1) == 0; ++id) {
      require(plan.items.back() == BatchItem{1, 8192, 0} && !scheduler.abandonsPasses(plan),
              "short arrivals abandoned a lane's group in passes");
      completePass(scheduler, plan, 500.0);
      scheduler.submit(shortPassingLane(id, 300));
      scheduler.resourcesReady(id, 20'000);
      plan = *scheduler.next({});
    }
    require(scheduler.promptProcessed(1) == 8192, "a lane's group did not commit under short arrivals");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(shortPassingLane(1, 300));
    scheduler.resourcesReady(1, 20'000);
    BatchPlan plan = *scheduler.next({});
    completePass(scheduler, plan, 500.0);
    std::vector<uint64_t> waiting;
    for (uint64_t id = 2; id <= 4; ++id) {
      scheduler.submit(shortPassingLane(id, 300));
      scheduler.resourcesReady(id, 20'000);
      plan = *scheduler.next(waiting);
      require(plan.pass == 0 && plan.width() == 2 && scheduler.abandonsPasses(plan),
              "a short arrival did not join a command early in its passes");
      scheduler.abandonPasses();
      plan.items.resize(1);
      waiting.push_back(id);
      completePass(scheduler, plan, 500.0, 0.0, waiting);
    }
    scheduler.submit(shortPassingLane(5, 300));
    scheduler.resourcesReady(5, 20'000);
    const BatchPlan next = *scheduler.next(waiting);
    require(next.pass == 1 && next.items == plan.items && !scheduler.abandonsPasses(next),
            "a command abandoned three times for arrivals was abandoned again");
  }
  {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    RequestSpec spec = passingLane(1, 48);
    spec.priority = RequestPriority::Background;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    for (uint64_t attempt = 1; attempt <= 5; ++attempt) {
      BatchPlan plan = *scheduler.next({});
      require(plan.items == std::vector<BatchItem>{{1, 8192, scheduler.promptProcessed(1)}} && plan.pass == 0,
              "the Background lane did not start its group");
      completePass(scheduler, plan, 500.0);
      const uint64_t foreground = 100 + attempt;
      scheduler.submit(request(foreground, 100, RequestPriority::Foreground));
      scheduler.resourcesReady(foreground, 0);
      plan = *scheduler.next({});
      if (attempt != 4) {
        require(plan.items[0].requestId == foreground && scheduler.abandonsPasses(plan),
                "a Foreground prompt did not supersede a Background command that yielded fewer than three times");
        scheduler.abandonPasses();
      } else {
        for (; plan.items[0].requestId == 1; plan = *scheduler.next({}))
          completePass(scheduler, plan, 500.0);
        require(scheduler.promptProcessed(1) == 8192 && plan.items[0].requestId == foreground,
                "a Background command that yielded three times did not run to its end first");
      }
      completePrefill(scheduler, plan);
      completeDecode(scheduler, true);
    }
  }
}

// A command between its passes yields, whole, to a prompt of a higher
// priority, planned before its next pass, which commits only once it is
// abandoned, after which its lanes run their rows again from a first pass.
// A Foreground lane's decode, once its first mask lands, runs before its
// next pass too, but between its passes, which use none of the memory
// decode does: the command goes on after it.
void testPassesYieldToAHigherPriority() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  scheduler.submit(request(3, 100, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  awaitFirstMask(scheduler, 3);
  for (uint64_t id = 1; id <= 2; ++id) {
    scheduler.submit(shortPassingLane(id, 300, RequestPriority::Background));
    scheduler.resourcesReady(id, 20'000);
  }
  const BatchPlan first = *scheduler.next({});
  require(first.width() == 2 && first.pass == 0 && first.passes > 1,
          "two Background prompts did not share a command in passes");
  completePass(scheduler, first, 500.0);
  scheduler.maskReady(3);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode && plan.items[0].requestId == 3 && !scheduler.abandonsPasses(plan),
          "a higher priority's decode waited for a command between its passes, or abandoned it");
  completeDecode(scheduler, true);
  plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.pass == 1 && plan.items == first.items,
          "a command did not go on with its next pass after a higher priority's decode");
  scheduler.submit(request(5, 100, RequestPriority::Foreground));
  scheduler.resourcesReady(5, 0);
  plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 5 && plan.width() == 1 &&
              scheduler.abandonsPasses(plan),
          "a higher priority's prompt waited for a command between its passes");
  rejects([&] { scheduler.commit(plan, {}); }, "batch no longer matches",
          "a higher priority's prompt committed beside a command between its passes");
  scheduler.abandonPasses();
  require(!scheduler.abandonsPasses(plan), "an abandoned command stayed between its passes");
  completePrefill(scheduler, plan);
  completeDecode(scheduler, true);
  plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.pass == 0 && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
}

// A command between its passes holds its lanes to its last pass: a pass
// before the last consumes no rows. It is not abandoned while a pass of it
// runs; abandoned, it leaves
// its lanes their rows, which they run again from a first pass, a lane
// suspended meanwhile too once it resumes.
void testCommandBetweenPassesHoldsItsLanes() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 4900.0, 3400.0);
  for (uint64_t id = 1; id <= 2; ++id) {
    scheduler.submit(shortPassingLane(id, 300));
    scheduler.resourcesReady(id, 20'000);
  }
  const BatchPlan first = *scheduler.next({});
  scheduler.commit(first, {});
  rejects([&] { scheduler.abandonPasses(); }, "no command between its passes",
          "a command was abandoned while a pass of it ran");
  const std::array results{StepResult{1, 0, false, DecodeStage::Regular},
                           StepResult{2, 0, false, DecodeStage::Regular}};
  scheduler.complete(first, results, 500.0, true);
  require(scheduler.phase(1) == engine::Phase::Prefill && scheduler.promptProcessed(1) == 20'000,
          "a pass before a command's last consumed its rows");
  scheduler.abandonPasses();
  BatchPlan plan = *scheduler.next({});
  require(plan.pass == 0 && plan.passes == first.passes && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
  completePass(scheduler, plan, 500.0);
  scheduler.abandonPasses();
  scheduler.suspendForResources(1);
  scheduler.resumeFromResources(1, 20'000, 20'300);
  plan = *scheduler.next({});
  require(plan.pass == 0 && plan.items == first.items,
          "a lane suspended between its command's passes did not run its rows again with the others");
}

// Completes a prefill command of one lane that streamed its history, which
// waited streamMilliseconds of its hostMilliseconds on the reads.
void completeStreamedPrefill(engine::Scheduler &scheduler, const BatchPlan &plan,
                             double wallMilliseconds, double hostMilliseconds,
                             double streamMilliseconds) {
  require(plan.kind == WorkKind::Prefill && plan.width() == 1, "expected a streaming lane alone");
  scheduler.commit(plan, {});
  const std::array result{
      StepResult{plan.items[0].requestId, plan.items[0].tokenCount, false, DecodeStage::Regular}};
  scheduler.complete(plan, result, wallMilliseconds, true, hostMilliseconds, streamMilliseconds);
}

// A lane that streams its history runs alone: leading, it takes no peers
// into its command; behind a leader, it is passed over while the others
// share the command, until later arrivals have overtaken it often enough
// that it leads one.
void testStreamingLaneRunsAlone() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 100));
  scheduler.submit(request(2, 300));
  scheduler.submit(request(3, 500));
  for (uint64_t id = 1; id <= 3; ++id)
    scheduler.resourcesReady(id, 0);
  scheduler.setStreamsHistory(1, true);
  BatchPlan plan = *scheduler.next({});
  require(plan.width() == 1 && plan.items[0].requestId == 1 && plan.items[0].tokenCount == 100,
          "a streaming leader shared its command");
  scheduler.setStreamsHistory(1, false);
  scheduler.setStreamsHistory(2, true);
  plan = *scheduler.next({});
  require(plan.width() == 2 && plan.items[0].requestId == 1 && plan.items[1].requestId == 3,
          "a streaming lane joined a command it did not lead");

  Scheduler overtaken(0.0);
  overtaken.submit(request(1, 4000));
  overtaken.resourcesReady(1, 0);
  overtaken.setStreamsHistory(1, true);
  uint64_t id = 2;
  for (; id <= model::ExecutionLimits::maximumBatchWidth; ++id) {
    overtaken.submit(request(id, 100));
    overtaken.resourcesReady(id, 0);
    plan = *overtaken.next({});
    require(plan.width() == 1 && plan.items[0].requestId == id,
            "a passed-over streaming lane led or joined a command");
    completePrefill(overtaken, plan);
    overtaken.cancel(id);
  }
  overtaken.submit(request(id, 100));
  overtaken.resourcesReady(id, 0);
  plan = *overtaken.next({});
  require(plan.width() == 1 && plan.items[0].requestId == 1,
          "an overtaken streaming lane did not lead a command of its own");
}

// Contended, a lane that streams its history takes the fewest rows, halving
// from its prefill group's, whose time still covers its estimated wait on
// the reads: fewer would not end the command sooner. It never takes fewer
// than another contended command, which it takes before it has streamed.
// Uncontended, it takes its group.
void testContendedStreamingLaneHalvesFromItsGroup() {
  // At 0.5 ms a row, a contended command takes 512 rows.
  const auto first = [](std::optional<double> streamMilliseconds, bool contended) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 1024.0);
    RequestSpec spec = request(1, 100'000);
    spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    scheduler.submit(spec);
    scheduler.resourcesReady(1, 0);
    scheduler.setStreamsHistory(1, true);
    if (streamMilliseconds) {
      const BatchPlan streamed = *scheduler.next({});
      completeStreamedPrefill(scheduler, streamed,
                              streamed.items[0].tokenCount * 0.5 + *streamMilliseconds,
                              *streamMilliseconds, *streamMilliseconds);
    }
    if (contended) {
      scheduler.submit(request(2, 1));
      scheduler.resourcesReady(2, 1);
      completeDecode(scheduler);
    }
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.width() == 1 && plan.items[0].requestId == 1,
            "expected the streaming lane alone");
    return plan.items[0].tokenCount;
  };
  require(first(std::nullopt, true) == 512,
          "a streaming lane's first contended command took more than another's");
  require(first(2800.0, true) == 8192, "a contended stream that dominates its group was cut");
  require(first(1000.0, true) == 2048,
          "a contended streaming lane did not take the fewest rows covering its stream");
  require(first(0.0, true) == 512, "a stream nothing waited on cut a command below another's");
  require(first(2800.0, false) == 8192, "an uncontended streaming lane did not take its group");
}

// A lane that streams its history may run commands that read none of it,
// before its first pages spill: they leave its estimate of the wait alone,
// which its first command that read the history sets.
void testStreamEstimateIgnoresCommandsThatReadNothing() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 1024.0);
  RequestSpec spec = request(1, 100'000);
  spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
  scheduler.submit(spec);
  scheduler.resourcesReady(1, 0);
  scheduler.setStreamsHistory(1, true);
  const BatchPlan unread = *scheduler.next({});
  completePrefill(scheduler, unread, unread.items[0].tokenCount * 0.5);
  const BatchPlan streamed = *scheduler.next({});
  completeStreamedPrefill(scheduler, streamed, streamed.items[0].tokenCount * 0.5 + 2800.0,
                          2800.0, 2800.0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == 8192,
          "a command that read no history lowered its lane's estimate of the reads");
}

// A streaming command's wait on its history is its lane's own cost: the
// command estimate other commands' slices come from leaves it out, and the
// rows it ran train the per-row estimate as any command's do.
void testStreamWaitStaysOutOfTheCommandEstimate() {
  Scheduler scheduler(0.0);
  scheduler.observePrefill(2048, 1024.0);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.setStreamsHistory(1, true);
  // 3 s of reads beside 2048 rows at 0.5 ms each.
  completeStreamedPrefill(scheduler, *scheduler.next({}), 4024.0, 3000.0, 3000.0);
  scheduler.setStreamsHistory(1, false);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].tokenCount == 512,
          "a streamed command's wait on its reads widened other commands' slices");
}

// The images a prefill command encodes run ahead of it as a command of their
// own: the command owes decode their time as its own, and the command
// estimate other commands' slices come from leaves it out.
void testImageEncodesStayOutOfTheCommandEstimate() {
  Scheduler scheduler(0.5);
  scheduler.observePrefill(2048, 1843.2);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].tokenCount == 512,
          "a contended command did not take the slice's rows");
  // 512 rows at 0.9 ms each, as the estimate gives them, behind 200 ms of
  // encodes: the rows alone keep it at 0.9 ms a row (512 rows a slice), where
  // with the encodes it would reach 1.29 (256). The command owes decode its
  // whole 661 ms by half, two decodes of 250 ms, where its rows alone would
  // owe one.
  scheduler.commit(plan, {});
  const std::array result{StepResult{1, 512, false, DecodeStage::Regular}};
  scheduler.complete(plan, result, 661.0, true, 0.0, std::nullopt, 200.0);
  uint32_t decodes = 0;
  for (plan = *scheduler.next({}); plan.kind == WorkKind::Decode; plan = *scheduler.next({})) {
    scheduler.commit(plan, {});
    const std::array decoded{StepResult{2, 0, false, DecodeStage::Regular}};
    scheduler.complete(plan, decoded, 250.0, true);
    ++decodes;
  }
  require(decodes == 2, "a command did not owe decode the time of its images' encodes");
  require(plan.items[0].requestId == 1 && plan.items[0].tokenCount == 512,
          "a command's image encodes trained other commands' slices");
}

// A command in passes whose first pass encoded images trains the estimate
// with its passes' time less the encodes: its next command runs in the
// passes and rows of one that encoded none.
void testImageEncodesStayOutOfAPassedCommandsEstimate() {
  std::array<BatchPlan, 2> next;
  for (const bool encodes : {false, true}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4900.0, 3400.0);
    scheduler.submit(passingLane(1, 48));
    scheduler.resourcesReady(1, 0);
    BatchPlan plan = *scheduler.next({});
    const uint32_t passes = plan.passes;
    for (uint32_t pass = 0; pass < passes; ++pass) {
      require(plan.items[0].requestId == 1 && plan.pass == pass && plan.passes == passes,
              "a command's next pass did not go on");
      const double encoded = encodes && !pass ? 5000.0 : 0.0;
      scheduler.commit(plan, {});
      const uint32_t consumed = pass + 1 == passes ? plan.items[0].tokenCount : 0;
      const std::array result{StepResult{1, consumed, false, DecodeStage::Regular}};
      scheduler.complete(plan, result, 500.0 + encoded, true, 300.0, std::nullopt, encoded);
      plan = *scheduler.next({});
    }
    next[encodes] = plan;
  }
  require(next[0].passes == 19 && next[1].passes == next[0].passes &&
              next[1].items[0].tokenCount == next[0].items[0].tokenCount,
          "a command's first pass's image encodes trained its next command's passes");
}

// Each gap a decoding lane sees between its commands across a streaming
// prefill command is recorded, and the command counts as contended; gaps
// across other commands are not recorded.
void testDecodeGapAcrossStreamingPrefill() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.resourcesReady(1, 1);
  scheduler.submit(request(2, 4096));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler, false, 40.0);
  completePrefill(scheduler, *scheduler.next({}), 1000.0);
  completeDecode(scheduler, false, 40.0);
  require(scheduler.snapshot().decodeGap.count == 0 &&
              scheduler.snapshot().contendedStreamingPrefillBatches == 0,
          "a gap across an ordinary prefill was recorded");
  scheduler.setStreamsHistory(2, true);
  completeStreamedPrefill(scheduler, *scheduler.next({}), 3000.0, 2500.0, 2500.0);
  completeDecode(scheduler, false, 40.0);
  const SchedulerSnapshot counts = scheduler.snapshot();
  require(counts.decodeGap.count == 1 && counts.decodeGap.maximumMilliseconds == 3040.0 &&
              counts.contendedStreamingPrefillBatches == 1,
          "the gap across a streaming prefill was not recorded");
}

void testShortestRemainingFirstUsesActualRows() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 17));
  scheduler.submit(request(2, 1000));
  scheduler.submit(request(3, 3000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  scheduler.resourcesReady(3, 0);
  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 3,
          "ragged prefill did not batch all ready sequences");
  uint32_t rows = 0;
  for (const BatchItem &item : plan.items)
    rows += item.tokenCount;
  require(rows == model::ExecutionLimits::prefillTokenBudget,
          "ragged prefill did not use the exact actual-row budget");
  require(plan.items[0].requestId == 1 && plan.items[0].tokenCount == 17 &&
              plan.items[1].requestId == 2 &&
              plan.items[1].tokenCount == 1000 &&
              plan.items[2].requestId == 3 &&
              plan.items[2].tokenCount == 1031,
          "prefill did not serve the shortest remaining sequences first");
  require(std::all_of(plan.items.begin(), plan.items.end(),
                      [](const BatchItem &item) {
                        return item.promptOffset == 0;
                      }),
          "initial prefill plan did not carry its absolute token offset");
}

// The engine ends a lane's command before its planned rows at a state
// boundary whose state or KV it could not reserve (Engine::prepare): the
// scheduler takes the rows the shortened item ran, and the lane resumes from
// them.
void testShortenedPrefillResumesFromItsRows() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.submit(request(2, 1000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[1].requestId == 1 && plan.items[1].tokenCount == 1048,
          "the long lane did not take the rest of the budget");
  plan.items[1].tokenCount = 32;
  completePrefill(scheduler, plan);
  require(scheduler.promptProcessed(1) == 32 && scheduler.phase(1) == engine::Phase::Prefill,
          "a shortened command did not keep the rows it ran");
  completeDecode(scheduler, true);
  const BatchPlan resumed = *scheduler.next({});
  require(resumed.items.size() == 1 && resumed.items[0].requestId == 1 &&
              resumed.items[0].promptOffset == 32 &&
              resumed.items[0].tokenCount == model::ExecutionLimits::prefillTokenBudget,
          "the shortened lane did not resume from the rows it ran");
}

void testEqualPromptsFinishInArrivalOrder() {
  // Equal cold prompts are served oldest first, one whole budget at a time,
  // instead of an equal water-fill share that finishes them all at once.
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 3; ++id) {
    scheduler.submit(request(id, 3000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "equal prompts were water-filled instead of served oldest first");
}

void testShortArrivalPrecedesLongColdPrompt() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 2 && plan.items[0].requestId == 2 &&
              plan.items[0].tokenCount == 700 &&
              plan.items[1].requestId == 1 &&
              plan.items[1].tokenCount == 1348,
          "a later short prompt waited behind an earlier long one");
}

void testEqualLanesRunInArrivalOrderWithoutOvertaking() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 5000));
  scheduler.submit(request(2, 5000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items.size() == 1 && first.items[0].requestId == 1 &&
              first.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "first command did not give the whole budget to the oldest lane");
  completePrefill(scheduler, first);
  // Lane 2 was left out by an older lane, which is not overtaking: lane 1
  // keeps the budget until it finishes, as under FIFO.
  const BatchPlan second = *scheduler.next({});
  require(second.items.size() == 1 && second.items[0].requestId == 1,
          "an older lane lost the budget to a younger equal lane");
}

void testLaneOvertakenThreeTimesLeadsTheNextCommand() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 6000));
  scheduler.resourcesReady(1, 0);
  // Three later short arrivals each take the whole budget ahead of lane 1;
  // each stops in prefill so no decode command interleaves.
  const auto finishShort = [&](const BatchPlan &plan) {
    scheduler.commit(plan, {});
    const std::vector<StepResult> results{
        {plan.items[0].requestId, plan.items[0].tokenCount, true,
         DecodeStage::Regular}};
    scheduler.complete(plan, results, 0.0, true);
  };
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, model::ExecutionLimits::prefillTokenBudget));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
                plan.items[0].requestId == id,
            "a short arrival did not run ahead of the long lane");
    finishShort(plan);
  }
  scheduler.submit(request(5, model::ExecutionLimits::prefillTokenBudget));
  scheduler.resourcesReady(5, 0);
  const BatchPlan overdue = *scheduler.next({});
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "a lane overtaken three times did not lead the next command");
  completePrefill(scheduler, overdue);
  // Served once, the long lane yields to short arrivals again.
  const BatchPlan resumed = *scheduler.next({});
  require(resumed.items.size() == 1 && resumed.items[0].requestId == 5,
          "a served lane kept its overdue priority");
}

void testServedLaneResetsOvertaking() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 3000));
  scheduler.submit(request(2, 700));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items.size() == 2,
          "both ready lanes did not share the command");
  scheduler.commit(first, {});
  // The short prompt stops in prefill; the long one keeps its remainder.
  std::vector<StepResult> results;
  for (const BatchItem &item : first.items) {
    results.push_back({item.requestId, item.tokenCount, item.requestId == 2,
                       DecodeStage::Regular});
  }
  scheduler.complete(first, results, 0.0, true);
  scheduler.submit(request(3, 200));
  scheduler.resourcesReady(3, 0);
  const BatchPlan second = *scheduler.next({});
  require(second.items.size() == 2 && second.items[0].requestId == 3 &&
              second.items[0].tokenCount == 200 &&
              second.items[1].requestId == 1 &&
              second.items[1].tokenCount == 1652,
          "a lane that received rows was treated as overtaken");
}

// The lane that leads the next prefill command once lane 1 of `earlier`
// tokens and lane 2 of `later`, submitted after it, are ready from their
// cached positions.
uint64_t prefillLeader(engine::Scheduler &scheduler, uint32_t earlier, uint32_t later,
                       uint32_t earlierCached = 0, uint32_t laterCached = 0,
                       RequestPriority earlierPriority = RequestPriority::Normal) {
  scheduler.submit(request(1, earlier, earlierPriority));
  scheduler.submit(request(2, later));
  scheduler.resourcesReady(1, earlierCached);
  scheduler.resourcesReady(2, laterCached);
  return scheduler.next({})->items.front().requestId;
}

uint64_t prefillLeader(uint32_t earlier, uint32_t later, uint32_t earlierCached = 0,
                       uint32_t laterCached = 0,
                       RequestPriority earlierPriority = RequestPriority::Normal) {
  engine::Scheduler scheduler(0.0);
  return prefillLeader(scheduler, earlier, later, earlierCached, laterCached, earlierPriority);
}

// The oldest lane leads whose remaining prompt exceeds the shortest by at most
// a command of the shortest's for each k = 3 it needs past its first.
// Unmeasured, a command is 2048 rows: 30,000 take 15 commands and allow
// 4 x 2048; 6,145 take 4 and allow 2048; 6,144 take 3 and allow none, as at
// 4,000 and 3,000.
void testOldestNearEqualPromptLeads() {
  require(prefillLeader(71'254, 71'220) == 1 && prefillLeader(38'192, 30'000) == 1 &&
              prefillLeader(8193, 6145) == 1,
          "a later prompt led an earlier one within the rows it would lead overdue");
  require(prefillLeader(38'193, 30'000) == 2 && prefillLeader(8192, 6144) == 2 &&
              prefillLeader(4000, 3000) == 2,
          "an earlier prompt led a later one past the rows it would lead overdue");
  // Remaining rows, after a cached prefix: 20,000 against 19,000 lead, and
  // 22,000 against 16,000 do not.
  require(prefillLeader(40'000, 19'000, 20'000, 0) == 1 &&
              prefillLeader(22'000, 40'000, 0, 24'000) == 2,
          "near-equal prompts were not measured by their remaining rows");
  // Beside a decoder, at 512-row slices: 2,000 rows take four commands and
  // allow 512; 500 take one.
  const auto contendedLeader = [](uint32_t earlier, uint32_t later) {
    engine::Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 1024.0);
    scheduler.submit(request(3, 1));
    scheduler.resourcesReady(3, 1);
    completeDecode(scheduler);
    return prefillLeader(scheduler, earlier, later);
  };
  require(contendedLeader(2512, 2000) == 1 && contendedLeader(2513, 2000) == 2 &&
              contendedLeader(600, 500) == 2,
          "near-equal prompts were not measured in their contended commands");
  // Prompts that finish within k commands keep shortest first.
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 3; ++id) {
    scheduler.submit(request(id, id == 1 ? 1300 : 1000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 3 && plan.items[0].requestId == 2 &&
              plan.items[1].requestId == 3 && plan.items[2].requestId == 1 &&
              plan.items[2].tokenCount == 48,
          "short prompts did not run shortest first");
}

// The band counts the shortest lane's own commands: in 8192-row prefill
// groups, 20,000 rows take three and allow none, where chunks would allow
// 6,144; and with the shortest streaming its history beside a ready peer,
// 2,000 rows take four 512-row slices and allow 512, where chunks would
// allow none.
void testNearEqualBandCountsTheShortestLanesCommands() {
  engine::Scheduler grouped(0.0);
  // A streamed target's chunk: 4.9 s, 3.4 s of it on expert reads.
  grouped.observePrefill(2048, 4900.0, 3400.0);
  for (uint64_t id = 1; id <= 2; ++id) {
    RequestSpec spec = request(id, id == 1 ? 21'000 : 20'000);
    spec.prefillGroupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    grouped.submit(spec);
    grouped.resourcesReady(id, 0);
  }
  require(grouped.next({})->items.front().requestId == 2,
          "the band counted chunks where the shortest lane runs prefill groups");
  engine::Scheduler streaming(0.0);
  // At 0.5 ms a row, a contended command takes 512 rows.
  streaming.observePrefill(2048, 1024.0);
  streaming.submit(request(1, 2512));
  streaming.submit(request(2, 2000));
  streaming.resourcesReady(1, 0);
  streaming.resourcesReady(2, 0);
  streaming.setStreamsHistory(2, true);
  require(streaming.next({})->items.front().requestId == 1,
          "the band counted chunks where the shortest lane streams its history");
}

// Lane 1 of 9,000 rows leads lane 3 of 7,000, within a command of it, and
// where attention halves its rows the rest join it shortest first.
void testNearEqualLeaderKeepsTheRestShortestFirst() {
  engine::Scheduler scheduler(0.0, 700'000);
  for (uint64_t id = 1; id <= 3; ++id) {
    scheduler.submit(request(id, 10'000 - 1000 * id));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.items.size() == 3 && plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == 1024 && plan.items[1].requestId == 3 &&
              plan.items[1].tokenCount == 512 && plan.items[2].requestId == 2 &&
              plan.items[2].tokenCount == 256,
          "the lanes after a near-equal leader did not stay shortest first");
}

// An overdue lane leads an older lane that is not overdue, which the band
// measured from the overdue lane would otherwise put first: lane 1 joins three
// short arrivals' commands that leave lane 2 out.
void testOverdueLaneLeadsAnOlderLane() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 8000));
  scheduler.submit(request(2, 9000));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);
  for (uint64_t id = 3; id <= 5; ++id) {
    scheduler.submit(request(id, 100));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next({});
    require(plan.items.size() == 2 && plan.items[0].requestId == id &&
                plan.items[1].requestId == 1,
            "a short arrival and the shorter long lane did not share the command");
    scheduler.commit(plan, {});
    const std::vector<StepResult> results{
        {id, 100, true, DecodeStage::Regular},
        {1, plan.items[1].tokenCount, false, DecodeStage::Regular}};
    scheduler.complete(plan, results, 0.0, true);
  }
  const BatchPlan overdue = *scheduler.next({});
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 2,
          "an overdue lane did not lead an older lane");
}

// Only lanes of the front lane's priority count: an older lane of a lower
// priority does not lead, and one overdue after lane 2's three commands does
// not keep lane 2 from leading lane 3.
void testNearEqualOrderStaysWithinAPriority() {
  require(prefillLeader(5000, 5000, 0, 0, RequestPriority::Background) == 2,
          "an older lane of a lower priority led a near-equal one");
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 5000, RequestPriority::Background));
  scheduler.resourcesReady(1, 0);
  scheduler.submit(request(2, 27'144));
  scheduler.resourcesReady(2, 0);
  for (int command = 0; command < 3; ++command)
    completePrefill(scheduler, *scheduler.next({}));
  scheduler.submit(request(3, 20'000));
  scheduler.resourcesReady(3, 0);
  require(scheduler.next({})->items.front().requestId == 2,
          "an overdue lane of a lower priority stopped the near-equal rule");
}

// Admission starts the lanes of the next plan, so the older of two queued
// near-equal prompts starts first, and none that the next command would leave
// out: without the shortest candidate, lane 3, which the plan leaves out,
// lane 1 leads lane 2.
void testAdmissionStartsTheLanesOfTheNextCommand() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 71'254));
  scheduler.submit(request(2, 71'220));
  const std::array candidates{PrefillAdmission{1, 0}, PrefillAdmission{2, 0}};
  require(scheduler.prefillAdmissionOrder(candidates) == std::vector<uint64_t>{1},
          "admission started a later near-equal prompt first");
  engine::Scheduler running(0.0);
  running.submit(request(1, 41'000));
  running.resourcesReady(1, 0);
  running.submit(request(2, 38'000));
  running.submit(request(3, 30'000));
  const std::array queued{PrefillAdmission{2, 0}, PrefillAdmission{3, 0}};
  require(running.prefillAdmissionOrder(queued).empty() &&
              running.next({})->items.front().requestId == 1,
          "admission started a lane the next command leaves out");
}

void testRealDecodeWidths() {
  for (uint32_t width = 1; width <= 4; ++width) {
    engine::Scheduler scheduler(0.0);
    for (uint32_t lane = 0; lane < width; ++lane) {
      scheduler.submit(request(lane + 1, 1));
      scheduler.resourcesReady(lane + 1, 1);
    }
    BatchPlan plan = *scheduler.next({});
    require(plan.kind == WorkKind::Decode && plan.width() == width,
            "ready decode width was delayed or rewritten");
    for (const BatchItem &item : plan.items) {
      require(item.tokenCount == 0, "decode plan carried a variable row count");
    }
    scheduler.commit(plan, {});
    std::vector<StepResult> results;
    for (const BatchItem &item : plan.items) {
      results.push_back({item.requestId, 0, false, DecodeStage::Regular});
    }
    scheduler.complete(plan, results, 0.0, true);
    require(scheduler.snapshot().decodeBatchesByWidth[width - 1] == 1,
            "decode width counter did not record the real command");
  }
}

void testUnconstrainedLanesShareDecode() {
  Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 4; ++id) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.width() == 4,
          "unconstrained requests did not share one decode command");
}

void testRejectedCommitCountsNothing() {
  Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);
  const BatchPlan stale = *scheduler.next({});
  scheduler.cancel(2);
  rejects([&] { scheduler.commit(stale, {}); }, "batch no longer matches scheduler state",
          "a commit of a cancelled request's batch was accepted");
  require(scheduler.snapshot().decodeBatches == 0,
          "rejected commit incremented decode counters");
  completeDecode(scheduler);
  require(scheduler.snapshot().decodeBatches == 1,
          "the remaining request's decode was not counted");
}

// Once its first token is selected, a lane that waited for its first mask
// decodes in the commands of the lanes that never did: two of each share one.
void testMaskedLanesShareDecode() {
  Scheduler scheduler(0.0);
  for (uint64_t id : {1, 2}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan prompts = *scheduler.next({});
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);
  scheduler.maskReady(2);
  completeDecode(scheduler);
  for (uint64_t id : {3, 4}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode && plan.width() == 4 &&
              plan.decodeStage == DecodeStage::Regular,
          "lanes that waited for their first masks decoded apart from the others");
}

void testPrefillAndDecodeAlternateWithoutStarvation() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 10'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan decode = *scheduler.next({});
  require(decode.kind == WorkKind::Decode && decode.items[0].requestId == 1,
          "decode did not retain the first ready command");
  scheduler.commit(decode, {});
  const std::array decodeResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(decode, decodeResult, 0.0, true);

  BatchPlan prefill = *scheduler.next({});
  require(prefill.kind == WorkKind::Prefill &&
              prefill.items[0].requestId == 2,
          "continuous decode starved a newly admitted prefill");
  completePrefill(scheduler, prefill);

  BatchPlan nextDecode = *scheduler.next({});
  require(nextDecode.kind == WorkKind::Decode &&
              nextDecode.items[0].requestId == 1,
          "prefill did not yield the next command back to decode");
}

// Lane 1 decodes beside lane 2's long prompt: one decode command, then one
// 500 ms prefill command.
void runContendedPrefill(engine::Scheduler &scheduler) {
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler, false, 50.0);
  const BatchPlan prefill = *scheduler.next({});
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "the long prompt did not follow the first decode command");
  completePrefill(scheduler, prefill, 500.0);
}

void testDecodeRepaysItsShareOfContendedPrefill() {
  for (const double share : {0.0, 0.5}) {
    engine::Scheduler scheduler(share);
    runContendedPrefill(scheduler);
    // At 0.5 the prefill owes 250 ms, five 50 ms decode commands; with no
    // share the kinds alternate one command each.
    for (uint32_t step = 0; step < (share > 0.0 ? 5u : 1u); ++step)
      completeDecode(scheduler, false, 50.0);
    require(scheduler.next({})->kind == WorkKind::Prefill,
            "decode did not return the next command once its share was repaid");
  }
}

// A constrained lane waits for its first mask while an equal-priority prompt
// prefills: decode could not have used that time, so once the mask arrives
// the kinds alternate without banked debt.
void testMaskWaitAccruesNoDecodeDebt() {
  engine::Scheduler scheduler(0.5);
  scheduler.submit(request(1, 1));
  scheduler.resourcesReady(1, 0);
  awaitFirstMask(scheduler, 1);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  for (uint32_t command = 0; command < 3; ++command)
    completePrefill(scheduler, *scheduler.next({}), 500.0);
  scheduler.maskReady(1);
  completeDecode(scheduler, false, 50.0);
  require(scheduler.next({})->kind == WorkKind::Prefill,
          "prefill beside a lane waiting for its mask banked decode debt");
}

// Lanes that wait for memory on its way back are left out of the plan, so
// the other kind or the other lanes run. A prefill that ran meanwhile owes
// the decoders it was planned without nothing, and a prefill lane left out
// was blocked, not overtaken.
void testExcludedLanesAreNotPlanned() {
  engine::Scheduler scheduler(0.5);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
  scheduler.submit(request(3, 20'000));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);
  scheduler.resourcesReady(3, 0);
  const std::array<uint64_t, 1> first{1};
  const BatchPlan decode = *scheduler.next(first);
  require(decode.kind == WorkKind::Decode && decode.width() == 1 &&
              decode.items[0].requestId == 2,
          "a decode plan included a lane left out of it");
  const std::array<uint64_t, 2> decoders{1, 2};
  const BatchPlan prefill = *scheduler.next(decoders);
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 3,
          "the prefill did not run while every decoder was left out");
  scheduler.commit(prefill, decoders);
  const std::array result{StepResult{3, prefill.items[0].tokenCount, false, DecodeStage::Regular}};
  scheduler.complete(prefill, result, 500.0, true);
  completeDecode(scheduler, false, 50.0);
  require(scheduler.next({})->kind == WorkKind::Prefill,
          "a prefill owed decode time to decoders it was planned without");

  engine::Scheduler blocked(0.0);
  blocked.submit(request(1, 6000));
  blocked.resourcesReady(1, 0);
  const std::array<uint64_t, 1> waiting{1};
  for (uint64_t id = 2; id <= 4; ++id) {
    blocked.submit(request(id, model::ExecutionLimits::prefillTokenBudget));
    blocked.resourcesReady(id, 0);
    const BatchPlan plan = *blocked.next(waiting);
    blocked.commit(plan, waiting);
    const std::array finished{
        StepResult{id, plan.items[0].tokenCount, true, DecodeStage::Regular}};
    blocked.complete(plan, finished, 0.0, true);
  }
  blocked.submit(request(5, model::ExecutionLimits::prefillTokenBudget));
  blocked.resourcesReady(5, 0);
  require(blocked.next({})->items[0].requestId == 5,
          "a lane left out of three commands was treated as overtaken");
}

void testDecodeDebtLeavesWithTheLastDecoder() {
  for (const bool finished : {true, false}) {
    engine::Scheduler scheduler(0.5);
    runContendedPrefill(scheduler);
    // Lane 1 leaves still owed 200 ms: its decode command finishes it, or it
    // is cancelled between commands.
    completeDecode(scheduler, finished, 50.0);
    if (!finished)
      scheduler.cancel(1);
    scheduler.submit(request(3, 1));
    scheduler.resourcesReady(3, 1);
    require(scheduler.next({})->kind == WorkKind::Prefill,
            "a later decoder inherited debt owed to one that left");
  }
}

void testHigherPriorityPrefillPrecedesDecodeDebt() {
  engine::Scheduler scheduler(0.5);
  runContendedPrefill(scheduler);
  scheduler.submit(request(3, 20'000, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 3,
          "decode debt overrode a higher-priority prefill");
}

void testMeasuredBudgetOnlyLimitsContendedWork() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  const BatchPlan first = *scheduler.next({});
  require(first.items[0].tokenCount == 2048,
          "unmeasured isolated work lost the full prefill budget");
  completePrefill(scheduler, first, 4096.0);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "slow isolated work lost the full prefill budget");

  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler, false, 1e9);
  const BatchPlan contended = *scheduler.next({});
  require(contended.kind == WorkKind::Prefill &&
              contended.items[0].tokenCount == 128,
          "measured slow prefill was not shortened for a decoding peer");
  completePrefill(scheduler, contended, 256.0);
  completeDecode(scheduler, true);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "prefill did not recover isolated throughput after its peer ended");
}

void testAuxiliaryWorkDoesNotTrainTextPrefillTiming() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  completePrefill(scheduler, *scheduler.next({}), 1e9, false);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 1);
  completeDecode(scheduler);
  require(scheduler.next({})->items.front().tokenCount == 128,
          "auxiliary encoding latency contaminated the text throughput estimate");
}

void testMeasuredBudgetUsesActualRowsAndRecovers() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 128));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 512.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  BatchPlan plan = *scheduler.next({});
  require(plan.items[0].tokenCount == 64,
          "prefill timing was normalized by capacity instead of actual rows");

  for (uint32_t sample = 0; sample < 8; ++sample) {
    completePrefill(scheduler, plan, 0.25 * plan.items[0].tokenCount);
    completeDecode(scheduler);
    plan = *scheduler.next({});
  }
  require(plan.items[0].tokenCount > 64 && plan.items[0].tokenCount < 2048,
          "prefill budget did not adapt when measured execution became faster");
}

void testMeasuredBudgetPreservesPriority() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(2, 1, RequestPriority::Background));
  scheduler.resourcesReady(2, 1);
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "lower-priority work reduced foreground throughput");

  scheduler.submit(request(3, 20'000, RequestPriority::Foreground));
  scheduler.resourcesReady(3, 0);
  scheduler.submit(request(4, 1, RequestPriority::Foreground));
  scheduler.resourcesReady(4, 1);
  completeDecode(scheduler);
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].promptOffset == 2048 &&
              plan.items[0].tokenCount == 128,
          "adaptive prefill exceeded its shared budget");
}

void testUnavailableTimingAndMinimumBudget() {
  for (double wallMilliseconds :
       {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(), 1.0, 1e9}) {
    engine::Scheduler scheduler(0.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), wallMilliseconds);
    scheduler.submit(request(2, 1));
    scheduler.resourcesReady(2, 1);
    completeDecode(scheduler);
    const BatchPlan plan = *scheduler.next({});
    require(plan.items[0].tokenCount ==
                (wallMilliseconds == 1e9 ? 64u : 2048u),
            "missing timing or extreme sample produced an invalid prefill budget");
  }
}

void testMeasuredBudgetDoesNotCountBlockedPeers() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(2, 20'000));
  scheduler.waitForResources(2);
  scheduler.submit(request(3, 20'000));
  require(scheduler.next({})->items[0].tokenCount == 2048,
          "queued or resource-blocked work reduced resident throughput");
}

// Peers that only prefill, none finishing within the measured slice, do not
// contend: the leader keeps the full budget.
void testMeasuredBudgetPreservesPurePrefillThroughput() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  for (uint64_t id = 2; id <= 4; ++id) {
    scheduler.submit(request(id, 20'000));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.width() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount == model::ExecutionLimits::prefillTokenBudget,
          "pure prefill contention lost throughput");
}

void testTinyTailDoesNotDistortPrefillThroughput() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 2056));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 2048.0);
  completePrefill(scheduler, *scheduler.next({}), 800.0);
  scheduler.submit(request(2, 20'000));
  scheduler.resourcesReady(2, 0);
  completeDecode(scheduler);
  require(scheduler.next({})->items[0].tokenCount == 256,
          "fixed overhead from a tiny tail distorted the prefill estimate");
}

void testMeasuredBudgetFinishesShortPrefillPromptly() {
  for (const auto priority : {RequestPriority::Normal,
                              RequestPriority::Background}) {
    engine::Scheduler scheduler(0.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), 4096.0);
    scheduler.submit(request(2, 31, priority));
    scheduler.resourcesReady(2, 0);
    const BatchPlan plan = *scheduler.next({});
    if (priority == RequestPriority::Background) {
      require(plan.width() == 1 && plan.items[0].requestId == 1 &&
                  plan.items[0].tokenCount == 2048,
              "a lower-priority short prefill throttled foreground work");
      continue;
    }
    require(plan.width() == 2 && plan.items[0].requestId == 2 &&
                plan.items[0].tokenCount == 31 &&
                plan.items[1].requestId == 1 &&
                plan.items[1].tokenCount == 97,
            "a finishing short prefill waited for a full long-prefill batch");
    completePrefill(scheduler, plan, 256.0);
    const BatchPlan next = *scheduler.next({});
    require(next.kind == WorkKind::Decode && next.items[0].requestId == 2,
            "the completed short prefill did not get its next decode turn");
  }
}

// At a measured 2.75 ms per row the slice is 128 rows. An arrival that
// finishes within the full budget, though not within one slice, takes a
// command of its own rows instead of a full one shared with a long prompt;
// one that does not finish in it still takes a full command.
void testShortArrivalBesideLongPrefillEndsAtItsLastRow() {
  const auto beside = [](Scheduler &scheduler, std::initializer_list<uint32_t> arrivals) {
    scheduler.observePrefill(2048, 5632.0);
    scheduler.submit(request(1, 20'000));
    scheduler.resourcesReady(1, 0);
    completePrefill(scheduler, *scheduler.next({}), 5632.0);
    uint64_t id = 2;
    for (uint32_t prompt : arrivals) {
      scheduler.submit(request(id, prompt));
      scheduler.resourcesReady(id++, 0);
    }
    return *scheduler.next({});
  };
  Scheduler chat(0.0);
  const BatchPlan plan = beside(chat, {300});
  require(plan.width() == 1 && plan.items[0].requestId == 2 && plan.items[0].tokenCount == 300,
          "a short arrival shared a full command with a long prefill");
  completePrefill(chat, plan, 825.0);
  completeDecode(chat);
  const BatchPlan slice = *chat.next({});
  require(slice.width() == 1 && slice.items[0].requestId == 1 &&
              slice.items[0].tokenCount == 128,
          "the long prefill did not return to slices beside the decoding arrival");

  Scheduler longer(0.0);
  const BatchPlan full = beside(longer, {3000});
  require(full.width() == 1 && full.items[0].requestId == 2 &&
              full.items[0].tokenCount == 2048,
          "an arrival that does not finish in one command lost the full budget");

  Scheduler two(0.0);
  const BatchPlan first = beside(two, {300, 300});
  require(first.width() == 1 && first.items[0].requestId == 2 &&
              first.items[0].tokenCount == 300,
          "two short arrivals shared a command instead of finishing in turn");
}

void testMeasuredBudgetRetainsOvertakingBound() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 20'000));
  scheduler.resourcesReady(1, 0);
  completePrefill(scheduler, *scheduler.next({}), 4096.0);
  scheduler.submit(request(100, 1));
  scheduler.resourcesReady(100, 1);
  for (uint64_t id = 2; id <= 4; ++id) {
    completeDecode(scheduler);
    scheduler.submit(request(id, 128));
    scheduler.resourcesReady(id, 0);
    const BatchPlan plan = *scheduler.next({});
    require(plan.items.size() == 1 && plan.items[0].requestId == id,
            "adaptive prefill did not serve a short arrival first");
    scheduler.commit(plan, {});
    const std::array result{StepResult{id, 128, true, DecodeStage::Regular}};
    scheduler.complete(plan, result, 256.0, true);
  }
  scheduler.submit(request(5, 128));
  scheduler.resourcesReady(5, 0);
  completeDecode(scheduler);
  const BatchPlan overdue = *scheduler.next({});
  require(overdue.items.size() == 1 && overdue.items[0].requestId == 1 &&
              overdue.items[0].tokenCount == 128,
          "adaptive prefill allowed short arrivals to starve an older lane");
}

void testPriorityPrecedesWorkKindAlternation() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, RequestPriority::Background));
  scheduler.submit(request(2, 100, RequestPriority::Foreground));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 0);

  BatchPlan foreground = *scheduler.next({});
  require(foreground.kind == WorkKind::Prefill &&
              foreground.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096, RequestPriority::Foreground));
  scheduler.submit(request(2, 4096, RequestPriority::Background));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);

  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Prefill && plan.items.size() == 1 &&
              plan.items[0].requestId == 1 &&
              plan.items[0].tokenCount ==
                  model::ExecutionLimits::prefillTokenBudget,
          "ragged prefill mixed priority tiers in one command");
}

void testDecodeCommandContainsOnePriorityTier() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1, RequestPriority::Foreground));
  scheduler.submit(request(2, 1, RequestPriority::Background));
  scheduler.resourcesReady(1, 1);
  scheduler.resourcesReady(2, 1);

  BatchPlan plan = *scheduler.next({});
  require(plan.kind == WorkKind::Decode && plan.width() == 1 &&
              plan.items[0].requestId == 1,
          "a decode command mixed priority tiers");
  scheduler.commit(plan, {});
  const std::array result{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(plan, result, 0.0, true);

  BatchPlan background = *scheduler.next({});
  require(background.kind == WorkKind::Decode && background.width() == 1 &&
              background.items[0].requestId == 2,
          "background decode did not run after foreground completion");
}

void testDecodeLanesRotate() {
  engine::Scheduler scheduler(0.0);
  for (uint64_t id = 1; id <= 6; ++id) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 1);
  }

  BatchPlan first = *scheduler.next({});
  require(first.kind == WorkKind::Decode && first.width() == 4,
          "the first unconstrained decode did not use the real B4 graph");
  scheduler.commit(first, {});
  std::vector<StepResult> firstResults;
  for (const BatchItem &item : first.items) {
    firstResults.push_back(
        {item.requestId, 0, false, DecodeStage::Regular});
  }
  scheduler.complete(first, firstResults, 0.0, true);

  BatchPlan second = *scheduler.next({});
  require(second.kind == WorkKind::Decode && second.width() == 4 &&
              second.items[0].requestId == 5 && second.items[1].requestId == 6,
          "decode did not prioritize lanes omitted by the previous batch");
}

void testMaskStagesNeverMix() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 1));
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(1, 0);
  scheduler.resourcesReady(2, 0);

  BatchPlan prompts = *scheduler.next({});
  require(prompts.kind == WorkKind::Prefill && prompts.width() == 2,
          "constrained prompts did not form one prefill");
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);

  BatchPlan initialSelection = *scheduler.next({});
  require(initialSelection.width() == 1 &&
              initialSelection.items[0].requestId == 1 &&
              initialSelection.decodeStage == DecodeStage::ApplyInitialMask,
          "a ready first mask did not get its selection plan");
  scheduler.commit(initialSelection, {});
  const std::array initialSelectionResult{
      StepResult{1, 0, false, DecodeStage::Regular}};
  scheduler.complete(initialSelection, initialSelectionResult, 0.0, true);

  scheduler.maskReady(2);
  BatchPlan otherInitial = *scheduler.next({});
  require(otherInitial.width() == 1 &&
              otherInitial.items[0].requestId == 2 &&
              otherInitial.decodeStage == DecodeStage::ApplyInitialMask,
          "a first token's selection mixed with a drafting lane");
  scheduler.commit(otherInitial, {});
  const std::array otherInitialResult{
      StepResult{2, 0, true, DecodeStage::Regular}};
  scheduler.complete(otherInitial, otherInitialResult, 0.0, true);

  BatchPlan verify = *scheduler.next({});
  require(verify.width() == 1 && verify.items[0].requestId == 1 &&
              verify.decodeStage == DecodeStage::Regular,
          "a drafting lane mixed with a first token's selection");
  scheduler.commit(verify, {});
  const std::array verifyResult{
      StepResult{1, 0, true, DecodeStage::Regular}};
  scheduler.complete(verify, verifyResult, 0.0, true);
}

// The first tokens of constrained requests of one priority whose masks have
// arrived are selected in one plan.
void testInitialSelectionsBatch() {
  engine::Scheduler scheduler(0.0);
  for (uint64_t id : {1, 2}) {
    scheduler.submit(request(id, 1));
    scheduler.resourcesReady(id, 0);
  }
  const BatchPlan prompts = *scheduler.next({});
  scheduler.commit(prompts, {});
  const std::array promptResults{
      StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
      StepResult{2, 1, false, DecodeStage::ApplyInitialMask},
  };
  scheduler.complete(prompts, promptResults, 0.0, true);
  scheduler.maskReady(1);
  scheduler.maskReady(2);
  const BatchPlan selection = *scheduler.next({});
  require(selection.kind == WorkKind::Decode && selection.width() == 2 &&
              selection.decodeStage == DecodeStage::ApplyInitialMask,
          "first-token selections of one priority did not share a plan");
}

void testWaitingMaskBoundsPeerPrefill() {
  for (const RequestPriority priority : {RequestPriority::Foreground,
                                         RequestPriority::Normal,
                                         RequestPriority::Background}) {
    Scheduler scheduler(0.0);
    scheduler.observePrefill(2048, 4096.0);
    scheduler.submit(request(1, 1, priority));
    scheduler.resourcesReady(1, 0);
    awaitFirstMask(scheduler, 1);

    scheduler.submit(request(2, 20'000));
    scheduler.resourcesReady(2, 0);
    const BatchPlan prefill = *scheduler.next({});
    const bool protectedPeer = priority <= RequestPriority::Normal;
    require(prefill.kind == WorkKind::Prefill &&
                prefill.items[0].requestId == 2 &&
                prefill.items[0].tokenCount == (protectedPeer ? 128u : 2048u),
            "mask wait lost prefill latency protection or priority ordering");
    // A mask arriving during the command must get the next decode turn.
    scheduler.maskReady(1);
    completePrefill(scheduler, prefill);
    if (protectedPeer) {
      const BatchPlan decode = *scheduler.next({});
      require(decode.kind == WorkKind::Decode &&
                  decode.items[0].requestId == 1,
              "ready mask did not resume after bounded prefill");
    }
    scheduler.cancel(1);
    require(scheduler.next({})->items[0].tokenCount == 2048,
            "cancelled mask request kept isolated prefill throttled");
  }
}

void testResourceSuspensionReplaysFromCache() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(1, 4096));
  scheduler.resourcesReady(1, 0);
  BatchPlan first = *scheduler.next({});
  completePrefill(scheduler, first);
  require(!scheduler.suspended(1), "a running request was suspended");
  scheduler.suspendForResources(1);
  require(scheduler.phase(1) == engine::Phase::WaitingResources && scheduler.suspended(1),
          "resident prefill did not enter resource wait");
  scheduler.resumeFromResources(1, 32, 4096);
  require(!scheduler.suspended(1), "a resumed request stayed suspended");
  BatchPlan resumed = *scheduler.next({});
  require(resumed.kind == WorkKind::Prefill &&
              resumed.items[0].promptOffset == 32,
          "resource resume did not start from its acquired cache boundary");

  // A request that ends while suspended stays so until it is removed, and
  // submission order follows the submits.
  engine::Scheduler ended(0.0);
  ended.submit(request(3, 64));
  ended.submit(request(4, 64));
  require(ended.submissionOrder(3) < ended.submissionOrder(4),
          "submission order did not follow the submits");
  ended.resourcesReady(3, 0);
  completePrefill(ended, *ended.next({}));
  ended.suspendForResources(3);
  ended.cancel(3);
  require(ended.phase(3) == engine::Phase::Cancelled && ended.suspended(3),
          "a cancelled request forgot it was suspended");
  ended.remove(3);
}

// A request suspended while it holds its first mask replays its history and
// selects its first token under that mask, without asking for another: the
// replay's prefill reports no mask wait.
void testReplayKeepsHeldInitialMask() {
  engine::Scheduler scheduler(0.0);
  scheduler.submit(request(2, 1));
  scheduler.resourcesReady(2, 0);
  awaitFirstMask(scheduler, 2);
  scheduler.maskReady(2);
  scheduler.suspendForResources(2);
  scheduler.resumeFromResources(2, 32, 40);
  const auto replay = *scheduler.next({});
  require(replay.kind == WorkKind::Prefill &&
              replay.items[0].promptOffset == 32 &&
              replay.items[0].tokenCount == 8,
          "decode resume did not replay its committed token history");
  completePrefill(scheduler, replay);
  require(scheduler.phase(2) == engine::Phase::Decode,
          "the replay waited for a second first mask");
  const auto continuation = *scheduler.next({});
  require(continuation.kind == WorkKind::Decode &&
              continuation.decodeStage == DecodeStage::ApplyInitialMask,
          "the replay dropped the held first mask's selection");
}

} // namespace

int main() {
  try {
    testAdmissionSharesDispatchOrderAndBudget();
    testAdmissionRespectsContendedBudgetAndDecodePriority();
    testHighestRunnablePriority();
    testQueuedPrefillCannotBeOvertakenIndefinitely();
    testWarmupTimingSeedsFirstContendedCommand();
    testFixedPrefillCostSizesContendedSlice();
    testPrefillGroupHasOneSpanningLane();
    testPrefillGroupsRunUncontended();
    testPrefillPassesBoundCommands();
    testShortPromptsShareACommandInPasses();
    testArrivalsRegroupACommandInPasses();
    testPassesShareTheRowsThatSpareACommand();
    testRegroupsKeepTheCommandsLanes();
    testRegroupsWeighTheRowsThatJoin();
    testPrefillEstimateFitsCommandsOfDifferentRows();
    testContendedCommandsStayWithinTheSliceAsTheyDeepen();
    testPrefillEstimateBoundsItsFit();
    testPassesShareOnlyWhatTheFixedPartOutweighs();
    testAHigherPriorityPromptsRowsDoNotCountAgainstIt();
    testAHigherPriorityWaitsForACommandPastItsMiddle();
    testACommandRunsAtThePriorityOfWhatItHoldsBack();
    testANearEqualLeaderKeepsThePassesRules();
    testCommandsInPassesEndUnderArrivals();
    testPassesYieldToAHigherPriority();
    testCommandBetweenPassesHoldsItsLanes();
    testDeepPrefillKeepsANativeChunksAttention();
    testStreamingLaneRunsAlone();
    testContendedStreamingLaneHalvesFromItsGroup();
    testStreamEstimateIgnoresCommandsThatReadNothing();
    testStreamWaitStaysOutOfTheCommandEstimate();
    testImageEncodesStayOutOfTheCommandEstimate();
    testImageEncodesStayOutOfAPassedCommandsEstimate();
    testDecodeGapAcrossStreamingPrefill();
    testShortestRemainingFirstUsesActualRows();
    testShortenedPrefillResumesFromItsRows();
    testEqualPromptsFinishInArrivalOrder();
    testShortArrivalPrecedesLongColdPrompt();
    testEqualLanesRunInArrivalOrderWithoutOvertaking();
    testLaneOvertakenThreeTimesLeadsTheNextCommand();
    testServedLaneResetsOvertaking();
    testOldestNearEqualPromptLeads();
    testNearEqualBandCountsTheShortestLanesCommands();
    testNearEqualLeaderKeepsTheRestShortestFirst();
    testOverdueLaneLeadsAnOlderLane();
    testNearEqualOrderStaysWithinAPriority();
    testAdmissionStartsTheLanesOfTheNextCommand();
    testRealDecodeWidths();
    testUnconstrainedLanesShareDecode();
    testRejectedCommitCountsNothing();
    testMaskedLanesShareDecode();
    testPrefillAndDecodeAlternateWithoutStarvation();
    testDecodeRepaysItsShareOfContendedPrefill();
    testMaskWaitAccruesNoDecodeDebt();
    testExcludedLanesAreNotPlanned();
    testDecodeDebtLeavesWithTheLastDecoder();
    testHigherPriorityPrefillPrecedesDecodeDebt();
    testMeasuredBudgetOnlyLimitsContendedWork();
    testAuxiliaryWorkDoesNotTrainTextPrefillTiming();
    testMeasuredBudgetUsesActualRowsAndRecovers();
    testMeasuredBudgetPreservesPriority();
    testUnavailableTimingAndMinimumBudget();
    testMeasuredBudgetDoesNotCountBlockedPeers();
    testMeasuredBudgetPreservesPurePrefillThroughput();
    testTinyTailDoesNotDistortPrefillThroughput();
    testMeasuredBudgetFinishesShortPrefillPromptly();
    testShortArrivalBesideLongPrefillEndsAtItsLastRow();
    testMeasuredBudgetRetainsOvertakingBound();
    testPriorityPrecedesWorkKindAlternation();
    testPrefillCommandContainsOnePriorityTier();
    testDecodeCommandContainsOnePriorityTier();
    testDecodeLanesRotate();
    testMaskStagesNeverMix();
    testInitialSelectionsBatch();
    testWaitingMaskBoundsPeerPrefill();
    testResourceSuspensionReplaysFromCache();
    testReplayKeepsHeldInitialMask();
    std::cout << "ragged scheduler tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "ragged scheduler tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
