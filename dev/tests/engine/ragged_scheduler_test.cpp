// The scheduler (engine::Scheduler): its phases, decode commands and debt,
// and the prefill rules that one price, CostModel's, decides: grouping, the
// bound and passes, the per-layer pair bound, joins, order and aging, and
// superseding a command between its passes. Prefill targets are the fakes'
// fitted profiles (TimingProfile.cpp).

#include "TestChecks.hpp"
#include "TestRegistry.hpp"
#include "TimingProfile.hpp"
#include "engine/Scheduler.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using test::rejects;
using test::require;

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
constexpr uint32_t kPageTokens = SPLASH_TARGET_KV_BLOCK_TOKENS;
constexpr double kGap = 500.0;

model::CommandLimits limits(uint32_t commandRows = kChunk, uint32_t passes = 1, uint32_t indexPageLanes = 1,
                            uint64_t attentionPairs = std::numeric_limits<uint64_t>::max()) {
  return {.commandRows = commandRows,
          .maximumPasses = passes,
          .lanesPastChunk = 1,
          .indexPageLanes = indexPageLanes,
          .attentionPairs = attentionPairs};
}

RequestSpec request(uint64_t id, uint32_t prompt, RequestPriority priority = RequestPriority::Normal) {
  return {id, priority, prompt};
}

// The turnarounds u the bound and pass tests run at until I-model measures
// it (TimingProfile.cpp: the profiles' 60 ms is the middle one).
constexpr std::array<double, 3> kTurnarounds{5.0, 60.0, 150.0};

test::TimingProfile withTurnaround(const test::TimingProfile &profile, double turnaround) {
  test::TimingProfile result = profile;
  result.turnaroundMilliseconds = turnaround;
  return result;
}

std::vector<test::TimingProfile> atEveryTurnaround(std::initializer_list<const test::TimingProfile *> profiles) {
  std::vector<test::TimingProfile> result;
  for (const test::TimingProfile *profile : profiles)
    for (const double turnaround : kTurnarounds)
      result.push_back(withTurnaround(*profile, turnaround));
  return result;
}

std::string at(const test::TimingProfile &profile) {
  return std::string(profile.name) + " at u " + std::to_string(int(profile.turnaroundMilliseconds)) + " ms";
}

// A target, its scheduler and what the engine gives the scheduler: the
// lanes' shapes, the lanes left out, the command between its passes, which
// run() keeps as the engine's PassCommand does, and for admission the starts
// restoring their prefix. Commands take the profile's time on a clock.
class Sim {
public:
  explicit Sim(const test::TimingProfile &profile, model::CommandLimits commandLimits = limits(),
               double decodeShare = 0.5, std::function<double()> residentShare = {})
      : work(profile, std::move(residentShare)),
        scheduler({.decodeShare = decodeShare, .decodeGapMilliseconds = kGap, .limits = commandLimits}, work) {}

  test::TimedWork work;
  Scheduler scheduler;
  std::vector<LaneShape> shapes;
  std::vector<uint64_t> excluded;
  std::optional<PassProgress> passes;
  std::vector<PrefillAdmission> restoring;
  double now = 0.0;

  // The startup's two prefill warm-ups (a prompt of one token, as
  // Runtime::warmupPrefill) and a load of the weights at the profile's SSD
  // rate.
  Sim &seed() {
    const std::array warmups{test::sampleOf(work.warmupTimes(kChunk)), test::sampleOf(work.warmupTimes(256))};
    const test::TimingProfile &profile = work.profile();
    const double rate =
        profile.weightBytesPerMillisecond ? profile.weightBytesPerMillisecond : profile.spilledKvBytesPerMillisecond;
    scheduler.seedCost(warmups, rate ? model::SsdReads{10'000'000'000, 1e10 / rate} : model::SsdReads{});
    return *this;
  }
  [[nodiscard]] PlanInputs in() const { return {excluded, shapes, passes ? &*passes : nullptr, restoring}; }
  [[nodiscard]] std::optional<BatchPlan> next() const { return scheduler.next(in()); }
  [[nodiscard]] BatchPlan plan() const {
    const std::optional<BatchPlan> result = next();
    require(result.has_value(), "the scheduler planned nothing");
    return *result;
  }
  // A resident prefill lane at `position` with `remaining` rows to go, and a
  // decoding lane.
  void lane(uint64_t id, uint32_t position, uint32_t remaining, RequestPriority priority = RequestPriority::Normal) {
    scheduler.submit(request(id, position + remaining, priority));
    scheduler.resourcesReady(id, position);
  }
  void decoder(uint64_t id, RequestPriority priority = RequestPriority::Normal) {
    scheduler.submit(request(id, 1, priority));
    scheduler.resourcesReady(id, 1);
  }
  void shape(uint64_t id, uint32_t indexPages, bool gainsIndexPages = false) {
    std::erase_if(shapes, [id](const LaneShape &shape) { return shape.requestId == id; });
    shapes.push_back({.requestId = id, .indexPages = indexPages, .gainsIndexPages = gainsIndexPages});
  }
  [[nodiscard]] std::vector<model::PrefillLane> lanes(std::span<const BatchItem> items) const {
    std::vector<model::PrefillLane> result;
    for (const BatchItem &item : items) {
      const auto found = std::ranges::find(shapes, item.requestId, &LaneShape::requestId);
      result.push_back({item.promptOffset, item.tokenCount, found == shapes.end() ? 0 : found->indexPages});
    }
    return result;
  }
  // The scheduler's price of pass `pass` of `passes` of these items, and of
  // the passes [first, passes).
  [[nodiscard]] double price(std::span<const BatchItem> items, uint32_t pass = 0, uint32_t count = 1) const {
    return scheduler.prefillMilliseconds(work.prefillWork(lanes(items), pass, count));
  }
  [[nodiscard]] double passesPrice(std::span<const BatchItem> items, uint32_t first, uint32_t count) const {
    double total = 0.0;
    for (uint32_t pass = first; pass < count; ++pass)
      total += price(items, pass, count);
    return total;
  }
  [[nodiscard]] double turnaround() const {
    return scheduler.snapshot().prefillEstimate.terms[static_cast<size_t>(CostTerm::Units)].millisecondsPerUnit;
  }
  [[nodiscard]] double bound() const { return std::max(kGap, turnaround() / kPassOverheadShare); }
  // Runs `plan` as the engine does, its time the profile's (times `scale`):
  // a prefill's lanes consume their rows at its last pass, a decode's commit
  // eight tokens each.
  double run(const BatchPlan &plan, double scale = 1.0) {
    // A plan that supersedes the command between its passes abandons it
    // first (Engine::tick).
    if (scheduler.abandonsPasses(plan, in()))
      abandon();
    const std::vector<uint64_t> leftOut = scheduler.commit(plan, in());
    if (plan.kind == WorkKind::Prefill && plan.passes > 1 && !plan.pass)
      passes = PassProgress{.plan = plan, .leftOut = leftOut};
    test::CommandTimes times;
    std::vector<StepResult> results;
    const bool last = plan.pass + 1 == plan.passes;
    if (plan.kind == WorkKind::Prefill) {
      times = test::prefillTimes(work.profile(), work.prefillWork(lanes(plan.items), plan.pass, plan.passes));
      for (const BatchItem &item : plan.items)
        results.push_back({item.requestId, last ? item.tokenCount : 0, false, DecodeStage::Regular});
    } else {
      times = test::decodeTimes(work.profile(), plan.width(), plan.width() * 8, work.residentShare(), 0);
      for (const BatchItem &item : plan.items)
        results.push_back({item.requestId, 0, false, DecodeStage::Regular, 8});
    }
    times.wallMilliseconds *= scale;
    scheduler.complete(plan, results, test::sampleOf(times));
    if (plan.kind == WorkKind::Prefill && passes) {
      if (last) {
        passes.reset();
      } else {
        ++passes->plan.pass;
        passes->elapsedMilliseconds += times.wallMilliseconds;
      }
    }
    now += times.wallMilliseconds;
    return times.wallMilliseconds;
  }
  double step(double scale = 1.0) { return run(plan(), scale); }
  // Abandons the command between its passes, as Engine::abandonPasses does.
  void abandon() {
    scheduler.abandoned(*passes);
    passes.reset();
  }
};

// A dense 27B target of one chunk a command, seeded: what joining and the
// order weigh is its price.
class DenseSim final : public Sim {
public:
  explicit DenseSim(double decodeShare = 0.0) : Sim(test::kDense27B, limits(), decodeShare) { seed(); }
};

uint32_t rowsOf(const BatchPlan &plan) {
  uint32_t rows = 0;
  for (const BatchItem &item : plan.items)
    rows += item.tokenCount;
  return rows;
}

void completeDecode(Sim &sim, bool finished = false) {
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode, "expected a decode command");
  sim.scheduler.commit(plan, sim.in());
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back({item.requestId, 0, finished, DecodeStage::Regular, 1});
  sim.scheduler.complete(plan, results, {.wallMilliseconds = 50.0});
}

// Commits a prefill unit and completes it in `sample`'s time, its lanes
// consuming their rows at its last pass.
void completePrefill(Sim &sim, const BatchPlan &plan, const model::CommandSample &sample = {}) {
  require(plan.kind == WorkKind::Prefill, "expected a prefill command");
  sim.scheduler.commit(plan, sim.in());
  std::vector<StepResult> results;
  for (const BatchItem &item : plan.items)
    results.push_back(
        {item.requestId, plan.pass + 1 == plan.passes ? item.tokenCount : 0, false, DecodeStage::Regular});
  sim.scheduler.complete(plan, results, sample);
}

// The M6 TTFT bench's rounds ahead of its four turns (DATA.md): the 8K
// prefix's command, then a turn of 300 rows on it alone and two together,
// each leaving at its first token, so the price has their commands.
void runTtftBenchRounds(Sim &sim) {
  struct Round final {
    uint32_t position, rows, turns;
  };
  uint64_t id = 1000;
  for (const Round round : {Round{0, 8192, 1}, Round{8192, 300, 1}, Round{8192, 300, 2}}) {
    std::vector<uint64_t> waiting;
    for (uint32_t turn = 0; turn < round.turns; ++turn) {
      sim.lane(id, round.position, round.rows);
      waiting.push_back(id++);
    }
    while (!waiting.empty()) {
      sim.step();
      std::erase_if(waiting, [&](uint64_t lane) {
        if (sim.scheduler.phase(lane) == Phase::Prefill)
          return false;
        sim.scheduler.cancel(lane);
        sim.scheduler.remove(lane);
        return true;
      });
    }
  }
}

// Lane 1 decodes beside lane 2's long prompt: a decode command, then a
// prefill unit of 500 ms, which owes decode its share of that.
void runContendedPrefill(Sim &sim) {
  sim.decoder(1);
  sim.lane(2, 0, 20'000);
  completeDecode(sim);
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "the long prompt did not follow the first decode command");
  completePrefill(sim, prefill, {.wallMilliseconds = 500.0});
}

// Prefills a constrained request's prompt, whose end waits for the mask of
// its first token.
void awaitFirstMask(Sim &sim, uint64_t id) {
  BatchPlan prompt = sim.plan();
  require(prompt.kind == WorkKind::Prefill && prompt.width() == 1 && prompt.items[0].requestId == id,
          "expected the constrained prompt");
  while (prompt.pass + 1 < prompt.passes) {
    sim.run(prompt);
    prompt = sim.plan();
  }
  sim.scheduler.commit(prompt, sim.in());
  const std::array result{StepResult{id, prompt.items[0].tokenCount, false, DecodeStage::ApplyInitialMask}};
  sim.scheduler.complete(prompt, result, {});
  sim.passes.reset();
  require(sim.scheduler.phase(id) == Phase::WaitingMask, "a constrained prompt did not wait for its first mask");
}

// --- Phases, decode and debt -------------------------------------------------------------------------------

void testRealDecodeWidths() {
  for (uint32_t width = 1; width <= 4; ++width) {
    DenseSim sim;
    for (uint32_t lane = 0; lane < width; ++lane)
      sim.decoder(lane + 1);
    const BatchPlan plan = sim.plan();
    require(plan.kind == WorkKind::Decode && plan.width() == width && rowsOf(plan) == 0,
            "ready decode width was delayed or rewritten");
    sim.run(plan);
    require(sim.scheduler.snapshot().decodeBatchesByWidth[width - 1] == 1,
            "decode width counter did not record the real command");
  }
}

void testRejectedCommitCountsNothing() {
  DenseSim sim;
  sim.decoder(1);
  sim.decoder(2);
  const BatchPlan stale = sim.plan();
  sim.scheduler.cancel(2);
  rejects([&] { static_cast<void>(sim.scheduler.commit(stale, sim.in())); },
          "batch no longer matches scheduler state", "a commit of a cancelled request's batch was accepted");
  require(sim.scheduler.snapshot().decodeBatches == 0, "rejected commit incremented decode counters");
  completeDecode(sim);
  require(sim.scheduler.snapshot().decodeBatches == 1, "the remaining request's decode was not counted");
}

void testDecodeLanesRotate() {
  DenseSim sim;
  for (uint64_t id = 1; id <= 6; ++id)
    sim.decoder(id);
  const BatchPlan first = sim.plan();
  require(first.kind == WorkKind::Decode && first.width() == 4, "the first decode did not use the real B4 graph");
  sim.run(first);
  const BatchPlan second = sim.plan();
  require(second.width() == 4 && second.items[0].requestId == 5 && second.items[1].requestId == 6,
          "decode did not prioritize lanes omitted by the previous batch");
}

void testMaskStagesNeverMix() {
  DenseSim sim;
  sim.lane(1, 0, 1);
  sim.lane(2, 0, 1);
  const BatchPlan prompts = sim.plan();
  require(prompts.kind == WorkKind::Prefill && prompts.width() == 2, "constrained prompts did not form one prefill");
  sim.scheduler.commit(prompts, sim.in());
  const std::array promptResults{StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
                                 StepResult{2, 1, false, DecodeStage::ApplyInitialMask}};
  sim.scheduler.complete(prompts, promptResults, {});
  sim.scheduler.maskReady(1);
  const BatchPlan initial = sim.plan();
  require(initial.width() == 1 && initial.items[0].requestId == 1 &&
              initial.decodeStage == DecodeStage::ApplyInitialMask,
          "a ready first mask did not get its selection plan");
  sim.scheduler.commit(initial, sim.in());
  const std::array selected{StepResult{1, 0, false, DecodeStage::Regular}};
  sim.scheduler.complete(initial, selected, {});
  sim.scheduler.maskReady(2);
  const BatchPlan other = sim.plan();
  require(other.width() == 1 && other.items[0].requestId == 2 && other.decodeStage == DecodeStage::ApplyInitialMask,
          "a first token's selection mixed with a drafting lane");
}

void testPrefillAndDecodeAlternateWithoutStarvation() {
  DenseSim sim;
  sim.decoder(1);
  sim.lane(2, 0, 10'000);
  require(sim.plan().kind == WorkKind::Decode, "decode did not retain the first ready command");
  completeDecode(sim);
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2,
          "continuous decode starved a newly admitted prefill");
  sim.run(prefill);
  require(sim.plan().kind == WorkKind::Decode, "prefill did not yield the next command back to decode");
}

// Lane 1 decodes beside lane 2's long prompt: one decode command, then one
// prefill unit of 500 ms owes half of it at a share of 0.5.
void testDecodeRepaysItsShareOfContendedPrefill() {
  for (const double share : {0.0, 0.5}) {
    DenseSim sim(share);
    sim.decoder(1);
    sim.lane(2, 0, 20'000);
    completeDecode(sim);
    const BatchPlan prefill = sim.plan();
    sim.scheduler.commit(prefill, sim.in());
    std::vector<StepResult> results;
    for (const BatchItem &item : prefill.items)
      results.push_back({item.requestId, prefill.pass + 1 == prefill.passes ? item.tokenCount : 0, false,
                         DecodeStage::Regular});
    sim.scheduler.complete(prefill, results, {.wallMilliseconds = 500.0});
    for (uint32_t step = 0; step < (share > 0.0 ? 5u : 1u); ++step)
      completeDecode(sim);
    require(sim.plan().kind == WorkKind::Prefill, "decode did not return the next command once its share was repaid");
  }
}

// The decode time per token is a decode command's wall over the tokens a
// lane committed, each command weighing 0.9 of the next; the tokens a lane
// commits per command are their mean the same way. A command whose lanes
// committed none trains neither.
void testDecodeTimePerTokenWeighsEachCommand() {
  DenseSim sim;
  sim.decoder(1);
  sim.decoder(2);
  const auto decode = [&](uint32_t committed, double wallMilliseconds) {
    const BatchPlan plan = sim.plan();
    sim.scheduler.commit(plan, sim.in());
    std::vector<StepResult> results;
    for (const BatchItem &item : plan.items)
      results.push_back({item.requestId, 0, false, DecodeStage::Regular, committed});
    sim.scheduler.complete(plan, results, {.wallMilliseconds = wallMilliseconds});
  };
  require(sim.scheduler.decodeTokenMilliseconds() == 0.0 && sim.scheduler.decodeTokensPerLaneCommand() == 0.0,
          "a decode time per token preceded the first decode");
  decode(9, 18.0);
  decode(4, 12.0);
  const double perToken = (0.9 * 2.0 + 3.0) / (0.9 + 1.0);
  const double tokens = (0.9 * 9.0 + 4.0) / (0.9 + 1.0);
  require(std::abs(sim.scheduler.decodeTokenMilliseconds() - perToken) < 1e-9 &&
              std::abs(sim.scheduler.decodeTokensPerLaneCommand() - tokens) < 1e-9,
          "the decode time per token did not weigh each command 0.9 of the next");
  decode(0, 30.0);
  require(std::abs(sim.scheduler.decodeTokenMilliseconds() - perToken) < 1e-9 &&
              sim.scheduler.snapshot().decodeEstimate.samples == 2,
          "a decode command whose lanes committed no tokens trained the decode time per token");
}

// Lanes left out of a plan are not planned; a prefill owes nothing to the
// decoders it was planned without.
void testExcludedLanesAreNotPlanned() {
  DenseSim sim(0.5);
  sim.decoder(1);
  sim.decoder(2);
  sim.lane(3, 0, 20'000);
  sim.excluded = {1};
  const BatchPlan decode = sim.plan();
  require(decode.kind == WorkKind::Decode && decode.width() == 1 && decode.items[0].requestId == 2,
          "a decode plan included a lane left out of it");
  sim.excluded = {1, 2};
  const BatchPlan prefill = sim.plan();
  require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 3,
          "the prefill did not run while every decoder was left out");
  sim.run(prefill);
  sim.excluded.clear();
  completeDecode(sim);
  require(sim.plan().kind == WorkKind::Prefill, "a prefill owed decode time to decoders it was planned without");
}

void testPriorityPrecedesWorkKindAlternation() {
  DenseSim sim;
  sim.decoder(1, RequestPriority::Background);
  sim.lane(2, 0, 100, RequestPriority::Foreground);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 2,
          "work-kind alternation overrode request priority");
}

void testPrefillCommandContainsOnePriorityTier() {
  DenseSim sim;
  sim.lane(1, 0, 100, RequestPriority::Foreground);
  sim.lane(2, 0, 100, RequestPriority::Background);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1, "a prefill command mixed priority tiers");
}

// The tier admission can select is set by the lanes that prefill or decode,
// not by those waiting for a mask or for admission.
void testHighestRunnablePriority() {
  DenseSim sim;
  require(!sim.scheduler.highestRunnablePriority(), "an idle scheduler had a runnable tier");
  sim.lane(1, 0, 1, RequestPriority::Foreground);
  awaitFirstMask(sim, 1);
  sim.decoder(2, RequestPriority::Background);
  sim.lane(3, 0, 100);
  sim.scheduler.submit(request(4, 100, RequestPriority::Foreground));
  require(sim.scheduler.highestRunnablePriority() == RequestPriority::Normal,
          "a mask wait or a queued request set the runnable tier");
}

void testResourceSuspensionReplaysFromCache() {
  DenseSim sim;
  sim.lane(1, 0, 4096);
  sim.step();
  sim.scheduler.suspendForResources(1);
  require(sim.scheduler.phase(1) == Phase::WaitingResources && sim.scheduler.suspended(1),
          "resident prefill did not enter resource wait");
  sim.scheduler.resumeFromResources(1, 32, 4096);
  const BatchPlan resumed = sim.plan();
  require(resumed.kind == WorkKind::Prefill && resumed.items[0].promptOffset == 32,
          "resource resume did not start from its acquired cache boundary");
}

// Admission starts only the candidates the next command serves.
void testAdmissionStartsTheLanesOfTheNextCommand() {
  DenseSim sim;
  sim.lane(1, 0, 41'000);
  sim.scheduler.submit(request(2, 38'000));
  sim.scheduler.submit(request(3, 30'000));
  const std::array queued{PrefillAdmission{2, 0}, PrefillAdmission{3, 0}};
  require(sim.scheduler.prefillAdmissionOrder(queued, sim.in()) == std::vector<uint64_t>{3},
          "admission did not start the candidate the next command leads with");
}

// Admission starts at most a full batch: of eight agent turns that share a
// command, the first four (after the TTFT bench's rounds, as there).
void testAdmissionStartsAtMostAFullBatch() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  runTtftBenchRounds(sim);
  std::vector<PrefillAdmission> turns;
  for (uint64_t id = 1; id <= 8; ++id) {
    sim.scheduler.submit(request(id, 8192 + 300));
    turns.push_back({id, 8192});
  }
  require(sim.scheduler.prefillAdmissionOrder(turns, sim.in()) == std::vector<uint64_t>({1, 2, 3, 4}),
          "admission of short prompts lost their sharing or exceeded the real width");
}

// Admission ranks candidates against every decoder: a higher priority's
// decode holds a lower priority's prompts back until it leaves.
void testAdmissionWaitsForAHigherPrioritysDecode() {
  DenseSim sim;
  std::vector<PrefillAdmission> candidates;
  for (uint64_t id = 1; id <= 2; ++id) {
    sim.scheduler.submit(request(id, 65));
    candidates.push_back({id, 0});
  }
  require(!sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(), "admission started nothing");
  sim.decoder(5, RequestPriority::Foreground);
  require(sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(),
          "a lower priority's prefill started ahead of a runnable Foreground decode");
  sim.scheduler.cancel(5);
  require(!sim.scheduler.prefillAdmissionOrder(candidates, sim.in()).empty(),
          "prefill admission did not resume after the Foreground decode left");
}

void testUnconstrainedLanesShareDecode() {
  DenseSim sim;
  for (uint64_t id = 1; id <= 4; ++id)
    sim.decoder(id);
  require(sim.plan().width() == 4, "unconstrained requests did not share one decode command");
}

// Two constrained prompts wait for their first masks; their first tokens'
// selections share a plan of their stage, and once selected they decode with
// the other lanes.
void testMaskedLanesShareDecode() {
  for (const bool selected : {false, true}) {
    DenseSim sim;
    sim.lane(1, 0, 1);
    sim.lane(2, 0, 1);
    const BatchPlan prompts = sim.plan();
    sim.scheduler.commit(prompts, sim.in());
    const std::array promptResults{StepResult{1, 1, false, DecodeStage::ApplyInitialMask},
                                   StepResult{2, 1, false, DecodeStage::ApplyInitialMask}};
    sim.scheduler.complete(prompts, promptResults, {});
    sim.scheduler.maskReady(1);
    sim.scheduler.maskReady(2);
    if (!selected) {
      const BatchPlan selection = sim.plan();
      require(selection.kind == WorkKind::Decode && selection.width() == 2 &&
                  selection.decodeStage == DecodeStage::ApplyInitialMask,
              "first-token selections of one priority did not share a plan");
      continue;
    }
    completeDecode(sim);
    sim.decoder(3);
    sim.decoder(4);
    const BatchPlan plan = sim.plan();
    require(plan.kind == WorkKind::Decode && plan.width() == 4 && plan.decodeStage == DecodeStage::Regular,
            "lanes that waited for their first masks decoded apart from the others");
  }
}

// A prefill beside a lane waiting for its mask owes decode nothing: the lane
// could not have decoded meanwhile.
void testMaskWaitAccruesNoDecodeDebt() {
  DenseSim sim(0.5);
  sim.lane(1, 0, 1);
  awaitFirstMask(sim, 1);
  sim.lane(2, 0, 20'000);
  for (uint32_t command = 0; command < 3; ++command)
    completePrefill(sim, sim.plan(), {.wallMilliseconds = 500.0});
  sim.scheduler.maskReady(1);
  completeDecode(sim);
  require(sim.plan().kind == WorkKind::Prefill, "prefill beside a lane waiting for its mask banked decode debt");
}

// Debt is owed only to lanes that decode: once the last leaves, by its
// decode command's end or a cancel, a later decoder starts without it.
void testDecodeDebtLeavesWithTheLastDecoder() {
  for (const bool finished : {true, false}) {
    DenseSim sim(0.5);
    runContendedPrefill(sim);
    completeDecode(sim, finished);
    if (!finished)
      sim.scheduler.cancel(1);
    sim.decoder(3);
    require(sim.plan().kind == WorkKind::Prefill, "a later decoder inherited debt owed to one that left");
  }
}

void testHigherPriorityPrefillPrecedesDecodeDebt() {
  DenseSim sim(0.5);
  runContendedPrefill(sim);
  sim.lane(3, 0, 20'000, RequestPriority::Foreground);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 3,
          "decode debt overrode a higher-priority prefill");
}

void testDecodeCommandContainsOnePriorityTier() {
  DenseSim sim;
  sim.decoder(1, RequestPriority::Foreground);
  sim.decoder(2, RequestPriority::Background);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode && plan.width() == 1 && plan.items[0].requestId == 1,
          "a decode command mixed priority tiers");
  completeDecode(sim, true);
  const BatchPlan background = sim.plan();
  require(background.kind == WorkKind::Decode && background.width() == 1 && background.items[0].requestId == 2,
          "background decode did not run after foreground completion");
}

// A request suspended while it holds its first mask replays its history and
// selects its first token under that mask, without asking for another.
void testReplayKeepsHeldInitialMask() {
  DenseSim sim;
  sim.lane(2, 0, 1);
  awaitFirstMask(sim, 2);
  sim.scheduler.maskReady(2);
  sim.scheduler.suspendForResources(2);
  sim.scheduler.resumeFromResources(2, 32, 40);
  const BatchPlan replay = sim.plan();
  require(replay.kind == WorkKind::Prefill && replay.items[0].promptOffset == 32 && replay.items[0].tokenCount == 8,
          "decode resume did not replay its committed token history");
  completePrefill(sim, replay);
  require(sim.scheduler.phase(2) == Phase::Decode, "the replay waited for a second first mask");
  const BatchPlan continuation = sim.plan();
  require(continuation.kind == WorkKind::Decode && continuation.decodeStage == DecodeStage::ApplyInitialMask,
          "the replay dropped the held first mask's selection");
}

// A lane waiting for its mask, of the prefill's priority or a higher one,
// bounds the prefill beside it as a decoder does; a lower one does not. A
// mask that arrives during the unit gets the next command; cancelled, the
// lane leaves the prefill unbounded.
void testWaitingMaskBoundsPeerPrefill() {
  for (const RequestPriority priority :
       {RequestPriority::Foreground, RequestPriority::Normal, RequestPriority::Background}) {
    DenseSim sim;
    sim.lane(1, 0, 1, priority);
    awaitFirstMask(sim, 1);
    sim.lane(2, 0, 20'000);
    const BatchPlan prefill = sim.plan();
    const bool bounded = priority <= RequestPriority::Normal;
    require(prefill.kind == WorkKind::Prefill && prefill.items[0].requestId == 2 &&
                (prefill.items[0].tokenCount < kChunk) == bounded,
            "a mask wait lost prefill latency protection or priority ordering");
    sim.scheduler.maskReady(1);
    completePrefill(sim, prefill);
    if (bounded) {
      const BatchPlan decode = sim.plan();
      require(decode.kind == WorkKind::Decode && decode.items[0].requestId == 1,
              "a ready mask did not resume after a bounded prefill");
    }
    sim.scheduler.cancel(1);
    require(sim.plan().items[0].tokenCount == kChunk, "a cancelled mask request kept the prefill bounded");
  }
}

// --- Contention --------------------------------------------------------------------------------------------

// Uncontended work runs whole; beside a decoder of its priority it is
// bounded, and once the decoder ends it runs whole again.
void testOnlyContendedUnitsAreBounded() {
  DenseSim sim;
  sim.lane(1, 0, 20'000);
  require(sim.plan().items[0].tokenCount == kChunk, "uncontended work lost its rows");
  sim.step();
  sim.decoder(2);
  completeDecode(sim);
  const BatchPlan contended = sim.plan();
  require(contended.kind == WorkKind::Prefill && contended.items[0].tokenCount < kChunk &&
              sim.price(contended.items) <= sim.bound() * (1 + 1e-9),
          "a contended unit was not bounded");
  sim.run(contended);
  completeDecode(sim, true);
  require(sim.plan().items[0].tokenCount == kChunk, "prefill did not run whole again once its peer ended");
}

// A lower priority's decode does not bound a prefill; one of its priority
// does.
void testALowerPrioritysDecodeDoesNotBoundAPrefill() {
  DenseSim sim;
  sim.lane(1, 0, 20'000, RequestPriority::Foreground);
  sim.decoder(2, RequestPriority::Background);
  require(sim.plan().items[0].tokenCount == kChunk, "lower-priority work bounded a Foreground prefill");
  sim.decoder(3, RequestPriority::Foreground);
  completeDecode(sim);
  require(sim.plan().items[0].tokenCount < kChunk, "a Foreground decoder left its priority's prefill unbounded");
}

// Neither lanes waiting for admission or memory nor lanes that only prefill
// bound a prefill: nothing of theirs waits for its units.
void testWaitingOrPrefillingPeersDoNotBoundAPrefill() {
  DenseSim sim;
  sim.lane(1, 0, 20'000);
  sim.scheduler.submit(request(2, 20'000));
  sim.scheduler.waitForResources(2);
  sim.scheduler.submit(request(3, 20'000));
  require(sim.plan().items[0].tokenCount == kChunk, "queued or resource-blocked work bounded a resident prefill");
  for (uint64_t id = 4; id <= 6; ++id)
    sim.lane(id, 0, 20'000);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1 && plan.items[0].tokenCount == kChunk &&
              plan.passes == 1,
          "lanes that only prefill bounded their leader");
}

// Behaviour change 7: a ready peer of its priority, with no decoder, does
// not bound a lane that streams its history: the Spill lane leads a group's
// rows beside a long Resident prompt as it does alone, and on a store-less
// target one whose short history does not group runs its chunk whole.
void testAReadyPeerDoesNotBoundAStreamingLeader() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 45056, 40'000);
  sim.shape(1, 45056 / kPageTokens, true);
  const BatchPlan alone = sim.plan();
  sim.lane(2, 0, 200'000);
  const BatchPlan beside = sim.plan();
  require(beside.items[0].requestId == 1 && beside.items[0].tokenCount == alone.items[0].tokenCount &&
              beside.items[0].tokenCount == 8192,
          "a streaming leader beside a ready peer ran " + std::to_string(beside.items[0].tokenCount) +
              " rows, alone " + std::to_string(alone.items[0].tokenCount));
  Sim storeLess(test::kStoreLessSpillFlashNext, limits(8192, 48));
  storeLess.seed();
  storeLess.lane(1, 16384, 20'000);
  storeLess.shape(1, 16384 / kPageTokens);
  storeLess.lane(2, 0, 200'000);
  const BatchPlan chunk = storeLess.plan();
  require(chunk.items[0].requestId == 1 && chunk.items[0].tokenCount == kChunk && chunk.passes == 1,
          "a streaming leader's chunk beside a ready peer ran in " + std::to_string(chunk.passes) + " passes of " +
              std::to_string(chunk.items[0].tokenCount) + " rows");
}

// A unit whose time is missing, not positive or not finite trains nothing;
// one whose completion added work (an auxiliary encoder's rerun) neither.
void testUntimedOrAuxiliaryUnitsTrainNothing() {
  for (const double wall : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(), 1e9}) {
    for (const bool representative : {true, false}) {
      if (std::isfinite(wall) && wall > 0.0 && representative)
        continue;
      DenseSim sim;
      sim.lane(1, 0, 20'000);
      const std::array probe{BatchItem{1, kChunk, 0}};
      const double before = sim.price(probe);
      const BatchPlan plan = sim.plan();
      completePrefill(sim, plan,
                      {.work = sim.work.prefillWork(sim.lanes(plan.items), 0, 1),
                       .wallMilliseconds = wall,
                       .representative = representative});
      require(sim.scheduler.snapshot().prefillEstimate.samples == 0 && sim.price(probe) == before,
              "a unit of " + std::to_string(wall) + " ms" + (representative ? "" : " that added work") +
                  " trained the price");
    }
  }
}

// One lane of a command may run past a chunk (model::CommandLimits::
// lanesPastChunk), and a command holds at most the model's command rows,
// whatever its lanes' depths and lengths.
void testACommandHoldsOneLanePastAChunk() {
  for (const uint32_t depth : {0u, 70'000u, 150'000u, 300'000u})
    for (const uint32_t remaining : {300u, 3000u, 9000u, 20'000u})
      for (const uint32_t peer : {0u, 100'000u})
        for (const uint32_t shortest : {1000u, remaining + 1000}) {
          Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
          sim.seed();
          sim.lane(1, depth, remaining);
          sim.lane(2, peer, shortest);
          sim.lane(3, peer, remaining + 2000);
          sim.lane(4, 0, remaining + 3000);
          const BatchPlan plan = sim.plan();
          uint32_t past = 0, rows = 0;
          for (const BatchItem &item : plan.items) {
            past += rows / kChunk != (rows + item.tokenCount - 1) / kChunk;
            rows += item.tokenCount;
          }
          require(past <= 1 && rows <= 8192, "a command held " + std::to_string(past) +
                                                 " lanes past a chunk or " + std::to_string(rows) + " rows");
        }
}

// --- The price and its seed ---------------------------------------------------------------------------------

// The prefill price is the cost model's for the work the model expects: a
// unit of more rows or deeper costs more, and the seed prices a chunk of
// the warm-ups as they ran.
void testAPrefillsPriceIsTheCostModels() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  const model::PrefillLane chunk{0, kChunk, 0};
  require(sim.scheduler.prefillMilliseconds(sim.work.prefillWork({&chunk, 1}, 0, 1)) == 0.0,
          "an unseeded price was not zero");
  sim.seed();
  const double warmup = sim.work.warmupTimes(kChunk).wallMilliseconds;
  require(std::abs(sim.scheduler.prefillMilliseconds(sim.work.warmupWork(kChunk)) - warmup) < 0.02 * warmup,
          "the seed did not price the warm-up as it ran");
  const model::PrefillLane more{0, 4 * kChunk, 0};
  require(sim.scheduler.prefillMilliseconds(sim.work.prefillWork({&more, 1}, 0, 1)) >
              sim.scheduler.prefillMilliseconds(sim.work.prefillWork({&chunk, 1}, 0, 1)),
          "more rows did not cost more");
}

// A byte of weights a prefill read costs the rate the fit learned of the
// bytes units read, what the memory ledger prices the bytes a pool's prefill
// reads at (MemoryRates::weightReadSecondsPerByte). The prefill price
// charges a model's expected bytes that rate scaled by the share of them
// units read, so on the M6 target, its units reading half the bytes its
// model expects (amend/I.md F1: 0.46-0.59), the bytes a turn reads cost what
// its price charges for the bytes it expects, not half of that.
void testAWeightReadByteIsPricedAsRead() {
  Sim sim(test::kM6FlashNextQ2);
  sim.seed();
  sim.lane(1, 0, 8 * kChunk);
  for (uint32_t command = 0; command < 8; ++command) {
    const BatchPlan plan = sim.plan();
    model::CommandWork read = sim.work.prefillWork(sim.lanes(plan.items), plan.pass, plan.passes);
    read.weightReadBytes /= 2;
    completePrefill(sim, plan, test::sampleOf(test::prefillTimes(sim.work.profile(), read)));
  }
  const CostTermSnapshot weights =
      sim.scheduler.snapshot().prefillEstimate.terms[static_cast<size_t>(CostTerm::WeightReadBytes)];
  require(std::abs(weights.realizedPerExpected - 0.5) < 1e-9, "the units did not read half the expected bytes");
  const model::PrefillLane turn{8192, 300, 0};
  const model::CommandWork expected = sim.work.prefillWork({&turn, 1}, 0, 1);
  model::CommandWork noReads = expected;
  noReads.weightReadBytes = 0;
  const double charged = sim.scheduler.prefillMilliseconds(expected) - sim.scheduler.prefillMilliseconds(noReads);
  const double priced = sim.scheduler.weightReadMillisecondsPerByte() * 0.5 * double(expected.weightReadBytes);
  require(charged > 0.0 && std::abs(priced - charged) < 1e-6 * charged,
          "the bytes a turn reads were priced " + std::to_string(priced) + " ms against the " +
              std::to_string(charged) + " ms its expected bytes are charged");
}

// S-P5: a cold start's first contended agent turn is never underpriced, as
// the base's host-wait seed prices it (1119.9 ms for a 3226.8 ms turn), and
// is priced within twice its time, so its first sample trains unclamped
// (CostModel::observe counts a time within half and twice the price, the
// clamp amend/S.md A10 keeps). Within 25 % (DESIGN §7.5) is out of this
// seed's reach: seeded from the load's read rate (A10) and the token-0
// warm-ups (A61), the price adds a unit's reads to its compute, where the
// profile hides 0.8 x min(reads, compute). On the M6 profile at u 60 ms the
// price, 4738.7 ms, is the warm-ups' line at no rows, 1632.5 (u 60 + chunk
// 934.5 + their experts' compute 12.5 + host floor 720 - their reads 94.5),
// plus 300 rows' 252.4, plus the turn's 14.27 GB of expert reads at the
// load's 5 GB/s, 2853.7; the turn takes 3226.8 = u 60 + compute 1565.2 +
// host 1601.6, its compute hiding 1252.2 ms of its reads. The warm-ups read
// 0.47 GB (ten experts a layer), under the host floor, so they cannot show
// that overlap. A turnaround adds to price and time alike.
void testColdSeedPricesTheFirstContendedTurn() {
  for (const test::TimingProfile &profile : atEveryTurnaround({&test::kM6FlashNextQ2})) {
    Sim sim(profile, limits(8192, 48));
    sim.seed();
    sim.decoder(1);
    sim.lane(2, 8192, 300);
    const std::array turn{BatchItem{2, 300, 8192}};
    const double truth =
        test::prefillTimes(sim.work.profile(), sim.work.prefillWork(sim.lanes(turn), 0, 1)).wallMilliseconds;
    const double price = sim.price(turn);
    require(price >= truth && price <= 2 * truth,
            at(profile) + ": the first turn's price " + std::to_string(price) + " ms is not between its " +
                std::to_string(truth) + " ms and twice that");
  }
}

// S-P1: contended commands of one shape, run in their passes, converge on
// their price: each pass of the next 300- or 8192-row one is priced within
// 15 % of what it takes. (One shape run in one count of passes does not
// separate the turnaround from the passes' share of the rest, so the price
// of that command run whole is not what this pins.)
void testEstimateConvergesUnderUniformContendedCommands() {
  for (const uint32_t rows : {300u, 8192u}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.seed();
    sim.decoder(1);
    for (uint32_t turn = 0; turn < 12; ++turn) {
      sim.lane(10 + turn, 8192, rows);
      while (sim.scheduler.phase(10 + turn) == Phase::Prefill)
        sim.step();
      sim.scheduler.cancel(10 + turn);
      sim.scheduler.remove(10 + turn);
    }
    sim.lane(99, 8192, rows);
    BatchPlan plan = sim.plan();
    while (plan.kind != WorkKind::Prefill) {
      sim.run(plan);
      plan = sim.plan();
    }
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
      const double truth =
          test::prefillTimes(sim.work.profile(), sim.work.prefillWork(sim.lanes(plan.items), pass, plan.passes))
              .wallMilliseconds;
      require(std::abs(sim.price(plan.items, pass, plan.passes) - truth) <= 0.15 * truth,
              std::to_string(rows) + "-row commands did not converge on their price: pass " + std::to_string(pass) +
                  " of " + std::to_string(plan.passes) + " priced " +
                  std::to_string(sim.price(plan.items, pass, plan.passes)) + " against " + std::to_string(truth) +
                  " ms");
    }
  }
}

// --- Groups, the bound, passes -----------------------------------------------------------------------------

// A dense prompt alone runs whole: nothing waits for its units.
void testALoneDenseCommandRunsWhole() {
  Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers));
  sim.seed();
  sim.lane(1, 0, 4096);
  const BatchPlan plan = sim.plan();
  require(plan.items[0].tokenCount == kChunk && plan.passes == 1, "a lone dense command did not run whole");
}

// A lone streamed request is sized as today: a turn in as many passes as
// keep each within the bound, and a long prompt in groups of the model's
// command rows; a lone dense prompt runs whole. At every turnaround.
void testALoneRequestIsSizedAsToday() {
  for (const double turnaround : kTurnarounds) {
    const test::TimingProfile profile = withTurnaround(test::kM6FlashNextQ2, turnaround);
    Sim sim(profile, limits(8192, 48));
    sim.seed();
    sim.lane(1, 8192, 300);
    const BatchPlan turn = sim.plan();
    // The fewest passes whose every pass is within the bound: about
    // ceil(cost / bound), more where the layers do not divide evenly.
    const auto within = [&](uint32_t passes) {
      for (uint32_t pass = 0; pass < passes; ++pass)
        if (sim.price(turn.items, pass, passes) > sim.bound() * (1 + 1e-9))
          return false;
      return true;
    };
    require(turn.items[0].tokenCount == 300 && turn.passes > 1 && within(turn.passes) && !within(turn.passes - 1),
            at(profile) + ": a lone turn ran in " + std::to_string(turn.passes) +
                " passes, not the fewest within the bound");
    sim.scheduler.cancel(1);
    sim.scheduler.remove(1);
    sim.lane(2, 0, 20'000);
    require(sim.plan().items[0].tokenCount == 8192, at(profile) + ": a lone long prompt did not take a group");
    const test::TimingProfile dense = withTurnaround(test::kDense27B, turnaround);
    Sim alone(dense, limits(kChunk, dense.layers));
    alone.seed();
    alone.lane(1, 0, 4096);
    const BatchPlan chunk = alone.plan();
    require(chunk.items[0].tokenCount == kChunk && chunk.passes == 1,
            at(dense) + ": a lone dense prompt did not run whole");
  }
}

// A contended agent turn on `profile` (the M6's at some turnaround), after
// commands of varied passes have taught the fit the turnaround: the sim and
// the turn's first plan.
std::pair<std::unique_ptr<Sim>, BatchPlan> learnedContendedTurn(const test::TimingProfile &profile) {
  auto sim = std::make_unique<Sim>(profile, limits(8192, 48));
  sim->seed();
  sim->decoder(1);
  for (uint32_t turn = 0; turn < 6; ++turn) {
    sim->lane(10 + turn, 8192, 300 + 1500 * turn);
    while (sim->scheduler.phase(10 + turn) == Phase::Prefill)
      sim->step();
    sim->scheduler.cancel(10 + turn);
    sim->scheduler.remove(10 + turn);
  }
  sim->lane(2, 8192, 300);
  BatchPlan plan = sim->plan();
  while (plan.kind != WorkKind::Prefill) {
    sim->run(plan);
    plan = sim->plan();
  }
  return {std::move(sim), plan};
}

// The passes of a contended turn are those the bound needs, and no more:
// the fewest whose every pass is within max(G, u / 0.1), at every
// turnaround.
void testContendedTurnPassesByTurnaround() {
  for (const double turnaround : kTurnarounds) {
    const test::TimingProfile profile = withTurnaround(test::kM6FlashNextQ2, turnaround);
    const auto [sim, plan] = learnedContendedTurn(profile);
    const auto least = static_cast<uint32_t>(std::ceil(sim->price(plan.items) / sim->bound()));
    require(plan.passes >= least && plan.passes <= least + 1,
            at(profile) + ": a contended turn ran in " + std::to_string(plan.passes) + " passes, not the " +
                std::to_string(least) + " its bound needs");
    for (uint32_t pass = 0; pass < plan.passes; ++pass)
      require(sim->price(plan.items, pass, plan.passes) <= sim->bound() * (1 + 1e-9),
              at(profile) + ": a contended turn's pass exceeds the bound");
  }
}

// Behaviour change 2: the bound follows the turnaround the units measure.
// At u = 150 ms it is u / 0.1 = 1.5 s, so a contended turn runs in fewer
// passes than G's 500 ms would cut it into (the base's slice).
void testTheBoundFollowsTheTurnaround() {
  const test::TimingProfile profile = withTurnaround(test::kM6FlashNextQ2, 150.0);
  const auto [sim, plan] = learnedContendedTurn(profile);
  const auto byGap = static_cast<uint32_t>(std::ceil(sim->price(plan.items) / kGap));
  require(plan.passes < byGap, at(profile) + ": a contended turn ran in " + std::to_string(plan.passes) +
                                   " passes, as many as G's " + std::to_string(byGap));
  require(sim->bound() > kGap, at(profile) + ": the bound did not follow the turnaround the units measured (" +
                                   std::to_string(sim->turnaround()) + " ms)");
}

// The fewest passes within the bound spend at most a tenth of the command on
// turnarounds: (P - 1) u <= (price - u) / 9, on every profile.
void testPassesAddAtMostATenthOfTheCommand() {
  for (const test::TimingProfile &tuned :
       atEveryTurnaround({&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kDense27B})) {
    const test::TimingProfile *profile = &tuned;
    const uint32_t rows = profile->experts.experts ? 8192 : kChunk;
    Sim sim(*profile, limits(rows, profile->layers));
    sim.seed();
    sim.decoder(1);
    for (uint32_t command = 0; command < 8; ++command) {
      sim.lane(10 + command, 4096 * command, rows);
      while (sim.scheduler.phase(10 + command) == Phase::Prefill)
        sim.step();
      sim.scheduler.cancel(10 + command);
      sim.scheduler.remove(10 + command);
    }
    sim.lane(2, 16384, rows);
    BatchPlan plan = sim.plan();
    while (plan.kind != WorkKind::Prefill) {
      sim.run(plan);
      plan = sim.plan();
    }
    // The fewest passes within u / 0.1: had P - 1 passes fit, P would not
    // run, so (P - 1) u stays under a ninth of what the command runs past
    // its turnaround, by the share of the layers the heaviest of P - 1
    // passes holds over an even one's.
    const double u = sim.turnaround(), whole = sim.price(plan.items);
    const uint32_t layers = profile->layers, fewer = plan.passes - 1;
    const double granularity = fewer ? double((layers + fewer - 1) / fewer) * fewer / layers : 1.0;
    require(plan.passes > 1 && fewer * u <= (whole - u) / 9.0 * granularity + 1e-6,
            at(*profile) + ": " + std::to_string(plan.passes) + " passes of " + std::to_string(u) +
                " ms turnaround for a " + std::to_string(whole) + " ms command");
  }
}

// S-P4: on a resident dense target a contended command over G runs in
// passes where the model offers them, and halves where it does not.
void testPassesFollowCostOnAResidentTarget() {
  for (const uint32_t most : {test::kDense27B.layers, 1u}) {
    Sim sim(test::kDense27B, limits(kChunk, most));
    sim.seed();
    sim.decoder(1);
    sim.lane(2, 0, 20'000);
    BatchPlan plan = sim.plan();
    while (plan.kind != WorkKind::Prefill) {
      sim.run(plan);
      plan = sim.plan();
    }
    if (most > 1)
      require(plan.items[0].tokenCount == kChunk && plan.passes > 1,
              "a contended dense command over G did not run in passes");
    else
      require(plan.items[0].tokenCount < kChunk && plan.passes == 1 &&
                  sim.price(plan.items) <= sim.bound() * (1 + 1e-9),
              "a contended dense command without passes did not halve to its bound");
  }
}

// Units of one pass over one chunk do not price the turnaround: a dense
// prompt alone runs whole, each command's Units and ChunkLayers one apiece,
// so its 20 commands running 8 % over the warm-ups (a GPU slower than at
// startup) leave the turnaround at zero and the bound at G, and the next
// contended command runs in passes of G (behaviour change 2: as today
// while u is small).
void testWholeCommandsLeaveTheTurnaroundAtZero() {
  Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers));
  sim.seed();
  sim.lane(1, 0, 20 * kChunk);
  while (sim.scheduler.phase(1) == Phase::Prefill) {
    const BatchPlan plan = sim.plan();
    require(plan.items[0].tokenCount == kChunk && plan.passes == 1, "a lone dense command did not run whole");
    sim.run(plan, 1.08);
  }
  require(sim.turnaround() < 1e-6 && sim.bound() == kGap,
          "whole commands priced a turnaround of " + std::to_string(sim.turnaround()) + " ms, a bound of " +
              std::to_string(sim.bound()) + " ms");
  sim.lane(2, 0, 4 * kChunk);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  for (uint32_t pass = 0; pass < plan.passes; ++pass)
    require(sim.price(plan.items, pass, plan.passes) <= kGap * (1 + 1e-9),
            "a contended dense command ran in " + std::to_string(plan.passes) + " passes of " +
                std::to_string(sim.price(plan.items, pass, plan.passes)) + " ms, past G");
}

// Behaviour change 5: groups follow the fixed part a group saves: on a
// store-less target a spill lane whose history reads cost a chunk's command
// a third of G or more groups, a short one does not, nor a Resident lane.
void testGroupRowsFollowTheFixedCost() {
  Sim sim(test::kStoreLessSpillFlashNext, limits(8192, 48));
  sim.seed();
  sim.lane(1, 262144, 20'000);
  sim.shape(1, 262144 / kPageTokens);
  require(sim.plan().items[0].tokenCount == 8192, "a spill lane with 256K of history did not group");
  sim.shape(1, 0);
  require(sim.plan().items[0].tokenCount == kChunk, "a Resident lane on a store-less target grouped");
  sim.shape(1, 16384 / kPageTokens);
  require(sim.plan().items[0].tokenCount == kChunk, "a spill lane with a short history grouped");
}

// Behaviour change 6: an expert store with 5 % of its experts out groups, a
// whole one does not, on the M6's SSD: grouping saves it nothing.
void testANearlyResidentStoreRunsChunks() {
  test::TimingProfile uniform = test::kM6FlashNextQ2;
  // Misses as the share not resident (amend/M.md B8's uniform marginal).
  uniform.curve.prefillHit = {{{0, 0.0}, {24576, 1.0}}};
  for (const double share : {0.95, 1.0}) {
    Sim sim(uniform, limits(8192, 48), 0.5, [share] { return share; });
    sim.seed();
    sim.lane(1, 0, 20'000);
    const uint32_t rows = sim.plan().items[0].tokenCount;
    require(rows == (share < 1.0 ? 8192u : kChunk),
            "a store " + std::to_string(share) + " resident ran " + std::to_string(rows) + " rows");
  }
}

// A row rule, not a pass rule: one attention layer's pairs stay within the
// cap only by fewer rows. Past a dense target's native window it halves as
// today; an 8192-row lane at 70,000 on 12 dense layers halves to 4096 (8192
// rows fit only to 61,183); a block-sparse layer's rows keep the command's.
void testThePairBoundIsPerLayerAndOnlyRowsMeetIt() {
  {
    Sim sim(test::kDense27B, limits(kChunk, test::kDense27B.layers, 1,
                                    model::denseAttentionPairs(131072 - kChunk, kChunk)));
    sim.seed();
    sim.lane(1, 131072 + 8192, 20'000);
    require(sim.plan().items[0].tokenCount == kChunk / 2, "a dense lane past its window did not halve once");
  }
  {
    test::TimingProfile full = test::kM6FlashNextQ2;
    full.reach = {};
    Sim sim(full, limits(8192, 19, 1, model::denseAttentionPairs(61183, 8192)));
    sim.seed();
    sim.lane(1, 70000, 20'000);
    require(sim.plan().items[0].tokenCount == 4096, "passes kept rows the per-layer pair bound refuses");
  }
  for (const uint32_t position : {0u, 131072u, 1u << 20}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48, 1, model::denseAttentionPairs(262144 - kChunk, kChunk)));
    sim.seed();
    sim.lane(1, position, 20'000);
    require(sim.plan().items[0].tokenCount == 8192, "a block-sparse lane lost rows to the pair bound");
  }
}

// S-P2: every contended unit is within max(G, u / 0.1) by its price, and by
// the profile's clock within that bound by the price's error once learned:
// at most bound × (1 + p95 relative error) (amend/S.md A09).
void testEveryContendedUnitWithinItsBound() {
  for (const test::TimingProfile &tuned :
       atEveryTurnaround({&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kDense27B})) {
    const test::TimingProfile *profile = &tuned;
    const bool streamed = profile->experts.experts;
    Sim sim(*profile, limits(streamed ? 8192 : kChunk, profile->layers));
    sim.seed();
    sim.decoder(1);
    sim.lane(2, 0, 120'000);
    uint32_t units = 0;
    double longest = 0.0;
    while (sim.scheduler.phase(2) == Phase::Prefill && units < 400) {
      const BatchPlan plan = sim.plan();
      if (plan.kind == WorkKind::Prefill) {
        // A plan is bounded by the prices it was made with: those of its first
        // pass's planning.
        for (uint32_t pass = 0; !plan.pass && pass < plan.passes; ++pass)
          require(sim.price(plan.items, pass, plan.passes) <= sim.bound() * (1 + 1e-9) ||
                      plan.passes == profile->layers,
                  at(*profile) + ": a contended unit's price exceeds the bound");
        const double limit =
            sim.bound() * (1.0 + sim.scheduler.snapshot().prefillEstimate.relativeErrorP95);
        const double wall = sim.run(plan);
        if (++units > 40)
          longest = std::max(longest, wall / limit);
      } else {
        sim.run(plan);
      }
    }
    require(longest <= 1.0 + 1e-9, at(*profile) + ": a contended unit took " + std::to_string(longest) +
                                       " of its bound by the price's p95 error once learned");
  }
}

// S-P7: a contended Spill lane leads commands of its group's rows, bounded
// by passes, never halved.
void testAContendedSpillLaneKeepsItsGroupRows() {
  for (const test::TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2}) {
    Sim sim(*profile, limits(8192, 48));
    sim.seed();
    sim.decoder(1);
    sim.decoder(2);
    sim.lane(3, 45056, 40'000);
    sim.shape(3, 45056 / kPageTokens, true);
    for (uint32_t command = 0; command < 30 && sim.scheduler.phase(3) == Phase::Prefill; ++command) {
      const BatchPlan plan = sim.plan();
      if (plan.kind == WorkKind::Prefill && !plan.pass) {
        const uint32_t remaining = 85056 - plan.items[0].promptOffset;
        require(plan.items[0].tokenCount == std::min(remaining, 8192u),
                std::string(profile->name) + ": a contended Spill lane lost its group's rows");
      }
      sim.run(plan);
    }
    require(sim.scheduler.snapshot().boundCuts == 0, "a contended Spill lane's rows were halved");
  }
}

// D5: a group no passes bound halves only when contended; alone it keeps
// its rows in the most passes.
void testAGroupBeyondMaximumPassesHalvesOnlyWhenContended() {
  test::TimingProfile slow = test::kM6FlashNextQ2;
  slow.chunkMilliseconds *= 3;
  slow.rowMilliseconds *= 3;
  slow.weightBytesPerMillisecond /= 3;
  for (const bool contend : {false, true}) {
    Sim sim(slow, limits(8192, 8));
    sim.seed();
    if (contend)
      sim.decoder(1);
    sim.lane(2, 0, 40'000);
    BatchPlan plan = sim.plan();
    while (plan.kind != WorkKind::Prefill) {
      sim.run(plan);
      plan = sim.plan();
    }
    if (!contend) {
      require(plan.items[0].tokenCount == 8192 && plan.passes == 8, "an uncontended group lost rows for time");
    } else {
      require(plan.items[0].tokenCount < 8192, "a contended group no passes bound kept its rows");
      static_cast<void>(sim.run(plan));
      require(sim.scheduler.snapshot().boundCuts > 0, "the halving was not counted");
    }
  }
}

// A contended unit whose cost per chunk no passes bring within the bound,
// nor any fewer rows, keeps its rows in the most passes: halving would cost
// as much a unit for fewer rows. (A dense target whose chunk takes 2.2 s,
// in four passes at most.)
void testAContendedUnitKeepsRowsNoFewerBound() {
  test::TimingProfile heavy = test::kDense27B;
  heavy.chunkMilliseconds = 2220.0;
  Sim sim(heavy, limits(kChunk, 4));
  sim.seed();
  sim.decoder(1);
  sim.lane(2, 0, 20'000);
  completeDecode(sim);
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].tokenCount == kChunk && plan.passes == 4,
          "a contended unit no fewer rows bound halved to " + std::to_string(plan.items[0].tokenCount) + " rows");
  sim.run(plan);
  require(sim.scheduler.snapshot().boundCuts == 0, "a halving no fewer rows could make was counted");
}

// S-P3: beside a Spill lane streaming its history, a decoder's gaps between
// its decode commands stay within a unit's bound, max(G, u / 0.1) by the
// price's error (p95), and one decode command, once the price has learned,
// at every turnaround.
void testDecodeGapsBesideAStreamingLaneStayWithinTheBound() {
  for (const test::TimingProfile *base : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2}) {
    for (const double turnaround : kTurnarounds) {
      const test::TimingProfile profile = withTurnaround(*base, turnaround);
      Sim sim(profile, limits(8192, profile.layers));
      sim.seed();
      sim.decoder(1);
      sim.lane(2, 45056, 80'000);
      sim.shape(2, 45056 / kPageTokens, true);
      std::optional<double> lastDecode;
      uint32_t units = 0;
      double worst = 0.0;
      while (sim.scheduler.phase(2) == Phase::Prefill && units < 800) {
        const BatchPlan plan = sim.plan();
        const double wall = sim.run(plan);
        if (plan.kind == WorkKind::Prefill) {
          ++units;
          continue;
        }
        if (lastDecode && units > 40) {
          const SchedulerSnapshot counts = sim.scheduler.snapshot();
          const double limit = sim.bound() * (1.0 + counts.prefillEstimate.relativeErrorP95) + wall;
          worst = std::max(worst, (sim.now - *lastDecode) / limit);
        }
        lastDecode = sim.now;
      }
      require(units > 40 && worst <= 1.0 + 1e-9,
              at(profile) + ": a decode gap beside a streaming lane reached " + std::to_string(worst) + " of B");
    }
  }
}

// The TTFT simulation (DESIGN §7.5, A74) on the M6's profile: agent turns
// of 300 rows at 8K, arriving at `arrivals`; each turn's first token ends
// its prompt (no admission here), after which it leaves, as what it waited
// for is what this measures. Returns each turn's TTFT, and in `first` the
// first plan.
std::vector<double> runTurns(std::span<const double> arrivals, std::optional<BatchPlan> *first = nullptr) {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  std::vector<double> ttft(arrivals.size(), -1.0);
  size_t arrived = 0, done = 0;
  while (done < arrivals.size()) {
    for (; arrived < arrivals.size() && arrivals[arrived] <= sim.now; ++arrived)
      sim.lane(arrived + 1, 8192, 300);
    const std::optional<BatchPlan> plan = sim.next();
    if (!plan) {
      sim.now = arrivals[arrived];
      continue;
    }
    if (first && !*first)
      *first = plan;
    sim.run(*plan);
    for (size_t index = 0; index < arrived; ++index)
      if (ttft[index] < 0.0 && sim.scheduler.phase(index + 1) != Phase::Prefill) {
        ttft[index] = sim.now - arrivals[index];
        sim.scheduler.cancel(index + 1);
        sim.scheduler.remove(index + 1);
        ++done;
      }
  }
  return ttft;
}

// N = 1, 2, 4 turns together share one command: the worst TTFT within 1.25
// times their command run whole. Two 1.03 s apart (the pass tree's split
// round, 6976 ms, A74): the second's TTFT is at most the rest of the
// first's command and its own, and the two together no more than that and
// the first's alone, regrouped or not. (On this profile a turn costs
// 3.2-3.8 s with its passes, not S.md §11.2's 2.47 s: the rule regroups
// the second while 2 R + K_J > 2 K', here up to about 1.6 s in.) Apart by
// 0.5, 1 or 1.5 s, each turn waits at most for the command in flight and a
// command of all of them.
void testTurnsTtftAloneTogetherAndApart() {
  const test::TimingProfile &profile = test::kM6FlashNextQ2;
  test::TimedWork work(profile);
  const auto whole = [&](uint32_t turns) {
    std::vector<model::PrefillLane> lanes(turns, model::PrefillLane{8192, 300, 0});
    return test::prefillTimes(profile, work.prefillWork(lanes, 0, 1)).wallMilliseconds;
  };
  const std::array lone{0.0};
  const double own = runTurns(lone)[0];
  for (const uint32_t count : {1u, 2u, 4u}) {
    const std::vector<double> together(count, 0.0);
    std::optional<BatchPlan> first;
    const std::vector<double> ttft = runTurns(together, &first);
    require(first && first->width() == count, std::to_string(count) + " turns together did not share a command");
    require(*std::ranges::max_element(ttft) <= 1.25 * whole(count),
            std::to_string(count) + " turns together: worst TTFT " +
                std::to_string(*std::ranges::max_element(ttft)) + " ms against their command's " +
                std::to_string(whole(count)) + " ms");
  }
  const std::array split{0.0, 1030.0};
  const std::vector<double> ttft = runTurns(split);
  require(ttft[1] <= (own - split[1]) + own + 1e-6 && ttft[0] + ttft[1] <= own + (own - split[1]) + own + 1e-6,
          "turns 1.03 s apart took " + std::to_string(ttft[0]) + " and " + std::to_string(ttft[1]) +
              " ms, past the rest of the first's command and their own, " + std::to_string(own) + " ms each");
  for (const uint32_t count : {2u, 4u})
    for (const double gap : {500.0, 1000.0, 1500.0}) {
      std::vector<double> arrivals;
      for (uint32_t index = 0; index < count; ++index)
        arrivals.push_back(index * gap);
      const std::vector<double> apart = runTurns(arrivals);
      for (size_t index = 0; index < apart.size(); ++index)
        require(apart[index] <= 1.25 * (own + whole(count)),
                std::to_string(count) + " turns " + std::to_string(int(gap)) + " ms apart: turn " +
                    std::to_string(index + 1) + " took " + std::to_string(apart[index]) + " ms");
    }
}

// --- Joins --------------------------------------------------------------------------------------------------

// A long lane's sparing rows do not join a short turn: what they add each
// lane exceeds what they save it.
void testALongLanesSparingRowsDoNotDelayAShortTurn() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 8192, 300);
  sim.lane(2, 0, 8192 * 3 + 5000);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1, "a long lane's 5000 sparing rows joined a turn");
  Sim small(test::kM6FlashNextQ2, limits(8192, 48));
  small.seed();
  small.lane(1, 8192, 300);
  small.lane(2, 0, 8192 * 3 + 300);
  const BatchPlan joined = small.plan();
  require(joined.width() == 2 && joined.items[1].tokenCount == 300, "a long lane's 300 sparing rows did not join");
}

// N=4: four agent turns share one command, as on 0bd50301 in the TTFT
// bench (A68: 5152/5247 ms), whose rounds before them trained its price; at
// a host-wait split (1.7 s + 4.3 ms a row) the third would not join.
void testFourShortTurnsShareOneCommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  runTtftBenchRounds(sim);
  for (uint64_t id = 1; id <= 4; ++id)
    sim.lane(id, 8192, 300);
  require(sim.plan().width() == 4, "four turns did not share one command");
}

// A dense leader that finishes its prompt runs without filling the rest of
// its chunk with a long lane's rows.
void testADenseFinishingLeaderRunsWithoutFill() {
  DenseSim sim;
  sim.lane(1, 0, 700);
  sim.lane(2, 0, 3000);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 1 && plan.items[0].requestId == 1 && plan.items[0].tokenCount == 700,
          "a finishing dense leader was filled");
}

// Behaviour change 4: a lane that streams its history shares a command: a
// Spill lane's last rows join a short turn's command, which runs in passes.
void testAStreamingLaneSharesACommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 8192, 300);
  sim.lane(2, 45056, 200);
  sim.shape(2, 45056 / kPageTokens);
  const BatchPlan plan = sim.plan();
  require(plan.width() == 2 && plan.items[0].requestId == 1 && plan.items[1].requestId == 2 &&
              plan.items[1].tokenCount == 200,
          "a lane with Index pages did not share a turn's command");
  require(plan.passes > 1, "the shared command did not run in passes");
}

// One lane whose history holds or may gain Index pages per command: a lane
// that may gain them does not join a command in passes beside one that holds
// them, and is counted; in one pass it may.
void testTwoSpillLanesNeverShareAMultiPassCommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.decoder(9);
  sim.lane(1, 45056, 300);
  sim.shape(1, 45056 / kPageTokens);
  sim.lane(2, 45056, 200);
  sim.shape(2, 0, true);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  require(plan.passes > 1 && plan.width() == 1, "two lanes with Index pages shared a command in passes");
  sim.run(plan);
  require(sim.scheduler.snapshot().indexPageConflicts == 1, "the lane left out for Index pages was not counted");
}

// What planning counted is the committed plan's, as next() made it: a lane
// prepare refused this tick, which the plan was made without but the commit
// does not exclude (it is no longer waiting for memory on its way back),
// changes nothing of the halvings and Index-page conflicts counted.
void testTheCountersAreThoseOfThePlanCommitted() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.decoder(9);
  sim.lane(1, 45056, 300);
  sim.shape(1, 45056 / kPageTokens);
  sim.lane(2, 45056, 200);
  sim.shape(2, 0, true);
  // Lane 3, the cheapest, which would lead, was refused.
  sim.lane(3, 8192, 100);
  completeDecode(sim);
  sim.excluded = {3};
  const BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId != 3 && plan.width() == 1,
          "the plan without the refused lane did not leave a lane out for Index pages");
  sim.excluded.clear();
  sim.run(plan);
  require(sim.scheduler.snapshot().indexPageConflicts == 1,
          "the committed plan's Index-page conflict was not counted");
}

// --- Order and aging ----------------------------------------------------------------------------------------

// Shortest expected remaining cost leads: 11K, 13K, 35K whatever their
// order, and of equal rows the shallower.
void testShortestExpectedRemainingLeads() {
  DenseSim sim;
  sim.lane(1, 0, 35'000);
  sim.lane(2, 0, 13'000);
  sim.lane(3, 0, 11'000);
  require(sim.plan().items[0].requestId == 3, "the least remaining cost did not lead");
  // Once commands at depth have priced attention, of equal rows the
  // shallower leads, the deeper submitted first.
  DenseSim depth;
  for (uint64_t id = 10; id < 16; ++id) {
    depth.lane(id, (id % 2) * 60'000, 2 * kChunk);
    while (depth.scheduler.phase(id) == Phase::Prefill)
      depth.step();
    depth.scheduler.cancel(id);
    depth.scheduler.remove(id);
  }
  depth.lane(1, 100'000, 4096);
  depth.lane(2, 0, 4096);
  require(depth.plan().items[0].requestId == 2, "of equal rows the deeper led");
}

// 87382087's case: of two near-equal long prompts, Spill lanes as at 18G,
// the younger, cheaper one leads; the older is never admitted while the
// younger prefills, its last command's passes included (amend/S.md A100
// (3)), and leads next. Started alone, the older leads once its first
// command commits: the younger, arriving while that command is between its
// passes, is judged against the older past the command's rows, which
// commit before any other prefill, and is never admitted while it
// prefills.
void testNearEqualLongPromptsDoNotInterleave() {
  Sim alone(test::kM6FlashNextQ2, limits(8192, 48));
  alone.seed();
  alone.lane(1, 0, 71'254);
  alone.shapes.push_back({.requestId = 1, .spill = true});
  alone.step();
  require(alone.passes.has_value(), "the older prompt's first command did not run in passes");
  alone.scheduler.submit(request(2, 71'220));
  const std::array younger{PrefillAdmission{2, 0}};
  while (alone.scheduler.phase(1) == Phase::Prefill) {
    require(alone.scheduler.prefillAdmissionOrder(younger, alone.in()).empty(),
            "the younger prompt was admitted while the older prefilled, at " +
                std::to_string(alone.scheduler.promptProcessed(1)) + " of its rows" +
                (alone.passes ? " and a pass of its command" : ""));
    alone.step();
  }
  require(alone.scheduler.prefillAdmissionOrder(younger, alone.in()) == std::vector<uint64_t>{2},
          "the younger prompt did not start once the older's prompt ended");
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.scheduler.submit(request(1, 71'254));
  sim.scheduler.submit(request(2, 71'220));
  const std::array both{PrefillAdmission{1, 0}, PrefillAdmission{2, 0}};
  require(sim.scheduler.prefillAdmissionOrder(both, sim.in()) == std::vector<uint64_t>{2},
          "admission did not start the cheaper of two near-equal prompts");
  sim.scheduler.resourcesReady(2, 0);
  sim.shapes.push_back({.requestId = 2, .spill = true});
  const std::array older{PrefillAdmission{1, 0}};
  while (sim.scheduler.phase(2) == Phase::Prefill) {
    require(sim.scheduler.prefillAdmissionOrder(older, sim.in()).empty(),
            "the older prompt was admitted while the younger prefilled");
    sim.step();
  }
  require(sim.scheduler.prefillAdmissionOrder(older, sim.in()) == std::vector<uint64_t>{1},
          "the older prompt did not start once the younger's prompt ended");
}

// A queued prompt starts only when the next command serves it and leaves it
// waiting for no other prompt: the command ends its prompt, it leads, it is
// overdue, or every lane ordered before it ends its prompt there and none
// is a Spill lane, two of which prefill together only as the later one
// leads or one is overdue (amend/S.md A100 (3), A103). Of a round's
// near-equal prompts on the Studio at 18G (SIM-churn-order), after the
// first, 59,043 tokens, now decoding, the second, a Spill lane, leads, and
// the third, which would join its commands with the rows past its whole
// ones, stays queued until it leads: the churn's 59,020 and 59,054, whose
// 1710 such rows would join commands of 6144 rows, and 57,644 and 57,664,
// whose 320 fit beside the second's last command, of 4396 rows, in the chunk
// it ends in. Beside a Resident prompt of 6144 rows, which its command ends,
// a prompt of three commands and 1710 rows starts with those and leads next;
// beside a Spill one it stays queued. One of 1048 rows beside a leader of
// 1000 in one chunk starts, the command ending it.
void testALongPromptStartsOnlyToLead() {
  const auto spill = [](Sim &sim, uint64_t id) { sim.shapes.push_back({.requestId = id, .spill = true}); };
  const auto round = [&](Sim &sim, uint32_t second, uint32_t third) {
    sim.seed();
    sim.lane(3, 0, 59'043);
    while (sim.scheduler.phase(3) == Phase::Prefill)
      sim.step();
    sim.lane(1, 0, second);
    spill(sim, 1);
    sim.scheduler.submit(request(2, third));
    const std::array later{PrefillAdmission{2, 0}};
    while (sim.scheduler.phase(1) == Phase::Prefill) {
      require(sim.scheduler.prefillAdmissionOrder(later, sim.in()).empty(),
              "the third prompt, " + std::to_string(third) + " tokens, started to join the second's command at " +
                  std::to_string(sim.scheduler.promptProcessed(1)) + " of its rows");
      sim.step();
    }
    require(sim.scheduler.prefillAdmissionOrder(later, sim.in()) == std::vector<uint64_t>{2},
            "the third prompt, " + std::to_string(third) + " tokens, did not start once it led");
  };
  Sim fits(test::kStudio18GFlashNextQ2, limits(8192, 48));
  round(fits, 57'644, 57'664);
  Sim sim(test::kStudio18GFlashNextQ2, limits(8192, 48));
  round(sim, 59'020, 59'054);
  sim.scheduler.cancel(2);
  sim.scheduler.remove(2);
  sim.lane(4, 0, 6144);
  sim.scheduler.submit(request(5, 3 * 8192 + 1710));
  const std::array next{PrefillAdmission{5, 0}};
  require(sim.scheduler.prefillAdmissionOrder(next, sim.in()) == std::vector<uint64_t>{5},
          "a prompt that leads once the command ends the Resident one ahead of it did not start beside it");
  spill(sim, 4);
  require(sim.scheduler.prefillAdmissionOrder(next, sim.in()).empty(),
          "a prompt that leads once the command ends the Spill one ahead of it started beside it");
  Sim chunk(test::kStudio18GFlashNextQ2, limits(8192, 48));
  chunk.seed();
  chunk.lane(1, 0, 1000);
  chunk.scheduler.submit(request(2, 1048));
  const std::array whole{PrefillAdmission{2, 0}};
  require(chunk.scheduler.prefillAdmissionOrder(whole, chunk.in()) == std::vector<uint64_t>{2},
          "a prompt the next command ends did not start beside its leader");
}

// A start restoring its prefix from the SSD cache runs in no command before
// its read lands, and holds back only a prompt that it would then lead
// (amend/S.md A103). On a dense target of one chunk a command, a cold prompt
// of 1500 rows starts beside one restoring with 640 to go, its first command
// ending it (a plan of both would not hold it), as does one of 3100 beside
// 3000, which that command leaves 1052; one of 5200, left 3152, stays
// queued. Beside a Foreground start restoring with 162 rows to go, a Normal
// prompt its first command ends starts; one of 3100 stays queued. Of the
// near-equal pair at 18G on the Studio, the second, 57,664 tokens, stays
// queued beside the first restoring its last 1000 rows, though its 320 rows
// past its whole commands fit beside them and the first's shape is no Spill
// lane's, as two Spill lanes prefill together only as the later one leads
// (A100 (3)).
void testARestoringStartHoldsBackWhatItWouldLead() {
  // Whether a cold prompt of `prompt` rows starts beside a start of
  // `priority` restoring to `cached` with `restoring` rows to go.
  const auto starts = [](Sim &sim, uint32_t cached, uint32_t restoring, uint32_t prompt,
                         RequestPriority priority = RequestPriority::Normal) {
    sim.scheduler.submit(request(1, cached + restoring, priority));
    sim.restoring.push_back({1, cached});
    sim.scheduler.submit(request(2, prompt));
    const std::array candidate{PrefillAdmission{2, 0}};
    return sim.scheduler.prefillAdmissionOrder(candidate, sim.in()) == std::vector<uint64_t>{2};
  };
  for (const auto &[restoring, prompt, priority, expected] :
       {std::tuple{640u, 1500u, RequestPriority::Normal, true},
        {3000u, 3100u, RequestPriority::Normal, true},
        {3000u, 5200u, RequestPriority::Normal, false},
        {162u, 226u, RequestPriority::Foreground, true},
        {162u, 3100u, RequestPriority::Foreground, false}}) {
    DenseSim sim;
    require(starts(sim, 64, restoring, prompt, priority) == expected,
            "a prompt of " + std::to_string(prompt) + " rows " + (expected ? "did not start" : "started") +
                " beside a start of priority " + std::to_string(static_cast<int>(priority)) + " restoring with " +
                std::to_string(restoring) + " rows to go");
  }
  Sim pair(test::kStudio18GFlashNextQ2, limits(8192, 48));
  pair.seed();
  pair.shapes.push_back({.requestId = 1});
  require(!starts(pair, 56'644, 1000, 57'664), "the second of the near-equal pair started beside the first's restore");
}

// A long lane under a stream of short arrivals of its priority waits at
// most K = 3 times its own remaining cost behind them, then leads to its
// end: its first token within (1 + K) times its cost plus a unit.
void testAgingBoundsALongLaneUnderArrivals() {
  DenseSim sim;
  sim.lane(1, 0, 6 * kChunk);
  double cost = 0.0;
  for (uint32_t command = 0; command < 6; ++command) {
    const std::array item{BatchItem{1, kChunk, command * kChunk}};
    cost += test::prefillTimes(sim.work.profile(), sim.work.prefillWork(sim.lanes(item), 0, 1)).wallMilliseconds;
  }
  // A shorter prompt arrives as each one before it ends.
  uint64_t arrival = 0;
  double longest = 0.0;
  while (sim.scheduler.phase(1) == Phase::Prefill && arrival < 1000) {
    if (!arrival || sim.scheduler.phase(arrival) != Phase::Prefill) {
      if (arrival) {
        sim.scheduler.cancel(arrival);
        sim.scheduler.remove(arrival);
      }
      sim.lane(arrival = arrival ? arrival + 1 : 100, 0, 1500);
    }
    longest = std::max(longest, sim.step());
  }
  require(sim.scheduler.phase(1) != Phase::Prefill && sim.now <= 4.0 * cost + longest,
          "the long lane's prompt ended after " + std::to_string(sim.now) + " ms, past (1 + K) of its " +
              std::to_string(cost) + " ms");
}

// Equal prompts are served oldest first, a whole command at a time, not
// filled side by side to finish together; the oldest keeps leading to its
// prompt's end, which leaves the younger nothing lost (not overtaken).
void testEqualPromptsFinishInArrivalOrder() {
  DenseSim sim;
  for (uint64_t id = 1; id <= 3; ++id)
    sim.lane(id, 0, 5000);
  for (uint32_t command = 0; command < 3; ++command) {
    const BatchPlan plan = sim.plan();
    require(plan.width() == 1 && plan.items[0].requestId == 1 &&
                plan.items[0].tokenCount == std::min(kChunk, 5000 - command * kChunk),
            "equal prompts were not served oldest first, a whole command at a time");
    sim.run(plan);
  }
  require(sim.scheduler.phase(1) == Phase::Decode, "the oldest prompt did not end in three commands");
  sim.scheduler.cancel(1);
  require(sim.plan().items[0].requestId == 2, "the next oldest equal prompt did not lead once the oldest's ended");
}

// A later short prompt leads an earlier long one.
void testShortArrivalPrecedesLongColdPrompt() {
  DenseSim sim;
  sim.lane(1, 0, 3000);
  sim.lane(2, 0, 700);
  const BatchPlan plan = sim.plan();
  require(plan.items[0].requestId == 2 && plan.items[0].tokenCount == 700,
          "a later short prompt waited behind an earlier long one");
}

// A higher priority's commands do not age a lower one: once they end, a
// later cheaper lane of its priority still leads.
void testHigherTierCommandsDoNotAgeALowerTier() {
  DenseSim sim;
  sim.lane(1, 0, 4096, RequestPriority::Background);
  for (uint64_t id = 10; id < 30; ++id) {
    sim.lane(id, 0, 1500, RequestPriority::Foreground);
    while (sim.scheduler.phase(id) == Phase::Prefill)
      sim.step();
    sim.scheduler.cancel(id);
    sim.scheduler.remove(id);
  }
  sim.lane(2, 0, 1000, RequestPriority::Background);
  require(sim.plan().items[0].requestId == 2, "a higher priority's commands aged a lower lane");
}

// --- Superseding a command between its passes --------------------------------------------------------------

// D2: a same-priority arrival abandons a command between its passes iff the
// summed completion falls: |A|(K' + W) + |O| W < |J| R.
void testASameTierArrivalSupersedesOnlyWhenTheSumFalls() {
  for (const bool other : {false, true}) {
    for (const double share : {0.5, 1.5}) {
      Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
      sim.seed();
      sim.lane(1, 0, 3 * 8192);
      const BatchPlan command = sim.plan();
      require(command.passes > 2, "the long command did not run in passes");
      sim.run(command);
      if (other)
        sim.lane(3, 0, 6 * 8192);
      sim.lane(2, 8192, 300);
      const std::array turn{BatchItem{2, 300, 8192}};
      const BatchPlan planned = [&] {
        PassProgress kept = *sim.passes;
        sim.passes.reset();
        const BatchPlan fresh = sim.plan();
        sim.passes = kept;
        return fresh;
      }();
      require(planned.width() == 1 && planned.items == std::vector<BatchItem>(turn.begin(), turn.end()),
              "the arrival did not lead a plan of its own");
      const double rest = sim.passesPrice(command.items, 1, command.passes);
      const double turnPrice = sim.passesPrice(turn, 0, planned.passes);
      const double threshold = (rest - turnPrice) / (other ? 2.0 : 1.0);
      require(threshold > 0.0, "the case has no threshold");
      sim.passes->elapsedMilliseconds = share * threshold;
      const BatchPlan next = sim.plan();
      const bool abandons = sim.scheduler.abandonsPasses(next, sim.in());
      require(abandons == (share < 1.0), std::string(other ? "with" : "without") + " a lane waiting behind both, W " +
                                             std::to_string(share) + " of the threshold did " +
                                             (abandons ? "" : "not ") + "abandon");
    }
  }
}

// A higher priority's start abandons a 19-pass command while (|A| + |O|) W
// < |J| R, with one lower lane waiting behind both.
void testAHigherTierAbandonCountsLanesWaitingBehindBoth() {
  test::TimingProfile even = test::kM6FlashNextQ2;
  even.layers = 38;
  Sim sim(even, limits(8192, 19));
  sim.seed();
  sim.lane(1, 0, 3 * 8192);
  sim.lane(3, 0, 6 * 8192);
  BatchPlan plan = sim.plan();
  require(plan.passes == 19 && plan.items[0].requestId == 1, "the command did not run in 19 passes");
  for (uint32_t pass = 0; pass + 1 < plan.passes; ++pass) {
    sim.run(pass ? sim.passes->plan : plan);
    const double done = sim.passes->elapsedMilliseconds;
    const double rest = sim.passesPrice(plan.items, pass + 1, plan.passes);
    require(sim.scheduler.yieldsTo(RequestPriority::Foreground, 1, sim.in()) == (2.0 * done < rest),
            "after " + std::to_string(pass + 1) + " passes a higher start's abandonment did not weigh the lower lane");
  }
}

// A regroup abandons the command only while the summed completion falls:
// keeping it, its lane ends with its passes, R, and the arrival after its own
// command; regrouped, both end with the new command, K'. Early in the
// command it regroups, near its end not.
void testARegroupHappensOnlyWhenTheSumFalls() {
  for (const bool early : {true, false}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.seed();
    sim.lane(1, 8192, 300);
    const BatchPlan command = sim.plan();
    require(command.passes > 3, "the turn did not run in passes");
    sim.run(command);
    while (!early && sim.passes->plan.pass + 1 < command.passes)
      sim.run(sim.passes->plan);
    sim.lane(2, 8192, 300);
    PassProgress kept = *sim.passes;
    sim.passes.reset();
    const BatchPlan fresh = sim.plan();
    sim.passes = kept;
    require(fresh.width() == 2, "the two turns did not plan together");
    const double rest = sim.passesPrice(command.items, sim.passes->plan.pass, command.passes);
    const double together = sim.passesPrice(fresh.items, 0, fresh.passes);
    const std::array second{fresh.items[1]};
    const double alone = sim.passesPrice(second, 0, fresh.passes);
    const double gain = rest + (rest + alone) - 2.0 * together;
    require((gain > 0.0) == early, "the case does not straddle the rule");
    const bool abandons = sim.scheduler.abandonsPasses(sim.plan(), sim.in());
    require(abandons == early, "a regroup did not follow the summed completion");
  }
}

// An overdue lane's command is never abandoned.
void testAnOverdueLaneIsNeverAbandoned() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 0, 3 * 8192);
  const BatchPlan command = sim.plan();
  sim.run(command);
  // Abandoned commands count as waited: enough of them make lane 1 overdue.
  sim.passes->elapsedMilliseconds = 1e7;
  PassProgress kept = *sim.passes;
  sim.abandon();
  sim.passes = kept;
  sim.passes->elapsedMilliseconds = 0.0;
  sim.lane(2, 8192, 300);
  require(!sim.scheduler.abandonsPasses(sim.plan(), sim.in()) &&
              !sim.scheduler.yieldsTo(RequestPriority::Foreground, 1, sim.in()),
          "an overdue lane's command was abandoned");
}

// While a command is between its passes, a start waiting for a quota does
// not lift it over a higher decoder; one the command held back does, and the
// kinds alternate.
void testOnlyAHeldStartLiftsTheCommand() {
  for (const bool held : {false, true}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.seed();
    sim.lane(1, 0, 3 * 8192, RequestPriority::Background);
    sim.run(sim.plan());
    sim.decoder(2);
    sim.scheduler.submit(request(3, 100));
    sim.scheduler.waitForResources(3);
    if (held)
      sim.passes->heldStarts.push_back(3);
    completeDecode(sim);
    const BatchPlan plan = sim.plan();
    require((plan.kind == WorkKind::Prefill) == held, held ? "a held start did not lift the command"
                                                           : "a waiting start lifted the command");
  }
}

// D2: under 300-row turns arriving every 5 s, each superseding the long
// lane's command partway through its passes, the long lane's abandoned
// commands and the turns' units count as its wait until it has waited K
// times its expected remaining cost, after which it leads to its end: its
// first token within K times that cost (its price, at its highest over the
// run), its own time alone and a unit. Without its abandoned time in its
// wait it is overdue only once the turns' units alone reach that, which
// here doubles the wait. (A turn takes 2.5-2.8 s on the profile: turns 3 s
// apart would leave the long lane a few tenths of a second a cycle, so its
// abandoned time would barely count.)
void testRepeatedSupersedesEndOnceTheLongLaneIsOverdue() {
  const auto alone = [] {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.seed();
    sim.lane(1, 0, 3 * 8192);
    while (sim.scheduler.phase(1) == Phase::Prefill)
      sim.step();
    return sim.now;
  }();
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 0, 3 * 8192);
  // The long lane's expected remaining cost: its commands of 8192 rows from
  // where it is, as the order prices it.
  const auto remaining = [&] {
    double total = 0.0;
    for (uint32_t position = sim.scheduler.promptProcessed(1); position < 3 * 8192; position += 8192) {
      const std::array command{BatchItem{1, 8192, position}};
      total += sim.price(command);
    }
    return total;
  };
  double nextArrival = 5000.0, longest = 0.0, priced = remaining();
  uint64_t turn = 100;
  std::vector<uint64_t> turns;
  while (sim.scheduler.phase(1) == Phase::Prefill && sim.now < 10 * alone) {
    if (sim.now >= nextArrival) {
      sim.lane(turn, 8192, 300);
      turns.push_back(turn++);
      nextArrival += 5000.0;
    }
    longest = std::max(longest, sim.step());
    priced = std::max(priced, sim.scheduler.phase(1) == Phase::Prefill ? remaining() : 0.0);
    // A turn ends at its first token: what it waited for is what this
    // pins, not its decode.
    std::erase_if(turns, [&](uint64_t id) {
      if (sim.scheduler.phase(id) == Phase::Prefill)
        return false;
      sim.scheduler.cancel(id);
      sim.scheduler.remove(id);
      return true;
    });
  }
  const SchedulerSnapshot counts = sim.scheduler.snapshot();
  require(counts.abandonedPassCommands > 3, "the turns did not supersede the long lane's command");
  require(sim.scheduler.phase(1) != Phase::Prefill && sim.now <= 3.0 * priced + alone + longest,
          "the long lane's prompt ended after " + std::to_string(sim.now) + " ms, past K times its " +
              std::to_string(priced) + " ms priced and its " + std::to_string(alone) + " ms alone, after " +
              std::to_string(counts.abandonedPassCommands) + " abandoned commands");
}

// A command between its passes holds its lanes to its last pass: a pass
// before the last consumes no rows. It is not abandoned while a pass of it
// runs; abandoned, it leaves its lanes their rows, which they run again
// from a first pass, a lane suspended meanwhile too once it resumes.
void testCommandBetweenPassesHoldsItsLanes() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 20'000, 300);
  sim.lane(2, 20'000, 300);
  const BatchPlan first = sim.plan();
  require(first.width() == 2 && first.passes > 1, "two turns did not share a command in passes");
  sim.scheduler.commit(first, sim.in());
  PassProgress progress{.plan = first};
  test::rejects([&] { sim.scheduler.abandoned(progress); }, "cannot be abandoned",
                "a command was abandoned while a pass of it ran");
  const std::array results{StepResult{1, 0, false, DecodeStage::Regular},
                           StepResult{2, 0, false, DecodeStage::Regular}};
  sim.scheduler.complete(first, results, {.wallMilliseconds = 500.0});
  require(sim.scheduler.phase(1) == Phase::Prefill && sim.scheduler.promptProcessed(1) == 20'000,
          "a pass before a command's last consumed its rows");
  progress.plan.pass = 1;
  progress.elapsedMilliseconds = 500.0;
  sim.scheduler.abandoned(progress);
  BatchPlan plan = sim.plan();
  require(plan.pass == 0 && plan.passes == first.passes && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
  sim.run(plan);
  sim.abandon();
  sim.scheduler.suspendForResources(1);
  sim.scheduler.resumeFromResources(1, 20'000, 20'300);
  plan = sim.plan();
  require(plan.pass == 0 && plan.items == first.items,
          "a lane suspended between its command's passes did not run its rows again with the others");
}

// A command's rows count once, with its first pass, in prefill_rows, the
// work the matrix compares with 1.5.0, which counts them so: the rows its
// lanes plan again after it was abandoned between its passes counted
// already. A lane suspended between its command's passes replays its rows
// once it resumes, and those count again, as 1.5.0's replays do.
void testAnAbandonedCommandsRowsCountOnce() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 0, 3 * 8192);
  sim.run(sim.plan());
  require(sim.passes.has_value(), "the long command did not run in passes");
  sim.lane(2, 8192, 300);
  require(sim.scheduler.abandonsPasses(sim.plan(), sim.in()), "the turn did not supersede the long command");
  while (sim.scheduler.phase(1) == Phase::Prefill || sim.scheduler.phase(2) == Phase::Prefill)
    sim.step();
  const SchedulerSnapshot counts = sim.scheduler.snapshot();
  require(counts.abandonedPassCommands > 0 && counts.prefillRows == 3 * 8192 + 300,
          std::to_string(counts.prefillRows) + " rows counted for the prompts' " + std::to_string(3 * 8192 + 300) +
              " after " + std::to_string(counts.abandonedPassCommands) + " abandoned commands");
  sim.lane(3, 0, 8192);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  sim.run(plan);
  require(sim.passes.has_value(), "the contended command did not run in passes");
  const uint64_t counted = sim.scheduler.snapshot().prefillRows;
  sim.abandon();
  sim.scheduler.suspendForResources(3);
  sim.scheduler.resumeFromResources(3, 0, 8192);
  while (sim.scheduler.phase(3) == Phase::Prefill)
    sim.step();
  require(sim.scheduler.snapshot().prefillRows == counted + 8192,
          "a replay counted " + std::to_string(sim.scheduler.snapshot().prefillRows - counted) + " of its 8192 rows");
}

// A higher priority's decode does not abandon a command between its passes,
// which goes on with its next pass after it; a higher priority's prompt
// early in the command supersedes it, cannot commit beside it, and once it
// is abandoned runs, and the command's lanes run their rows again from a
// first pass.
void testPassesYieldToAHigherPriority() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(3, 0, 100, RequestPriority::Foreground);
  awaitFirstMask(sim, 3);
  sim.lane(1, 20'000, 300, RequestPriority::Background);
  sim.lane(2, 20'000, 300, RequestPriority::Background);
  const BatchPlan first = sim.plan();
  require(first.width() == 2 && first.pass == 0 && first.passes > 1,
          "two Background prompts did not share a command in passes");
  sim.run(first);
  sim.scheduler.maskReady(3);
  BatchPlan plan = sim.plan();
  require(plan.kind == WorkKind::Decode && plan.items[0].requestId == 3 &&
              !sim.scheduler.abandonsPasses(plan, sim.in()),
          "a higher priority's decode waited for a command between its passes, or abandoned it");
  completeDecode(sim, true);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.pass == 1 && plan.items == first.items,
          "a command did not go on with its next pass after a higher priority's decode");
  sim.lane(5, 0, 100, RequestPriority::Foreground);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.items[0].requestId == 5 && plan.width() == 1 &&
              sim.scheduler.abandonsPasses(plan, sim.in()),
          "a higher priority's prompt waited for a command between its passes");
  test::rejects([&] { static_cast<void>(sim.scheduler.commit(plan, sim.in())); }, "batch no longer matches",
                "a higher priority's prompt committed beside a command between its passes");
  sim.abandon();
  require(!sim.scheduler.abandonsPasses(plan, sim.in()), "an abandoned command stayed between its passes");
  while (sim.scheduler.phase(5) == Phase::Prefill)
    sim.step();
  completeDecode(sim, true);
  plan = sim.plan();
  require(plan.kind == WorkKind::Prefill && plan.pass == 0 && plan.items == first.items,
          "an abandoned command's lanes did not run its rows again from a first pass");
}

// A command between its passes runs at the priority of the most urgent lane
// it holds back: with a Foreground lane decoding, a Background command's
// next pass waits for that decode while it holds nothing back, or only a
// Foreground prompt memory on its way back holds; beside a Foreground prompt
// it does not yield to (its lane is overdue) it runs as Foreground work,
// after the decode command.
void testACommandRunsAtThePriorityOfWhatItHoldsBack() {
  enum class Held { Nothing, ReadyPrompt, WaitingPrompt };
  for (const Held held : {Held::Nothing, Held::ReadyPrompt, Held::WaitingPrompt}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48), 0.5);
    sim.seed();
    sim.decoder(2, RequestPriority::Foreground);
    sim.lane(1, 0, 3 * 8192, RequestPriority::Background);
    // Planned without the decoder, whose memory is on its way back.
    sim.excluded = {2};
    const BatchPlan first = sim.plan();
    require(first.kind == WorkKind::Prefill && first.passes > 1, "the Background command did not run in passes");
    sim.run(first);
    sim.excluded.clear();
    // Its abandoned commands' time made its lane overdue: it yields to
    // nothing.
    PassProgress kept = *sim.passes;
    sim.passes->elapsedMilliseconds = 1e7;
    sim.abandon();
    sim.passes = kept;
    if (held != Held::Nothing)
      sim.lane(3, 0, 100, RequestPriority::Foreground);
    if (held == Held::WaitingPrompt)
      sim.excluded = {3};
    completeDecode(sim);
    const BatchPlan plan = sim.plan();
    require((plan.kind == WorkKind::Prefill && plan.items == first.items && plan.pass == 1) ==
                (held == Held::ReadyPrompt),
            held == Held::ReadyPrompt ? "a command did not run at the priority of a prompt it holds back"
                                      : "a command ran ahead of a higher priority's decode holding back no ready lane");
  }
}

// A command's first pass's image encodes stay out of the price its next
// command is planned with.
void testImageEncodesStayOutOfAPassedCommandsEstimate() {
  std::array<BatchPlan, 2> next;
  std::array<double, 2> prices{};
  for (const bool encodes : {false, true}) {
    Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
    sim.seed();
    sim.lane(1, 0, 20'000);
    BatchPlan plan = sim.plan();
    require(plan.passes > 1, "the command did not run in passes");
    for (uint32_t pass = 0; pass < plan.passes; ++pass) {
      const model::CommandWork work = sim.work.prefillWork(sim.lanes(plan.items), pass, plan.passes);
      const double encoded = encodes && !pass ? 5000.0 : 0.0;
      BatchPlan unit = plan;
      unit.pass = pass;
      if (pass)
        sim.passes->plan = unit;
      completePrefill(sim, unit,
                      {.work = work,
                       .wallMilliseconds = test::prefillTimes(sim.work.profile(), work).wallMilliseconds + encoded,
                       .imageEncodeMilliseconds = encoded});
      if (!pass && plan.passes > 1)
        sim.passes = PassProgress{.plan = plan};
    }
    sim.passes.reset();
    next[encodes] = sim.plan();
    prices[encodes] = sim.price(next[encodes].items);
  }
  require(next[1].passes == next[0].passes && next[1].items == next[0].items &&
              std::abs(prices[1] - prices[0]) <= 1e-9 * prices[0],
          "a command's first pass's image encodes trained its next command's price");
}

// The image encodes ahead of a command stay out of its price: a command
// whose rows ran as priced trains nothing new, encodes or not.
void testImageEncodesStayOutOfTheCommandEstimate() {
  std::array<double, 2> prices{};
  for (const bool encodes : {false, true}) {
    DenseSim sim;
    sim.lane(1, 0, 4096);
    const BatchPlan plan = sim.plan();
    sim.scheduler.commit(plan, sim.in());
    const model::CommandWork work = sim.work.prefillWork(sim.lanes(plan.items), 0, 1);
    const double rows = test::prefillTimes(sim.work.profile(), work).wallMilliseconds;
    const std::array result{StepResult{1, plan.items[0].tokenCount, false, DecodeStage::Regular}};
    sim.scheduler.complete(plan, result,
                           {.work = work,
                            .wallMilliseconds = rows + (encodes ? 5000.0 : 0.0),
                            .imageEncodeMilliseconds = encodes ? 5000.0 : 0.0});
    prices[encodes] = sim.price(sim.plan().items);
  }
  require(prices[0] == prices[1], "a command's image encodes trained its price");
}

constexpr test::TestCase kTests[] = {
    {"testRealDecodeWidths", testRealDecodeWidths},
    {"testRejectedCommitCountsNothing", testRejectedCommitCountsNothing},
    {"testDecodeLanesRotate", testDecodeLanesRotate},
    {"testMaskStagesNeverMix", testMaskStagesNeverMix},
    {"testPrefillAndDecodeAlternateWithoutStarvation", testPrefillAndDecodeAlternateWithoutStarvation},
    {"testDecodeRepaysItsShareOfContendedPrefill", testDecodeRepaysItsShareOfContendedPrefill},
    {"testDecodeTimePerTokenWeighsEachCommand", testDecodeTimePerTokenWeighsEachCommand},
    {"testExcludedLanesAreNotPlanned", testExcludedLanesAreNotPlanned},
    {"testPriorityPrecedesWorkKindAlternation", testPriorityPrecedesWorkKindAlternation},
    {"testPrefillCommandContainsOnePriorityTier", testPrefillCommandContainsOnePriorityTier},
    {"testHighestRunnablePriority", testHighestRunnablePriority},
    {"testResourceSuspensionReplaysFromCache", testResourceSuspensionReplaysFromCache},
    {"testAdmissionStartsTheLanesOfTheNextCommand", testAdmissionStartsTheLanesOfTheNextCommand},
    {"testAdmissionStartsAtMostAFullBatch", testAdmissionStartsAtMostAFullBatch},
    {"testAdmissionWaitsForAHigherPrioritysDecode", testAdmissionWaitsForAHigherPrioritysDecode},
    {"testUnconstrainedLanesShareDecode", testUnconstrainedLanesShareDecode},
    {"testMaskedLanesShareDecode", testMaskedLanesShareDecode},
    {"testMaskWaitAccruesNoDecodeDebt", testMaskWaitAccruesNoDecodeDebt},
    {"testDecodeDebtLeavesWithTheLastDecoder", testDecodeDebtLeavesWithTheLastDecoder},
    {"testHigherPriorityPrefillPrecedesDecodeDebt", testHigherPriorityPrefillPrecedesDecodeDebt},
    {"testDecodeCommandContainsOnePriorityTier", testDecodeCommandContainsOnePriorityTier},
    {"testReplayKeepsHeldInitialMask", testReplayKeepsHeldInitialMask},
    {"testWaitingMaskBoundsPeerPrefill", testWaitingMaskBoundsPeerPrefill},
    {"testOnlyContendedUnitsAreBounded", testOnlyContendedUnitsAreBounded},
    {"testALowerPrioritysDecodeDoesNotBoundAPrefill", testALowerPrioritysDecodeDoesNotBoundAPrefill},
    {"testWaitingOrPrefillingPeersDoNotBoundAPrefill", testWaitingOrPrefillingPeersDoNotBoundAPrefill},
    {"testAReadyPeerDoesNotBoundAStreamingLeader", testAReadyPeerDoesNotBoundAStreamingLeader},
    {"testUntimedOrAuxiliaryUnitsTrainNothing", testUntimedOrAuxiliaryUnitsTrainNothing},
    {"testACommandHoldsOneLanePastAChunk", testACommandHoldsOneLanePastAChunk},
    {"testAPrefillsPriceIsTheCostModels", testAPrefillsPriceIsTheCostModels},
    {"testAWeightReadByteIsPricedAsRead", testAWeightReadByteIsPricedAsRead},
    {"testColdSeedPricesTheFirstContendedTurn", testColdSeedPricesTheFirstContendedTurn},
    {"testEstimateConvergesUnderUniformContendedCommands", testEstimateConvergesUnderUniformContendedCommands},
    {"testALoneDenseCommandRunsWhole", testALoneDenseCommandRunsWhole},
    {"testALoneRequestIsSizedAsToday", testALoneRequestIsSizedAsToday},
    {"testContendedTurnPassesByTurnaround", testContendedTurnPassesByTurnaround},
    {"testTheBoundFollowsTheTurnaround", testTheBoundFollowsTheTurnaround},
    {"testPassesAddAtMostATenthOfTheCommand", testPassesAddAtMostATenthOfTheCommand},
    {"testPassesFollowCostOnAResidentTarget", testPassesFollowCostOnAResidentTarget},
    {"testWholeCommandsLeaveTheTurnaroundAtZero", testWholeCommandsLeaveTheTurnaroundAtZero},
    {"testGroupRowsFollowTheFixedCost", testGroupRowsFollowTheFixedCost},
    {"testANearlyResidentStoreRunsChunks", testANearlyResidentStoreRunsChunks},
    {"testThePairBoundIsPerLayerAndOnlyRowsMeetIt", testThePairBoundIsPerLayerAndOnlyRowsMeetIt},
    {"testEveryContendedUnitWithinItsBound", testEveryContendedUnitWithinItsBound},
    {"testAContendedSpillLaneKeepsItsGroupRows", testAContendedSpillLaneKeepsItsGroupRows},
    {"testAGroupBeyondMaximumPassesHalvesOnlyWhenContended", testAGroupBeyondMaximumPassesHalvesOnlyWhenContended},
    {"testAContendedUnitKeepsRowsNoFewerBound", testAContendedUnitKeepsRowsNoFewerBound},
    {"testDecodeGapsBesideAStreamingLaneStayWithinTheBound", testDecodeGapsBesideAStreamingLaneStayWithinTheBound},
    {"testTurnsTtftAloneTogetherAndApart", testTurnsTtftAloneTogetherAndApart},
    {"testALongLanesSparingRowsDoNotDelayAShortTurn", testALongLanesSparingRowsDoNotDelayAShortTurn},
    {"testFourShortTurnsShareOneCommand", testFourShortTurnsShareOneCommand},
    {"testADenseFinishingLeaderRunsWithoutFill", testADenseFinishingLeaderRunsWithoutFill},
    {"testAStreamingLaneSharesACommand", testAStreamingLaneSharesACommand},
    {"testTwoSpillLanesNeverShareAMultiPassCommand", testTwoSpillLanesNeverShareAMultiPassCommand},
    {"testTheCountersAreThoseOfThePlanCommitted", testTheCountersAreThoseOfThePlanCommitted},
    {"testShortestExpectedRemainingLeads", testShortestExpectedRemainingLeads},
    {"testNearEqualLongPromptsDoNotInterleave", testNearEqualLongPromptsDoNotInterleave},
    {"testALongPromptStartsOnlyToLead", testALongPromptStartsOnlyToLead},
    {"testARestoringStartHoldsBackWhatItWouldLead", testARestoringStartHoldsBackWhatItWouldLead},
    {"testAgingBoundsALongLaneUnderArrivals", testAgingBoundsALongLaneUnderArrivals},
    {"testEqualPromptsFinishInArrivalOrder", testEqualPromptsFinishInArrivalOrder},
    {"testShortArrivalPrecedesLongColdPrompt", testShortArrivalPrecedesLongColdPrompt},
    {"testHigherTierCommandsDoNotAgeALowerTier", testHigherTierCommandsDoNotAgeALowerTier},
    {"testASameTierArrivalSupersedesOnlyWhenTheSumFalls", testASameTierArrivalSupersedesOnlyWhenTheSumFalls},
    {"testAHigherTierAbandonCountsLanesWaitingBehindBoth", testAHigherTierAbandonCountsLanesWaitingBehindBoth},
    {"testARegroupHappensOnlyWhenTheSumFalls", testARegroupHappensOnlyWhenTheSumFalls},
    {"testAnOverdueLaneIsNeverAbandoned", testAnOverdueLaneIsNeverAbandoned},
    {"testOnlyAHeldStartLiftsTheCommand", testOnlyAHeldStartLiftsTheCommand},
    {"testRepeatedSupersedesEndOnceTheLongLaneIsOverdue", testRepeatedSupersedesEndOnceTheLongLaneIsOverdue},
    {"testCommandBetweenPassesHoldsItsLanes", testCommandBetweenPassesHoldsItsLanes},
    {"testAnAbandonedCommandsRowsCountOnce", testAnAbandonedCommandsRowsCountOnce},
    {"testPassesYieldToAHigherPriority", testPassesYieldToAHigherPriority},
    {"testACommandRunsAtThePriorityOfWhatItHoldsBack", testACommandRunsAtThePriorityOfWhatItHoldsBack},
    {"testImageEncodesStayOutOfAPassedCommandsEstimate", testImageEncodesStayOutOfAPassedCommandsEstimate},
    {"testImageEncodesStayOutOfTheCommandEstimate", testImageEncodesStayOutOfTheCommandEstimate},
};

} // namespace

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"scheduler", kTests}};
  return test::runTests("ragged-scheduler", groups, argc, argv);
}
