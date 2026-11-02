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
#include <limits>
#include <optional>
#include <span>
#include <string>
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

// A target, its scheduler and what the engine gives the scheduler: the
// lanes' shapes, the lanes left out, and the command between its passes,
// which run() keeps as the engine's PassCommand does. Commands take the
// profile's time on a clock.
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
  [[nodiscard]] PlanInputs in() const { return {excluded, shapes, passes ? &*passes : nullptr}; }
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

// Prefills a constrained request's prompt, whose end waits for the mask of
// its first token.
void awaitFirstMask(Sim &sim, uint64_t id) {
  const BatchPlan prompt = sim.plan();
  require(prompt.kind == WorkKind::Prefill && prompt.width() == 1 && prompt.items[0].requestId == id,
          "expected the constrained prompt");
  sim.scheduler.commit(prompt, sim.in());
  const std::array result{StepResult{id, prompt.items[0].tokenCount, false, DecodeStage::ApplyInitialMask}};
  sim.scheduler.complete(prompt, result, {});
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

// S-P5: a cold start's first contended agent turn is not underpriced as
// with a host-wait seed (base: 1.09 s for a 3.2 s turn): seeded from the
// warm-ups and the load, it prices within the clamp above its truth, which
// the turn's own sample then trains in full.
void testColdSeedPricesTheFirstContendedTurn() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.decoder(1);
  sim.lane(2, 8192, 300);
  const std::array turn{BatchItem{2, 300, 8192}};
  const double truth =
      test::prefillTimes(sim.work.profile(), sim.work.prefillWork(sim.lanes(turn), 0, 1)).wallMilliseconds;
  const double price = sim.price(turn);
  require(price >= truth && price <= 2.0 * truth,
          "the first turn's price " + std::to_string(price) + " ms is not within the clamp above its " +
              std::to_string(truth) + " ms");
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
// command rows; a lone dense prompt runs whole.
void testALoneRequestIsSizedAsToday() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.lane(1, 8192, 300);
  const BatchPlan turn = sim.plan();
  const double cost = sim.price(turn.items);
  const auto expected = static_cast<uint32_t>(std::ceil(cost / sim.bound()));
  require(turn.items[0].tokenCount == 300 && turn.passes + 1 >= expected && turn.passes <= expected + 1,
          "a lone turn ran in " + std::to_string(turn.passes) + " passes, not about " + std::to_string(expected));
  sim.scheduler.cancel(1);
  sim.scheduler.remove(1);
  sim.lane(2, 0, 20'000);
  require(sim.plan().items[0].tokenCount == 8192, "a lone long prompt did not take a group");
}

// The passes of a contended turn are those the bound needs: with the
// turnaround the units measure, max(G, u / 0.1).
void testContendedTurnPassesByTurnaround() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.decoder(1);
  // Commands of varied passes teach the fit the turnaround.
  for (uint32_t turn = 0; turn < 6; ++turn) {
    sim.lane(10 + turn, 8192, 300 + 1500 * turn);
    while (sim.scheduler.phase(10 + turn) == Phase::Prefill)
      sim.step();
    sim.scheduler.cancel(10 + turn);
    sim.scheduler.remove(10 + turn);
  }
  sim.lane(2, 8192, 300);
  BatchPlan plan = sim.plan();
  while (plan.kind != WorkKind::Prefill) {
    sim.run(plan);
    plan = sim.plan();
  }
  const auto least = static_cast<uint32_t>(std::ceil(sim.price(plan.items) / sim.bound()));
  require(plan.passes >= least, "a contended turn ran in fewer passes than its bound needs");
  for (uint32_t pass = 0; pass < plan.passes; ++pass)
    require(sim.price(plan.items, pass, plan.passes) <= sim.bound() * (1 + 1e-9),
            "a contended turn's pass exceeds the bound");
}

// The fewest passes within the bound spend at most a tenth of the command on
// turnarounds: (P - 1) u <= (price - u) / 9, on every profile.
void testPassesAddAtMostATenthOfTheCommand() {
  for (const test::TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kDense27B}) {
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
    const double u = sim.turnaround(), whole = sim.price(plan.items);
    require(plan.passes > 1 && (plan.passes - 1) * u <= (whole - u) / 9.0 + 1e-6,
            std::string(profile->name) + ": " + std::to_string(plan.passes) + " passes of " + std::to_string(u) +
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

// Groups follow the fixed part a group saves: a spill lane whose history
// reads cost a chunk's command a third of G or more groups, a short one
// does not; an expert store with 5 % of its experts out groups, a whole one
// does not, on the M6's SSD.
void testGroupRowsFollowTheFixedCost() {
  {
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

// S-P2: every contended unit is within max(G, u / 0.1) by its price, and
// takes about that by the profile's clock once the price has learned.
void testEveryContendedUnitWithinItsBound() {
  for (const test::TimingProfile *profile : {&test::kM6FlashNextQ2, &test::kStudio18GFlashNextQ2, &test::kDense27B}) {
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
                  std::string(profile->name) + ": a contended unit's price exceeds the bound");
        const double wall = sim.run(plan);
        if (++units > 40)
          longest = std::max(longest, wall / sim.bound());
      } else {
        sim.run(plan);
      }
    }
    require(longest <= 1.25, std::string(profile->name) + ": a contended unit took " + std::to_string(longest) +
                                 " of its bound once learned");
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

// N=4: four agent turns share one command; at today's host split the third
// would not join.
void testFourShortTurnsShareOneCommand() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
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

// 87382087's case: of two near-equal long prompts the younger, cheaper one
// leads; the older is never admitted while the younger prefills, and leads
// next.
void testNearEqualLongPromptsDoNotInterleave() {
  Sim sim(test::kM6FlashNextQ2, limits(8192, 48));
  sim.seed();
  sim.scheduler.submit(request(1, 71'254));
  sim.scheduler.submit(request(2, 71'220));
  const std::array both{PrefillAdmission{1, 0}, PrefillAdmission{2, 0}};
  require(sim.scheduler.prefillAdmissionOrder(both, sim.in()) == std::vector<uint64_t>{2},
          "admission did not start the cheaper of two near-equal prompts");
  sim.scheduler.resourcesReady(2, 0);
  const std::array older{PrefillAdmission{1, 0}};
  while (sim.scheduler.phase(2) == Phase::Prefill) {
    require(sim.scheduler.prefillAdmissionOrder(older, sim.in()).empty(),
            "the older prompt was admitted while the younger prefilled");
    sim.step();
  }
  require(sim.scheduler.prefillAdmissionOrder(older, sim.in()) == std::vector<uint64_t>{1},
          "the older prompt did not start once the younger's prompt ended");
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
    {"testAPrefillsPriceIsTheCostModels", testAPrefillsPriceIsTheCostModels},
    {"testColdSeedPricesTheFirstContendedTurn", testColdSeedPricesTheFirstContendedTurn},
    {"testEstimateConvergesUnderUniformContendedCommands", testEstimateConvergesUnderUniformContendedCommands},
    {"testALoneDenseCommandRunsWhole", testALoneDenseCommandRunsWhole},
    {"testALoneRequestIsSizedAsToday", testALoneRequestIsSizedAsToday},
    {"testContendedTurnPassesByTurnaround", testContendedTurnPassesByTurnaround},
    {"testPassesAddAtMostATenthOfTheCommand", testPassesAddAtMostATenthOfTheCommand},
    {"testPassesFollowCostOnAResidentTarget", testPassesFollowCostOnAResidentTarget},
    {"testGroupRowsFollowTheFixedCost", testGroupRowsFollowTheFixedCost},
    {"testThePairBoundIsPerLayerAndOnlyRowsMeetIt", testThePairBoundIsPerLayerAndOnlyRowsMeetIt},
    {"testEveryContendedUnitWithinItsBound", testEveryContendedUnitWithinItsBound},
    {"testAContendedSpillLaneKeepsItsGroupRows", testAContendedSpillLaneKeepsItsGroupRows},
    {"testAGroupBeyondMaximumPassesHalvesOnlyWhenContended", testAGroupBeyondMaximumPassesHalvesOnlyWhenContended},
    {"testALongLanesSparingRowsDoNotDelayAShortTurn", testALongLanesSparingRowsDoNotDelayAShortTurn},
    {"testFourShortTurnsShareOneCommand", testFourShortTurnsShareOneCommand},
    {"testADenseFinishingLeaderRunsWithoutFill", testADenseFinishingLeaderRunsWithoutFill},
    {"testTwoSpillLanesNeverShareAMultiPassCommand", testTwoSpillLanesNeverShareAMultiPassCommand},
    {"testShortestExpectedRemainingLeads", testShortestExpectedRemainingLeads},
    {"testNearEqualLongPromptsDoNotInterleave", testNearEqualLongPromptsDoNotInterleave},
    {"testAgingBoundsALongLaneUnderArrivals", testAgingBoundsALongLaneUnderArrivals},
    {"testHigherTierCommandsDoNotAgeALowerTier", testHigherTierCommandsDoNotAgeALowerTier},
    {"testASameTierArrivalSupersedesOnlyWhenTheSumFalls", testASameTierArrivalSupersedesOnlyWhenTheSumFalls},
    {"testAHigherTierAbandonCountsLanesWaitingBehindBoth", testAHigherTierAbandonCountsLanesWaitingBehindBoth},
    {"testARegroupHappensOnlyWhenTheSumFalls", testARegroupHappensOnlyWhenTheSumFalls},
    {"testAnOverdueLaneIsNeverAbandoned", testAnOverdueLaneIsNeverAbandoned},
    {"testOnlyAHeldStartLiftsTheCommand", testOnlyAHeldStartLiftsTheCommand},
    {"testImageEncodesStayOutOfTheCommandEstimate", testImageEncodesStayOutOfTheCommandEstimate},
};

} // namespace

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"scheduler", kTests}};
  return test::runTests("ragged-scheduler", groups, argc, argv);
}
