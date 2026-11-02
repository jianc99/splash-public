// The prefill price (engine::CostModel): its seed from the warm-ups and the
// load, the rates it learns from the units that ran, and the calibration of
// the read bytes a model expects.

#include "TestChecks.hpp"
#include "TestRegistry.hpp"
#include "engine/CostModel.hpp"
#include "model/Model.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using test::require;

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
constexpr uint32_t kLayers = 48;
constexpr size_t kWeightReads = static_cast<size_t>(CostTerm::WeightReadBytes);
constexpr size_t kSpilledReads = static_cast<size_t>(CostTerm::SpilledKvReadBytes);

// The true rates of a streamed target's units, per CostTerm: the M6's per
// command (60 ms), per chunk (934.5 ms) and per row (0.8413 ms) over its 48
// layers (TimingProfile.cpp, kM6FlashNextQ2), a microsecond per attention
// pair and a fiftieth of that per index pair, its SSD's 5 GB/s for weights and
// half of that for spilled K/V.
constexpr CostVector kTruth{60.0, 934.5 / kLayers, 0.8413 / kLayers, 1e-6, 2e-8, 2e-7, 4e-7};

double truth(const model::CommandWork &work) {
  const CostVector terms = costTerms(work);
  double milliseconds = 0.0;
  for (size_t term = 0; term < kCostTerms; ++term)
    milliseconds += kTruth[term] * terms[term];
  return milliseconds;
}

// One pass of `passes` of a command of `rows` rows from `position`, its
// layers' share of every count.
model::CommandWork unit(uint32_t rows, uint32_t position, uint64_t weightBytes, uint64_t spilledBytes,
                        uint32_t passes = 1) {
  const double share = 1.0 / passes;
  const uint64_t pairs = rows * uint64_t{position} + uint64_t{rows} * (rows + 1) / 2;
  return {.lanes = 1,
          .rows = rows,
          .chunks = (rows + kChunk - 1) / kChunk,
          .layers = kLayers / passes,
          .attentionPairs = static_cast<uint64_t>(12 * pairs * share),
          .layerAttentionPairs = pairs,
          .indexPairs = static_cast<uint64_t>(12 * double(rows) * (position + rows) / 4 * share),
          .weightReadBytes = static_cast<uint64_t>(weightBytes * share),
          .spilledKvReadBytes = static_cast<uint64_t>(spilledBytes * share)};
}

model::CommandSample sample(const model::CommandWork &work, double milliseconds) {
  return {.work = work, .wallMilliseconds = milliseconds};
}

// The two prefill warm-ups at the truth, a prompt of one token reading few
// experts, and a load at the weights' true rate.
CostModel seeded() {
  CostModel cost;
  const std::array warmups{sample(unit(kChunk, 0, 400'000'000, 0), truth(unit(kChunk, 0, 400'000'000, 0))),
                           sample(unit(256, 0, 400'000'000, 0), truth(unit(256, 0, 400'000'000, 0)))};
  cost.seed(warmups, {.bytes = 10'000'000'000, .milliseconds = 10'000'000'000 * kTruth[kWeightReads]});
  return cost;
}

double relative(double value, double expected) { return std::fabs(value - expected) / std::fabs(expected); }

// (7) Two exact warm-ups and the load: RowLayers the slope of the line
// through them and ChunkLayers what it leaves at no rows over the chunk's
// layers, both read terms the load's rate, the rest zero; a line through two
// points that would cross zero keeps it at zero. A model that reports no
// layers leaves the line's constant to Units.
void testTheSeedReproducesAnExactAffineTruth() {
  const CostModelStats stats = seeded().stats();
  // Two one-chunk prompts over every layer in one pass do not separate the
  // turnaround from the chunk's cost: ChunkLayers carries both.
  const double constant = kTruth[0] + kTruth[1] * kLayers;
  require(relative(stats.seedRates[1] * kLayers, constant) < 0.02 && stats.rates[1] == stats.seedRates[1],
          "the seed's ChunkLayers is not the line through the warm-ups at no rows");
  require(relative(stats.seedRates[2], kTruth[2]) < 0.02, "the seed's RowLayers is not the line's slope");
  require(stats.seedRates[kWeightReads] == kTruth[kWeightReads] &&
              stats.seedRates[kSpilledReads] == kTruth[kWeightReads],
          "the seed's read terms are not the load's time per byte");
  require(stats.seedRates[0] == 0.0 && stats.seedRates[3] == 0.0 && stats.seedRates[4] == 0.0,
          "a term no startup measurement measured did not seed at zero");
  // Warm-ups whose short one costs more than the long: the slope would be
  // negative, so the constant is their mean and RowLayers zero.
  CostModel inverted;
  const std::array warmups{sample(unit(kChunk, 0, 0, 0), 1000.0), sample(unit(256, 0, 0, 0), 2000.0)};
  inverted.seed(warmups, {});
  const CostModelStats clamped = inverted.stats();
  require(clamped.seedRates[1] * kLayers == 1500.0 && clamped.seedRates[2] == 0.0,
          "a negative seed slope was not clamped to a mean");
  // One whose line would cross zero at a positive row count: no constant,
  // the rows carry it all.
  CostModel steep;
  const std::array steepWarmups{sample(unit(kChunk, 0, 0, 0), 2048.0), sample(unit(256, 0, 0, 0), 64.0)};
  steep.seed(steepWarmups, {});
  require(steep.stats().seedRates[1] == 0.0 && steep.stats().seedRates[2] > 0.0,
          "a negative seed intercept was not clamped to zero");
  // Warm-ups over no layers: rows and chunks are all there is, the constant
  // is Units'.
  CostModel flat;
  model::CommandWork noLayers = unit(kChunk, 0, 0, 0);
  noLayers.layers = 0;
  const std::array flatWarmups{sample(noLayers, 300.0), sample(noLayers, 300.0)};
  flat.seed(flatWarmups, {});
  require(flat.stats().seedRates[0] == 300.0 && flat.stats().seedRates[1] == 0.0,
          "warm-ups over no layers did not seed Units");
}

// (10) Both SSD terms take the load's time per byte on every target, whatever
// the warm-ups read: their host-step time is not a price.
void testSeedReadRatesComeFromTheLoad() {
  for (const uint64_t warmupBytes : {uint64_t{0}, uint64_t{2'000'000'000}}) {
    CostModel cost;
    const std::array warmups{sample(unit(kChunk, 0, warmupBytes, 0), 3213.0),
                             sample(unit(256, 0, warmupBytes, 0), 2400.0)};
    cost.seed(warmups, {.bytes = 30'000'000'000, .milliseconds = 4000.0});
    const CostModelStats stats = cost.stats();
    require(stats.seedRates[kWeightReads] == 4000.0 / 30e9 && stats.seedRates[kSpilledReads] == 4000.0 / 30e9,
            "a read term was not seeded from the load");
  }
}

// (9) A unit's price is linear in each read term, at its rate: no floor.
void testPriceIsLinearInReadBytes() {
  const CostModel cost = seeded();
  const model::CommandWork base = unit(300, 8192, 0, 0);
  for (const size_t term : {kWeightReads, kSpilledReads}) {
    for (const uint64_t bytes : {uint64_t{1} << 20, uint64_t{1} << 30, uint64_t{40} << 30}) {
      model::CommandWork more = base;
      (term == kWeightReads ? more.weightReadBytes : more.spilledKvReadBytes) = bytes;
      const double added = cost.price(more) - cost.price(base);
      require(relative(added, cost.stats().rates[term] * double(bytes)) < 1e-9,
              "a unit's price is not linear in its read bytes");
    }
  }
}

// (1) Varied units in passes of varied counts: the fit recovers what the
// terms that vary most cost, and prices new units within S-P1's 15 % of the
// truth, 5 % on average, with or without 3 % of noise. Over its
// decayed window of some ten commands each seed still weighs a unit's worth:
// Units, which only the passes' counts separate from the chunks' cost, and
// SpilledKvReadBytes, seeded at the weights' rate, move from their seeds
// toward the truth, not all the way. (IndexPairs, a two-hundredth of
// AttentionPairs here and as deep, is not separated.)
void testRecoversTheRatesVariedUnitsShow() {
  for (const double spread : {0.0, 0.03}) {
    CostModel cost = seeded();
    std::mt19937_64 random(7);
    std::uniform_int_distribution<uint32_t> rows(64, 4 * kChunk), position(0, 131072);
    std::uniform_int_distribution<uint64_t> weights(0, 16'000'000'000), spilled(0, 4'000'000'000);
    std::normal_distribution<double> noise(0.0, spread);
    const std::array<uint32_t, 5> passes{1, 2, 4, 8, 24};
    for (uint32_t command = 0; command < 400; ++command) {
      const uint32_t count = passes[command % passes.size()];
      const uint32_t r = rows(random), p = position(random);
      const uint64_t w = weights(random), s = spilled(random);
      for (uint32_t pass = 0; pass < count; ++pass) {
        const model::CommandWork work = unit(r, p, w, s, count);
        cost.observe(work, work, truth(work) * (1.0 + noise(random)), 1.0 / count);
      }
    }
    const CostModelStats stats = cost.stats();
    std::string rates = " (noise " + std::to_string(spread) + ")";
    for (size_t term = 0; term < kCostTerms; ++term)
      rates += " " + std::to_string(stats.rates[term] / kTruth[term]);
    double sum = 0.0;
    for (uint32_t probe = 0; probe < 100; ++probe) {
      const model::CommandWork work = unit(rows(random), position(random), weights(random), spilled(random));
      const double error = relative(cost.price(work), truth(work));
      sum += error;
      require(error < 0.15, "a new unit's price strayed from the truth: " +
                                std::to_string(cost.price(work) / truth(work)) + " of it; rates over the truth" +
                                rates);
    }
    require(sum / 100 < 0.05, "new units' prices strayed from the truth on average" + rates);
    // With noise the read terms trade against each other within the window;
    // the prices above hold.
    if (spread)
      continue;
    for (const size_t term : {size_t{1}, size_t{2}, size_t{3}})
      require(relative(stats.rates[term], kTruth[term]) < 0.05,
              "the fit did not recover " + std::string(costTermName(static_cast<CostTerm>(term))) + ":" + rates);
    require(relative(stats.rates[kWeightReads], kTruth[kWeightReads]) < 0.1,
            "the fit did not recover the weights' reads:" + rates);
    require(stats.rates[kSpilledReads] > 1.2 * stats.seedRates[kSpilledReads] && stats.rates[0] > 0.25 * kTruth[0],
            "the spilled reads or the turnaround did not move from their seeds toward the truth:" + rates);
  }
}

// (2) Units of one size and depth: the price of that size is right, and the
// seed's split of it between Units and RowLayers holds, as the data say
// nothing of it.
void testOneSizeKeepsTheSeedsSplit() {
  CostModel cost = seeded();
  const CostModelStats seed = cost.stats();
  const model::CommandWork work = unit(300, 8192, 0, 0);
  for (uint32_t command = 0; command < 200; ++command)
    cost.observe(work, work, truth(work) * 1.1, 1.0);
  require(relative(cost.price(work), truth(work) * 1.1) < 0.03, "one size's price did not follow its units");
  const CostModelStats stats = cost.stats();
  // What a one-chunk command pays whatever its rows, against its rows.
  const auto share = [](const CostModelStats &of) {
    const double constant = of.rates[0] + of.rates[1] * kLayers;
    return constant / (constant + of.rates[2] * 300 * kLayers);
  };
  require(relative(share(stats), share(seed)) < 0.05, "one size moved the seed's split of its constant and rows");
}

// (3) The model's expected bytes price at once: 30 % more of them adds their
// rate times the difference, before any unit trains.
void testMoreExpectedBytesPriceAtOnce() {
  const CostModel cost = seeded();
  const model::CommandWork work = unit(2048, 0, 10'000'000'000, 0);
  model::CommandWork more = work;
  more.weightReadBytes = work.weightReadBytes * 13 / 10;
  require(relative(cost.price(more) - cost.price(work), cost.stats().rates[kWeightReads] * 3e9) < 1e-9,
          "more expected bytes did not add their price");
}

// (4) A unit ten times its price moves the rates no further than one at
// twice it.
void testAStallMovesTheFitAsTwiceItsPrice() {
  const model::CommandWork work = unit(2048, 4096, 8'000'000'000, 0);
  CostModel stalled = seeded(), doubled = seeded();
  stalled.observe(work, work, 10.0 * stalled.price(work), 1.0);
  doubled.observe(work, work, 2.0 * doubled.price(work), 1.0);
  require(stalled.stats().rates == doubled.stats().rates && stalled.stats().clamped == 1 &&
              doubled.stats().clamped == 0,
          "a stall moved the fit beyond twice its price");
}

// (5) A command in 24 passes, each a 24th of it, moves the fit as the whole
// command does: varied commands at 1.2 times the truth, run in passes or
// whole, price a whole command alike, and as they ran.
void testPassesMoveTheFitAsTheirCommand() {
  CostModel passed = seeded(), whole = seeded();
  std::mt19937_64 random(11);
  std::uniform_int_distribution<uint32_t> rows(256, 4 * kChunk), position(0, 65536);
  std::uniform_int_distribution<uint64_t> weights(4'000'000'000, 16'000'000'000);
  for (uint32_t command = 0; command < 30; ++command) {
    const uint32_t r = rows(random), p = position(random);
    const uint64_t w = weights(random);
    for (uint32_t pass = 0; pass < 24; ++pass)
      passed.observe(unit(r, p, w, 0, 24), unit(r, p, w, 0, 24), 1.2 * truth(unit(r, p, w, 0, 24)), 1.0 / 24);
    whole.observe(unit(r, p, w, 0), unit(r, p, w, 0), 1.2 * truth(unit(r, p, w, 0)), 1.0);
  }
  const model::CommandWork probe = unit(4096, 16384, 10'000'000'000, 0);
  require(relative(passed.price(probe), whole.price(probe)) < 0.05 &&
              relative(passed.price(probe), 1.2 * truth(probe)) < 0.1,
          "a command's passes moved the fit unlike the command: " + std::to_string(passed.price(probe)) +
              " against " + std::to_string(whole.price(probe)) + " ms, the commands' " +
              std::to_string(1.2 * truth(probe)));
}

// (6) Collinear terms leave no rate negative.
void testCollinearTermsLeaveNoRateNegative() {
  CostModel cost = seeded();
  for (uint32_t command = 0; command < 100; ++command) {
    // Rows and pairs grow together, and the time falls with them.
    const model::CommandWork work = unit(64 + command, 0, 0, 0);
    cost.observe(work, work, 4000.0 - 10.0 * command, 1.0);
  }
  for (const double rate : cost.stats().rates)
    require(rate >= 0.0 && std::isfinite(rate), "a rate went negative or not finite");
}

// (8) A term no unit has (spilled K/V on a target whose K/V never spills)
// stays out of the fit: its seed holds, nothing is NaN.
void testATermNoUnitHasKeepsItsSeed() {
  CostModel cost = seeded();
  const double seed = cost.stats().seedRates[kSpilledReads];
  for (uint32_t command = 0; command < 50; ++command) {
    const model::CommandWork work = unit(512 + 64 * command, 1024 * command, 1'000'000'000, 0);
    cost.observe(work, work, truth(work), 1.0);
  }
  const CostModelStats stats = cost.stats();
  require(stats.rates[kSpilledReads] == seed, "a term no unit has left its seed");
  for (const double rate : stats.rates)
    require(std::isfinite(rate), "a rate is not finite");
}

// The model's expected weight bytes, priced at the share of them the units
// realize: a fake that reads 0.59 of the uniform expectation at 300 rows and
// 0.46 at 8192 (amend/I.md F1), commands of either size in turn, prices
// within 15 % of what is read after eight units; without the scale it is 1.7
// and 2.2 times off.
void testExpectedWeightReadsCalibrateToRealized() {
  CostModel cost = seeded();
  const std::array<std::pair<uint32_t, double>, 2> sizes{{{300, 0.59}, {8192, 0.46}}};
  const auto expected = [](uint32_t rows) { return unit(rows, 0, 20'000'000'000, 0); };
  for (uint32_t command = 0; command < 8; ++command) {
    const auto [rows, share] = sizes[command % sizes.size()];
    model::CommandWork realized = expected(rows);
    realized.weightReadBytes = static_cast<uint64_t>(share * double(realized.weightReadBytes));
    cost.observe(realized, expected(rows), truth(realized), 1.0);
  }
  const CostModelStats stats = cost.stats();
  for (const auto &[rows, share] : sizes) {
    const double priced = stats.realizedPerExpected[kWeightReads] * double(expected(rows).weightReadBytes);
    const double read = share * double(expected(rows).weightReadBytes);
    require(relative(priced, read) < 0.15, "the priced bytes did not follow the realized ones");
    require(double(expected(rows).weightReadBytes) / read > 1.6, "the fixture's expectation is not biased");
  }
  require(stats.realizedPerExpected[kSpilledReads] == 1.0 && stats.realizedPerExpected[0] == 1.0,
          "a term without expected bytes was scaled");
}

// The warm-ups seed the price and never calibrate it: a warm-up prompt of
// one token reads 4 % of what a text prompt would.
void testWarmupsDoNotCalibrate() {
  CostModel cost;
  const std::array warmups{sample(unit(kChunk, 0, 600'000'000, 0), 3213.0),
                           sample(unit(256, 0, 600'000'000, 0), 2400.0)};
  cost.seed(warmups, {.bytes = 30'000'000'000, .milliseconds = 4000.0});
  require(cost.stats().realizedPerExpected[kWeightReads] == 1.0, "a warm-up calibrated the expected bytes");
}

// What /status names the terms.
void testTheTermsAreNamed() {
  const std::array<std::string_view, kCostTerms> names{"units",       "chunk_layers",      "row_layers",
                                                       "attention_pairs", "index_pairs", "weight_read_bytes",
                                                       "spilled_kv_read_bytes"};
  for (size_t term = 0; term < kCostTerms; ++term)
    require(costTermName(static_cast<CostTerm>(term)) == names[term], "a term is misnamed");
}

// The latest units' relative errors, before each trained the fit: the
// median and the 95th percentile.
void testRelativeErrorsAreTheLatestUnits() {
  CostModel cost = seeded();
  const model::CommandWork work = unit(2048, 0, 400'000'000, 0);
  const double price = cost.price(work);
  cost.observe(work, work, price, 1.0);
  CostModelStats stats = cost.stats();
  require(stats.samples == 1 && stats.relativeErrorP50 < 1e-9, "an exact unit's error was not zero");
  cost.observe(work, work, 2.0 * cost.price(work), 1.0);
  stats = cost.stats();
  require(stats.samples == 2 && std::fabs(stats.relativeErrorP95 - 0.5) < 1e-9,
          "a unit at twice its price did not err by a half");
}

constexpr test::TestCase kTests[] = {
    {"testTheSeedReproducesAnExactAffineTruth", testTheSeedReproducesAnExactAffineTruth},
    {"testSeedReadRatesComeFromTheLoad", testSeedReadRatesComeFromTheLoad},
    {"testPriceIsLinearInReadBytes", testPriceIsLinearInReadBytes},
    {"testRecoversTheRatesVariedUnitsShow", testRecoversTheRatesVariedUnitsShow},
    {"testOneSizeKeepsTheSeedsSplit", testOneSizeKeepsTheSeedsSplit},
    {"testMoreExpectedBytesPriceAtOnce", testMoreExpectedBytesPriceAtOnce},
    {"testAStallMovesTheFitAsTwiceItsPrice", testAStallMovesTheFitAsTwiceItsPrice},
    {"testPassesMoveTheFitAsTheirCommand", testPassesMoveTheFitAsTheirCommand},
    {"testCollinearTermsLeaveNoRateNegative", testCollinearTermsLeaveNoRateNegative},
    {"testATermNoUnitHasKeepsItsSeed", testATermNoUnitHasKeepsItsSeed},
    {"testExpectedWeightReadsCalibrateToRealized", testExpectedWeightReadsCalibrateToRealized},
    {"testWarmupsDoNotCalibrate", testWarmupsDoNotCalibrate},
    {"testTheTermsAreNamed", testTheTermsAreNamed},
    {"testRelativeErrorsAreTheLatestUnits", testRelativeErrorsAreTheLatestUnits},
};

} // namespace

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"cost", kTests}};
  return test::runTests("cost-model", groups, argc, argv);
}
