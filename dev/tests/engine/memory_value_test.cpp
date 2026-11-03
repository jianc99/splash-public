// What memory is worth (engine/MemoryValue.hpp), on inputs from recorded
// runs and the fakes' fitted profiles.

#include "FakeElasticPool.hpp"
#include "TestChecks.hpp"
#include "TestRegistry.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryValue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

std::string &recordedLoadsDirectory() {
  static std::string directory = "dev/tests/fixtures/recorded-loads";
  return directory;
}

namespace {

using test::FakeElasticPool;
using test::FakeMemory;
using test::require;

constexpr uint64_t kMiB = 1ULL << 20;
constexpr double kGiB = double(1ULL << 30);

// Flash-Next Q2_0's streamed experts: 24,576 routed experts of 1,382,400 B a
// slot (streamed_experts.arena_bytes / slots in every recorded status).
constexpr uint64_t kExperts = 24576;
constexpr uint64_t kSlotBytes = 1382400;
// DFlash2 commits 7.94 tokens a lane per decode command (TimingProfile.cpp,
// from mMNa's 9000 output tokens over its decode commands' lanes).
constexpr double kTokensPerLaneCommand = 7.94;

// A pool observed every 500 ms for `seconds` with commands running: decode
// ghost hits, hits and prefill scan reads at those rates.
value::PoolRate observed(double ghostsPerSecond, double hitsPerSecond, double scanBytesPerSecond, uint64_t ghostBytes,
                         double seconds = 30.0) {
  value::PoolRate rate;
  model::PoolValue counters{.ghostBytes = ghostBytes};
  for (double t = 0.0; t <= seconds; t += 0.5) {
    rate.observe(counters, 0, t, true);
    counters.ghostHits += static_cast<uint64_t>(std::llround(ghostsPerSecond * 0.5));
    counters.hits += static_cast<uint64_t>(std::llround(hitsPerSecond * 0.5));
    counters.scanReadBytes += static_cast<uint64_t>(std::llround(scanBytesPerSecond * 0.5));
  }
  return rate;
}

// Before its first interval a rate is zero, and the first interval with
// commands sets it, with no prior; a cache unit is worth nothing before the
// probes measure reuse, and the spilled decode's read time starts from the
// measured 9 ms.
void testRatesStartFromTheirFirstMeasurement() {
  value::PoolRate rate;
  require(rate.ghostHitsPerSecond() == 0.0 && rate.hitsPerSecond() == 0.0 && rate.scanReadBytesPerSecond() == 0.0 &&
              rate.takesPerSecond() == 0.0 && rate.ghostBytes() == 0.0,
          "a rate is not zero before its first interval");
  rate.observe({.hits = 10, .ghostHits = 20, .ghostBytes = 4096, .scanReadBytes = 100}, 1, 0.0, true);
  rate.observe({.hits = 60, .ghostHits = 70, .ghostBytes = 4096, .scanReadBytes = 600}, 3, 0.5, false);
  require(rate.ghostHitsPerSecond() == 0.0, "an interval without commands set a rate");
  rate.observe({.hits = 110, .ghostHits = 170, .ghostBytes = 8192, .scanReadBytes = 1100}, 4, 1.0, true);
  require(rate.hitsPerSecond() == 100.0 && rate.ghostHitsPerSecond() == 200.0 &&
              rate.scanReadBytesPerSecond() == 1000.0 && rate.takesPerSecond() == 2.0 && rate.ghostBytes() == 8192.0,
          "the first interval with commands did not set the rates");
  const MemoryRates rates;
  require(value::cached(8192, 108 * kMiB, false, false, rates).at(60.0) == 0.0 &&
              value::cached(8192, 108 * kMiB, false, true, {.prefillSecondsPerRow = 0.001}).at(60.0) == 8.192,
          "a cache unit was priced at a prior reuse, or certain reuse at less than its recompute");
  require(rates.spilledReadMillisecondsPerLaneCommand == 9.0 && rates.tierReadBytesPerSecond == 0.0 &&
              rates.reuse == 0.0,
          "a rate other than the spilled decode's measured 9 ms has a prior");
}

// A unit's cycle is its refill, key by key at the pool's miss time, and its
// release at the pool's measured time per byte: on the M6 a 1023 MiB store
// chunk is 776 experts at 0.1846 ms (1,382,400 B at the profile's 7.49 GB/s)
// and 62.7 ms of compaction (the median of 40 recorded 1023 MiB shrinks),
// 0.21 s. Without a release measured, no release is priced.
void testCycleCostIncludesCompaction() {
  const uint64_t unit = 1023 * kMiB;
  model::PoolValue store{
      .missMilliseconds = 0.1846, .releasedBytes = 2 * unit, .releaseMilliseconds = 125.4, .keyBytes = kSlotBytes};
  const double refill = double(unit) / kSlotBytes * 0.1846 / 1000.0;
  require(std::abs(value::cycleSeconds(store, unit) - (refill + 0.0627)) < 1e-9 &&
              std::abs(value::cycleSeconds(store, unit) - 0.206) < 0.005,
          "the cycle is not the refill and the compaction");
  store.keyBytes = 0;
  require(std::abs(value::cycleSeconds(store, unit) - 0.0627) < 1e-9,
          "with no key reported the cycle is not the compaction alone");
  require(value::cycleSeconds({.missMilliseconds = 0.1846}, unit) == 0.0, "an unmeasured release was priced");
}

// Units no lane uses cost nothing: the hot block cache while no lane decodes
// spilled K/V (no ghost hits), a store whose slots hold no evicted key (no
// ghosts) about one miss, embeddings one encode if reused within the hold.
void testUnitsWithoutDemandCostAboutNothing() {
  const MemoryRates rates;
  const value::PoolRate idle = observed(0.0, 0.0, 0.0, 0);
  require(value::poolUnit(idle, {.missMilliseconds = 9.0}, 64 * kMiB, 0, 60.0, rates).at(60.0) == 0.0,
          "an idle hot block cache's extent costs something");
  const value::PoolRate hits = observed(0.0, 500.0, 0.0, 0);
  require(std::abs(value::poolUnit(hits, {.missMilliseconds = 0.1846}, 1023 * kMiB, 0, 60.0, rates).at(60.0) -
                   0.0001846) < 1e-12,
          "a store without ghosts does not cost one miss");
  const value::PoolRate encodes = observed(0.0, 2.0, 0.0, 0);
  require(std::abs(value::poolUnit(encodes, {.missMilliseconds = 250.0}, 8 * kMiB, 0, 60.0, rates).at(60.0) - 0.25) <
              1e-9,
          "embeddings reused within the hold do not cost one encode");
}

// The experts' value counts prefill reads (amend/M.md B8, A18): on the M6,
// with no decoder and no ghost hit, scans reading 3.2 GB a prefill-second
// (amend/M.md D10) at the profile's weight price ((1 - 0.8 hidden) / 5 GB/s)
// make a 1023 MiB chunk of the 5722-slot store dearer than an 8K copied
// chain read back at 5 GB/s at reuse 0.25; without the scan term the chunk
// costs one miss and goes first.
void testExpertValueCountsPrefillReads() {
  const uint64_t unit = 1023 * kMiB;
  const uint64_t missing = (kExperts - 5722) * kSlotBytes;
  const MemoryRates rates{.weightReadSecondsPerByte = 0.2 / 5.0e9, .tierReadBytesPerSecond = 5.0e9, .reuse = 0.25};
  // Prefill hits at the store's 0.288 share: 3.2 GB / 1,382,400 B of misses
  // at 0.712 of the uses.
  const double hits = 3.2e9 / kSlotBytes * 0.288 / 0.712;
  const value::PoolRate scanning = observed(0.0, hits, 3.2e9, 0);
  const value::PoolRate blind = observed(0.0, hits, 0.0, 0);
  const model::PoolValue store{.missMilliseconds = 0.1846};
  const double hold = 200.0;
  const double chunk = value::poolUnit(scanning, store, unit, missing, hold, rates).at(hold) / double(unit);
  const double chunkBlind = value::poolUnit(blind, store, unit, missing, hold, rates).at(hold) / double(unit);
  const uint64_t extent = 108527616;  // 8192 tokens of one lane's KV (status kv_extent_bytes)
  const double chain = value::cached(8192, extent, true, false, rates).at(hold) / double(extent);
  require(chain < chunk, "a copied 8K chain is not cheaper than a store chunk prefill scans read");
  require(chunkBlind < chain, "without the scan term the store chunk is not the cheaper (the mutant's choice)");
}

// HiSparse on, Studio 36 GB, in a prefill phase (amend/M.md B8 (b)): the
// scans' 4.5 GB a prefill-second (D10) at the profile's weight price
// ((1 - 0.05 hidden) / 12.4 GB/s) over the 11,360 slots not resident make a
// 678 MiB chunk dearer per byte than converting a 128K lane (1.09 GiB, D9)
// with 1500 tokens left at Spill's +2.0 ms a token; without the scan term
// the chunk is the cheaper and would go.
void testHiSparseOnStudio36GWeighsPrefillScans() {
  const uint64_t chunkBytes = 678 * kMiB;
  const uint64_t missing = (kExperts - 13216) * kSlotBytes;
  const MemoryRates rates{.weightReadSecondsPerByte = 0.95 / 12.4e9,
                          .spilledReadMillisecondsPerLaneCommand = 2.0 * kTokensPerLaneCommand,
                          .decodeTokensPerLaneCommand = kTokensPerLaneCommand};
  // Decode: +2-3 % a GiB at 5.9 ms a token (0.15 ms a token a GiB), 21.6
  // tokens a second, at 0.2222 ms an expert (1,382,400 B at 6.22 GB/s).
  const double ghosts = 21.6 * 0.15 / 0.2222 * double(chunkBytes) / kGiB;
  const value::PoolRate prefill = observed(ghosts, 0.0, 4.5e9, chunkBytes);
  const value::PoolRate decodeOnly = observed(ghosts, 0.0, 0.0, chunkBytes);
  const model::PoolValue store{.missMilliseconds = 0.2222};
  const double hold = 200.0;
  const double chunk = value::poolUnit(prefill, store, chunkBytes, missing, hold, rates).at(hold) / double(chunkBytes);
  const double chunkBlind =
      value::poolUnit(decodeOnly, store, chunkBytes, missing, hold, rates).at(hold) / double(chunkBytes);
  const double lane = value::conversion(value::spillDecodeSeconds(1500, rates)).at(hold) / (1.09 * kGiB);
  require(lane < chunk, "the prefill's scans do not make the chunk dearer than the 128K lane's conversion");
  require(chunkBlind < lane, "without the scan term the chunk is not the cheaper (the mutant's choice)");
}

// The spilled decode's term (DESIGN section 0 Q5): projected tokens over the
// tokens a lane commits per command, times the engine's mean hot-block time
// per lane and command, 9 ms until its first sample; a lane commits at least
// one token a command.
void testSpillDecodeTermUsesTheEnginesMeanAndItsPrior() {
  MemoryRates rates{.decodeTokensPerLaneCommand = kTokensPerLaneCommand};
  require(std::abs(value::spillDecodeSeconds(1500, rates) - 1500 / kTokensPerLaneCommand * 0.009) < 1e-12,
          "the term does not start from the 9 ms prior");
  rates.spilledReadMillisecondsPerLaneCommand = 10.0;
  require(std::abs(value::spillDecodeSeconds(1500, rates) - 1500 / kTokensPerLaneCommand * 0.010) < 1e-12,
          "the term does not follow the engine's mean");
  rates.decodeTokensPerLaneCommand = 0.0;
  require(std::abs(value::spillDecodeSeconds(1500, rates) - 15.0) < 1e-12,
          "with no decode measured the term does not take one token a command");
  require(value::conversion(2.5).at(1e6) == 2.5, "a conversion's cost is not its penalty, once");
}

// Free room goes back to a pool claims retake only while a unit is worth its
// expected retake: none taken grows unconditionally; one taken every 20 s
// asks 95 % of its cycle (1 - e^-3 over the 60 s horizon).
void testGrowthPaysOnlyWhatARetakeCosts() {
  value::PoolRate untaken;
  require(value::growsIntoFreeRoom(untaken, 0.0, 1.0), "an untaken pool did not grow unconditionally");
  value::PoolRate retaken;
  for (uint64_t step = 0; step <= 400; ++step)
    retaken.observe({}, step / 40, step * 0.5, true);
  const double share = 1.0 - std::exp(-retaken.takesPerSecond() * kValueHorizonSeconds);
  require(std::abs(retaken.takesPerSecond() - 0.05) < 0.02 && share > 0.9,
          "a unit taken every 20 s is not taken at about 0.05 a second");
  require(value::growsIntoFreeRoom(retaken, share * 1.0 + 1e-9, 1.0) &&
              !value::growsIntoFreeRoom(retaken, share * 1.0 - 1e-6, 1.0),
          "growth into free room does not ask the expected retake's cost");
}

// A recorded M6 run, sampled every 2 s.
struct RecordedRun final {
  std::map<std::string, double> constants;
  struct Sample final {
    double seconds = 0.0;
    uint64_t misses = 0, prefillBytes = 0, commands = 0;
  };
  std::vector<Sample> samples;
};

RecordedRun readRun(const std::string &name) {
  const std::string path = recordedLoadsDirectory() + "/" + name;
  std::ifstream file(path);
  require(file.good(), "cannot read " + path);
  RecordedRun run;
  for (std::string line; std::getline(file, line);) {
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream fields(line);
    if (std::isalpha(static_cast<unsigned char>(line[0]))) {
      std::string key;
      double number = 0.0;
      fields >> key >> number;
      run.constants[key] = number;
      continue;
    }
    RecordedRun::Sample sample;
    fields >> sample.seconds >> sample.misses >> sample.prefillBytes >> sample.commands;
    run.samples.push_back(sample);
  }
  require(run.samples.size() > 400, path + " holds too few samples");
  return run;
}

// The rates' time constant (amend/M.md section 2): replayed through
// PoolRate at the control pass's 500 ms, a recorded M6 run's decode misses
// (the ghost-hit proxy) and prefill reads change by less, at P95 over the
// passes of steady load, than half the relative hysteresis a 1023 MiB
// chunk's cycle gives it at the run's value: the refill and the median
// compaction against the misses a chunk saves (D2: 21.9 a token for the 866
// slots between floor and cap) at the run's decode rate over the horizon.
void testRecordedLoadsDoNotFlipFlop() {
  for (const char *name : {"m6-mMNa.tsv", "m6-pFa.tsv"}) {
    const RecordedRun run = readRun(name);
    const double missMilliseconds = run.constants.at("miss_ms");
    const double slots = 1023.0 * double(kMiB) / run.constants.at("slot_bytes");
    const double cycle = slots * missMilliseconds / 1000.0 + run.constants.at("compaction_ms_per_1023_mib") / 1000.0;
    const double perSlot = (60.75 - 38.85) / (5722 - 4856);
    const double worth = run.constants.at("decode_tokens_per_second") * perSlot * slots * missMilliseconds / 1000.0 *
                         kValueHorizonSeconds;
    const double halfHysteresis = 0.5 * cycle / worth;

    value::PoolRate rate;
    std::vector<double> missChanges, scanChanges;
    bool steadyMisses = false, steadyScans = false;
    rate.observe({.ghostHits = run.samples.front().misses, .scanReadBytes = run.samples.front().prefillBytes}, 0,
                 run.samples.front().seconds, false);
    for (size_t index = 1; index < run.samples.size(); ++index) {
      const RecordedRun::Sample &a = run.samples[index - 1], &b = run.samples[index];
      const double span = b.seconds - a.seconds;
      if (span <= 0.0)
        continue;
      const bool commands = b.commands != a.commands;
      const int passes = std::max(1, static_cast<int>(std::lround(span / 0.5)));
      const bool missing = b.misses > a.misses, scanning = b.prefillBytes > a.prefillBytes;
      for (int pass = 1; pass <= passes; ++pass) {
        const double share = double(pass) / passes;
        const double misses = rate.ghostHitsPerSecond(), scans = rate.scanReadBytesPerSecond();
        rate.observe({.ghostHits = a.misses + static_cast<uint64_t>(double(b.misses - a.misses) * share),
                      .scanReadBytes =
                          a.prefillBytes + static_cast<uint64_t>(double(b.prefillBytes - a.prefillBytes) * share)},
                     0, a.seconds + span * share, commands);
        if (commands && missing && steadyMisses && misses > 0.0)
          missChanges.push_back(std::abs(rate.ghostHitsPerSecond() - misses) / misses);
        if (commands && scanning && steadyScans && scans > 0.0)
          scanChanges.push_back(std::abs(rate.scanReadBytesPerSecond() - scans) / scans);
      }
      if (commands) {
        steadyMisses = missing;
        steadyScans = scanning;
      }
    }
    const auto p95 = [](std::vector<double> changes) {
      std::ranges::sort(changes);
      return changes[static_cast<size_t>(0.95 * double(changes.size() - 1))];
    };
    const double worst = std::max(p95(missChanges), p95(scanChanges));
    require(worst < halfHysteresis, std::string(name) + ": the rates change by " + std::to_string(worst) +
                                        " a pass at P95, past half the hysteresis " + std::to_string(halfHysteresis));
  }
}

// value_per_gib against a pinned store (amend/M.md section 2, D2). On the M6
// 1.5.0's store held 5722 slots and missed 37.3 and 40.4 experts a decode
// token (m150a, m150b); mMNa's ran at its 4856-slot floor. Replayed through
// a store at that floor that a ledger observes every 500 ms, mMNa's misses
// beyond 1.5.0's are the hits of ghosts spanning the 866 slots between,
// each at mMNa's miss time (0.2095 ms, its mean host-step work a read): the
// account's value_per_gib over the passes with commands, per decode token,
// is what the slots above the floor saved a decode token's wall time, D2's
// 1.1-3.6 ms a GiB. At 37.3 it replays to 3.79, and at 40.4 to 3.25, 1.59
// and 1.75 times what those runs' own wall times show (2.66 and 2.07 ms a
// token, 2.39 and 1.86 a GiB): a mean read costs more than the one a
// smaller store adds (a decode token's host work differs by 3.34 and 2.40
// ms for 21.6 and 18.5 reads, 0.155 and 0.130 ms a read). Whether a miss
// is priced at the marginal read is the owner's to decide (open); until
// then this fails at 37.3. pFa is not replayed: its 5128-slot store is not
// 1.5.0's floor pair, and it misses more a token (61.7) than mMNa's
// smaller one.
void testValuePerGiBReproducesThePinnedStore() {
  constexpr uint32_t kFloorSlots = 4856, kPinnedSlots = 5722;
  // What the store's slots above the floor saved a decode token's wall time
  // on the M6, ms a GiB (amend/M.md D2: 1.5.0's whole store at 18.2-19.1 ms
  // a token, main's and intpart's floor at 20.3-22.2, 1140 MiB between).
  constexpr double kSavedLeast = 1.1, kSavedMost = 3.6;
  // 1.5.0's misses a decode token: m150a and m150b.
  constexpr double kPinnedMisses[] = {37.3, 40.4};
  const RecordedRun run = readRun("m6-mMNa.tsv");
  const double missMilliseconds = run.constants.at("miss_ms");
  const double tokens =
      run.constants.at("decode_tokens_per_second") * (run.samples.back().seconds - run.samples.front().seconds);
  const double missesPerToken = double(run.samples.back().misses - run.samples.front().misses) / tokens;
  for (const double pinnedPerToken : kPinnedMisses) {
    const double ghostShare = 1.0 - pinnedPerToken / missesPerToken;
    FakeMemory memory({});
    FakeElasticPool store("experts", memory, kSlotBytes, kFloorSlots, kPinnedSlots, kPinnedSlots, kFloorSlots,
                          missMilliseconds, 0.0);
    MemoryLedger ledger;
    ledger.addPool(store);
    ledger.observePools(run.samples.front().seconds, false);
    // The ghost hits due and served so far; value_per_gib / H summed over
    // the passes with commands, in seconds a GiB.
    double due = 0.0, worth = 0.0;
    uint64_t served = 0;
    for (size_t index = 1; index < run.samples.size(); ++index) {
      const RecordedRun::Sample &a = run.samples[index - 1], &b = run.samples[index];
      const double span = b.seconds - a.seconds;
      if (span <= 0.0)
        continue;
      const bool commands = b.commands != a.commands;
      const int passes = std::max(1, static_cast<int>(std::lround(span / 0.5)));
      for (int pass = 1; pass <= passes; ++pass) {
        due += double(b.misses - a.misses) * ghostShare / passes;
        const uint64_t ghosts = static_cast<uint64_t>(std::llround(due)) - served;
        store.served(0, ghosts, 0, 0);
        served += ghosts;
        ledger.observePools(a.seconds + span * pass / passes, commands);
        if (!commands)
          continue;
        MemoryGovernorSnapshot snapshot;
        ledger.fill(snapshot, 0);
        worth += snapshot.accounts.at(1).valuePerGiB / kValueHorizonSeconds * span / passes;
      }
    }
    const double perToken = worth / tokens * 1000.0;
    require(perToken >= kSavedLeast && perToken <= kSavedMost,
            "at 1.5.0's " + std::to_string(pinnedPerToken) + " misses a token value_per_gib replays to " +
                std::to_string(perToken) + " ms a decode token a GiB, outside the " + std::to_string(kSavedLeast) +
                " to " + std::to_string(kSavedMost) + " the store saved (amend/M.md D2)");
  }
}

} // namespace

std::span<const test::TestCase> valueTests() {
  static constexpr test::TestCase kTests[] = {
      {"testRatesStartFromTheirFirstMeasurement", testRatesStartFromTheirFirstMeasurement},
      {"testCycleCostIncludesCompaction", testCycleCostIncludesCompaction},
      {"testUnitsWithoutDemandCostAboutNothing", testUnitsWithoutDemandCostAboutNothing},
      {"testExpertValueCountsPrefillReads", testExpertValueCountsPrefillReads},
      {"testHiSparseOnStudio36GWeighsPrefillScans", testHiSparseOnStudio36GWeighsPrefillScans},
      {"testSpillDecodeTermUsesTheEnginesMeanAndItsPrior", testSpillDecodeTermUsesTheEnginesMeanAndItsPrior},
      {"testGrowthPaysOnlyWhatARetakeCosts", testGrowthPaysOnlyWhatARetakeCosts},
      {"testRecordedLoadsDoNotFlipFlop", testRecordedLoadsDoNotFlipFlop},
      {"testValuePerGiBReproducesThePinnedStore", testValuePerGiBReproducesThePinnedStore},
  };
  return kTests;
}
