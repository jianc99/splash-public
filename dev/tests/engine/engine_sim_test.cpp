// Integration scenarios (DESIGN §7.11, amend/V.md binding 8): the engine and
// its native loop on the fakes, at the fitted profiles of the Macs the matrix
// runs on and the memory their recorded /status shows, each asserting the
// rules the design makes the parts keep together. CPU only; seconds each.
//
// The machines' numbers are the recorded churn runs' first status samples
// (notes/churn: sL for the Studio at --max-memory 18G, fn15b for the 24 GB
// M6, m35M for 35B at 18G): the limit, the fixed runtime and the dynamic
// budget, the store's arena at its floor and cap, the host's spare memory and
// reserves, the KV page bytes and the spill plan. Each store is a pool of
// whole chunks; the chunk is the size that puts its floor and cap at the
// recorded slots (the Studio's 954 slots: 5724 and 6678 against 5672 and
// 6679; the M6's 817: 4902 and 5719 against 4856 and 5721; 35B's 560: 5600
// and 6720 against 5684 and 6640), near memory_ledger_test's chunks.

#include "CostSeeds.hpp"
#include "EngineHarness.hpp"
#include "FakeElasticPool.hpp"
#include "FakeMemory.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "TimingProfile.hpp"
#include "engine/CostModel.hpp"
#include "engine/Engine.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;

namespace {

using test::require;

constexpr uint64_t kSlotBytes = 1'382'400;  // a Flash-Next expert as the store streams it
constexpr uint32_t kTokensPerCommand = 8;   // DFlash2 commits 7.93-7.98 tokens a lane command (TimingProfile.cpp)

// --- The machines -------------------------------------------------------------------------------------------

// The expert store as a pool of whole chunks (FakeElasticPool), its misses
// priced at an expert's read (memory_ledger_test's HiSparse fixtures:
// 1,382,400 B at 6.22 GB/s on the Studio, 7.49 GB/s on the M6; 1,911,603 B
// at 6.22 GB/s for 35B) and a chunk's compaction at those fixtures' 62.7 ms.
struct Store final {
  uint64_t unitBytes = 0;
  uint32_t floor = 0, cap = 0, whole = 0;
  double missMilliseconds = 0.0;
};
constexpr double kReleaseMillisecondsPerUnit = 62.7;

struct Machine final {
  const char *name = "";
  const test::TimingProfile *profile = nullptr;
  // The governor's limit and the reserves it leaves untracked (pipelines
  // and runtime overhead), and what the runtime holds outside the KV, the
  // states and the store's arena: fixed_runtime_bytes less those reserves
  // and the arena at its cap.
  uint64_t limitBytes = 0, untrackedBytes = 0, fixedBytes = 0;
  // The Mac: its memory, what other applications held at the run's first
  // sample (physical less allocated less spare), and the reserves.
  uint64_t hostBytes = 0, otherBytes = 0, hostReserveBytes = 0, hostSpareReserveBytes = 0;
  uint64_t pageBytes = 0, indexPageBytes = 0;
  uint32_t extentPages = 256;
  // A lane's state buffers: lane_state_bytes over the four lanes it plans.
  uint64_t laneStateBytes = 0;
  Store store;
  // The spill plan (none for a target that cannot spill), and whether every
  // request spills (--kv-spill always).
  std::optional<engine::SpillGeometry> spill;
  bool spillAlways = false;
  // Commands of a streamed target: its command rows in up to a pass a layer.
  uint32_t commandRows = model::ExecutionLimits::prefillTokenBudget;
  // The churn's work: prefill rows / R + output tokens / D (DATA.md).
  double rowsPerWork = 1000.0, tokensPerWork = 75.0;
};

// Flash-Next's spill plan as both Macs' runs recorded it (kv_spill: dense 64,
// window 64, group 256 pages; the hot floor a decoding Spill lane's readers
// take; context_bounds.resident 147,449 tokens).
engine::SpillGeometry flashNextSpill() {
  return {.fullPageBytes = 423'936,
          .indexPageBytes = 24'576,
          .slotBytes = 100,  // the test tier's slots (TestKvTier::slotBytes), as its quota counts them
          .residentTokens = 147'449,
          .densePages = 64,
          .windowPages = 64,
          .groupPages = 256,
          .laneReadBytes = 408'944'640};
}

// The Studio (36 GB) at --max-memory 18G, Flash-Next Q2_0 (sL).
const Machine kStudio18G{
    .name = "studio-18g",
    .profile = &test::kStudio18GFlashNextQ2,
    .limitBytes = 19'327'352'832,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 17'048'316'698 - 805'306'368 - 7ULL * 954 * kSlotBytes,
    .hostBytes = 38'654'705'664,
    .otherBytes = 38'654'705'664 - 16'091'482'912 - 12'678'086'656,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'705'829'392,
    .pageBytes = 423'936,
    .indexPageBytes = 24'576,
    .laneStateBytes = 276'496'384 / 4,
    .store = {.unitBytes = 954 * kSlotBytes, .floor = 6, .cap = 7, .whole = 26, .missMilliseconds = 0.2222},
    .spill = flashNextSpill(),
    .commandRows = 8192,
};

// The 24 GB M6, Flash-Next Q2_0 at its default memory (fn15b).
const Machine kM6{
    .name = "m6",
    .profile = &test::kM6FlashNextQ2,
    .limitBytes = 17'995'923'456,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 15'716'314'026 - 805'306'368 - 7ULL * 817 * kSlotBytes,
    .hostBytes = 25'769'803'776,
    .otherBytes = 25'769'803'776 - 14'759'478'192 - 4'537'188'352,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'147'483'648,
    .pageBytes = 423'936,
    .indexPageBytes = 24'576,
    .laneStateBytes = 276'496'384 / 4,
    .store = {.unitBytes = 817 * kSlotBytes, .floor = 6, .cap = 7, .whole = 30, .missMilliseconds = 0.1846},
    .spill = flashNextSpill(),
    .commandRows = 8192,
    .rowsPerWork = 585.0,
    .tokensPerWork = 46.0,
};

// The Studio at its default memory: the store at sA's 13,216 slots in
// chunks of memory_ledger_test's 678 MiB (514 slots), its floor the plan's
// 0.85 of the cap; the runtime and the rest of the Mac as at 18G (a test
// sets its limit).
const Machine kStudio36{
    .name = "studio-36g",
    .profile = &test::kStudio36FlashNextQ2,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 17'048'316'698 - 805'306'368 - 7ULL * 954 * kSlotBytes,
    .hostBytes = 38'654'705'664,
    .otherBytes = 38'654'705'664 - 16'091'482'912 - 12'678'086'656,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'705'829'392,
    .pageBytes = 423'936,
    .indexPageBytes = 24'576,
    .laneStateBytes = 276'496'384 / 4,
    .store = {.unitBytes = 514 * kSlotBytes, .floor = 22, .cap = 26, .whole = 48, .missMilliseconds = 0.2222},
    .spill = flashNextSpill(),
    .commandRows = 8192,
};

// Qwen3.6-35B-A3B at --max-memory 18G on the Studio, streaming its experts
// without HiSparse (m35M): no Index pages, no Spill.
const Machine k35BStudio18G{
    .name = "studio-18g-35b",
    .profile = &test::k35BStudio18G,
    .limitBytes = 19'327'352'832,
    .untrackedBytes = 805'306'368,
    .fixedBytes = 18'378'746'256 - 805'306'368 - 12ULL * 560 * 1'911'603,
    .hostBytes = 38'654'705'664,
    .otherBytes = 38'654'705'664 - 16'091'482'912 - 12'678'086'656,
    .hostReserveBytes = 2'147'483'648,
    .hostSpareReserveBytes = 2'705'829'392,
    .pageBytes = 332'800,
    .laneStateBytes = 181'469'184 / 4,
    .store = {.unitBytes = 560 * 1'911'603, .floor = 10, .cap = 12, .whole = 18, .missMilliseconds = 0.3073},
    .commandRows = 8192,
};

std::string at(const Machine &machine) { return std::string(machine.name) + ": "; }

// A scenario's numbers, beside its verdict.
void note(const std::string &line) { std::printf("engine-sim: %s\n", line.c_str()); }

// --- Requests -----------------------------------------------------------------------------------------------

// A prompt of `tokens` tokens no other prompt shares a block of.
std::vector<uint32_t> prompt(uint64_t seed, uint32_t tokens) {
  std::vector<uint32_t> result(tokens);
  for (uint32_t index = 0; index < tokens; ++index)
    result[index] = static_cast<uint32_t>(seed << 20 | index);
  return result;
}

protocol::RequestFrame frame(uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = engine::RequestPriority::Normal;
  result.constraint = ConstraintMode::None;
  // An hour, on the fixture's unix clock (1 s).
  result.absoluteDeadlineUnixMicros = 1'000'000 + 3'600'000'000ULL;
  result.remainingDeadlineMicros = 3'600'000'000ULL;
  result.logicalMaxOutputTokens = output;
  result.promptTokens = std::move(tokens);
  return result;
}

// churn_ord.py's prompts at the matrix's 18G cells (BASE 5800, STEP 1500
// words, about 8.09 tokens a word): worker 2's first, the first request of
// order A, 59,043 tokens, and 12,200 more a round, the three workers a few
// tokens apart (round 2's pair, 71,220 and 71,254); every fourth round
// resends the previous conversation, its prompt that prefix and 8 tokens
// more. Output: max_tokens 1500, in whole commands.
struct ChurnPrompts final {
  uint32_t first = 59'043, step = 12'200;
  std::array<int32_t, 3> offsets{-23, 11, 0};
  uint32_t output = 1504;
  [[nodiscard]] uint32_t tokens(uint32_t worker, uint32_t round) const {
    return static_cast<uint32_t>(int64_t{first} + int64_t{step} * (round - 1) + offsets[worker]);
  }
};

// --- A loop on the fakes at a machine's memory ----------------------------------------------------------------

// The native loop (LoopFixture) at `machine`'s memory and profile: FakeMemory
// holding the fixed runtime, the governor at the machine's limit and host,
// the store as a FakeElasticPool, the KV tier and spill plan of a target
// that can spill, the production control pass between commands, and the
// accounts and pools a start books (RuntimeResources: the KV pages and the
// states; Bootstrap: the model's pools). Its price seeded as Bootstrap seeds
// it, from the two warm-ups and the load's reads.
struct Serving final {
  explicit Serving(const Machine &machine, std::vector<test::HostEvent> script = {},
                   std::optional<uint32_t> startUnits = std::nullopt, uint64_t limitBytes = 0)
      : machine(machine),
        fixture(config(machine), shape(machine, limitBytes ? limitBytes : machine.limitBytes), {},
                metal::kResidencyKeepAliveSeconds, options(machine, script, limitBytes)) {
    // The fixture's weights stand for part of the fixed runtime.
    const uint64_t weights = fixture.memory->held("weights");
    require(machine.fixedBytes >= weights, at(machine) + "the fixed runtime is less than the fixture's weights");
    fixture.memory->hold("runtime", machine.fixedBytes - weights);
    experts.emplace("experts", *fixture.memory, machine.store.unitBytes, machine.store.floor, machine.store.cap,
                    machine.store.whole, startUnits.value_or(machine.store.cap), machine.store.missMilliseconds,
                    kReleaseMillisecondsPerUnit);
    test::Executor &model = fixture.executor;
    model.experts = &*experts;
    model.decodeTokens = kTokensPerCommand;
    model.decodeFinishes = false;
    model.firstTokens = true;
    model.laneFootprintBytes = machine.laneStateBytes;
    if (machine.spill) {
      model.spilledReadFloorBytes = machine.spill->laneReadBytes;
      // sL's history staging, 281,149,440 B for its floor's 8192 pages.
      model.stagingPageBytes = 281'149'440 / 8192;
    }
    engine::MemoryLedger &ledger = fixture.governor->ledger();
    static_cast<void>(ledger.addAccount("kv", engine::AccountRole::Kv,
                                        [this] { return fixture.memory->held("kv"); }));
    static_cast<void>(ledger.addAccount("states", engine::AccountRole::States,
                                        [this] { return fixture.memory->held("states"); }));
    static_cast<void>(ledger.addPool(*experts));
    const test::TimedWork &work = *model.timing;
    const std::array warmups{test::sampleOf(work.warmupTimes(model::ExecutionLimits::prefillTokenBudget)),
                             test::sampleOf(work.warmupTimes(256))};
    const double rate = machine.profile->weightBytesPerMillisecond;
    fixture.loop.seedCost(warmups, model::SsdReads{10'000'000'000, 1e10 / rate});
    model.commandTimed = [this](const BatchPlan &plan, const test::CommandTimes &times) {
      commands.push_back({.kind = plan.kind, .pass = plan.pass, .passes = plan.passes,
                          .startMilliseconds = fixture.monotonic, .wallMilliseconds = times.wallMilliseconds});
      for (const BatchItem &item : plan.items) {
        commands.back().lanes.push_back(item.requestId);
        commands.back().rows.push_back(item.tokenCount);
      }
    };
  }

  static engine::NativeLoopConfig config(const Machine &machine) {
    engine::NativeLoopConfig result;
    result.engine.maxContext = 262'144;
    if (machine.spill)
      result.engine.spill = {.capable = true,
                             .always = machine.spillAlways,
                             .denseTokens = machine.spill->densePages * engine::KvCache::pageTokens,
                             .geometry = *machine.spill};
    return result;
  }
  // Pages for the whole limit, in the machine's extents.
  static test::PoolShape shape(const Machine &machine, uint64_t limitBytes) {
    const auto extents = static_cast<uint32_t>(limitBytes / machine.pageBytes / machine.extentPages + 1);
    return {.pages = extents * machine.extentPages,
            .extentPages = machine.extentPages,
            .runwayPages = 0,
            .pageBytes = machine.pageBytes,
            .indexPageBytes = machine.indexPageBytes};
  }
  static test::LoopOptions options(const Machine &machine, std::vector<test::HostEvent> script, uint64_t limitBytes) {
    test::LoopOptions result{
        .memory = test::FakeMemory::Config{.hostBytes = machine.hostBytes,
                                           .otherBytes = machine.otherBytes,
                                           .script = std::move(script)},
        .limits = {.limitBytes = limitBytes ? limitBytes : machine.limitBytes,
                   .hostReserveBytes = machine.hostReserveBytes,
                   .hostSpareReserveBytes = machine.hostSpareReserveBytes,
                   .untrackedReserveBytes = machine.untrackedBytes},
        .timing = machine.profile,
        .shape = [rows = machine.commandRows, layers = machine.profile->layers](test::Executor &model) {
          model.groupRows = rows;
          model.passes = layers;
        }};
    // Its slots' quota beyond any churn's KV; 64 transfers of a page each in
    // flight land at the profile's SSD rate together (TestKvTier's slots are
    // 100 bytes).
    result.tier = test::TierOptions{.capacity = 4'000'000,
                                    .transferLimit = 64,
                                    .bytesPerMillisecond = 100.0 * machine.profile->weightBytesPerMillisecond /
                                                           (64.0 * double(machine.pageBytes))};
    if (machine.spill)
      result.cacheSpill = engine::CacheSpillPolicy{machine.spill->densePages, machine.spill->windowPages};
    return result;
  }

  // Sends a request now.
  void submit(uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    submittedAt[id] = fixture.monotonic;
    require(fixture.loop.receive(protocol::peer::serialize(frame(id, std::move(tokens), output))),
            "a request frame was refused");
  }
  [[nodiscard]] bool active(uint64_t id) const { return submittedAt.contains(id) && !ended.contains(id); }
  [[nodiscard]] std::optional<double> ttft(uint64_t id) const {
    const auto found = tokenTimes.find(id);
    if (found == tokenTimes.end())
      return std::nullopt;
    return found->second.front() - submittedAt.at(id);
  }
  // What the loop wrote since the last step, at the clock's time: each
  // request's starts, token events and end.
  void drain() {
    if (fixture.output.empty())
      return;
    for (const protocol::EngineEvent &event : reader.feed(fixture.output)) {
      if (const auto *start = std::get_if<protocol::StartEvent>(&event)) {
        startedAt.try_emplace(start->requestId, fixture.monotonic);
        ++starts;
      } else if (const auto *tokens = std::get_if<protocol::TokensEvent>(&event)) {
        tokenTimes[tokens->requestId].push_back(fixture.monotonic);
      } else if (const auto *done = std::get_if<protocol::DoneEvent>(&event)) {
        ended.insert(done->requestId);
      } else if (const auto *error = std::get_if<protocol::ErrorEvent>(&event)) {
        ended.insert(error->requestId);
        errors.push_back(std::to_string(error->requestId) + " " + error->code + ": " + error->message);
      }
    }
    fixture.output.clear();
  }
  // Ticks until done(), observing each step.
  void run(const std::function<bool()> &done, const std::function<void()> &observe, std::string_view what) {
    fixture.run(
        [&] {
          drain();
          if (observe)
            observe();
          return done();
        },
        what, 100'000'000);
  }
  // What FakeMemory holds by holder, and the requests' phases, for a
  // failure's message.
  [[nodiscard]] std::string holders() const {
    std::string result = " (MiB held:";
    for (const char *holder : {"runtime", "experts", "kv", "states", "spilled_reads", "weights"})
      result += std::string(" ") + holder + " " + std::to_string(fixture.memory->held(holder) >> 20);
    const engine::MemoryGovernorSnapshot governor = memory();
    result += "; charged " + std::to_string(governor.chargedBytes >> 20) + " of " +
              std::to_string(governor.limitBytes >> 20) + "; requests:";
    for (const engine::ActiveRequestSnapshot &request : fixture.loop.statusSnapshot().activeRequests)
      result += " " + std::to_string(request.id) + ":" + std::to_string(int(request.phase)) + "/" +
                std::to_string(int(request.kvMode)) + "/" + std::to_string(request.promptProcessed) + "of" +
                std::to_string(request.promptTokens);
    return result + ")";
  }
  [[nodiscard]] engine::MemoryGovernorSnapshot memory() const { return fixture.governor->snapshot(); }
  [[nodiscard]] double now() const noexcept { return fixture.monotonic; }
  // The experts account as the governor reports it.
  [[nodiscard]] uint64_t expertsAccount() const {
    for (const engine::MemoryAccountSnapshot &account : memory().accounts)
      if (account.name == "experts")
        return account.currentBytes;
    return 0;
  }

  const Machine &machine;
  test::LoopFixture fixture;
  std::optional<test::FakeElasticPool> experts;
  std::vector<test::EngineHarness::Command> commands;
  std::map<uint64_t, double> submittedAt, startedAt;
  std::map<uint64_t, std::vector<double>> tokenTimes;
  std::set<uint64_t> ended;
  std::vector<std::string> errors;
  uint64_t starts = 0;
  protocol::peer::EventReader reader;
};

// A churn of three closed-loop workers (churn_ord.py), arriving in `order`
// 0.6 s apart, until `endMilliseconds`; each step observed. Returns the work
// done by then: prefill rows / R + output tokens / D.
struct Churn final {
  Churn(Serving &serving, std::array<uint32_t, 3> order, ChurnPrompts prompts = {})
      : serving(serving), prompts(prompts) {
    for (uint32_t rank = 0; rank < 3; ++rank)
      next[order[rank]] = serving.now() + 600.0 * rank;
  }
  void run(double endMilliseconds, const std::function<void()> &observe) {
    while (serving.now() < endMilliseconds) {
      for (uint32_t worker = 0; worker < 3; ++worker) {
        if (inFlight[worker] && !serving.active(*inFlight[worker])) {
          finished.push_back(*inFlight[worker]);
          inFlight[worker].reset();
          next[worker] = serving.now();
        }
        if (!inFlight[worker] && next[worker] <= serving.now()) {
          const uint32_t round = ++rounds[worker];
          const uint64_t id = 100 * (worker + 1) + round;
          std::vector<uint32_t> tokens;
          if (round % 4 == 0 && !last[worker].empty()) {
            tokens = last[worker];
            const std::vector<uint32_t> again = prompt(9000 + id, 8);
            tokens.insert(tokens.end(), again.begin(), again.end());
          } else {
            tokens = prompt(id, prompts.tokens(worker, round));
            last[worker] = tokens;
          }
          serving.submit(id, std::move(tokens), prompts.output);
          inFlight[worker] = id;
        }
      }
      const double nextArrival = std::min({next[0], next[1], next[2]});
      serving.run(
          [&] {
            if (serving.now() >= endMilliseconds)
              return true;
            for (uint32_t worker = 0; worker < 3; ++worker)
              if ((inFlight[worker] && !serving.active(*inFlight[worker])) ||
                  (!inFlight[worker] && next[worker] <= serving.now()))
                return true;
            return !inFlight[0] && !inFlight[1] && !inFlight[2] && serving.now() >= nextArrival;
          },
          observe, "the churn did not move");
      require(serving.errors.empty(), at(serving.machine) + "a churn request failed: " +
                                          (serving.errors.empty() ? std::string() : serving.errors.front()) +
                                          serving.holders());
    }
  }
  // Lets the requests in flight end, sending no more.
  void finish() {
    serving.run(
        [&] {
          return std::ranges::none_of(inFlight,
                                      [&](const std::optional<uint64_t> &id) { return id && serving.active(*id); });
        },
        {}, "the churn's last requests did not end");
  }
  [[nodiscard]] double work() const {
    const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
    return double(counts.scheduler.prefillRows) / serving.machine.rowsPerWork +
           double(serving.fixture.metrics.snapshot().decodeOutputTokens) / serving.machine.tokensPerWork;
  }

  Serving &serving;
  ChurnPrompts prompts;
  std::array<double, 3> next{};
  std::array<uint32_t, 3> rounds{};
  std::array<std::optional<uint64_t>, 3> inFlight{};
  std::array<std::vector<uint32_t>, 3> last{};
  std::vector<uint64_t> finished;
};

// The bound of a contended prefill unit as the run's own status reports it
// (amend/V.md A09, G4a): max(G, u / kPassOverheadShare), u the units term's
// price, by the price's p95 error.
double unitBound(const engine::SchedulerSnapshot &scheduler) {
  double turnaround = 0.0;
  for (const engine::CostTermSnapshot &term : scheduler.prefillEstimate.terms)
    if (term.quantity == "units")
      turnaround = term.millisecondsPerUnit;
  return std::max(scheduler.decodeGapTargetMilliseconds, turnaround / engine::kPassOverheadShare) *
         (1.0 + scheduler.prefillEstimate.relativeErrorP95);
}

// The longest a decode command of the profile takes: its widest at the
// store's floor, every lane reading spilled K/V.
double longestDecode(const test::TimingProfile &profile, double floorShare) {
  constexpr uint32_t width = model::ExecutionLimits::maximumBatchWidth;
  return test::decodeTimes(profile, width, width * model::ExecutionLimits::targetVerifyRows, floorShare, width)
      .wallMilliseconds;
}

// --- SIM-churn ------------------------------------------------------------------------------------------------

// SIM-churn (V.md §4.1, amend/V.md A09; HiSparse on, the Studio at 18G): the
// churn for 15 minutes beside a host episode: other applications leave the
// host a store chunk short of its spare reserve at 300 s, macOS warns from
// 300 to 420 s, and they give it back at 600 s.
// Every tick keeps I1 (the fixture); every decode gap stays within the run's
// unit bound and one decode command; no request waits for memory past the
// resource wait limit and none fails for capacity; the experts account
// shrinks only inside the episode, changes direction at most once per change
// of the host's level, and is back at its size before the episode within
// 60 s of its end (#645: a store that shrinks and never regrows, 32.9 against
// 48.9 tokens a second).
void testChurnRegrowsAfterTheHostEpisode() {
  const Machine &machine = kStudio18G;
  constexpr double episode = 300'000.0, warningEnd = 420'000.0, release = 600'000.0, end = 900'000.0;
  // The episode leaves the host a store chunk short of its spare reserve as
  // the engine started, beside its margin: the other applications take the
  // rest.
  const uint64_t started = machine.fixedBytes + uint64_t{machine.store.cap} * machine.store.unitBytes;
  const uint64_t episodeOther = machine.hostBytes - started - machine.hostSpareReserveBytes + machine.store.unitBytes;
  Serving serving(machine, {{episode, episodeOther, engine::MemoryPressure::Warning},
                            {warningEnd, episodeOther, engine::MemoryPressure::Normal},
                            {release, machine.otherBytes, engine::MemoryPressure::Normal}});
  Churn churn(serving, {2, 0, 1});
  uint64_t last = serving.experts->state().currentBytes, before = last;
  int direction = 0;
  std::map<int, uint32_t> turnsByLevel;  // per host level: 0 before, 1 warning, 2 short, 3 after
  std::optional<double> outsideShrink, backAt;
  bool episodeShrank = false;
  double oldestClaim = 0.0;
  churn.run(end, [&] {
    const double now = serving.now();
    const uint64_t current = serving.experts->state().currentBytes;
    const int level = now < episode ? 0 : now < warningEnd ? 1 : now < release ? 2 : 3;
    if (level == 0)
      before = current;
    if (current != last) {
      const int moved = current > last ? 1 : -1;
      if (moved < 0 && (now < episode || now >= release + 60'000.0) && !outsideShrink)
        outsideShrink = now;
      episodeShrank |= moved < 0 && level && level < 3;
      if (direction && moved != direction)
        ++turnsByLevel[level];
      direction = moved;
      last = current;
    }
    if (now >= release && !backAt && current >= before)
      backAt = now;
    oldestClaim = std::max(oldestClaim, serving.memory().claims.oldestMilliseconds);
  });
  const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
  const double gapLimit = unitBound(counts.scheduler) +
                          longestDecode(*machine.profile, double(machine.store.floor) / machine.store.whole);
  const double longestGap = serving.fixture.metrics.snapshot().decodeGap.maximumMilliseconds;
  require(churn.finished.size() >= 6, at(machine) + "the churn finished " + std::to_string(churn.finished.size()) +
                                          " requests in 15 minutes");
  require(longestGap <= gapLimit, at(machine) + "a decode gap of " + std::to_string(longestGap) +
                                      " ms passed the run's bound and a decode command, " + std::to_string(gapLimit) +
                                      " ms");
  require(oldestClaim < engine::kResourceWaitTimeoutMilliseconds && counts.failed == 0,
          at(machine) + "a request waited " + std::to_string(oldestClaim) + " ms for memory, or failed");
  require(!outsideShrink, at(machine) + "the store shrank outside the host's episode, at " +
                              std::to_string(outsideShrink.value_or(0.0) / 1000.0) + " s");
  for (const auto &[level, turns] : turnsByLevel)
    require(turns <= 1, at(machine) + "the store changed direction " + std::to_string(turns) +
                            " times at one host level (" + std::to_string(level) + ")");
  require(episodeShrank, at(machine) + "the host's episode took no store unit");
  require(backAt && *backAt <= release + 60'000.0,
          at(machine) + "the store was not back at its size before the episode within 60 s of its end (" +
              (backAt ? std::to_string((*backAt - release) / 1000.0) + " s" : std::string("never")) + ")");
  note(at(machine) + std::to_string(churn.finished.size()) + " requests, work " + std::to_string(churn.work()) +
       ", longest decode gap " + std::to_string(longestGap) + " ms of " + std::to_string(gapLimit) +
       ", store back " + std::to_string((*backAt - release) / 1000.0) + " s after the episode");
}

// --- SIM-churn-order ------------------------------------------------------------------------------------------

// SIM-churn-order (amend/V.md A100, AMENDMENTS X4; replaces SIM-arrival and
// SIM-two-long): the churn's arrivals in orders A (2,0,1) and B (0,1,2) on
// the Studio at 18G, its 59,043-token first request and round 2's
// 71,254/71,220 pair. (1) No start converts a running lane or shrinks the
// store: a Start claim takes Ordinary units only (no tick that starts a
// request converts a lane or gives a store unit). (2) A leader keeps leading
// to its prompt's end unless a lane of less expected remaining cost
// supersedes it or one overdue displaces it. (3) Two Spill lanes run prefill
// together only when the later has the less expected remaining cost or one is
// overdue, never the near-equal pair. (4) The bound cuts rows only of units
// already at their most passes. Run as the plan places the prompts, and with
// every request spilling (--kv-spill always), which makes the prompts Spill
// lanes whatever the placement decides.
void churnOrder(const Machine &machine) {
  const test::TimingProfile &profile = *machine.profile;
  const auto own = [&](uint32_t rows) {
    const std::array lanes{model::PrefillLane{.rows = rows}};
    return test::prefillTimes(profile, test::TimedWork(profile).prefillWork(lanes, 0, 1)).wallMilliseconds;
  };
  for (const auto &[label, order] : {std::pair{"A", std::array<uint32_t, 3>{2, 0, 1}},
                                     std::pair{"B", std::array<uint32_t, 3>{0, 1, 2}}}) {
    const std::string where = at(machine) + "order " + label + ": ";
    Serving serving(machine);
    Churn churn(serving, order);
    // At each step that started a request, ran a command or took a store
    // unit: what each request has left to prefill, and the counts.
    std::map<uint64_t, uint32_t> remaining, prompts;
    uint64_t converted = 0, shrinks = 0, starts = 0, boundCuts = 0;
    size_t seen = 0;
    std::optional<uint64_t> leader;
    std::string broken;
    // Waited the aging factor times its own cost (the scheduler's
    // kAgingSlowdown, the batch width less one, AMENDMENTS "K 2 -> 3").
    const auto overdue = [&](uint64_t id) {
      constexpr double aging = model::ExecutionLimits::maximumBatchWidth - 1;
      return serving.now() - serving.submittedAt.at(id) >= aging * own(remaining[id]);
    };
    churn.run(480'000.0, [&] {
      const bool shrank = serving.experts->shrinks > shrinks;
      if (!broken.empty() || (serving.starts == starts && seen == serving.commands.size() && !shrank))
        return;
      const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
      for (const engine::ActiveRequestSnapshot &request : counts.activeRequests) {
        remaining[request.id] = request.promptTokens - request.promptProcessed;
        prompts[request.id] = request.promptTokens;
      }
      if (serving.starts > starts && (counts.spill.converted > converted || shrank))
        broken = "a start converted a running lane or took a store unit";
      converted = counts.spill.converted;
      shrinks = serving.experts->shrinks;
      starts = serving.starts;
      for (; seen < serving.commands.size(); ++seen) {
        const test::EngineHarness::Command &command = serving.commands[seen];
        if (command.kind != WorkKind::Prefill || command.pass)
          continue;
        const uint64_t lead = command.lanes.front();
        if (broken.empty() && leader && *leader != lead && remaining[*leader] > 0 &&
            remaining[lead] >= remaining[*leader] && !overdue(lead))
          broken = "lane " + std::to_string(lead) + " took the lead from " + std::to_string(*leader) + " with " +
                   std::to_string(remaining[lead]) + " rows left against " + std::to_string(remaining[*leader]);
        leader = lead;
        if (broken.empty() && counts.scheduler.boundCuts > boundCuts && command.passes < profile.layers)
          broken = "the bound cut a unit in " + std::to_string(command.passes) + " passes";
        boundCuts = counts.scheduler.boundCuts;
      }
      // Spill lanes prefilling together: both part way through their prompts.
      std::vector<uint64_t> spilling;
      for (const engine::ActiveRequestSnapshot &request : counts.activeRequests)
        if (request.kvMode == engine::KvMode::Spill && request.phase == engine::Phase::Prefill &&
            request.promptProcessed > 0 && request.promptProcessed < request.promptTokens)
          spilling.push_back(request.id);
      if (broken.empty() && spilling.size() >= 2) {
        std::ranges::sort(spilling, {}, [&](uint64_t id) { return serving.submittedAt.at(id); });
        const uint64_t earlier = spilling[0], later = spilling[1];
        const bool nearEqual = std::max(prompts[earlier], prompts[later]) -
                                   std::min(prompts[earlier], prompts[later]) <
                               engine::KvCache::pageTokens * 8;
        if (nearEqual || (remaining[later] >= remaining[earlier] && !overdue(later) && !overdue(earlier)))
          broken = "Spill lanes " + std::to_string(earlier) + " and " + std::to_string(later) + " (prompts " +
                   std::to_string(prompts[earlier]) + " and " + std::to_string(prompts[later]) +
                   ") prefilled together with " + std::to_string(remaining[earlier]) + " and " +
                   std::to_string(remaining[later]) + " rows left";
      }
    });
    require(broken.empty(), where + broken);
    require(serving.errors.empty(), where + "a request failed: " + (serving.errors.empty() ? "" : serving.errors[0]));
    require(churn.finished.size() >= 3, where + "the churn finished " + std::to_string(churn.finished.size()) +
                                            " requests in 8 minutes");
    const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
    note(where + std::to_string(churn.finished.size()) + " requests, " + std::to_string(counts.spill.spill) +
         " started Spill, " + std::to_string(counts.spill.converted) + " converted, " +
         std::to_string(serving.experts->shrinks) + " store units given, " +
         std::to_string(counts.scheduler.passCommands) + " commands in passes, " +
         std::to_string(counts.scheduler.boundCuts) + " bound cuts");
    if (machine.spillAlways)
      require(counts.spill.spill >= 4, where + "only " + std::to_string(counts.spill.spill) + " requests ran Spill");
  }
}
void testChurnOrderKeepsLeadersAndSpillLanes() {
  Machine always = kStudio18G;
  always.name = "studio-18g-spill-always";
  always.spillAlways = true;
  churnOrder(kStudio18G);
  churnOrder(always);
}

// --- SIM-warning-M6 -------------------------------------------------------------------------------------------

// Ticks the idle loop until `done` or `milliseconds` pass; whether done.
bool idleUntil(Serving &serving, double milliseconds, const std::function<bool()> &done,
               const std::function<void()> &observe = {}) {
  const double end = serving.now() + milliseconds;
  serving.run([&] { return done() || serving.now() >= end; }, observe, "the idle loop did not move");
  return done();
}

// SIM-warning-M6 (amend/V.md A15/A31, X19): the M6 under macOS's warning
// from the start, the host's headroom below one store chunk (443 MB, fxL),
// the store at its floor after warm-up as measured. Growth under the warning
// displaces: the store regrows from cache memory, net zero for the host, a
// unit at most each control pass the host has settled from the last (cap -
// floor passes, the first settled too). With idle cached prompts to
// displace, the store reaches its cap within that; after a long Resident
// prompt ends, it is back at its size before the prompt within as long; and
// the host's footprint (FakeMemory's total) never rises while it regrows.
void testStoreRegrowsUnderPersistentWarning() {
  const Machine &machine = kM6;
  const uint64_t headroom = 443'000'000;
  const uint64_t free = machine.hostSpareReserveBytes + headroom;
  const uint64_t other =
      machine.hostBytes - free - (machine.fixedBytes + uint64_t{machine.store.floor} * machine.store.unitBytes);
  Serving serving(machine, test::persistentWarning(other), machine.store.floor);
  // Two finished prompts leave their KV cached and idle.
  serving.submit(1, prompt(1, 20'000), kTokensPerCommand);
  serving.submit(2, prompt(2, 20'000), kTokensPerCommand);
  serving.run([&] { return !serving.active(1) && !serving.active(2); }, {}, "the cached prompts did not end");
  const double allowed = (machine.store.cap - machine.store.floor + 1) * engine::kHostSettleMilliseconds;
  const auto regrows = [&](uint32_t target, std::string_view when) {
    uint64_t lowest = serving.fixture.memory->total();
    bool rose = false;
    const bool reached = idleUntil(serving, allowed, [&] { return serving.experts->units() >= target; }, [&] {
      const uint64_t total = serving.fixture.memory->total();
      rose |= total > lowest;
      lowest = std::min(lowest, total);
    });
    require(reached, at(machine) + std::string(when) + ": under macOS's warning the store stayed at " +
                         std::to_string(serving.experts->units()) + " chunks of " + std::to_string(target) +
                         " for " + std::to_string(allowed) + " ms");
    require(!rose, at(machine) + std::string(when) + ": the host's footprint rose while the store regrew");
  };
  regrows(machine.store.cap, "after warm-up");
  const uint32_t before = serving.experts->units();
  serving.submit(3, prompt(3, 45'056), kTokensPerCommand);
  serving.run([&] { return !serving.active(3); }, {}, "the long prompt did not end");
  regrows(before, "after the long prompt");
}

// --- SIM-low-host ---------------------------------------------------------------------------------------------

// SIM-low-host (amend/V.md A109, X19): an engine that starts short of host
// memory, its store at its floor, other applications leaving less than the
// store's cap until 60 s. Once the host frees, free room grows every unit it
// admits in one pass: the experts account reaches its cap within two
// settled control passes and the host's settling (3 kHostSettleMilliseconds)
// of the release. A first request arriving then reaches its first token
// within the same request's on a start at the cap and the store's refill
// reads (the units it lacked at the profile's read rate).
void testShortStartRegrowsWhenTheHostFrees() {
  const Machine &machine = k35BStudio18G;
  constexpr double release = 60'000.0;
  const uint64_t lacking = uint64_t{machine.store.cap - machine.store.floor} * machine.store.unitBytes;
  const auto firstTtft = [&](bool constrained) {
    std::vector<test::HostEvent> script;
    if (constrained) {
      // The host's spare memory a unit short of the cap's beyond the margin.
      const uint64_t spare = machine.hostBytes - machine.otherBytes - machine.fixedBytes -
                             uint64_t{machine.store.cap} * machine.store.unitBytes;
      script = test::lowHostAtStart(machine.otherBytes + spare - machine.hostSpareReserveBytes, machine.otherBytes,
                                    release);
    }
    Serving serving(machine, std::move(script), constrained ? machine.store.floor : machine.store.cap);
    idleUntil(serving, release - serving.now(), [] { return false; });
    if (constrained) {
      idleUntil(serving, 3 * engine::kHostSettleMilliseconds,
                [&] { return serving.experts->units() == machine.store.cap; });
      require(serving.experts->units() == machine.store.cap,
              at(machine) + "the store stayed at " + std::to_string(serving.experts->units()) + " chunks of " +
                  std::to_string(machine.store.cap) + " for two settled passes after the host freed");
    }
    serving.submit(1, prompt(1, 8192), 64);
    serving.run([&] { return !serving.active(1); }, {}, "the first request did not end");
    return serving.fixture.metrics.snapshot().ttftP50Milliseconds;
  };
  const double free = firstTtft(false), shortStart = firstTtft(true);
  const double refill = double(lacking) / machine.profile->weightBytesPerMillisecond;
  require(shortStart <= free + refill, at(machine) + "the first request after a short start took " +
                                           std::to_string(shortStart) + " ms to its first token against " +
                                           std::to_string(free) + " ms and the refill's " + std::to_string(refill));
  note(at(machine) + "first TTFT " + std::to_string(shortStart) + " ms after a short start, " + std::to_string(free) +
       " ms at the cap");
}

// --- SIM-value ------------------------------------------------------------------------------------------------

// What memory took first when both decoders' growth found no room: a lane's
// conversion to Spill or a store chunk; and when both decoders ended.
struct ValueRun final {
  enum class Action { None, Conversion, StoreUnit } action = Action::None;
  double endMilliseconds = 0.0;
  // A decoder failed: its run did the least work.
  std::optional<std::string> failure;
};

// Two prompts of `context` tokens, the first decoding while the second
// prefills: the moment both decode, what the engine holds then (with the
// reserves, the limit that leaves their growth no room) and the tokens the
// first has generated by then. Commands ask memory every 128 tokens (extents
// of 4 pages).
struct ValueSetup final {
  uint64_t limitBytes = 0;
  uint32_t generated = 0;
};
ValueSetup valueSetup(const Machine &machine, uint32_t context) {
  Machine roomy = machine;
  roomy.extentPages = 4;
  Serving serving(roomy, {}, std::nullopt, machine.hostBytes);
  serving.submit(1, prompt(1, context), 100'000);
  serving.submit(2, prompt(2, context), 100'000);
  ValueSetup setup;
  serving.run([&] { return setup.limitBytes != 0; },
              [&] {
                if (!serving.ttft(2))
                  return;
                const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
                setup.limitBytes = serving.fixture.memory->total() + machine.untrackedBytes;
                setup.generated = counts.activeRequests.front().generatedTokens;
              },
              "the two prompts did not both decode");
  require(serving.errors.empty(), at(machine) + "a sizing request failed: " +
                                      (serving.errors.empty() ? std::string() : serving.errors.front()));
  return setup;
}

// The two decoders with `output` tokens left each once both decode, on
// `machine` with Spill's decode costing `spillMilliseconds` a token, the
// store at its cap and no free room: free, the store pinned at its cap, or
// Spill unavailable.
enum class Counterfactual { Free, StorePinned, NoSpill };
ValueRun runValue(const Machine &base, double spillMilliseconds, uint32_t context, uint32_t output,
                  Counterfactual which, const ValueSetup &setup) {
  test::TimingProfile profile = *base.profile;
  profile.spillDecodeMilliseconds = spillMilliseconds * kTokensPerCommand;
  Machine machine = base;
  machine.profile = &profile;
  machine.extentPages = 4;
  if (which == Counterfactual::StorePinned)
    machine.store.floor = machine.store.cap;
  if (which == Counterfactual::NoSpill)
    machine.spill.reset();
  Serving serving(machine, {}, std::nullopt, setup.limitBytes);
  serving.submit(1, prompt(1, context), setup.generated + output);
  serving.submit(2, prompt(2, context), output);
  ValueRun result;
  serving.run([&] { return !serving.active(1) && !serving.active(2); },
              [&] {
                if (result.action != ValueRun::Action::None || !serving.ttft(2))
                  return;
                if (serving.experts->shrinks)
                  result.action = ValueRun::Action::StoreUnit;
                else if (serving.fixture.loop.statusSnapshot().spill.converted)
                  result.action = ValueRun::Action::Conversion;
              },
              "the decoders did not end");
  if (!serving.errors.empty())
    result.failure = serving.errors.front();
  result.endMilliseconds = result.failure ? std::numeric_limits<double>::infinity() : serving.now();
  return result;
}

// SIM-value (amend/V.md §3.4; owner 20:45 OQ1: the value outcome): on the
// M6 and the 36 GB Studio, Spill's decode costing the D3 median and worst
// (M6 1.0 and 2.7 ms a token, Studio 1.7 and 3.8), the decoders with 300 or
// 1500 tokens left over 8K or 45K of context, and 128K, where amend/V.md
// §3.3 puts the M6's choice: what the free run takes first
// (a conversion or a store chunk) is what the counterfactual that does
// better takes (the store pinned, so the lane converts; Spill unavailable,
// so the store gives a chunk), unless the two end within a decode command
// of each other. On the 36 GB Studio this follows AMENDMENTS OQ1's answer:
// the value outcome, no spill-capability order.
void testValueChoiceFollowsTheCalibratedCost() {
  struct Point final {
    const Machine *machine;
    std::array<double, 2> spill;
  };
  for (const Point &point : {Point{&kM6, {1.0, 2.7}}, Point{&kStudio36, {1.7, 3.8}}}) {
    const Machine &machine = *point.machine;
    for (const uint32_t context : {8192u, 45'056u, 131'072u}) {
      const ValueSetup setup = valueSetup(machine, context);
      for (const double spill : point.spill)
        for (const uint32_t output : {304u, 1504u}) {
          const ValueRun free = runValue(machine, spill, context, output, Counterfactual::Free, setup);
          const ValueRun pinned = runValue(machine, spill, context, output, Counterfactual::StorePinned, setup);
          const ValueRun noSpill = runValue(machine, spill, context, output, Counterfactual::NoSpill, setup);
          const std::string where = at(machine) + std::to_string(context) + " tokens, " + std::to_string(output) +
                                    " left, Spill +" + std::to_string(spill) + " ms a token: ";
          require(!free.failure, where + "a decoder failed: " + free.failure.value_or(""));
          require(free.action != ValueRun::Action::None, where + "the decoders' growth took nothing");
          const double command = longestDecode(*machine.profile, double(machine.store.floor) / machine.store.whole);
          note(where + (free.action == ValueRun::Action::Conversion ? "converted" : "gave a store chunk") +
               "; ends free " + std::to_string(free.endMilliseconds / 1000.0) + " s, converted " +
               std::to_string(pinned.endMilliseconds / 1000.0) + " s, store chunk " +
               std::to_string(noSpill.endMilliseconds / 1000.0) + " s");
          if (std::abs(pinned.endMilliseconds - noSpill.endMilliseconds) <= command)
            continue;
          const ValueRun::Action better = pinned.endMilliseconds < noSpill.endMilliseconds
                                              ? ValueRun::Action::Conversion
                                              : ValueRun::Action::StoreUnit;
          require(free.action == better,
                  where + "memory took " +
                      (free.action == ValueRun::Action::Conversion ? "a conversion" : "a store chunk") +
                      " where the other ends first (converted " + std::to_string(pinned.endMilliseconds / 1000.0) +
                      " s, store chunk " + std::to_string(noSpill.endMilliseconds / 1000.0) + " s)");
        }
    }
  }
}

// --- SIM-35b-churn --------------------------------------------------------------------------------------------

// SIM-35b-churn (amend/V.md §3.4; HiSparse off: no conversion exists):
// 35B at 18G on the Studio, the churn at churn_ord.py's default growth
// (BASE 1100, STEP 900 words: 16,186 tokens and 7,281 a round at 35B's
// tokens a word). Against the counterfactual whose store never grows back
// into free room, the free run does at least as much work, reads the KV tier
// for no more than a store chunk's refill more, and takes back no chunk it
// regrew within kValueHorizonSeconds (the cool-down's 8 shrinks for 7
// regrows, A30).
void testNoSpillTargetChurnFollowsValue() {
  const Machine &machine = k35BStudio18G;
  const ChurnPrompts prompts{.first = 16'186, .step = 7'281, .offsets = {-17, 0, 105}};
  struct Result final {
    double work = 0.0;
    uint64_t tierReads = 0, shrinks = 0, grows = 0;
    uint32_t retaken = 0;
    bool regrown = false;
  };
  const auto churnOf = [&](bool regrowth) {
    Serving serving(machine);
    // The counterfactual's cap follows the store down: it grows no further
    // than it stands.
    serving.experts->regrows = regrowth;
    Churn churn(serving, {2, 0, 1}, prompts);
    std::vector<double> grownAt;
    uint64_t grows = serving.experts->grows, shrinks = serving.experts->shrinks;
    Result result;
    churn.run(600'000.0, [&] {
      if (serving.experts->grows > grows)
        grownAt.push_back(serving.now());
      if (serving.experts->shrinks > shrinks && !grownAt.empty() &&
          serving.now() - grownAt.back() < 1000.0 * engine::kValueHorizonSeconds)
        ++result.retaken;
      grows = serving.experts->grows;
      shrinks = serving.experts->shrinks;
    });
    result.work = churn.work();
    result.tierReads = uint64_t{serving.fixture.tier->restores} * machine.pageBytes;
    result.shrinks = serving.experts->shrinks;
    result.grows = serving.experts->grows;
    // The churn's last requests end; idle, the store takes back what it gave
    // (free room or idle cache) within two settled passes and the settling.
    churn.finish();
    result.regrown = idleUntil(serving, 3 * engine::kHostSettleMilliseconds,
                               [&] { return serving.experts->units() == machine.store.cap; });
    return result;
  };
  const Result free = churnOf(true), counterfactual = churnOf(false);
  require(free.work >= counterfactual.work,
          at(machine) + "regrowth did " + std::to_string(free.work) + " work against " +
              std::to_string(counterfactual.work) + " without it");
  require(free.tierReads <= counterfactual.tierReads + machine.store.unitBytes,
          at(machine) + "regrowth read " + std::to_string(free.tierReads) + " bytes of the KV tier against " +
              std::to_string(counterfactual.tierReads));
  require(free.retaken == 0, at(machine) + std::to_string(free.retaken) +
                                 " store chunks were taken back within the value horizon of their regrowth");
  require(free.regrown, at(machine) + "idle after the churn, the store stayed below its cap");
  note(at(machine) + "work " + std::to_string(free.work) + " against " + std::to_string(counterfactual.work) +
       ", tier reads " + std::to_string(free.tierReads >> 20) + " against " +
       std::to_string(counterfactual.tierReads >> 20) + " MiB, store units given/taken back " +
       std::to_string(free.shrinks) + "/" + std::to_string(free.grows) + " against " +
       std::to_string(counterfactual.shrinks) + "/" + std::to_string(counterfactual.grows));
}

// --- SIM-turns ------------------------------------------------------------------------------------------------

// Agent turns on the harness: the profile's commands on its clock, the
// model's limits (commandRows rows in up to `passes` passes), its price
// seeded as a start seeds it, its first token at the prompt's end.
struct Turns final {
  Turns(const test::TimingProfile &profile, uint32_t commandRows, uint32_t passes)
      : harness({.pages = 8192,
                 .engine = {.maxContext = 262'144},
                 .timing = &profile,
                 .shape = [=](test::Executor &model) {
                   model.groupRows = commandRows;
                   model.passes = passes;
                 }}) {
    harness.model.decodeTokens = kTokensPerCommand;
    harness.model.decodeFinishes = false;
    harness.model.firstTokens = true;
    test::seedAsStartup(harness.engine, harness.model, profile);
  }
  void submit(uint64_t id, std::vector<uint32_t> tokens, uint32_t output) {
    engine::EngineRequest request = harness.request(id, std::move(tokens));
    request.maxNewTokens = output;
    request.deadlineMilliseconds = harness.now() + 3'600'000.0;
    submitted[id] = harness.now();
    harness.engine.submit(std::move(request));
  }
  [[nodiscard]] std::optional<double> ttft(uint64_t id) const {
    const auto found = harness.events.tokenTimes.find(id);
    if (found == harness.events.tokenTimes.end() || found->second.empty())
      return std::nullopt;
    return found->second.front() - submitted.at(id);
  }
  // The prefill units that ran while `decoder` decoded, and the decoder's
  // gaps, as the harness's clock saw them.
  [[nodiscard]] std::vector<double> contendedUnits(uint64_t decoder) const {
    const std::vector<double> &tokens = harness.events.tokenTimes.at(decoder);
    std::vector<double> walls;
    for (const test::EngineHarness::Command &command : harness.commands())
      if (command.kind == WorkKind::Prefill && command.startMilliseconds > tokens.front() &&
          command.startMilliseconds < tokens.back())
        walls.push_back(command.wallMilliseconds);
    return walls;
  }

  test::EngineHarness harness;
  std::map<uint64_t, double> submitted;
};

// Runs N turns of 300 rows past the shared 8K prefix arriving `gap` apart,
// beside the decoder, until each has its first token; their TTFTs.
std::vector<double> turnsTogether(Turns &turns, uint64_t firstId, uint32_t count, double gap,
                                  const std::vector<uint32_t> &prefix) {
  std::vector<double> ttft;
  for (uint32_t index = 0; index < count; ++index) {
    if (index)
      turns.harness.runUntil([&] { return turns.harness.now() >= turns.submitted.at(firstId) + gap * index; },
                             "the next turn's arrival did not come");
    std::vector<uint32_t> tokens = prefix;
    const std::vector<uint32_t> suffix = prompt(firstId + index, 300);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    turns.submit(firstId + index, std::move(tokens), kTokensPerCommand);
  }
  turns.harness.runUntil(
      [&] {
        for (uint32_t index = 0; index < count; ++index)
          if (!turns.ttft(firstId + index))
            return false;
        return true;
      },
      "the turns did not reach their first tokens");
  for (uint32_t index = 0; index < count; ++index)
    ttft.push_back(*turns.ttft(firstId + index));
  return ttft;
}

// SIM-turns (V.md §4.1, amend/V.md A09, A74) on the M6's profile: beside one
// decoder, four 300-row turns 2 s apart past an 8K cached prefix: the
// contended prefill units' p95 within the run's unit bound by its price's
// p95 error (G4a), the decoder's gaps within that and a decode command.
// Then, as the TTFT bench runs (no decoder, A74): N = 1, 2 and 4 turns
// together, the worst of four within 1.71 times one turn's (the pass tree's
// N=4 against N=1, 5247 / 3065 ms); and 2 and 4 arriving 0.5, 1 and 1.5 s
// apart, each within its own command, the one in flight and one of all of
// them.
void testTurnsSettleAndShareACommand() {
  const test::TimingProfile &profile = test::kM6FlashNextQ2;
  Turns turns(profile, 8192, profile.layers);
  const std::vector<uint32_t> prefix = prompt(1, 8192);
  turns.submit(1, prefix, kTokensPerCommand);
  turns.harness.runUntil([&] { return turns.ttft(1).has_value(); }, "the prefix did not prefill");
  turns.harness.runUntilIdle();
  turns.submit(2, prompt(2, 2048), 200'000);
  turns.harness.runUntil([&] { return turns.ttft(2).has_value(); }, "the decoder did not start");
  uint64_t id = 10;
  for (uint32_t turn = 0; turn < 4; ++turn, ++id) {
    static_cast<void>(turnsTogether(turns, id, 1, 0.0, prefix));
    turns.harness.runUntil([&, start = turns.harness.now()] { return turns.harness.now() >= start + 2000.0; },
                           "the turns' spacing did not pass");
  }
  const engine::SchedulerSnapshot scheduler = turns.harness.engine.snapshot().scheduler;
  const double bound = unitBound(scheduler);
  std::vector<double> units = turns.contendedUnits(2);
  std::ranges::sort(units);
  require(units.size() >= 20, "only " + std::to_string(units.size()) + " prefill units ran beside the decoder");
  const double p95 = units[(units.size() * 95 + 99) / 100 - 1];
  double turnaround = 0.0;
  for (const engine::CostTermSnapshot &term : scheduler.prefillEstimate.terms)
    if (term.quantity == "units")
      turnaround = term.millisecondsPerUnit;
  require(p95 <= bound, "the contended prefill units' p95, " + std::to_string(p95) + " ms of " +
                            std::to_string(units.size()) + " (longest " + std::to_string(units.back()) +
                            "), passed the bound's " + std::to_string(bound) + " ms (G " +
                            std::to_string(scheduler.decodeGapTargetMilliseconds) + ", turnaround " +
                            std::to_string(turnaround) + " ms, p95 error " +
                            std::to_string(scheduler.prefillEstimate.relativeErrorP95) + ")");
  const double command = longestDecode(profile, profile.measuredResidentShare);
  require(turns.harness.longestDecodeGap(2) <= bound + command,
          "the decoder's longest gap, " + std::to_string(turns.harness.longestDecodeGap(2)) + " ms, passed " +
              std::to_string(bound + command) + " ms");
  turns.harness.engine.cancel(2);
  turns.harness.runUntilIdle();
  const double alone = turnsTogether(turns, id, 1, 0.0, prefix)[0];
  id += 1;
  turns.harness.runUntilIdle();
  static_cast<void>(turnsTogether(turns, id, 2, 0.0, prefix));
  id += 2;
  turns.harness.runUntilIdle();
  const size_t before = turns.harness.commands().size();
  const std::vector<double> four = turnsTogether(turns, id, 4, 0.0, prefix);
  id += 4;
  const double worstFour = *std::ranges::max_element(four);
  std::string ran;
  for (size_t index = before; index < turns.harness.commands().size(); ++index) {
    const test::EngineHarness::Command &unit = turns.harness.commands()[index];
    if (unit.kind == WorkKind::Prefill)
      ran += " " + std::to_string(unit.lanes.size()) + "x" + std::to_string(unit.rows.front()) + "@" +
             std::to_string(unit.pass) + "/" + std::to_string(unit.passes) + ":" +
             std::to_string(int(unit.wallMilliseconds));
  }
  require(worstFour <= 1.71 * alone, "four turns together: worst TTFT " + std::to_string(worstFour) +
                                         " ms against one turn's " + std::to_string(alone) +
                                         " ms; prefill units (lanes x rows @ pass/passes: ms):" + ran);
  test::TimedWork work(profile);
  for (const uint32_t count : {2u, 4u})
    for (const double gap : {500.0, 1000.0, 1500.0}) {
      turns.harness.runUntilIdle();
      std::vector<model::PrefillLane> lanes(count, model::PrefillLane{8192, 300, 0});
      const double whole = test::prefillTimes(profile, work.prefillWork(lanes, 0, 1)).wallMilliseconds;
      const std::vector<double> apart = turnsTogether(turns, id, count, gap, prefix);
      id += count;
      for (size_t index = 0; index < apart.size(); ++index)
        require(apart[index] <= 1.25 * (alone + whole),
                std::to_string(count) + " turns " + std::to_string(int(gap)) + " ms apart: turn " +
                    std::to_string(index + 1) + " took " + std::to_string(apart[index]) + " ms against " +
                    std::to_string(1.25 * (alone + whole)));
    }
}

// --- SIM-turns-dense and SIM-long-prompt ----------------------------------------------------------------------

// N turns of 300 rows past a 2048-token prefix, arriving together beside
// one decoder, on `profile` with up to `passes` passes a command; their
// worst TTFT and the work the run did by the last first token: prefill rows
// and decode tokens a second.
struct DenseRun final {
  double worstTtft = 0.0, rowsPerSecond = 0.0, tokensPerSecond = 0.0;
};
DenseRun denseTurns(const test::TimingProfile &profile, uint32_t passes, uint32_t count) {
  Turns turns(profile, model::ExecutionLimits::prefillTokenBudget, passes);
  const std::vector<uint32_t> prefix = prompt(1, 2048);
  turns.submit(1, prefix, kTokensPerCommand);
  turns.harness.runUntilIdle();
  turns.submit(2, prompt(2, 1024), 200'000);
  turns.harness.runUntil([&] { return turns.ttft(2).has_value(); }, "the decoder did not start");
  const double start = turns.harness.now();
  const uint64_t rows = turns.harness.engine.snapshot().scheduler.prefillRows;
  const size_t tokens = turns.harness.events.tokenTimes.at(2).size();
  const std::vector<double> ttft = turnsTogether(turns, 10, count, 0.0, prefix);
  const double seconds = (turns.harness.now() - start) / 1000.0;
  return {.worstTtft = *std::ranges::max_element(ttft),
          .rowsPerSecond = double(turns.harness.engine.snapshot().scheduler.prefillRows - rows) / seconds,
          .tokensPerSecond =
              double(turns.harness.events.tokenTimes.at(2).size() - tokens) * kTokensPerCommand / seconds};
}

// SIM-turns-dense (DESIGN §7.11; behaviour changes 1 and 7, G10): on the
// dense 27B and on Flint, N = 1, 2, 4 turns together beside a decoder with
// a pass a layer allowed, against the base's policy on these targets (one
// pass: a contended command halves instead). Passes add at most a tenth of
// a command (kPassOverheadShare): each worst TTFT within 1/0.9 of the
// base policy's, and prefill rows and decode tokens a second at least 0.9
// of its.
void testDenseTurnsKeepTtftAndThroughput() {
  for (const test::TimingProfile *profile : {&test::kDense27B, &test::kFlint})
    for (const uint32_t count : {1u, 2u, 4u}) {
      const DenseRun passes = denseTurns(*profile, profile->layers, count);
      const DenseRun base = denseTurns(*profile, 1, count);
      const std::string where = std::string(profile->name) + ", " + std::to_string(count) + " turns: ";
      require(passes.worstTtft <= base.worstTtft / (1.0 - engine::kPassOverheadShare),
              where + "worst TTFT " + std::to_string(passes.worstTtft) + " ms in passes against " +
                  std::to_string(base.worstTtft) + " ms whole");
      require(passes.rowsPerSecond >= (1.0 - engine::kPassOverheadShare) * base.rowsPerSecond &&
                  passes.tokensPerSecond >= (1.0 - engine::kPassOverheadShare) * base.tokensPerSecond,
              where + "throughput in passes " + std::to_string(passes.rowsPerSecond) + " rows/s and " +
                  std::to_string(passes.tokensPerSecond) + " tokens/s against " + std::to_string(base.rowsPerSecond) +
                  " and " + std::to_string(base.tokensPerSecond));
    }
}

// SIM-long-prompt (V.md §4.1) on the dense 27B: a 64K prompt beside two
// decoders. Every decoder gap within the run's unit bound and a decode
// command; the prompt's prefill, its decode share taken out, at least 0.9
// (1 - kPassOverheadShare) of its rate alone.
void testLongPromptBesideTwoDecoders() {
  const test::TimingProfile &profile = test::kDense27B;
  std::string shape;
  const auto prefillOf = [&](bool decoders) {
    Turns turns(profile, model::ExecutionLimits::prefillTokenBudget, profile.layers);
    if (decoders) {
      turns.submit(1, prompt(1, 1024), 60'000);
      turns.submit(2, prompt(2, 1024), 60'000);
      turns.harness.runUntil([&] { return turns.ttft(1) && turns.ttft(2); }, "the decoders did not start");
    }
    const size_t first = turns.harness.commands().size();
    turns.submit(3, prompt(3, 65'536), kTokensPerCommand);
    turns.harness.runUntil([&] { return turns.ttft(3).has_value(); }, "the long prompt did not prefill");
    double decode = 0.0;
    for (size_t index = first; index < turns.harness.commands().size(); ++index)
      if (turns.harness.commands()[index].kind == WorkKind::Decode)
        decode += turns.harness.commands()[index].wallMilliseconds;
    if (decoders) {
      const engine::SchedulerSnapshot scheduler = turns.harness.engine.snapshot().scheduler;
      for (const engine::CostTermSnapshot &term : scheduler.prefillEstimate.terms)
        if (term.quantity == "units")
          shape = "; the price's turnaround " + std::to_string(term.millisecondsPerUnit) + " ms, " +
                  std::to_string(scheduler.prefillUnits) + " units of " + std::to_string(scheduler.passCommands) +
                  " commands in passes, " + std::to_string(scheduler.boundCuts) + " cut";
      const double bound = unitBound(scheduler);
      const double command = longestDecode(profile, 1.0);
      for (const uint64_t decoder : {1u, 2u})
        require(turns.harness.longestDecodeGap(decoder) <= bound + command,
                "a decoder's gap beside the long prompt, " + std::to_string(turns.harness.longestDecodeGap(decoder)) +
                    " ms, passed " + std::to_string(bound + command) + " ms");
    }
    return *turns.ttft(3) - decode;
  };
  const double alone = prefillOf(false), opposed = prefillOf(true);
  require(opposed <= alone / (1.0 - engine::kPassOverheadShare),
          "the long prompt's prefill beside two decoders took " + std::to_string(opposed) +
              " ms of its own against " + std::to_string(alone) + " ms alone" + shape);
}

// --- Placement once a candidate a state ------------------------------------------------------------------------

// DESIGN §4.4, §6.11: each candidate's placement is found once per admission
// state (Request::placement): over the Studio's churn, no candidate asks the
// model's spilled-read shortfall more than once a kind (prefill, decode) in
// one admission state, and Spill-capable candidates were placed by it.
void testPlacementOncePerCandidatePerState() {
  Serving serving(kStudio18G);
  // The admission state changes with every admission or refusal, as the
  // harness counts it (EngineHarness::admissionState).
  serving.fixture.executor.admissionState = [&] {
    return serving.fixture.memory->changes().size() + serving.memory().deniedReservations;
  };
  Churn churn(serving, {2, 0, 1});
  churn.run(240'000.0, {});
  uint32_t most = 0;
  for (const auto &[key, calls] : serving.fixture.executor.shortfallQueries)
    most = std::max(most, calls);
  require(!serving.fixture.executor.shortfallQueries.empty(),
          "no candidate's placement asked the model's spilled-read shortfall");
  require(most <= 2, "a candidate asked the spilled-read shortfall " + std::to_string(most) +
                         " times in one admission state");
}

} // namespace

std::span<const test::TestCase> simTests() {
  static constexpr test::TestCase kTests[] = {
      {"testTurnsSettleAndShareACommand", testTurnsSettleAndShareACommand},
      {"testDenseTurnsKeepTtftAndThroughput", testDenseTurnsKeepTtftAndThroughput},
      {"testLongPromptBesideTwoDecoders", testLongPromptBesideTwoDecoders},
      {"testChurnRegrowsAfterTheHostEpisode", testChurnRegrowsAfterTheHostEpisode},
      {"testChurnOrderKeepsLeadersAndSpillLanes", testChurnOrderKeepsLeadersAndSpillLanes},
      {"testStoreRegrowsUnderPersistentWarning", testStoreRegrowsUnderPersistentWarning},
      {"testShortStartRegrowsWhenTheHostFrees", testShortStartRegrowsWhenTheHostFrees},
      {"testValueChoiceFollowsTheCalibratedCost", testValueChoiceFollowsTheCalibratedCost},
      {"testNoSpillTargetChurnFollowsValue", testNoSpillTargetChurnFollowsValue},
      {"testPlacementOncePerCandidatePerState", testPlacementOncePerCandidatePerState},
  };
  return kTests;
}

int main(int argc, char **argv) {
  const test::TestGroup groups[] = {{"sim", simTests()}};
  return test::runTests("engine-sim", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
