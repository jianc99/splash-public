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
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
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

// A scenario's checks, each judged on its own so that one that fails leaves
// the others' verdicts known: the scenario fails at its end with every check
// that failed, one a line.
struct Checks final {
  void expect(bool holds, const std::string &message) {
    if (!holds)
      failures.push_back(message);
  }
  void verdict() const {
    std::string message;
    for (const std::string &failure : failures)
      message += (message.empty() ? "" : "\n  ") + failure;
    require(failures.empty(), message);
  }
  std::vector<std::string> failures;
};

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
    // The ledger's line per decision: each claim's class, request and what
    // it took (MemoryLedger::log).
    ledger.setDecisionLog([this](std::string_view line) { decisions.emplace_back(line); });
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
        commands.back().positions.push_back(item.promptOffset);
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
  std::vector<std::string> errors, decisions;
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

// One decode command of the scenario's own decoders on the profile (the
// "one decode command" of amend/V.md A09): `decoders` lanes verifying their
// rows at the experts' `share`, `spillLanes` of them reading spilled K/V.
double decodeCommand(const test::TimingProfile &profile, uint32_t decoders, double share, uint32_t spillLanes) {
  return test::decodeTimes(profile, decoders, decoders * model::ExecutionLimits::targetVerifyRows, share, spillLanes)
      .wallMilliseconds;
}

// The prefill estimate's price of `work`, from the rates the scheduler
// reports (CostModel::price over its snapshot): each term's rate times the
// work's quantity, the read terms scaled by their realized share.
double estimatedPrice(const engine::CostEstimateSnapshot &estimate, const model::CommandWork &work) {
  const engine::CostVector quantities = engine::costTerms(work);
  double price = 0.0;
  for (size_t term = 0; term < engine::kCostTerms; ++term)
    for (const engine::CostTermSnapshot &reported : estimate.terms)
      if (reported.quantity == engine::costTermName(static_cast<engine::CostTerm>(term)))
        price += reported.millisecondsPerUnit * reported.realizedPerExpected * quantities[term];
  return std::max(price, 0.0);
}

// --- SIM-churn ------------------------------------------------------------------------------------------------

// SIM-churn (V.md §4.1, amend/V.md A09; HiSparse on, the Studio at 18G): the
// churn for 15 minutes beside a host episode: other applications leave the
// host a store chunk short of its spare reserve at 300 s, macOS warns from
// 300 to 420 s, and they give it back at 600 s.
// Every tick keeps I1, I3, I4 and I5 (LoopFixture::run); every decode gap
// stays within the run's unit bound and one decode command of the churn's
// three lanes, each reading spilled K/V at the store's floor; no request
// waits for memory past the resource wait limit and none fails for
// capacity. Inside the episode (to 60 s past its end) the experts account
// changes direction at most once per level of the host, and it is back at
// its size before the episode within 60 s of its end (#645: a store that
// shrinks and never regrows, 32.9 against 48.9 tokens a second). Outside
// it, value may take a store unit for a long prompt's prefill (brief §2.12,
// DESIGN §0 19:45; which one is SIM-value's question, on calibrated
// counterfactuals), and what it takes comes back right after: once no
// request that prefilled when it went still prefills, the store is back at
// its size before within two control passes after the host's settling (the
// free-room bound, AMENDMENTS X19).
void testChurnRegrowsAfterTheHostEpisode() {
  const Machine &machine = kStudio18G;
  constexpr double episode = 300'000.0, warningEnd = 420'000.0, release = 600'000.0, end = 900'000.0;
  constexpr double episodeEnd = release + 60'000.0;
  // The episode leaves the host a store chunk short of its spare reserve as
  // the engine started, beside its margin: the other applications take the
  // rest.
  const uint64_t started = machine.fixedBytes + uint64_t{machine.store.cap} * machine.store.unitBytes;
  const uint64_t episodeOther = machine.hostBytes - started - machine.hostSpareReserveBytes + machine.store.unitBytes;
  Serving serving(machine, {{episode, episodeOther, engine::MemoryPressure::Warning},
                            {warningEnd, episodeOther, engine::MemoryPressure::Normal},
                            {release, machine.otherBytes, engine::MemoryPressure::Normal}});
  Churn churn(serving, {2, 0, 1});
  // A unit given outside the episode: when, the store's bytes before, the
  // requests that prefilled then, when the last of them stopped, and the
  // control passes since the host settled after that.
  struct Given final {
    double at = 0.0;
    uint64_t bytesBefore = 0;
    std::set<uint64_t> prefilling;
    std::optional<double> claimEnd;
    uint32_t passes = 0;
  };
  std::vector<Given> given;
  uint64_t last = serving.experts->state().currentBytes, before = last;
  int direction = 0;
  std::map<int, uint32_t> turnsByLevel;  // per level of the host: 1 warning, 2 short, 3 the minute after
  std::optional<double> backAt;
  std::string notBack;
  uint32_t givenOutside = 0;
  bool episodeShrank = false;
  double oldestClaim = 0.0;
  const auto prefilling = [&] {
    std::set<uint64_t> ids;
    for (const engine::ActiveRequestSnapshot &request : serving.fixture.loop.statusSnapshot().activeRequests)
      if (request.phase == engine::Phase::Prefill)
        ids.insert(request.id);
    return ids;
  };
  churn.run(end, [&] {
    const double now = serving.now();
    const uint64_t current = serving.experts->state().currentBytes;
    const bool inside = now >= episode && now < episodeEnd;
    const int level = now < warningEnd ? 1 : now < release ? 2 : 3;
    if (now < episode) {
      before = current;
      for (const Given &unit : given)
        before = std::max(before, unit.bytesBefore);
    }
    if (current != last) {
      const int moved = current > last ? 1 : -1;
      if (moved < 0 && !inside) {
        given.push_back({.at = now, .bytesBefore = last, .prefilling = prefilling()});
        ++givenOutside;
      }
      episodeShrank |= moved < 0 && now >= episode && now < release;
      if (inside && direction && moved != direction)
        ++turnsByLevel[level];
      direction = moved;
      last = current;
    }
    // What went outside the episode comes back, unless the episode begins
    // before its bound: the host's level decides then.
    const auto settled = [&](const Given &unit) {
      return current >= unit.bytesBefore || (unit.at < episode && now >= episode);
    };
    std::erase_if(given, settled);
    for (Given &unit : given) {
      if (!unit.claimEnd) {
        const std::set<uint64_t> still = prefilling();
        if (std::ranges::none_of(unit.prefilling, [&](uint64_t id) { return still.contains(id); }))
          unit.claimEnd = now;
      }
      if (unit.claimEnd && now >= *unit.claimEnd + engine::kHostSettleMilliseconds &&
          !serving.fixture.loop.commandInFlight() && ++unit.passes > 2 && notBack.empty())
        notBack = "a store unit given at " + std::to_string(unit.at / 1000.0) +
                  " s, outside the host's episode, was not back two control passes after its claim ended at " +
                  std::to_string(*unit.claimEnd / 1000.0) + " s and the host settled";
    }
    if (now >= release && !backAt && current >= before)
      backAt = now;
    oldestClaim = std::max(oldestClaim, serving.memory().claims.oldestMilliseconds);
  });
  const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
  const double gapLimit = unitBound(counts.scheduler) +
                          decodeCommand(*machine.profile, 3, double(machine.store.floor) / machine.store.whole, 3);
  const double longestGap = serving.fixture.metrics.snapshot().decodeGap.maximumMilliseconds;
  require(churn.finished.size() >= 6, at(machine) + "the churn finished " + std::to_string(churn.finished.size()) +
                                          " requests in 15 minutes");
  require(longestGap <= gapLimit, at(machine) + "a decode gap of " + std::to_string(longestGap) +
                                      " ms passed the run's bound and a decode command, " + std::to_string(gapLimit) +
                                      " ms");
  require(oldestClaim < engine::kResourceWaitTimeoutMilliseconds && counts.failed == 0,
          at(machine) + "a request waited " + std::to_string(oldestClaim) + " ms for memory, or failed");
  require(notBack.empty(), at(machine) + notBack + serving.holders());
  for (const auto &[level, turns] : turnsByLevel)
    require(turns <= 1, at(machine) + "the store changed direction " + std::to_string(turns) +
                            " times at one host level (" + std::to_string(level) + ")");
  require(episodeShrank, at(machine) + "the host's episode took no store unit");
  require(backAt && *backAt <= release + 60'000.0,
          at(machine) + "the store was not back at its size before the episode within 60 s of its end (" +
              (backAt ? std::to_string((*backAt - release) / 1000.0) + " s" : std::string("never")) + ")");
  note(at(machine) + std::to_string(churn.finished.size()) + " requests, work " + std::to_string(churn.work()) +
       ", longest decode gap " + std::to_string(longestGap) + " ms of " + std::to_string(gapLimit) +
       ", store back " + std::to_string((*backAt - release) / 1000.0) + " s after the episode, " +
       std::to_string(givenOutside) + " units given outside it");
}

// --- SIM-churn-order ------------------------------------------------------------------------------------------

// SIM-churn-order (amend/V.md A100, AMENDMENTS X4; replaces SIM-arrival and
// SIM-two-long): the churn's arrivals in orders A (2,0,1) and B (0,1,2) on
// the Studio at 18G, its 59,043-token first request and round 2's
// 71,254/71,220 pair. (1) No start converts a running lane or shrinks the
// store: every start claim the ledger decides takes the cache's Ordinary
// units only (its decision log, MemoryLedger::log: no pool unit, conversion,
// pass command, in-use unit or other engine victim). (2) A leader keeps
// leading to its prompt's end unless a lane of less expected remaining cost
// supersedes it or an overdue one displaces it. (3) Two Spill lanes are in
// prefill together (in one command, or one's first pass while the other has
// prefilled part of its prompt) only when the later has the less expected
// remaining cost or one is overdue, and never the near-equal pair (two
// workers' fresh prompts of one round, at most 34 tokens apart). (4) The
// bound cuts rows only of units already at their most passes. Overdue as the
// scheduler judges it (Scheduler.cpp:425-432): the time a lane waited without
// rows behind the units that served later arrivals, against kAgingSlowdown
// (the batch width less one, amend/S.md A10) times its expected remaining
// cost, its remaining commands alone at the profile's price, less the run's
// p95 price error (a lane the scheduler prices that much lower is overdue to
// it already). Run as the plan places the prompts, and with every request
// spilling (--kv-spill always), which makes the prompts Spill lanes whatever
// the placement decides.
void churnOrder(const Machine &machine, Checks &checks) {
  const test::TimingProfile &profile = *machine.profile;
  constexpr double aging = model::ExecutionLimits::maximumBatchWidth - 1;
  // The pages a Spill lane keeps Full (kv_spill's dense and window pages).
  const uint32_t fullPages = machine.spill ? machine.spill->densePages + machine.spill->windowPages : 0;
  for (const auto &[label, order] : {std::pair{"A", std::array<uint32_t, 3>{2, 0, 1}},
                                     std::pair{"B", std::array<uint32_t, 3>{0, 1, 2}}}) {
    const std::string where = at(machine) + "order " + label + ": ";
    Serving serving(machine);
    Churn churn(serving, order);
    const test::TimedWork work(profile, [&] { return serving.experts->residentShare(); });
    // A lane's remaining commands alone, of the machine's command rows from
    // where it stands, a Spill lane re-reading its Index pages' history.
    const auto own = [&](const engine::ActiveRequestSnapshot &request) {
      const uint32_t pages = request.promptProcessed / engine::KvCache::pageTokens;
      const uint32_t indexPages = request.kvMode == engine::KvMode::Spill && pages > fullPages ? pages - fullPages : 0;
      double total = 0.0;
      for (uint32_t position = request.promptProcessed; position < request.promptTokens;
           position += machine.commandRows) {
        const std::array lanes{model::PrefillLane{.position = position,
                                                  .rows = std::min(machine.commandRows, request.promptTokens - position),
                                                  .indexPages = indexPages}};
        total += test::prefillTimes(profile, work.prefillWork(lanes, 0, 1)).wallMilliseconds;
      }
      return total;
    };
    // At each step that ran a command: the requests in flight, and the time
    // each waited as the scheduler counts it.
    std::map<uint64_t, engine::ActiveRequestSnapshot> active;
    std::map<uint64_t, double> waited;
    uint64_t boundCuts = 0;
    size_t seen = 0;
    std::optional<uint64_t> leader, lastSpill;
    std::string broken;
    const auto unfinished = [&](uint64_t id) {
      const auto found = active.find(id);
      return found != active.end() && found->second.phase == engine::Phase::Prefill &&
             found->second.promptProcessed < found->second.promptTokens;
    };
    const auto overdue = [&](uint64_t id, double e95) {
      const auto found = active.find(id);
      return found != active.end() && waited[id] >= aging * own(found->second) * (1.0 - e95);
    };
    // Churn ids are 100 times the worker plus the round; every fourth round
    // resends.
    const auto nearEqual = [](uint64_t a, uint64_t b) { return a % 100 == b % 100 && a % 100 % 4 != 0; };
    const auto together = [&](uint64_t a, uint64_t b, double e95) -> std::string {
      const uint64_t earlier = serving.submittedAt.at(a) <= serving.submittedAt.at(b) ? a : b;
      const uint64_t later = earlier == a ? b : a;
      const double earlierCost = own(active.at(earlier)), laterCost = own(active.at(later));
      if (!nearEqual(a, b) && (laterCost < earlierCost || overdue(a, e95) || overdue(b, e95)))
        return {};
      const auto left = [&](uint64_t id) { return active.at(id).promptTokens - active.at(id).promptProcessed; };
      return "Spill lanes " + std::to_string(earlier) + " and " + std::to_string(later) + " (prompts " +
             std::to_string(active.at(earlier).promptTokens) + " and " + std::to_string(active.at(later).promptTokens) +
             ") prefilled together with " + std::to_string(left(earlier)) + " and " + std::to_string(left(later)) +
             " rows left, expected " + std::to_string(earlierCost) + " and " + std::to_string(laterCost) +
             " ms, waited " + std::to_string(waited[earlier]) + " and " + std::to_string(waited[later]) + " ms" +
             (nearEqual(a, b) ? " (the near-equal pair)" : "");
    };
    churn.run(480'000.0, [&] {
      if (!broken.empty() || seen == serving.commands.size())
        return;
      const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
      active.clear();
      for (const engine::ActiveRequestSnapshot &request : counts.activeRequests)
        active.emplace(request.id, request);
      const double e95 = counts.scheduler.prefillEstimate.relativeErrorP95;
      for (; seen < serving.commands.size() && broken.empty(); ++seen) {
        const test::EngineHarness::Command &command = serving.commands[seen];
        if (command.kind != WorkKind::Prefill)
          continue;
        const auto serves = [&](uint64_t id) { return std::ranges::find(command.lanes, id) != command.lanes.end(); };
        // The lanes submitted before the youngest it serves wait its time
        // without rows (Scheduler::completed).
        double youngest = 0.0;
        for (const uint64_t lane : command.lanes)
          youngest = std::max(youngest, serving.submittedAt.at(lane));
        for (const auto &[id, request] : active)
          if (serving.submittedAt.at(id) < youngest && !serves(id) &&
              (request.phase == engine::Phase::Queued || request.phase == engine::Phase::WaitingResources ||
               request.phase == engine::Phase::WaitingPrefix || request.phase == engine::Phase::Prefill))
            waited[id] += command.wallMilliseconds;
        if (command.pass)
          continue;
        if (counts.scheduler.boundCuts > boundCuts && command.passes < profile.layers)
          broken = "the bound cut a unit in " + std::to_string(command.passes) + " passes";
        boundCuts = counts.scheduler.boundCuts;
        const uint64_t lead = command.lanes.front();
        if (broken.empty() && leader && *leader != lead && unfinished(*leader) && active.contains(lead) &&
            own(active.at(lead)) >= own(active.at(*leader)) && !overdue(lead, e95))
          broken = "lane " + std::to_string(lead) + " took the lead from " + std::to_string(*leader) + " expecting " +
                   std::to_string(own(active.at(lead))) + " ms left against " +
                   std::to_string(own(active.at(*leader))) + " ms";
        leader = lead;
        std::vector<uint64_t> spill;
        for (const uint64_t lane : command.lanes)
          if (active.contains(lane) && active.at(lane).kvMode == engine::KvMode::Spill)
            spill.push_back(lane);
        for (size_t first = 0; first < spill.size() && broken.empty(); ++first)
          for (size_t second = first + 1; second < spill.size() && broken.empty(); ++second)
            broken = together(spill[first], spill[second], e95);
        if (broken.empty() && lastSpill && !serves(*lastSpill) && unfinished(*lastSpill) && !spill.empty())
          broken = together(*lastSpill, spill.front(), e95);
        if (!spill.empty())
          lastSpill = spill.front();
      }
    });
    // (1) from the ledger's decisions: a start claim's take names nothing
    // but the cache (its Ordinary units) beside the MiB it took of each.
    uint32_t startClaims = 0;
    std::string startTook;
    for (const std::string &line : serving.decisions) {
      if (!line.starts_with("Memory: start claim"))
        continue;
      ++startClaims;
      const size_t took = line.find(", took ");
      if (took == std::string::npos)
        continue;
      std::istringstream items(line.substr(took + 7, line.find(" for ", took) - took - 7));
      for (std::string word; items >> word && startTook.empty();)
        if (word != "cache" && !std::isdigit(static_cast<unsigned char>(word.front())))
          startTook = line;
    }
    checks.expect(startTook.empty(), where + "a start claim took more than the cache's Ordinary units: " + startTook);
    checks.expect(broken.empty(), where + broken);
    checks.expect(serving.errors.empty(),
                  where + "a request failed: " + (serving.errors.empty() ? "" : serving.errors[0]));
    checks.expect(churn.finished.size() >= 3, where + "the churn finished " + std::to_string(churn.finished.size()) +
                                                  " requests in 8 minutes");
    const engine::EngineSnapshot counts = serving.fixture.loop.statusSnapshot();
    note(where + std::to_string(churn.finished.size()) + " requests, " + std::to_string(counts.spill.spill) +
         " started Spill, " + std::to_string(counts.spill.converted) + " converted, " +
         std::to_string(serving.experts->shrinks) + " store units given, " +
         std::to_string(counts.scheduler.passCommands) + " commands in passes, " +
         std::to_string(counts.scheduler.boundCuts) + " bound cuts, " + std::to_string(startClaims) +
         " start claims decided");
    if (machine.spillAlways)
      checks.expect(counts.spill.spill >= 4,
                    where + "only " + std::to_string(counts.spill.spill) + " requests ran Spill");
  }
}
void testChurnOrderKeepsLeadersAndSpillLanes() {
  Machine always = kStudio18G;
  always.name = "studio-18g-spill-always";
  always.spillAlways = true;
  Checks checks;
  churnOrder(kStudio18G, checks);
  churnOrder(always, checks);
  checks.verdict();
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

// SIM-low-host (amend/V.md A109, X19; HiSparse on, the Studio at 18G, whose
// prefill reads exceed its host floor, so a command's time follows the
// store's residency): other applications leave less than the store's cap
// until 60 s. A store sized from the boot sample at start
// (ExpertStore.cpp:1091-1103, A109's base defect) is pressure-serving's
// question, on Metal through RuntimeBootstrap::start; here the store starts
// at its floor, the size that sizing gives, and the questions are its
// regrowth and what the short store costs the first request. The store is
// still at its floor when the host frees. Idle, free room grows every unit
// it admits in one pass: the experts account reaches its cap within two
// settled control passes and the host's settling (3 kHostSettleMilliseconds)
// of the release. A first request of six of the machine's commands arriving
// at the release prefills while the store regrows: its first token within
// the same request's at the cap and the refill's reads (the units the store
// lacked at the profile's read rate). Its TTFT on another run is not that
// same request: the scheduler's units follow its price, seeded at the
// store's size at start, and their shapes differ between runs by far more
// than a chunk's reads (on this profile about 40 ms an 8192-row command;
// nothing on 35B at 18G, whose reads its host floor hides). So each of the
// request's prefill units is set against the same unit at the cap's
// residency on the profile, and what they took beyond it is held within the
// refill. The bound tells regrowth apart: with the store held at its floor
// through the prompt, the request's units pass it. A guard: the store starts
// at its floor by construction, so a base that regrows from there passes it
// (the tree before M3 does) and the boot sample's sizing never reaches it;
// A109's base failure is pressure-serving's. It fails when the store does not
// regrow once the host frees, or regrows too late for the first request.
void testShortStartRegrowsWhenTheHostFrees() {
  const Machine &machine = kStudio18G;
  constexpr double release = 60'000.0;
  const uint64_t lacking = uint64_t{machine.store.cap - machine.store.floor} * machine.store.unitBytes;
  // Until the release, the host's spare memory with the store at its cap is
  // its spare reserve alone: a unit's growth would take the margin.
  const uint64_t spare = machine.hostBytes - machine.otherBytes - machine.fixedBytes -
                         uint64_t{machine.store.cap} * machine.store.unitBytes;
  const std::vector<test::HostEvent> shortHost =
      test::lowHostAtStart(machine.otherBytes + spare - machine.hostSpareReserveBytes, machine.otherBytes, release);
  // Idles to the release; a short start's store is still at its floor then.
  const auto toRelease = [&](Serving &serving, bool startedShort) {
    idleUntil(serving, release - serving.now(), [] { return false; });
    if (startedShort)
      require(serving.experts->units() == machine.store.floor,
              at(machine) + "the store stood at " + std::to_string(serving.experts->units()) +
                  " chunks when the host freed, not at its floor of " + std::to_string(machine.store.floor));
  };
  {
    Serving serving(machine, shortHost, machine.store.floor);
    toRelease(serving, true);
    idleUntil(serving, 3 * engine::kHostSettleMilliseconds,
              [&] { return serving.experts->units() == machine.store.cap; });
    require(serving.experts->units() == machine.store.cap,
            at(machine) + "the store stayed at " + std::to_string(serving.experts->units()) + " chunks of " +
                std::to_string(machine.store.cap) + " for two settled passes after the host freed");
  }
  // What the first request's prefill units took beyond the same units at
  // the cap's residency, its TTFT, and its units (rows @ pass/passes: ms, the
  // store's chunks as each began).
  struct First final {
    double beyondCap = 0.0, ttft = 0.0;
    std::string units;
  };
  const test::TimedWork atCap(*machine.profile, [&] { return double(machine.store.cap) / machine.store.whole; });
  const auto firstRequest = [&](bool held) {
    Serving serving(machine, shortHost, machine.store.floor);
    serving.experts->regrows = !held;
    toRelease(serving, true);
    serving.submit(1, prompt(1, 6 * machine.commandRows), 64);
    First result;
    size_t seen = serving.commands.size();
    serving.run([&] { return !serving.active(1); },
                [&] {
                  for (; seen < serving.commands.size(); ++seen) {
                    const test::EngineHarness::Command &unit = serving.commands[seen];
                    if (unit.kind != WorkKind::Prefill || unit.lanes != std::vector<uint64_t>{1})
                      continue;
                    const std::array lanes{model::PrefillLane{.position = unit.positions.front(),
                                                              .rows = unit.rows.front()}};
                    result.beyondCap +=
                        unit.wallMilliseconds -
                        test::prefillTimes(*machine.profile, atCap.prefillWork(lanes, unit.pass, unit.passes))
                            .wallMilliseconds;
                    result.units += " " + std::to_string(unit.rows.front()) + "@" + std::to_string(unit.pass) + "/" +
                                    std::to_string(unit.passes) + ":" + std::to_string(int(unit.wallMilliseconds)) +
                                    "(" + std::to_string(serving.experts->units()) + ")";
                  }
                },
                "the first request did not end");
    require(serving.errors.empty() && serving.ttft(1), at(machine) + "the first request failed: " +
                                                           (serving.errors.empty() ? "" : serving.errors.front()));
    result.ttft = *serving.ttft(1);
    return result;
  };
  const First regrown = firstRequest(false), held = firstRequest(true);
  const double refill = double(lacking) / machine.profile->weightBytesPerMillisecond;
  require(held.beyondCap > refill, at(machine) + "the bound cannot tell regrowth: beside a store held at its floor "
                                                 "the first request's units took " +
                                       std::to_string(held.beyondCap) + " ms beyond the cap's, the refill " +
                                       std::to_string(refill) + " ms; its units:" + held.units);
  require(regrown.beyondCap <= refill, at(machine) + "the first request after a short start took " +
                                           std::to_string(regrown.beyondCap) +
                                           " ms beyond its units at the cap, the refill " + std::to_string(refill) +
                                           " ms; its units:" + regrown.units);
  note(at(machine) + "the first request's units took " + std::to_string(regrown.beyondCap) +
       " ms beyond the cap's after a short start (TTFT " + std::to_string(regrown.ttft) + " ms), " +
       std::to_string(held.beyondCap) + " ms beside a store held at its floor (" + std::to_string(held.ttft) +
       " ms); refill " + std::to_string(refill) + " ms");
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
// so the store gives a chunk), unless the two end within one decode token's
// work of each other (amend/V.md §3.4): a token of the two decoders'
// command, Resident, at the store's cap. On the 36 GB Studio this follows
// AMENDMENTS OQ1's answer: the value outcome, no spill-capability order.
void testValueChoiceFollowsTheCalibratedCost() {
  struct Point final {
    const Machine *machine;
    std::array<double, 2> spill;
  };
  Checks checks;
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
          if (free.failure || free.action == ValueRun::Action::None) {
            checks.expect(false, where + (free.failure ? "a decoder failed: " + *free.failure
                                                       : std::string("the decoders' growth took nothing")));
            continue;
          }
          const double token =
              decodeCommand(*machine.profile, 2, double(machine.store.cap) / machine.store.whole, 0) /
              kTokensPerCommand;
          note(where + (free.action == ValueRun::Action::Conversion ? "converted" : "gave a store chunk") +
               "; ends free " + std::to_string(free.endMilliseconds / 1000.0) + " s, converted " +
               std::to_string(pinned.endMilliseconds / 1000.0) + " s, store chunk " +
               std::to_string(noSpill.endMilliseconds / 1000.0) + " s");
          if (std::abs(pinned.endMilliseconds - noSpill.endMilliseconds) <= token)
            continue;
          const ValueRun::Action better = pinned.endMilliseconds < noSpill.endMilliseconds
                                              ? ValueRun::Action::Conversion
                                              : ValueRun::Action::StoreUnit;
          checks.expect(free.action == better,
                        where + "memory took " +
                            (free.action == ValueRun::Action::Conversion ? "a conversion" : "a store chunk") +
                            " where the other ends first (converted " +
                            std::to_string(pinned.endMilliseconds / 1000.0) + " s, store chunk " +
                            std::to_string(noSpill.endMilliseconds / 1000.0) + " s)");
        }
    }
  }
  checks.verdict();
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
// decoder, four 300-row turns 2 s apart past an 8K cached prefix. After
// those five commands and more, the prefill estimate prices a 300-row turn
// past the prefix and an 8192-row command from nothing within 15 % of the
// profile's walls; every contended prefill unit within the run's unit bound
// by its price's p95 error (G4a, A09: each unit, not their p95); the
// decoder's gaps within that and one decode command of its one lane. Then,
// as the TTFT bench runs (no decoder, A74): N = 1, 2 and 4 turns together,
// the worst of two within 1.51 times one turn's and the worst of four within
// 1.71 times (the pass tree's statistics against its N=1, 4622 and 5247
// against 3065 ms, A74), the four within 1.2 of each other (passes' gain,
// DATA.md: 5152/5247 ms); and 2 and 4 arriving 0.5, 1 and 1.5 s apart, each
// within its own command, the one in flight and one of all of them.
void testTurnsSettleAndShareACommand() {
  const test::TimingProfile &profile = test::kM6FlashNextQ2;
  Checks checks;
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
  checks.expect(scheduler.prefillEstimate.samples >= 5,
                "only " + std::to_string(scheduler.prefillEstimate.samples) + " prefill units trained the estimate");
  test::TimedWork work(profile);
  for (const model::PrefillLane lane : {model::PrefillLane{8192, 300, 0}, model::PrefillLane{0, 8192, 0}}) {
    const model::CommandWork shape = work.prefillWork({&lane, 1}, 0, 1);
    const double predicted = estimatedPrice(scheduler.prefillEstimate, shape);
    const double wall = test::prefillTimes(profile, shape).wallMilliseconds;
    checks.expect(std::abs(predicted - wall) <= 0.15 * wall,
                  "the prefill estimate priced " + std::to_string(lane.rows) + " rows from " +
                      std::to_string(lane.position) + " at " + std::to_string(predicted) + " ms against the profile's " +
                      std::to_string(wall) + " ms");
  }
  const double bound = unitBound(scheduler);
  std::vector<double> units = turns.contendedUnits(2);
  std::ranges::sort(units);
  checks.expect(units.size() >= 20, "only " + std::to_string(units.size()) + " prefill units ran beside the decoder");
  if (!units.empty()) {
    const double p95 = units[(units.size() * 95 + 99) / 100 - 1];
    double turnaround = 0.0;
    for (const engine::CostTermSnapshot &term : scheduler.prefillEstimate.terms)
      if (term.quantity == "units")
        turnaround = term.millisecondsPerUnit;
    checks.expect(units.back() <= bound,
                  "the longest of " + std::to_string(units.size()) + " contended prefill units, " +
                      std::to_string(units.back()) + " ms (p95 " + std::to_string(p95) + "), passed the bound's " +
                      std::to_string(bound) + " ms (G " + std::to_string(scheduler.decodeGapTargetMilliseconds) +
                      ", turnaround " + std::to_string(turnaround) + " ms, p95 error " +
                      std::to_string(scheduler.prefillEstimate.relativeErrorP95) + ")");
  }
  const double command = decodeCommand(profile, 1, profile.measuredResidentShare, 0);
  checks.expect(turns.harness.longestDecodeGap(2) <= bound + command,
                "the decoder's longest gap, " + std::to_string(turns.harness.longestDecodeGap(2)) + " ms, passed " +
                    std::to_string(bound + command) + " ms");
  turns.harness.engine.cancel(2);
  turns.harness.runUntilIdle();
  const double alone = turnsTogether(turns, id, 1, 0.0, prefix)[0];
  id += 1;
  turns.harness.runUntilIdle();
  const std::vector<double> two = turnsTogether(turns, id, 2, 0.0, prefix);
  id += 2;
  const double worstTwo = *std::ranges::max_element(two);
  checks.expect(worstTwo <= 4622.0 / 3065.0 * alone, "two turns together: worst TTFT " + std::to_string(worstTwo) +
                                                         " ms against one turn's " + std::to_string(alone) + " ms");
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
  checks.expect(worstFour <= 5247.0 / 3065.0 * alone,
                "four turns together: worst TTFT " + std::to_string(worstFour) + " ms against one turn's " +
                    std::to_string(alone) + " ms; prefill units (lanes x rows @ pass/passes: ms):" + ran);
  const double bestFour = *std::ranges::min_element(four);
  checks.expect(worstFour <= 1.2 * bestFour, "four turns together: TTFTs from " + std::to_string(bestFour) + " to " +
                                                 std::to_string(worstFour) +
                                                 " ms; prefill units (lanes x rows @ pass/passes: ms):" + ran);
  for (const uint32_t count : {2u, 4u})
    for (const double gap : {500.0, 1000.0, 1500.0}) {
      turns.harness.runUntilIdle();
      std::vector<model::PrefillLane> lanes(count, model::PrefillLane{8192, 300, 0});
      const double whole = test::prefillTimes(profile, work.prefillWork(lanes, 0, 1)).wallMilliseconds;
      const std::vector<double> apart = turnsTogether(turns, id, count, gap, prefix);
      id += count;
      for (size_t index = 0; index < apart.size(); ++index)
        checks.expect(apart[index] <= 1.25 * (alone + whole),
                      std::to_string(count) + " turns " + std::to_string(int(gap)) + " ms apart: turn " +
                          std::to_string(index + 1) + " took " + std::to_string(apart[index]) + " ms against " +
                          std::to_string(1.25 * (alone + whole)));
    }
  checks.verdict();
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
// of its. A guard: it compares the tree with itself, its passes against one
// pass, so no base can fail it; it fails when passes cost dense targets more
// than their tenth.
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
// decoders. Every decoder gap within the run's unit bound and one decode
// command of the two lanes; the prompt's prefill, its decode share taken
// out, at least 0.9 (1 - kPassOverheadShare) of its rate alone.
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
      const double command = decodeCommand(profile, 2, 1.0, 0);
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
// state (Request::placement): over the Studio's churn, no candidate (a
// request that holds no lane: the conversions the ledger prices ask about
// lanes in service, Executor::spilledReadShortfall) asks the model's
// spilled-read shortfall of one kind (a prefill's or a decode's need) more
// than once in one admission state, and Spill-capable candidates were placed
// by it. The state changes with what memory allocated or gave back, every
// refusal the governor counts (EngineHarness::admissionState) and every
// control pass, whose landings change what a placement reads. A take the
// ledger makes for a refusal outside the governor's count, which the engine
// counts too, is not visible here; nor are the governor's available() calls:
// it gives no hook to count them by.
void testPlacementOncePerCandidatePerState() {
  Serving serving(kStudio18G);
  serving.fixture.executor.admissionState = [&] {
    return serving.fixture.memory->changes().size() + serving.memory().deniedReservations +
           serving.fixture.controlPasses;
  };
  Churn churn(serving, {2, 0, 1});
  churn.run(240'000.0, {});
  require(!serving.fixture.executor.shortfallQueries.empty(),
          "no candidate's placement asked the model's spilled-read shortfall");
  for (const auto &[key, calls] : serving.fixture.executor.shortfallQueries) {
    const auto &[id, state, kind] = key;
    require(calls == 1, "request " + std::to_string(id) + " asked the spilled-read shortfall of its " +
                            (kind == WorkKind::Prefill ? "prefill" : "decode") + " " + std::to_string(calls) +
                            " times in admission state " + std::to_string(state));
  }
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
