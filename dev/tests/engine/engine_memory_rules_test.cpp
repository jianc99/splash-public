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

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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

// An engine whose prompts run in passes (test::commandShape), its KV pages
// of 16 bytes and the model's states of 64 allocated through FakeMemory
// under a governor of `limitBytes` (GovernedHost); the prefill commands'
// first lane and pass, in order.
struct PassesUnderALimit {
  explicit PassesUnderALimit(uint64_t limitBytes) : host({.limitBytes = limitBytes}) {
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
  test::TestKvStorage storage{512, 16, 4};
  KvPool pool{(host.through(storage), storage), 0};
  engine::Cache cache{pool, nullptr, nullptr};
  Executor model;
  bool shaped = (test::commandShape(4 * model::ExecutionLimits::prefillTokenBudget, 6)(model), true);
  Events events;
  engine::Engine engine{host.connect({}), cache, model, events};
  std::vector<std::pair<uint64_t, uint32_t>> prefills;
  double now = 1;
};

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
// disk. A restore that lands while a command is in flight takes nothing for
// its copy (L8): refused, the state stays on disk and the request runs from
// it.
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
    const StateCacheSnapshot states = harness.cache.snapshot().stateCache;
    require(harness.engine.commandInFlight() && states.promotions == 0 && states.promotionsSkipped == 1,
            "a restore that landed beside a command in flight was promoted with no room");
    harness.memory->release("limit", harness.memory->held("limit"));
    *release = true;
    harness.runUntilIdle();
    require(harness.events.completedCount == 2 && harness.events.failedCount == 0 &&
                harness.model.heldStateBytes == 0,
            "the requests did not finish, or the refused copy held memory");
  }
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

} // namespace

std::span<const test::TestCase> rulesTests() {
  static constexpr test::TestCase kTests[] = {
      {"testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds",
       testALowerPriorityClaimTakesNothingAHigherPriorityLaneNeeds},
      {"testSupersedingClaimTakesThePassCommand", testSupersedingClaimTakesThePassCommand},
      {"testLandingPaysItsOwnerFirst", testLandingPaysItsOwnerFirst},
      {"testQuotaWaitIsNotAMemoryClaim", testQuotaWaitIsNotAMemoryClaim},
      {"testConversionVsStoreFollowsValue", testConversionVsStoreFollowsValue},
      {"testPlacementOncePerCandidatePerState", testPlacementOncePerCandidatePerState},
      {"testARestoredStateIsPromotedInAClaim", testARestoredStateIsPromotedInAClaim},
      {"testAReplayPointBesideACheckpointIsReservedInAClaim", testAReplayPointBesideACheckpointIsReservedInAClaim},
      {"testEnginesOnOneGovernorOfferTheCacheOnce", testEnginesOnOneGovernorOfferTheCacheOnce},
      {"testAJunctionTakesNothingWorthMoreThanItSaves", testAJunctionTakesNothingWorthMoreThanItSaves},
      {"testCacheReuseFollowsTheRecentProbes", testCacheReuseFollowsTheRecentProbes},
  };
  return kTests;
}
