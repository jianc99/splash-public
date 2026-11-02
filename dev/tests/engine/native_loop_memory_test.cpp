#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestExecutor.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"

#include <optional>
#include <utility>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::Executor;
using splash::test::hostMemory;
using splash::test::LoopFixture;
using splash::test::PoolShape;
using splash::test::releaseAnswer;
using splash::test::request;
using splash::test::require;
using splash::test::runUntilIdle;

// Between commands the control pass runs what the host's pressure asks for:
// with nothing waiting, a paced pass toward the warning margin, which keeps
// the resume point and the empty runway extent, and nothing past it; nothing
// once the host recovers; and under critical pressure every cache entry and
// extent. None waits for a transfer.
void testControlPassReclaimsUnderHostPressure() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  const engine::Cache &resources = fixture.cache;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  // Two finished requests with different prompts leave two replay states.
  for (uint64_t id : {1, 2}) {
    auto input = request(id);
    for (uint32_t &token : input.promptTokens)
      token += static_cast<uint32_t>(100 * id);
    require(loop.receive(protocol::peer::serialize(input)),
            "control pass request failed");
    runUntilIdle(loop);
  }
  require(resources.snapshot().stateCache.entries == 2,
          "the fixture did not cache two replay states");

  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  std::optional<uint64_t> available = hostReserve + kHostWarningMarginBytes / 2;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  require(!control.run(MemoryPressure::Normal), "a paced pass waited for a transfer");
  const auto paced = resources.snapshot();
  require(paced.stateCache.entries == 1 && paced.stateCache.evictions == 1 &&
              paced.pool.reclaimableBytes == 4 * 4096,
          "host pressure did not evict toward its target, keeping the resume "
          "point and the runway extent");

  // Past the warning margin the hysteresis still holds the pressure, but a
  // pass that nothing waits for evicts nothing there: the state a third
  // request caches stays beside the resume point.
  auto third = request(3);
  for (uint32_t &token : third.promptTokens)
    token += 300;
  require(loop.receive(protocol::peer::serialize(third)), "control pass request failed");
  runUntilIdle(loop);
  available = hostReserve + kHostWarningMarginBytes + kHostWarningMarginBytes / 2;
  fixture.monotonic += 1000.0;
  require(!control.run(MemoryPressure::Normal) &&
              governor.snapshot().pressure == MemoryPressure::Warning &&
              resources.snapshot().stateCache.entries == 2,
          "a pass that nothing waited for evicted past the warning margin");

  available = hostReserve + kHostRecoveryMarginBytes + kHostWarningMarginBytes;
  const uint64_t releases = pool.snapshot().extentReleases;
  // The host's recovery lifts the pressure, so no pass runs: the untargeted
  // pass the settle window would otherwise run finds nothing to take either.
  require(!control.run(MemoryPressure::Normal) &&
              governor.snapshot().pressure == MemoryPressure::Normal &&
              resources.snapshot().stateCache.entries == 2 &&
              pool.snapshot().extentReleases == releases,
          "a recovered host still reclaimed");

  require(!control.run(MemoryPressure::Critical), "a critical pass waited for a transfer");
  const auto critical = resources.snapshot();
  require(critical.stateCache.entries == 0 && critical.pool.pagesPrefix == 0 &&
              critical.pool.pagesAllocated == 0,
          "critical pressure did not empty the cache and release every extent");
}

// macOS's warning asks caches to give memory back: however much room the
// host has, a control pass under it returns cache, keeping the resume point
// and the runway extent while nothing waits, and pauses growth. Once the
// level is normal again growth resumes and no pass runs: the state a later
// request caches stays beside the resume point.
void testControlPassGivesCacheBackUnderSystemWarning() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  const engine::Cache &resources = fixture.cache;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  // Finished requests with different prompts leave a replay state each.
  const auto finish = [&](uint64_t id) {
    auto input = request(id);
    for (uint32_t &token : input.promptTokens)
      token += static_cast<uint32_t>(100 * id);
    require(loop.receive(protocol::peer::serialize(input)), "control pass request failed");
    runUntilIdle(loop);
  };
  finish(1);
  finish(2);
  require(resources.snapshot().stateCache.entries == 2,
          "the fixture did not cache two replay states");

  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  const std::optional<uint64_t> available = hostReserve + (16ULL << 30);
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  require(!control.run(MemoryPressure::Warning), "a pass under macOS's warning waited for a transfer");
  const auto warned = resources.snapshot();
  require(warned.stateCache.entries == 1 && warned.stateCache.evictions == 1 &&
              warned.pool.reclaimableBytes == 4 * 4096 && !governor.snapshot().hostGrowthAllowed,
          "macOS's warning did not give cache back, keeping the resume point and the runway "
          "extent, or left growth open");

  finish(3);
  fixture.monotonic += 1000.0;
  const uint64_t releases = pool.snapshot().extentReleases;
  require(!control.run(MemoryPressure::Normal) && governor.snapshot().hostGrowthAllowed &&
              resources.snapshot().stateCache.entries == 2 &&
              pool.snapshot().extentReleases == releases,
          "a pass ran once macOS's level was normal again");
}

// A start marks what it allocates as memory a request in service needs until
// serving begins. An allocation the engine marks meanwhile restores the mark
// it found, so the start's mark outlives it, and once serving begins the
// engine's marks still end with their allocations.
void testServingMarkRestoresTheMarkItFound() {
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  const std::optional<uint64_t> available = hostReserve + (16ULL << 30);
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  // Under macOS's warning only the mark admits growth.
  governor.setPressure(MemoryPressure::Warning);
  const auto admits = [&governor] {
    return static_cast<bool>(governor.allocationAdmission()(4096, [] {}));
  };
  const std::function<bool(bool)> mark = [&governor](bool serving) {
    return governor.setServing(serving);
  };
  static_cast<void>(governor.setServing(true));
  {
    const Serving allocation(mark, true);
    require(admits(), "an allocation the engine marked was refused");
  }
  require(admits(), "an allocation the engine marked ended the start's mark");
  static_cast<void>(governor.setServing(false));
  {
    const Serving allocation(mark, true);
    require(admits(), "an allocation the engine marked was refused once serving began");
  }
  require(!admits(), "an allocation the engine marked left its mark behind");
}

// The control pass tells the governor whether a request waits for memory:
// only for one does growth wait for the recovery margin while the host's
// headroom sits between the margins.
void testControlPassHoldsGrowthForAWaitingRequest() {
  LoopFixture fixture;
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  // A finished request leaves cache a pass can free.
  require(loop.receive(protocol::peer::serialize(request(1))), "the first request failed");
  runUntilIdle(loop);
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  // A page short of the recovery margin, after growth the host refused.
  std::optional<uint64_t> available = hostReserve + kHostRecoveryMarginBytes - 4096;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  require(!governor.allocationAdmission()(kHostWarningMarginBytes, [] {}),
          "growth past the warning margin was admitted");
  MemoryControl control(governor, backend, loop);
  static_cast<void>(control.run(MemoryPressure::Normal));
  require(governor.snapshot().pressure == MemoryPressure::Warning &&
              governor.snapshot().hostGrowthAllowed,
          "growth waited for the recovery margin with no request waiting");

  // The host refuses the next request's lane, which then waits for memory.
  fixture.executor.beginGrowthBlocked = [] { return true; };
  fixture.executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  require(loop.receive(protocol::peer::serialize(request(2))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request did not wait for memory");
  // The next control pass finds it waiting.
  static_cast<void>(control.run(MemoryPressure::Normal));
  require(!governor.snapshot().hostGrowthAllowed,
          "growth did not wait for the recovery margin while a request waited");

  // Its retry, within its deadline, finds its lane.
  fixture.executor.beginGrowthBlocked = {};
  fixture.monotonic += 2 * engine::kResourceRetryBackoffMilliseconds;
  runUntilIdle(loop);
  static_cast<void>(control.run(MemoryPressure::Normal));
  require(loop.statusSnapshot().completed == 2 && governor.snapshot().hostGrowthAllowed,
          "the hold outlived the waiting request");
}

// The reporter logs a change in what requests wait for, never a retry or the
// depth of a queue.
void testMemoryStatusReporterLogsTransitionsOnly() {
  ResourceWaitSnapshot wait{.memory = 2, .concurrency = 1, .heldBehindRefusal = 4,
                            .restoring = 1, .suspended = 1,
                            .oldestWaitMilliseconds = 1250.0, .draining = true};
  MemoryStatusReporter reporter;
  require(reporter.update({}, true).empty(), "healthy idle engine logged pressure");
  require(!reporter.update(wait, false).empty(), "pressure transition was silent");
  ++wait.memory;
  wait.oldestWaitMilliseconds += 1000;
  require(reporter.update(wait, false).empty(), "pressure retries flooded the log");
  wait = {};
  wait.concurrency = 4;
  require(!reporter.update(wait, true).empty(), "end of memory wait was silent");
  require(reporter.update(wait, true).empty(), "concurrency queue logged pressure");
  // The refused request is deferred for scheduling, out of the memory
  // count, while it still holds the others back.
  wait.heldBehindRefusal = 2;
  const std::string held = reporter.update(wait, true);
  require(held.find("held=2") != std::string::npos &&
              held.find("cleared") == std::string::npos,
          "requests held behind a refusal were reported as no wait");
}

// Ticks until no command is in flight, where the transport runs the control
// pass.
void settleCommand(engine::NativeRuntime &loop) {
  for (uint32_t step = 0; step < 8 && loop.commandInFlight(); ++step)
    static_cast<void>(loop.tick());
  require(!loop.commandInFlight(), "the command in flight did not complete");
}

// A control pass the policy asks nothing of takes back one chunk of the
// expert store toward its cap while memory is ample: the host has the
// recovery margin to spare beyond the chunk, or the engine returns the rest
// to the system first, no request waits for memory, the weights are neither
// released nor being restored, and the cool-down after reclaim last gave a
// chunk back has passed, twice as long after a chunk it took back went
// again. Beside a resident lane, with no cache that covers what the engine's
// limit lacks, it takes only room the limit has free, and asks for none it
// lacks.
void testControlPassGrowsTheExpertStoreWhenMemoryIsAmple() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  loop.announceReady();
  constexpr uint64_t chunk = 1ULL << 30;
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk, chunk, chunk};
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t limit = 40ULL << 30;
  // Half the chunk short of the recovery margin, which the empty extents
  // the engine returns do not cover.
  std::optional<uint64_t> available = hostReserve + kHostRecoveryMarginBytes + chunk / 2;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  const auto pass = [&](double elapsed = 0.0) {
    fixture.monotonic += elapsed;
    require(!control.run(MemoryPressure::Normal), "a pass waited for a transfer");
    return executor.expertChunks.size();
  };
  require(pass() == 1 && executor.expertGrowthAttempts == 0,
          "the store grew without the recovery margin to spare beyond its chunk");
  available = hostReserve + (16ULL << 30);
  require(pass() == 2, "a pass with memory to spare did not take one chunk back");

  // A request waiting for memory holds the growth back.
  executor.beginGrowthBlocked = [] { return true; };
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  require(loop.receive(protocol::peer::serialize(request(1))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request did not wait for memory");
  require(pass() == 2, "the store grew while a request waited for memory");
  executor.beginGrowthBlocked = {};
  fixture.monotonic += 2 * engine::kResourceRetryBackoffMilliseconds;
  runUntilIdle(loop);

  // Beside a resident lane, with no cache that covers half a chunk, only
  // room the engine's limit has free.
  auto running = request(2, 900);
  for (uint32_t &token : running.promptTokens)
    token += 200;
  fixture.executor.decodeFinishes = false;
  require(loop.receive(protocol::peer::serialize(running)), "the long request failed");
  for (uint32_t step = 0; step < 8; ++step)
    static_cast<void>(loop.tick());
  settleCommand(loop);
  require(fixture.executor.requests.contains(2) && fixture.executor.requests.at(2).resident,
          "the long request did not decode");
  test::metalStatistics().allocatedBytes = limit - chunk / 2;
  const uint32_t attempts = executor.expertGrowthAttempts;
  require(pass() == 2 && executor.expertGrowthAttempts == attempts,
          "growth beside a resident lane asked for room the engine's limit lacked");
  test::metalStatistics().allocatedBytes = limit - 2 * chunk;
  require(pass() == 3, "growth beside a resident lane did not take free room");
  test::metalStatistics().allocatedBytes = 0;
  fixture.executor.decodeFinishes = true;
  runUntilIdle(loop);

  // A host short of the warning margin takes a chunk back; the store grows
  // again only once the cool-down after it has passed. The chunk last taken
  // back has stayed for a cool-down, which keeps it at its 30 s.
  fixture.monotonic += kExpertGrowthCooldownMilliseconds;
  available = hostReserve + kHostWarningMarginBytes / 2;
  require(pass() == 2 && loop.expertReleases() == 1,
          "a short host did not take a chunk of the store");
  available = hostReserve + (16ULL << 30);
  require(pass(1000.0) == 2 && pass(kExpertGrowthCooldownMilliseconds - 1001.0) == 2,
          "the store grew back within the cool-down after a chunk went");
  require(pass(1.0) == 3, "the store did not grow back after the cool-down");

  // Released weights hold the growth back, until their restore is over.
  executor.releasedExpertChunks.push_back(chunk);
  executor.expertChunks.pop_back();
  require(!releaseAnswer(fixture, 9).restoring && fixture.weights.released(),
          "the weights were not released");
  require(pass() == 2, "the store grew while the weights were released");
  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              fixture.weights.released() && pass() == 2,
          "the store grew while the weights were being restored");
  runUntilIdle(loop);
  require(!fixture.weights.released() && pass() == 3,
          "the store did not grow once the weights were back");

  // A chunk taken back that goes again within the cool-down doubles the
  // next; one that stays for a whole cool-down restores it.
  available = hostReserve + kHostWarningMarginBytes / 2;
  const uint64_t releases = loop.expertReleases();
  require(pass(1000.0) == 2 && loop.expertReleases() == releases + 1,
          "a short host did not take the chunk taken back");
  available = hostReserve + (16ULL << 30);
  require(pass(kExpertGrowthCooldownMilliseconds) == 2 &&
              pass(kExpertGrowthCooldownMilliseconds - 1.0) == 2,
          "the store grew back within twice the cool-down after a chunk it took back went");
  require(pass(1.0) == 3, "the store did not grow back after twice the cool-down");
  require(pass(2 * kExpertGrowthCooldownMilliseconds) == 4 && pass(kExpertGrowthCooldownMilliseconds) == 4,
          "the store did not grow to its cap");
  available = hostReserve + kHostWarningMarginBytes / 2;
  require(pass(1000.0) == 3, "a short host did not take a chunk");
  available = hostReserve + (16ULL << 30);
  require(pass(kExpertGrowthCooldownMilliseconds - 1001.0) == 3 && pass(1001.0) == 4,
          "a chunk that stayed for the cool-down did not restore it");
}

// A conversation that finishes leaves its replay point, and a prompt of its
// own shares no prefix with another's.
void finishConversation(LoopFixture &fixture, uint64_t id) {
  auto input = request(id);
  for (uint32_t &token : input.promptTokens)
    token += static_cast<uint32_t>(100 * id);
  require(fixture.loop.receive(protocol::peer::serialize(input)), "a conversation failed");
  runUntilIdle(fixture.loop);
}

// The pool of the exchange tests: extents of two pages, one allocated up
// front as the runway, and chunks of the expert store of two extents' bytes.
constexpr PoolShape kExchangePool{.pages = 64, .extentPages = 2, .runwayPages = 2};
constexpr uint64_t kExchangeExtentBytes = 2 * 4096;
constexpr uint64_t kExchangeChunkBytes = 2 * kExchangeExtentBytes;

// With no request in the engine, a control pass whose host's headroom lacks
// a chunk of the expert store beyond the recovery margin exchanges cache for
// it: the engine returns the cache nothing in use holds to the system,
// releasing the extents it empties, until what it returned covers the
// chunk, and keeps the resume point; the governor's growth admission counts
// those bytes as the host's already and admits the chunk, which the host's
// headroom alone does not. One chunk comes back per pass. Without such
// cache none does.
void testIdleControlPassExchangesCacheForTheExpertStore() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // Between the warning margin and the recovery margin.
  const std::optional<uint64_t> available =
      hostReserve + kHostWarningMarginBytes + kHostWarningMarginBytes / 2;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk, chunk};
  const auto pass = [&] {
    require(!control.run(MemoryPressure::Normal), "a pass waited for a transfer");
    return executor.expertChunks.size();
  };
  require(pass() == 1 && executor.expertGrowthAttempts == 0 &&
              pool.snapshot().allocatedBytes == kExchangeExtentBytes,
          "the store grew with no cache to return for it, or the runway extent went");

  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  const auto before = resources.snapshot();
  const uint64_t buffers = executor.reclaimedIdleStateBytes;
  require(pass() == 2 && executor.releasedExpertChunks.size() == 1,
          "an idle pass did not exchange cache for one chunk of the store");
  const auto after = resources.snapshot();
  const uint64_t extents = before.pool.allocatedBytes - after.pool.allocatedBytes;
  require(extents + executor.reclaimedIdleStateBytes - buffers >= chunk && extents <= chunk &&
              after.pool.extentReleases > before.pool.extentReleases &&
              after.stateCache.entries < before.stateCache.entries,
          "the exchange did not release about a chunk of cache to the system");
  require(!governor.growthAdmission()(chunk, [] {}),
          "the host's headroom alone admitted the chunk, or the exchange's credit outlived it");
  require(pass() == 3 && pass() == 3, "the store did not grow a chunk per pass up to its cap");
  const uint64_t releases = pool.snapshot().extentReleases;
  require(pass() == 3 && pool.snapshot().extentReleases == releases,
          "a store at its cap took cache");
  const uint64_t hits = loop.statusSnapshot().cacheHits;
  finishConversation(fixture, 6);
  require(loop.statusSnapshot().cacheHits == hits + 1,
          "the newest conversation did not resume from the state the exchange kept");
}

// The exchange waits where the store's growth does: while a request waits
// for memory, within the cool-down after a chunk went, and under critical
// pressure, whose passes give cache back but take no chunk back. None
// returns cache for the store or asks for a chunk. It waits neither beside
// a running lane nor under macOS's warning alone: the store's cap is a
// reservation the cache repays from what no request needs
// (testControlPassExchangesCacheBesideARunningLane), and an exchange that
// returns the whole chunk first leaves the host's footprint where it was
// (testWarningPassExchangesOnlyWhatItReturns).
void testExpertExchangeWaitsWhereGrowthDoes() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // The host lacks the whole chunk beyond the recovery margin.
  const uint64_t between = hostReserve + kHostWarningMarginBytes + kHostWarningMarginBytes / 2;
  std::optional<uint64_t> available = between;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk, chunk};
  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  // A pass that neither took a chunk back nor asked for one, and, unless
  // its pressure gives cache back, returned none.
  const auto waits = [&](MemoryPressure pressure = MemoryPressure::Normal) {
    const size_t chunks = executor.expertChunks.size();
    const uint32_t attempts = executor.expertGrowthAttempts;
    const uint64_t releases = pool.snapshot().extentReleases;
    require(!control.run(pressure), "a pass waited for a transfer");
    return executor.expertChunks.size() <= chunks && executor.expertGrowthAttempts == attempts &&
           (pressure != MemoryPressure::Normal || pool.snapshot().extentReleases == releases);
  };

  // While a request waits for memory: the host has the recovery margin it
  // waits for, beside half a chunk.
  available = hostReserve + kHostRecoveryMarginBytes + chunk / 2;
  executor.beginGrowthBlocked = [] { return true; };
  executor.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  require(loop.receive(protocol::peer::serialize(request(21))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request did not wait for memory");
  require(waits(), "the exchange took cache while a request waited for memory");
  executor.beginGrowthBlocked = {};
  fixture.monotonic += 2 * engine::kResourceRetryBackoffMilliseconds;
  runUntilIdle(loop);

  // Within the cool-down after a short host took the store's chunks.
  available = hostReserve + kHostWarningMarginBytes / 2;
  require(!control.run(MemoryPressure::Normal) && executor.expertChunks.empty() &&
              loop.expertReleases() != 0,
          "a short host did not take the store's chunks");
  for (uint64_t id = 7; id <= 12; ++id)
    finishConversation(fixture, id);
  available = hostReserve + kHostRecoveryMarginBytes + chunk / 2;
  fixture.monotonic += kExpertGrowthCooldownMilliseconds - 1.0;
  require(waits(), "the exchange took cache within the cool-down");
  fixture.monotonic += 1.0;
  require(!control.run(MemoryPressure::Normal) && executor.expertChunks.size() == 1,
          "the exchange did not take a chunk back after the cool-down");

  // Under critical pressure.
  available = between;
  require(waits(MemoryPressure::Critical), "the store took a chunk back under critical pressure");
}

// Beside a running lane a control pass exchanges cache for a chunk of the
// expert store as an idle one does: the pass returns the cache nothing in
// use holds to the system, keeps the replay point the lane uses and the
// resume point, and takes the chunk; the lane decodes on to its end, and the
// newest conversation's next turn resumes where it was.
void testControlPassExchangesCacheBesideARunningLane() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // The host lacks the whole chunk beyond the recovery margin.
  const std::optional<uint64_t> available =
      hostReserve + kHostWarningMarginBytes + kHostWarningMarginBytes / 2;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk};
  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  auto running = request(20, 900);
  for (uint32_t &token : running.promptTokens)
    token += 2000;
  executor.decodeFinishes = false;
  require(loop.receive(protocol::peer::serialize(running)), "the long request failed");
  for (uint32_t step = 0; step < 8; ++step)
    static_cast<void>(loop.tick());
  settleCommand(loop);
  require(executor.requests.contains(20) && executor.requests.at(20).resident,
          "the long request did not decode");
  const auto before = resources.snapshot();
  require(!control.run(MemoryPressure::Normal) && executor.expertChunks.size() == 2,
          "a pass beside a running lane did not exchange cache for a chunk of the store");
  const auto after = resources.snapshot();
  require(after.pool.extentReleases > before.pool.extentReleases &&
              after.stateCache.entries < before.stateCache.entries &&
              after.stateCache.inUse == 1 && after.stateCache.inUseEvictions == 0,
          "the exchange beside a running lane did not return unused cache alone");
  executor.decodeFinishes = true;
  runUntilIdle(loop);
  require(loop.statusSnapshot().failed == 0, "the running request failed beside the exchange");
  const uint64_t hits = loop.statusSnapshot().cacheHits;
  finishConversation(fixture, 6);
  require(loop.statusSnapshot().cacheHits == hits + 1,
          "the newest conversation did not resume from the state the exchange kept");
}

// macOS's warning holds back growth that adds to the host's footprint, not
// an exchange that returns the whole chunk first. Under the warning a pass
// takes none of the host's headroom for a chunk of the expert store,
// however much there is: a chunk no cache covers stays out, though a pass
// at normal pressure takes it from that headroom. A chunk the cache covers
// comes back, the cache leaving the engine before the chunk is taken, which
// the governor admits only for those bytes; the pass then gives back the
// cache the warning asks for, the resume point kept.
void testWarningPassExchangesOnlyWhatItReturns() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  const KvPool &pool = fixture.pool;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // More than the whole pool and every cached state's buffers.
  constexpr uint64_t large = 1ULL << 30;
  // The host has the large chunk and more to spare.
  const std::optional<uint64_t> available = hostReserve + (16ULL << 30);
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk, large};
  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  const uint32_t attempts = executor.expertGrowthAttempts;
  require(!control.run(MemoryPressure::Warning) && executor.expertChunks.size() == 1 &&
              executor.expertGrowthAttempts == attempts,
          "a pass under macOS's warning took the host's headroom for a chunk of the store");
  require(!control.run(MemoryPressure::Normal) && executor.expertChunks.size() == 2,
          "a pass at normal pressure did not take the chunk from the host's headroom");

  for (uint64_t id = 7; id <= 12; ++id)
    finishConversation(fixture, id);
  const auto before = resources.snapshot();
  const uint64_t buffers = executor.reclaimedIdleStateBytes;
  require(!control.run(MemoryPressure::Warning) && executor.expertChunks.size() == 3,
          "a pass under macOS's warning did not exchange cache for a chunk the cache covers");
  const auto after = resources.snapshot();
  require(before.pool.allocatedBytes - after.pool.allocatedBytes + executor.reclaimedIdleStateBytes -
                  buffers >= chunk &&
              after.stateCache.entries == 1 && pool.snapshot().reclaimableBytes == kExchangeExtentBytes,
          "the warning pass did not return the chunk's cache and then the cache the warning asks for, "
          "keeping the resume point and the runway extent");
  require(!governor.growthAdmission()(chunk, [] {}),
          "the governor admitted under macOS's warning a chunk the engine returned nothing for");
}

// An idle pass whose cache cannot cover what the host lacks of a chunk of the
// expert store leaves that cache in place: the idle buffers beyond a lane's
// and every entry nothing in use holds return less than the chunk, so the
// chunk stays out, and an exchange that buys no chunk evicts nothing.
void testExpertExchangeKeepsCacheThatCannotCoverTheChunk() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  // More than the whole pool and every cached state's buffers.
  constexpr uint64_t chunk = 1ULL << 30;
  // The host lacks the whole chunk beyond the recovery margin.
  const std::optional<uint64_t> available =
      hostReserve + kHostWarningMarginBytes + kHostWarningMarginBytes / 2;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk};
  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  const auto before = resources.snapshot();
  require(before.stateCache.entries > 1, "the fixture kept no cache beside the resume point");
  const uint32_t attempts = executor.expertGrowthAttempts;
  require(!control.run(MemoryPressure::Normal), "a pass waited for a transfer");
  require(executor.expertChunks.size() == 1 && executor.expertGrowthAttempts == attempts,
          "the store asked for a chunk the cache could not cover");
  const auto after = resources.snapshot();
  require(after.stateCache.entries == before.stateCache.entries &&
              after.pool.allocatedBytes == before.pool.allocatedBytes &&
              after.pool.extentReleases == before.pool.extentReleases,
          "an exchange that bought no chunk evicted cache");
}

// The chunk takes from the engine only the room its limit has free. The
// pass exchanges cache for what the limit lacks of it, as for what the host
// lacks, and the extents the exchange releases make the room. Where both lack some of the chunk, what the exchange returns adds
// to both, so it returns the larger lack alone. Without cache that covers
// the lack the pass neither takes cache nor asks for the chunk.
void testExpertExchangeCoversTheEngineLimit() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t limit = 40ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // More than the whole pool and every cached state's buffers.
  constexpr uint64_t large = 1ULL << 30;
  // The host has the chunk and more to spare.
  constexpr uint64_t ample = hostReserve + (16ULL << 30);
  std::optional<uint64_t> available = ample;
  MemoryGovernor governor(backend, limit, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {large, chunk, chunk};
  // The pool's extents are the backend's memory, which the governor charges.
  fixture.storage.released = [] { test::metalStatistics().allocatedBytes -= kExchangeExtentBytes; };
  // A pass while the engine's limit lacks `lacked` of the next chunk: whether
  // it took the chunk back, and how many extents the cache released for it.
  const auto pass = [&](uint64_t lacked) {
    test::metalStatistics().allocatedBytes = limit - (executor.expertGrowthBytes() - lacked);
    const size_t chunks = executor.expertChunks.size();
    const uint64_t releases = resources.snapshot().pool.extentReleases;
    require(!control.run(MemoryPressure::Normal), "a pass waited for a transfer");
    return std::pair{executor.expertChunks.size() > chunks,
                     resources.snapshot().pool.extentReleases - releases};
  };
  uint32_t attempts = executor.expertGrowthAttempts;
  require(pass(kExchangeExtentBytes) == std::pair{false, uint64_t{0}} &&
              executor.expertGrowthAttempts == attempts,
          "the store asked for a chunk the engine's limit lacked with no cache to make room");

  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);
  const auto [grown, released] = pass(kExchangeExtentBytes);
  require(grown && released != 0,
          "an idle pass did not exchange cache for the room the engine's limit lacked");
  const uint64_t hits = loop.statusSnapshot().cacheHits;
  finishConversation(fixture, 6);
  require(loop.statusSnapshot().cacheHits == hits + 1,
          "the newest conversation did not resume from the state the exchange kept");

  // The host lacks one extent of the chunk beyond the recovery margin, the
  // engine's limit the whole chunk, two extents.
  available = hostReserve + kHostRecoveryMarginBytes + chunk - kExchangeExtentBytes;
  require(pass(chunk) == std::pair{true, uint64_t{2}},
          "an exchange did not return the larger of what the host and the engine's limit lacked");
  available = ample;

  const uint32_t entries = resources.snapshot().stateCache.entries;
  attempts = executor.expertGrowthAttempts;
  require(pass(large) == std::pair{false, uint64_t{0}} && executor.expertGrowthAttempts == attempts &&
              resources.snapshot().stateCache.entries == entries,
          "an exchange that could not cover what the engine's limit lacked took cache");
}

// An exchange whose chunk is refused after all, as by memory another process
// took meanwhile, leaves what it returned with the system and takes no more
// than the chunk lacked: the next pass's sample shows those bytes, and the
// chunk comes back from them without taking cache again.
void testRefusedExchangeLeavesItsBytesToTheNextPass() {
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  LoopFixture fixture(config, kExchangePool);
  engine::NativeRuntime &loop = fixture.loop;
  Executor &executor = fixture.executor;
  const engine::Cache &resources = fixture.cache;
  loop.announceReady();
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  constexpr uint64_t hostReserve = 2ULL << 30;
  constexpr uint64_t chunk = kExchangeChunkBytes;
  // The host has the recovery margin and nothing beyond it: it lacks the
  // whole chunk.
  std::optional<uint64_t> available = hostReserve + kHostRecoveryMarginBytes;
  MemoryGovernor governor(backend, 40ULL << 30, hostReserve, hostReserve,
                          [&available] { return hostMemory(available); }, 0);
  MemoryControl control(governor, backend, loop);
  executor.expertAdmission = governor.growthAdmission();
  executor.expertChunks = {chunk};
  executor.releasedExpertChunks = {chunk};
  // The host refuses the first chunk asked for and admits every later one.
  bool askedBefore = false;
  executor.expertGrowthRoom = [&askedBefore] { return std::exchange(askedBefore, true); };
  executor.expertGrowthRefusal = metal::AllocationFailure::HostPressure;
  for (uint64_t id = 1; id <= 6; ++id)
    finishConversation(fixture, id);

  const auto before = resources.snapshot();
  const uint64_t buffers = executor.reclaimedIdleStateBytes;
  const uint32_t attempts = executor.expertGrowthAttempts;
  require(!control.run(MemoryPressure::Normal), "a pass waited for a transfer");
  const auto after = resources.snapshot();
  const uint64_t returned = before.pool.allocatedBytes - after.pool.allocatedBytes +
                            executor.reclaimedIdleStateBytes - buffers;
  require(executor.expertChunks.size() == 1 && executor.expertGrowthAttempts == attempts + 1 &&
              returned >= chunk && after.stateCache.entries > 1,
          "a refused exchange did not return the chunk's bytes alone");

  available = *available + returned;
  require(!control.run(MemoryPressure::Normal) && executor.expertChunks.size() == 2 &&
              resources.snapshot().pool.extentReleases == after.pool.extentReleases &&
              resources.snapshot().stateCache.entries == after.stateCache.entries,
          "the next pass took cache again for the bytes the refused exchange returned");
}

} // namespace

std::span<const test::TestCase> memoryTests() {
  static constexpr test::TestCase kTests[] = {
      {"testControlPassReclaimsUnderHostPressure", testControlPassReclaimsUnderHostPressure},
      {"testControlPassGivesCacheBackUnderSystemWarning", testControlPassGivesCacheBackUnderSystemWarning},
      {"testServingMarkRestoresTheMarkItFound", testServingMarkRestoresTheMarkItFound},
      {"testControlPassHoldsGrowthForAWaitingRequest", testControlPassHoldsGrowthForAWaitingRequest},
      {"testControlPassGrowsTheExpertStoreWhenMemoryIsAmple", testControlPassGrowsTheExpertStoreWhenMemoryIsAmple},
      {"testIdleControlPassExchangesCacheForTheExpertStore", testIdleControlPassExchangesCacheForTheExpertStore},
      {"testExpertExchangeWaitsWhereGrowthDoes", testExpertExchangeWaitsWhereGrowthDoes},
      {"testControlPassExchangesCacheBesideARunningLane", testControlPassExchangesCacheBesideARunningLane},
      {"testWarningPassExchangesOnlyWhatItReturns", testWarningPassExchangesOnlyWhatItReturns},
      {"testExpertExchangeKeepsCacheThatCannotCoverTheChunk", testExpertExchangeKeepsCacheThatCannotCoverTheChunk},
      {"testExpertExchangeCoversTheEngineLimit", testExpertExchangeCoversTheEngineLimit},
      {"testRefusedExchangeLeavesItsBytesToTheNextPass", testRefusedExchangeLeavesItsBytesToTheNextPass},
      {"testMemoryStatusReporterLogsTransitionsOnly", testMemoryStatusReporterLogsTransitionsOnly},
  };
  return kTests;
}
