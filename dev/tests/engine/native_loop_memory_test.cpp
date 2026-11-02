#include "FakeElasticPool.hpp"
#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestExecutor.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "engine/Cache.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"

#include <algorithm>
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

// The store's account in the governor's snapshot.
MemoryAccountSnapshot storeAccount(const MemoryGovernor &governor) {
  const MemoryGovernorSnapshot snapshot = governor.snapshot();
  const auto found = std::ranges::find(snapshot.accounts, std::string("experts"), &MemoryAccountSnapshot::name);
  require(found != snapshot.accounts.end(), "the store has no account");
  return *found;
}

// One control pass at the fixture's clock `elapsed` milliseconds on, at
// FakeMemory's pressure then.
void controlPassAfter(LoopFixture &fixture, double elapsed) {
  fixture.monotonic += elapsed;
  fixture.memory->advanceTo(fixture.monotonic);
  static_cast<void>(fixture.control->run(fixture.memory->pressure()));
}

// A store that came back at its floor, as the weights' restore leaves it,
// grows back to its cap through the control pass's growth into free room,
// a unit at a time through the ledger, while the host admits it (DESIGN
// section 7.8 M3.6, amend/I.md A12): nothing else grows it.
void testRestoreBringsTheFloorThenGrowthTheRest() {
  LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                      test::LoopOptions{.memory = test::FakeMemory::Config{}});
  constexpr uint64_t unit = 1ULL << 30;
  test::FakeElasticPool store("experts", *fixture.memory, unit, 1, 4, 8, 1, 1.0, 0.0);
  static_cast<void>(fixture.governor->ledger().addPool(store));
  for (uint32_t pass = 0; pass < 4 && store.state().currentBytes < store.state().capBytes; ++pass)
    controlPassAfter(fixture, 500.0);
  const MemoryAccountSnapshot account = storeAccount(*fixture.governor);
  require(store.state().currentBytes == 4 * unit && account.grows == 3 && account.shrinks == 0,
          "the control pass did not grow the store back from its floor to its cap a unit at a time");
}

// Critical pressure drops the store to its floor; once the pressure is over,
// the store takes the free room back within the passes that follow, with no
// cool-down and no exchange (DESIGN section 7.8 M3.4).
void testStoreRetakesFreeRoomAfterCriticalPressure() {
  LoopFixture fixture({}, {}, {}, metal::kResidencyKeepAliveSeconds,
                      test::LoopOptions{.memory = test::FakeMemory::Config{
                                            .script = {{1'000.0, 8ULL << 30, MemoryPressure::Critical},
                                                       {2'000.0, 8ULL << 30, MemoryPressure::Normal}}}});
  constexpr uint64_t unit = 1ULL << 30;
  test::FakeElasticPool store("experts", *fixture.memory, unit, 1, 4, 8, 4, 1.0, 0.0);
  static_cast<void>(fixture.governor->ledger().addPool(store));
  controlPassAfter(fixture, 1'000.0);
  require(store.state().currentBytes == unit, "critical pressure did not drop the store to its floor");
  controlPassAfter(fixture, 1'000.0);
  controlPassAfter(fixture, 500.0);
  const MemoryAccountSnapshot account = storeAccount(*fixture.governor);
  require(store.state().currentBytes == 4 * unit && account.grows == 3,
          "the store did not take the free room back once the pressure was over");
}

} // namespace

std::span<const test::TestCase> memoryTests() {
  static constexpr test::TestCase kTests[] = {
      {"testServingMarkRestoresTheMarkItFound", testServingMarkRestoresTheMarkItFound},
      {"testMemoryStatusReporterLogsTransitionsOnly", testMemoryStatusReporterLogsTransitionsOnly},
      {"testRestoreBringsTheFloorThenGrowthTheRest", testRestoreBringsTheFloorThenGrowthTheRest},
      {"testStoreRetakesFreeRoomAfterCriticalPressure", testStoreRetakesFreeRoomAfterCriticalPressure},
  };
  return kTests;
}
