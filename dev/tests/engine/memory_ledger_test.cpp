// The memory ledger and the governor's rule over it: accounts, claims and
// the room they hold, the one host rule, free(), available(), growth and the
// control pass, on FakeMemory, FakeElasticPool and victim sources of the
// test's own.

#include "FakeElasticPool.hpp"
#include "FakeMemory.hpp"
#include "TestChecks.hpp"
#include "TestMetalMemory.hpp"
#include "TestRegistry.hpp"
#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"
#include "engine/MemoryValue.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace splash;
using namespace splash::engine;

std::span<const test::TestCase> valueTests();
std::string &recordedLoadsDirectory();

namespace {

using test::FakeElasticPool;
using test::FakeMemory;
using test::HostEvent;
using test::require;

constexpr uint64_t kGiB = 1ULL << 30;
constexpr uint64_t kMiB = 1ULL << 20;

// An allocation the test expects granted.
void require(const metal::AllocationResult &granted, std::string_view message) {
  test::require(static_cast<bool>(granted), message);
}

// Units in the order the test lists them. A unit that frees memory now gives
// its bytes back to FakeMemory's holder when taken; one that lands later
// does when the test lands it.
class TestSource final : public VictimSource {
public:
  explicit TestSource(std::string name, FakeMemory *memory = nullptr) : name_(std::move(name)), memory_(memory) {}

  [[nodiscard]] std::string_view name() const noexcept override { return name_; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    ++peeks;
    auto unit = units.begin();
    if (after)
      unit = std::next(std::ranges::find(units, after->key, &Victim::key));
    return unit == units.end() ? std::nullopt : std::optional<Victim>(*unit);
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    const auto unit = std::ranges::find(units, victim.key, &Victim::key);
    if (unit == units.end())
      throw std::logic_error("took a unit the source does not hold");
    taken.push_back(unit->key);
    const Victim gone = *unit;
    units.erase(unit);
    if (gone.later) {
      landing.push_back(gone);
      return {.laterBytes = gone.bytes};
    }
    const uint64_t bytes = gone.sized ? gone.bytes : unsizedBytes;
    if (memory_ && bytes)
      memory_->release(name_, bytes);
    return {.nowBytes = bytes};
  }
  // The units taken to land later free their memory. With
  // landsIntoExtents, as the cache's demotions do, a landing only empties
  // the page's extent: a unit of its own that costs nothing, which a later
  // take releases.
  void land() {
    for (const Victim &unit : landing) {
      if (landsIntoExtents)
        units.push_back(Victim{.key = 1000 + unit.key, .bytes = unit.bytes});
      else if (memory_ && unit.bytes)
        memory_->release(name_, unit.bytes);
    }
    landing.clear();
  }

  std::vector<Victim> units;
  std::vector<uint64_t> taken;
  std::vector<Victim> landing;
  uint64_t unsizedBytes = 0;
  bool landsIntoExtents = false;
  mutable uint32_t peeks = 0;

private:
  std::string name_;
  FakeMemory *memory_ = nullptr;
};

// A pool whose units above its floor are listed, the last given back first,
// the fewest whose bytes reach what release() asks, as the expert store
// gives its last live chunk back, whatever its size (ExpertStorePool). It
// reports what the store reports of its keys: the bytes that would hold all
// of them, a key's bytes and the bytes its prefill scans read, which the
// test moves with its hits.
class ListedPool final : public model::ElasticPool {
public:
  ListedPool(uint64_t floorBytes, std::vector<uint64_t> units, uint64_t wholeBytes, uint64_t keyBytes,
             double missMilliseconds)
      : floorBytes_(floorBytes), units_(std::move(units)), wholeBytes_(wholeBytes), keyBytes_(keyBytes),
        missMilliseconds_(missMilliseconds) {}

  [[nodiscard]] model::PoolState state() const override {
    const uint64_t current = currentBytes();
    return {.name = "experts",
            .floorBytes = floorBytes_,
            .currentBytes = current,
            .capBytes = current,
            .releaseUnitBytes = units_.empty() ? 0 : units_.back(),
            .wholeBytes = wholeBytes_};
  }
  [[nodiscard]] model::PoolValue value() const override {
    return {.hits = hits, .missMilliseconds = missMilliseconds_, .keyBytes = keyBytes_,
            .scanReadBytes = scanReadBytes, .nearGhostHits = nearGhostHits, .tailHits = tailHits,
            .marginBytes = marginBytes};
  }
  [[nodiscard]] uint64_t release(uint64_t bytes) override {
    uint64_t released = 0;
    while (released < bytes && !units_.empty()) {
      released += units_.back();
      units_.pop_back();
    }
    return released;
  }
  [[nodiscard]] metal::AllocationResult grow(const metal::AllocationAdmission &) override { return {}; }

  [[nodiscard]] uint64_t currentBytes() const noexcept {
    uint64_t bytes = floorBytes_;
    for (const uint64_t unit : units_)
      bytes += unit;
    return bytes;
  }

  uint64_t hits = 0, scanReadBytes = 0;
  // The marginal counts and the bytes they count in, an elastic chunk's.
  uint64_t nearGhostHits = 0, tailHits = 0, marginBytes = 0;

private:
  uint64_t floorBytes_;
  std::vector<uint64_t> units_;
  uint64_t wholeBytes_, keyBytes_;
  double missMilliseconds_;
};

Victim unit(uint64_t key, uint64_t bytes, double onceSeconds, bool later = false) {
  return {.key = key, .bytes = bytes, .later = later, .cost = {.onceSeconds = onceSeconds}};
}

// Lanes a claim may convert, each listed while its pages fit the quota left,
// each lane alone: as EngineVictims listed them, a conversion's commitment
// lowering the quota the others are listed against.
class QuotaSource final : public VictimSource {
public:
  struct Lane final {
    uint64_t id = 0;
    uint64_t pages = 0;
  };
  [[nodiscard]] std::string_view name() const noexcept override { return "engine"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    auto lane = lanes.begin();
    if (after)
      lane = std::next(std::ranges::find(lanes, after->key, &Lane::id));
    for (; lane != lanes.end(); ++lane)
      if (std::ranges::find(converted, lane->id) == converted.end() && lane->pages <= quota)
        return Victim{.key = lane->id,
                      .bytes = kGiB,
                      .later = true,
                      .cost = {.onceSeconds = 1.0},
                      .kind = VictimKind::Conversion};
    return std::nullopt;
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    const auto lane = std::ranges::find(lanes, victim.key, &Lane::id);
    converted.push_back(lane->id);
    quota -= lane->pages;
    return {FreeOutcome::Pending, 0, victim.bytes};
  }

  std::vector<Lane> lanes;
  uint64_t quota = 0;
  std::vector<uint64_t> converted;
};

// A host of 64 GiB whose headroom, its spare memory above the governor's
// 2 GiB spare reserve, is `headroom` while the engine holds nothing.
FakeMemory::Config host(uint64_t headroom, MemoryPressure pressure = MemoryPressure::Normal,
                        std::vector<HostEvent> later = {}) {
  std::vector<HostEvent> script{{0.0, 62 * kGiB - headroom, pressure}};
  script.insert(script.end(), later.begin(), later.end());
  return {.hostBytes = 64 * kGiB, .otherBytes = 62 * kGiB - headroom, .script = std::move(script)};
}
HostEvent at(double milliseconds, uint64_t headroom, MemoryPressure pressure = MemoryPressure::Normal) {
  return {milliseconds, 62 * kGiB - headroom, pressure};
}

// The governor over FakeMemory's host, told macOS's pressure as the control
// pass tells it.
struct Fixture final {
  explicit Fixture(FakeMemory::Config config, uint64_t limit = 40 * kGiB)
      : memory(std::move(config)), governor(test::governorOver(backend, memory, {.limitBytes = limit})) {
    governor->setPressure(memory.pressure());
  }
  MemoryLedger &ledger() { return governor->ledger(); }
  // An allocation of `holder` for `claim`, in its scope.
  metal::AllocationResult allocate(ClaimId claim, uint64_t bytes, std::string_view holder = "kv") {
    governor->setPressure(memory.pressure());
    const MemoryGovernor::Scope scope(*governor, claim);
    return memory.allocate(governor->allocationAdmission(), holder, bytes);
  }
  // Runs the control pass at `seconds`, observing the pools first as the
  // engine does each pass.
  bool pass(double seconds, bool commandsRan = true) {
    memory.advanceTo(seconds * 1000.0);
    governor->setPressure(memory.pressure());
    ledger().observePools(seconds, commandsRan);
    return governor->controlPass(seconds * 1000.0);
  }
  // The allocation as the engine makes it: refused, free() what it lacks
  // and try again, while free() meets it.
  metal::AllocationResult allocateFreeing(ClaimId claim, uint64_t bytes) {
    metal::AllocationResult result = allocate(claim, bytes);
    for (int attempt = 0; !result && attempt < 3; ++attempt) {
      if (governor->free(claim, governor->need(claim)).outcome != FreeOutcome::Met)
        break;
      result = allocate(claim, bytes);
    }
    return result;
  }

  FakeMemory memory;
  metal::MetalBackend backend{""};
  std::optional<MemoryGovernor> governor;
};

ClaimSpec request(uint64_t id, RequestPriority priority, ClaimClass cls = ClaimClass::InService,
                  double holdSeconds = 0.0) {
  return {.cls = cls,
          .requestId = id,
          .rank = {.priority = static_cast<uint8_t>(2 - static_cast<uint8_t>(priority)),
                   .earliness = std::numeric_limits<uint64_t>::max() - id},
          .holdSeconds = holdSeconds,
          .priority = priority};
}

// Leaves a request's claim waiting for `need`, which its own lane's owed
// copies meet and no other claim counts: the free() it waits on holds it.
void waitForItsCopies(MemoryLedger &ledger, ClaimId claim, const Need &need) {
  const uint64_t id = ledger.claim(claim).spec.requestId;
  Returning own{.owedBytes = need.bytes};
  own.owedPages[static_cast<size_t>(need.form)] = need.pages;
  ledger.setReturning([] { return Returning{}; }, [=](std::span<const uint64_t> lanes) {
    return std::ranges::find(lanes, id) != lanes.end() ? own : Returning{};
  });
  require(ledger.free(claim, need, 0, false, false).outcome == FreeOutcome::Pending && ledger.claim(claim).waiting,
          "a claim its lane's owed copies cover did not wait");
}

// A pool's decode value: `ghostsPerSecond` ghost hits for `seconds`, the
// pool observed every 500 ms with commands running.
void serve(MemoryLedger &ledger, FakeElasticPool &pool, double ghostsPerSecond, double from, double seconds) {
  for (double t = from; t < from + seconds; t += 0.5) {
    pool.served(0, static_cast<uint64_t>(std::llround(ghostsPerSecond * 0.5)), 0, 0);
    ledger.observePools(t + 0.5, true);
  }
}

uint64_t accountCurrent(const MemoryGovernorSnapshot &snapshot, std::string_view name) {
  for (const MemoryAccountSnapshot &account : snapshot.accounts)
    if (account.name == name)
      return account.currentBytes;
  throw std::runtime_error("no account " + std::string(name));
}

// L1: over seeded sequences of allocations and releases on two pulled
// accounts, a pool and memory no account owns, the accounts sum to the
// charged bytes, the fixed account holding the rest.
void testAccountsSumToCharged() {
  for (uint64_t seed = 1; seed <= 200; ++seed) {
    std::mt19937_64 random(seed);
    Fixture fixture(host(32 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, 256 * kMiB, 2, 8, 16, 4, 0.2, 60.0);
    fixture.ledger().addAccount("kv", AccountRole::Kv, [&] { return fixture.memory.held("kv"); });
    fixture.ledger().addAccount("states", AccountRole::States, [&] { return fixture.memory.held("states"); });
    fixture.ledger().addPool(experts);
    for (int step = 0; step < 60; ++step) {
      const char *holders[] = {"kv", "states", "weights"};
      const char *holder = holders[random() % 3];
      const uint64_t bytes = (1 + random() % 64) * kMiB;
      switch (random() % 4) {
      case 0:
        fixture.memory.hold(holder, bytes);
        break;
      case 1:
        if (fixture.memory.held(holder) >= bytes)
          fixture.memory.release(holder, bytes);
        break;
      case 2:
        static_cast<void>(experts.release(1));
        break;
      default: {
        // The pool grows inside a growth claim's scope, as the ledger grows it.
        const ClaimId growth = fixture.ledger().open({.cls = ClaimClass::Growth});
        {
          const MemoryGovernor::Scope scope(*fixture.governor, growth);
          static_cast<void>(experts.grow(fixture.governor->allocationAdmission()));
        }
        fixture.ledger().close(growth);
      }
      }
      const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
      uint64_t sum = 0;
      for (const MemoryAccountSnapshot &account : snapshot.accounts)
        sum += account.currentBytes;
      require(sum == snapshot.chargedBytes, "seed " + std::to_string(seed) + ": the accounts do not sum to charged");
      require(accountCurrent(snapshot, "fixed") == fixture.memory.held("weights") &&
                  accountCurrent(snapshot, "experts") == fixture.memory.held("experts") &&
                  accountCurrent(snapshot, "kv") == fixture.memory.held("kv"),
              "an account does not pull its owner's bytes");
    }
  }
}

// Each account renders its floor, current, cap, unit, reserved bytes, value
// per GiB and moves; owed bytes come from what is returning, claims from the
// waiting ones.
void testSnapshotRendersEveryAccount() {
  Fixture fixture(host(32 * kGiB));
  fixture.memory.hold("kv", 3 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 4, 3, 0.25, 60.0);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addAccount("kv", AccountRole::Kv, [&] { return fixture.memory.held("kv"); });
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.landingBytes = 3 * kMiB, .owedBytes = 5 * kMiB, .transfers = 2,
                                            .oldestMilliseconds = 40.0}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  serve(ledger, experts, 400.0, 0.0, 10.0);
  const ClaimId waiting = ledger.open(request(7, RequestPriority::Normal));
  require(ledger.free(waiting, {.bytes = 2 * kMiB}, 0, false, false).outcome == FreeOutcome::Pending,
          "a claim the landings cover did not wait");
  ledger.observePools(14.0, false);
  const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
  require(snapshot.accounts.size() == 3 && snapshot.accounts[0].name == "fixed" && snapshot.accounts[1].name == "kv" &&
              snapshot.accounts[2].name == "experts",
          "the accounts are not fixed, kv and the pool in order");
  const MemoryAccountSnapshot &pool = snapshot.accounts[2];
  // 400 ghost hits a second at 0.25 ms over a span of one unit: 0.1 s a
  // second, 6 s over the 60 s horizon for 1 GiB.
  require(pool.floorBytes == 2 * kGiB && pool.currentBytes == 3 * kGiB && pool.capBytes == 4 * kGiB &&
              pool.unitBytes == kGiB && std::abs(pool.valuePerGiB - 6.0) < 1e-9,
          "the pool's row does not render its state and value");
  require(snapshot.owed.transfers == 2 && snapshot.owed.bytes == 8 * kMiB && snapshot.owed.toClaimsBytes == 2 * kMiB &&
              snapshot.owed.oldestMilliseconds == 40.0,
          "owed bytes do not render what is returning");
  require(snapshot.claims.waiting == 1 && snapshot.claims.bytes == 2 * kMiB &&
              std::abs(snapshot.claims.oldestMilliseconds - 4000.0) < 1e-6,
          "the waiting claim does not render");
}

// A waiting claim holds what it lacks against every claim ranked below it:
// a lower priority, or at its priority a later submission. Each waits for
// what lands.
void testWaitingClaimHoldsRoomForItsRank() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.setReturning([] { return Returning{.landingBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const auto waits = [&](ClaimId claim) {
    return fixture.governor->free(claim, fixture.governor->need(claim)).outcome == FreeOutcome::Pending;
  };
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(!fixture.allocate(high, 2 * kGiB), "2 GiB fit in 1 GiB of room");
  require(waits(high), "a claim the landings cover did not wait");
  require(fixture.allocate(low, 512 * kMiB).failure == metal::AllocationFailure::EngineBudget,
          "a lower-ranked claim took room a waiting claim lacks");
  require(fixture.allocate(high, 512 * kMiB), "the waiting claim was refused its own room");
  const ClaimId earlier = ledger.open(request(3, RequestPriority::Normal));
  const ClaimId later = ledger.open(request(4, RequestPriority::Normal));
  ledger.setWaiting(high, false);
  require(!fixture.allocate(earlier, 2 * kGiB), "2 GiB fit in the room left");
  require(waits(earlier), "a claim the landings cover did not wait");
  require(!fixture.allocate(later, 256 * kMiB), "a later submission took room an earlier one waits for");
  require(fixture.allocate(earlier, 256 * kMiB), "the earlier submission was refused room");
}

// A claim that does not wait holds nothing.
void testShortClaimHoldsNothing() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  const ClaimId high = fixture.ledger().open(request(1, RequestPriority::Foreground));
  const ClaimId low = fixture.ledger().open(request(2, RequestPriority::Background));
  require(!fixture.allocate(high, 2 * kGiB), "2 GiB fit in 1 GiB of room");
  require(fixture.allocate(low, 512 * kMiB), "a claim that does not wait held room");
}

// A scope restores the one it replaced: refusals go to the innermost claim,
// and with none open admission is as before scopes.
void testNestedScopesRestore() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  const ClaimId outer = ledger.open(request(1, RequestPriority::Normal));
  const ClaimId inner = ledger.open(request(2, RequestPriority::Normal));
  {
    const MemoryGovernor::Scope a(*fixture.governor, outer);
    {
      const MemoryGovernor::Scope b(*fixture.governor, inner);
      require(!fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 2 * kGiB), "2 GiB fit");
    }
    require(!fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 3 * kGiB), "3 GiB fit");
  }
  require(ledger.claim(inner).refusal.bytes == 2 * kGiB && ledger.claim(outer).refusal.bytes == 3 * kGiB,
          "a refusal was recorded on the wrong claim");
  // Once they end, an admission is for the claim whose scope is open then: a
  // claim ranked above the one that waits, for what lands, is not held back
  // by it.
  ledger.setReturning([] { return Returning{.landingBytes = 3 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  require(fixture.governor->free(outer, fixture.governor->need(outer)).outcome == FreeOutcome::Pending,
          "a claim the landings cover did not wait");
  const ClaimId above = ledger.open(request(3, RequestPriority::Foreground));
  {
    const MemoryGovernor::Scope scope(*fixture.governor, above);
    require(fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 512 * kMiB),
            "an admission of a claim ranked above the waiting one was held back by it");
  }
  ledger.close(above);
}

// KV pages a waiting claim lacks are held, by form, against claims ranked
// below it.
void testHeldPagesFollowRank() {
  MemoryLedger ledger;
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId mid = ledger.open(request(2, RequestPriority::Normal));
  const ClaimId low = ledger.open(request(3, RequestPriority::Background));
  TestSource none("cache");
  ledger.addSource(none);
  waitForItsCopies(ledger, high, {.pages = 6, .form = kv::PageForm::Index});
  waitForItsCopies(ledger, mid, {.pages = 4});
  const std::array<uint32_t, 3> forLow = ledger.heldPages(low);
  const std::array<uint32_t, 3> forMid = ledger.heldPages(mid);
  require(forLow[static_cast<size_t>(kv::PageForm::Index)] == 6 && forLow[static_cast<size_t>(kv::PageForm::Full)] == 4,
          "a lower-ranked claim is not held the pages of both claims above it");
  require(forMid[static_cast<size_t>(kv::PageForm::Index)] == 6 && forMid[static_cast<size_t>(kv::PageForm::Full)] == 0,
          "a claim is held its own pages or a lower claim's");
}

// The bytes a waiting claim lacks are held, in each form's pages, against
// the free pages a claim ranked below it is handed: 10,000 bytes are three
// Full pages of 4096 bytes and four Slim pages of 3000; a form without pages
// holds none. A claim of its rank or above it is held nothing, nor is any
// claim by one that free() found short.
void testWaitingBytesHoldPagesOfEachForm() {
  MemoryLedger ledger;
  ledger.setPageBytes({4096, 0, 3000});
  const ClaimId waiting = ledger.open(request(5, RequestPriority::Normal));
  const ClaimId shortClaim = ledger.open(request(4, RequestPriority::Normal));
  const ClaimId same = ledger.open(request(5, RequestPriority::Normal));
  const ClaimId above = ledger.open(request(6, RequestPriority::Foreground));
  const ClaimId below = ledger.open(request(7, RequestPriority::Background));
  require(ledger.free(shortClaim, {.bytes = 20'000}, 0, false, false).outcome == FreeOutcome::Short,
          "a claim nothing covers did not find free() short");
  waitForItsCopies(ledger, waiting, {.bytes = 10'000});
  require(ledger.heldPages(below) == std::array<uint32_t, 3>{3, 0, 4},
          "a lower-ranked claim was not held the pages of the bytes a claim above it waits for");
  require(ledger.heldPages(same) == std::array<uint32_t, 3>{} && ledger.heldPages(above) == std::array<uint32_t, 3>{},
          "a claim of the waiting claim's rank or above it was held its pages");
}

// A request's wait spans its allocations: a waiting claim whose free() for
// another of them is met keeps waiting and holds the need it waits for;
// one free() finds short waits for nothing.
void testAMetNeedKeepsTheWaitAShortOneEndsIt() {
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim leaf = unit(1, 0, 0.2);
  leaf.pages = 4;
  cache.units = {leaf};
  ledger.addSource(cache);
  const ClaimId waiting = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId below = ledger.open(request(2, RequestPriority::Background));
  waitForItsCopies(ledger, waiting, {.bytes = kGiB});
  require(ledger.free(waiting, {.pages = 4}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "the leaf did not meet the claim's need of pages");
  require(ledger.claim(waiting).waiting && ledger.claim(waiting).need.bytes == kGiB &&
              ledger.heldBytes(below, std::nullopt) == kGiB,
          "a claim whose free() for another need was met stopped waiting for its copies");
  require(ledger.free(waiting, {.bytes = 2 * kGiB}, 0, false, false).outcome == FreeOutcome::Short &&
              !ledger.claim(waiting).waiting && ledger.heldBytes(below, std::nullopt) == 0,
          "a claim free() found short still waits");
}

// A pool whose displacement waits for landings holds the unit as pending
// growth: another account is refused it, the pool's own growth takes it once
// the victims land (V's M-P2 at the unit level).
void testPendingGrowthIsReservedAgainstOthers() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 2 * kGiB);
  fixture.memory.hold("weights", 2 * kGiB);
  cache.units = {unit(1, kGiB, 0.5, true), unit(2, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  const AccountId pool = ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  // A room of 256 MiB, short of a unit: the displacement demotes a cached
  // unit, whose bytes land later.
  fixture.memory.hold("weights", 768 * kMiB);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken.size() == 1 && experts.units() == 3, "the displacement did not demote one unit");
  const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
  require(snapshot.accounts[static_cast<size_t>(pool.value)].reservedBytes == kGiB && snapshot.reservedBytes == kGiB,
          "the pool's growth was not reserved");
  cache.land();
  const ClaimId other = ledger.open(request(9, RequestPriority::Foreground));
  require(fixture.allocate(other, 512 * kMiB).failure == metal::AllocationFailure::EngineBudget,
          "another account took the pool's pending growth");
  static_cast<void>(fixture.pass(11.0));
  require(experts.units() == 4 && fixture.governor->snapshot().reservedBytes == 0,
          "the pool did not grow into its pending growth once the victims landed");
}

// Pending growth keeps its bytes until it grows, on the host's credit of
// what its victims freed (amend/M.md B4): the host consents to a store unit
// and the displacement demotes 1 GiB of cache for it; other applications
// then take 612 MiB, so that the host would not admit the unit on its own.
// While other transfers in flight could still hold the victims' bytes, the
// growth waits, pending; once they land, it releases the extent its
// demotion emptied and grows on what that gave back, the engine's
// footprint back where it was.
void testPendingGrowthGrowsOnItsLandedVictims() {
  // 2148 MiB of headroom beside the engine's 7.5 GiB, then 1.5 GiB.
  Fixture fixture(host(2148 * kMiB + 7680 * kMiB, MemoryPressure::Normal, {at(11000.0, 1536 * kMiB + 7680 * kMiB)}),
                  8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  cache.landsIntoExtents = true;
  fixture.memory.hold("cache", kGiB);
  fixture.memory.hold("weights", 3584 * kMiB);
  cache.units = {unit(1, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  Returning inFlight{.landingBytes = kGiB};
  ledger.setReturning([&] { return inFlight; }, [](std::span<const uint64_t>) { return Returning{}; });
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken.size() == 1 && fixture.governor->snapshot().reservedBytes == kGiB,
          "the displacement did not pend on its demotion");
  cache.land();
  static_cast<void>(fixture.pass(11.5));
  require(experts.units() == 3 && fixture.governor->snapshot().reservedBytes == kGiB,
          "growth the host refused gave up its pending bytes");
  inFlight = {};
  static_cast<void>(fixture.pass(12.0));
  require(experts.units() == 4 && fixture.governor->snapshot().reservedBytes == 0 &&
              fixture.memory.total() == 7680 * kMiB,
          "the pending growth did not grow on its landed victims' credit");
}

// Pending growth without the host's consent grows only on what its own
// take gives the host back (amend/M.md B4: its footprint where it was): its
// demotion lands into a kept extent, which a warning's host pass, short of
// its margin, takes first at no cost; the growth's take then finds nothing
// else that frees memory now within the unit's worth, and the store stays
// as it was, the host's footprint where the pass left it.
void testPendingGrowthCreditsOnlyWhatItsTakeReleases() {
  // 3 GiB of headroom beside the engine's 7.5 GiB, then 512 MiB under
  // macOS's warning.
  Fixture fixture(host(3 * kGiB + 7680 * kMiB, MemoryPressure::Normal,
                       {at(11000.0, 512 * kMiB + 7680 * kMiB, MemoryPressure::Warning)}),
                  8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  cache.landsIntoExtents = true;
  fixture.memory.hold("cache", kGiB);
  fixture.memory.hold("weights", 3584 * kMiB);
  cache.units = {unit(1, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken == std::vector<uint64_t>{1} && fixture.governor->snapshot().reservedBytes == kGiB,
          "the displacement did not pend on its demotion");
  cache.land();
  static_cast<void>(fixture.pass(11.5));
  require(cache.taken == std::vector<uint64_t>{1, 1001} && experts.units() == 3 &&
              fixture.memory.total() == 6656 * kMiB,
          "the pending growth grew on the extent the host's pass gave back");
}

// Pending growth whose take finds what it still lacks worth more than the
// unit beyond its cycle holds nothing (R3, amend/M.md B4: growth displaces
// only below that worth): the lanes took the pages its demotion freed, and
// the cache left costs 9 s against the unit's 6 s. A start then fits the
// room the pending growth held.
void testAShortPendingGrowthHoldsNothing() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 2 * kGiB);
  fixture.memory.hold("weights", 2560 * kMiB);
  cache.units = {unit(1, kGiB, 0.5, true), unit(2, kGiB, 9.0)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken == std::vector<uint64_t>{1} && fixture.governor->snapshot().reservedBytes == kGiB,
          "the displacement did not pend on its demotion");
  // Its page landed, and a lane took it.
  cache.landing.clear();
  static_cast<void>(fixture.pass(11.0));
  const ClaimId start = ledger.open(request(9, RequestPriority::Normal, ClaimClass::Start));
  require(fixture.allocate(start, 256 * kMiB), "a start was refused the room a pending growth found short held");
  require(fixture.governor->snapshot().reservedBytes == 0 && experts.units() == 3 &&
              cache.taken == std::vector<uint64_t>{1},
          "a pending growth found short took something or kept its bytes");
}

// Pending growth takes what its victims leave once they land: a demotion's
// landing only empties its page's extent, which the cache releases when a
// claim takes it (Cache.cpp's landing drops the page into its extent), so the
// growth frees that extent for itself at a later pass and grows into it.
void testPendingGrowthTakesWhatItsLandedVictimsLeave() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  cache.landsIntoExtents = true;
  fixture.memory.hold("cache", 2 * kGiB);
  fixture.memory.hold("weights", 2816 * kMiB);
  cache.units = {unit(1, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken == std::vector<uint64_t>{1} && fixture.governor->snapshot().reservedBytes == kGiB,
          "the displacement did not pend on its demotion");
  cache.land();
  for (double seconds = 11.0; seconds <= 13.0; seconds += 0.5)
    static_cast<void>(fixture.pass(seconds));
  require(experts.units() == 4 && fixture.governor->snapshot().reservedBytes == 0 &&
              cache.taken == std::vector<uint64_t>{1, 1001},
          "the pending growth did not take the extent its landed victim emptied and grow into it");
}

// Without the host's consent a displacement takes units that free memory
// only once their copies land too, as pending growth (amend/M.md B4): in
// the M6's band (1.1 GiB of headroom) a store a unit short of its cap,
// beside uncopied cache, regrows once the cache's copies land, on the
// extent its take releases (its credit), rather than staying short while
// no claim asks for copies.
void testDisplacementWithoutTheHostsConsentPendsOnUncopiedCache() {
  Fixture fixture(host(1100 * kMiB + 8 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  cache.landsIntoExtents = true;
  fixture.memory.hold("cache", kGiB);
  fixture.memory.hold("weights", 4 * kGiB);
  cache.units = {unit(1, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(cache.taken == std::vector<uint64_t>{1} && fixture.governor->snapshot().reservedBytes == kGiB,
          "a displacement the host did not consent to took no cache whose copies land later");
  cache.land();
  static_cast<void>(fixture.pass(11.5));
  require(experts.units() == 4 && fixture.governor->snapshot().reservedBytes == 0 &&
              cache.taken == std::vector<uint64_t>{1, 1001},
          "the store did not regrow in the band on the extent its landed victim emptied");
}

// A claim that takes a pool's pending growth cancels it: the pool keeps its
// units and the room is the claim's.
void testTakenPendingGrowthIsCancelled() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", kGiB);
  fixture.memory.hold("weights", 3584 * kMiB);
  cache.units = {unit(1, kGiB, 0.5, true)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(fixture.governor->snapshot().reservedBytes == kGiB, "no pending growth to take");
  cache.land();
  const ClaimId claim = ledger.open(request(9, RequestPriority::Foreground));
  require(!fixture.allocate(claim, kGiB), "the claim fit beside the pending growth");
  const FreeResult freed = fixture.governor->free(claim, fixture.governor->need(claim));
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == kGiB && experts.units() == 3 &&
              experts.shrinks == 0 && fixture.governor->snapshot().reservedBytes == 0,
          "taking the pending growth did not cancel it and leave the pool as it was");
  require(fixture.allocate(claim, kGiB), "the room the pending growth held was not the claim's");
}

// One host rule for every class (V's M-P7): the floor passes; in-service
// work passes with the margin at normal pressure, and once exempt; starts,
// optional work and growth need the margin beyond their bytes outside the
// band (1 GiB; the base asked 2 GiB of optional growth); the band begins
// below the margin and ends past it by the largest growth unit; macOS's
// warning refuses elective work, its critical pressure everything.
void testOneHostRule() {
  const auto admits = [](uint64_t headroom, MemoryPressure pressure, ClaimClass cls, uint64_t bytes,
                         bool exempt = false) {
    Fixture fixture(host(headroom, pressure));
    const ClaimId claim = fixture.ledger().open({.cls = cls});
    if (exempt)
      fixture.ledger().setHostExempt(claim, true);
    return fixture.allocate(claim, bytes);
  };
  const MemoryPressure normal = MemoryPressure::Normal, warning = MemoryPressure::Warning,
                       critical = MemoryPressure::Critical;
  require(admits(512 * kMiB, warning, ClaimClass::Floor, kGiB), "the floor was refused under warning");
  require(!admits(8 * kGiB, critical, ClaimClass::Floor, kGiB), "critical pressure admitted the floor");
  require(admits(2560 * kMiB, normal, ClaimClass::InService, 1536 * kMiB), "in-service work keeping M was refused");
  require(admits(2560 * kMiB, normal, ClaimClass::InService, 1792 * kMiB).failure ==
              metal::AllocationFailure::HostPressure,
          "in-service work inside the margin was admitted unexempt");
  require(admits(512 * kMiB, warning, ClaimClass::InService, kGiB, true), "an exempt in-service claim was refused");
  require(!admits(8 * kGiB, critical, ClaimClass::InService, kGiB, true), "critical pressure admitted exempt work");
  for (const ClaimClass cls : {ClaimClass::Start, ClaimClass::Optional, ClaimClass::Growth}) {
    require(admits(2560 * kMiB, normal, cls, 1536 * kMiB), "elective work keeping the margin was refused");
    require(!admits(2560 * kMiB, normal, cls, 1792 * kMiB), "elective work inside the margin was admitted");
    require(!admits(8 * kGiB, warning, cls, kGiB), "macOS's warning admitted elective work");
  }
  // The query the engine asks of growth, with no bytes and no credit: the
  // margin, the band and macOS's warning hold it as they hold growth.
  const auto growthAllowed = [](uint64_t headroom, MemoryPressure pressure) {
    Fixture fixture(host(headroom, pressure));
    static_cast<void>(fixture.governor->evaluate());
    return fixture.governor->admits(ClaimClass::Growth);
  };
  require(growthAllowed(8 * kGiB, normal), "growth was not allowed outside the band");
  require(!growthAllowed(900 * kMiB, normal), "growth was allowed in the band");
  require(!growthAllowed(8 * kGiB, warning), "growth was allowed under macOS's warning");
  require(!growthAllowed(8 * kGiB, critical), "growth was allowed under critical pressure");
  // The band: in below M, out at M + H, H the largest growth unit. The
  // pool's two units are held on the host too.
  for (const uint64_t unit : {kGiB, 512 * kMiB}) {
    const uint64_t pool = 2 * unit;
    Fixture fixture(host(900 * kMiB + pool, normal,
                         {at(100.0, kGiB + unit - 100 * kMiB + pool), at(200.0, kGiB + unit + pool)}));
    FakeElasticPool experts("experts", fixture.memory, unit, 1, 4, 5, 2, 0.2, 60.0);
    fixture.ledger().addPool(experts);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open({.cls = ClaimClass::Start});
    require(!fixture.allocate(start, 64 * kMiB), "a start was admitted below the margin");
    fixture.memory.advanceTo(100.0);
    require(fixture.allocate(start, 64 * kMiB).failure == metal::AllocationFailure::HostPressure,
            "the band ended short of M + H");
    fixture.memory.advanceTo(200.0);
    require(fixture.allocate(start, 64 * kMiB), "the band outlasted M + the largest unit");
  }
  // Growth within its credit leaves the host's footprint where it was.
  {
    Fixture fixture(host(512 * kMiB, warning));
    const ClaimId growth = fixture.ledger().open({.cls = ClaimClass::Growth});
    const MemoryGovernor::Scope scope(*fixture.governor, growth, std::nullopt, kGiB);
    require(fixture.memory.allocate(fixture.governor->allocationAdmission(), "experts", kGiB),
            "growth within its credit was refused under warning");
    require(!fixture.memory.allocate(fixture.governor->allocationAdmission(), "experts", 2 * kGiB),
            "growth past its credit was admitted under warning");
  }
}

// The band ends once a host pass finds nothing to take: only other
// applications could return the rest.
void testBandWaivedWhenTheHostClaimFindsNothing() {
  // A pool of one unit at its floor that could grow one more: H = 1 GiB.
  Fixture fixture(host(800 * kMiB + kGiB, MemoryPressure::Normal, {at(100.0, 1536 * kMiB + kGiB)}));
  FakeElasticPool pool("experts", fixture.memory, kGiB, 1, 2, 3, 1, 0.2, 60.0);
  fixture.ledger().addPool(pool);
  static_cast<void>(fixture.governor->evaluate());
  const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, kGiB), "a start was admitted below the margin");
  fixture.ledger().setWaiting(start, true);
  fixture.memory.advanceTo(100.0);
  const ClaimId probe = fixture.ledger().open(request(2, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(probe, 128 * kMiB), "the band ended short of M + H");
  static_cast<void>(fixture.pass(0.1));
  require(fixture.allocate(probe, 128 * kMiB), "the band held after the host pass found nothing");
}

// Every host pass gives back the units that cost nothing and are sized only
// when taken, whatever else it finds (amend/M.md B3; plan/M3a.md S2: the
// idle state buffers, which the base gave back first on every pass,
// EngineMemory.cpp:991-995): under macOS's warning, with no cache to take,
// two finished lanes' pooled buffers go at the first pass.
void testEveryHostPassGivesBackIdleBuffers() {
  Fixture fixture(host(4 * kGiB, MemoryPressure::Warning));
  TestSource engine("engine", &fixture.memory);
  Victim buffers = unit(1, 0, 0.0);
  buffers.sized = false;
  engine.units = {buffers};
  engine.unsizedBytes = 226 * kMiB;
  fixture.memory.hold("engine", 226 * kMiB);
  fixture.ledger().addSource(engine);
  static_cast<void>(fixture.pass(1.0));
  require(engine.taken == std::vector<uint64_t>{1} && fixture.memory.total() == 0,
          "a host pass that found no sized unit kept the idle buffers");
}

// The pass's deficit is the margin plus what host-refused waiting claims
// lack, not a recovery margin: with nothing waiting a band pass frees only
// what restores M.
void testDeficitCountsHostRefusedClaimsNotAnIdleMargin() {
  // 800 MiB of headroom beside the cache's 2 GiB: the deficit is 224 MiB of
  // M, and 512 MiB more for a start the host refused and that waits.
  for (const bool waits : {false, true}) {
    Fixture fixture(host(800 * kMiB + 2 * kGiB));
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB, 0.1));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    if (waits) {
      const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
      require(fixture.allocate(start, 512 * kMiB).failure == metal::AllocationFailure::HostPressure,
              "the start was not refused by the host");
      fixture.ledger().setWaiting(start, true);
    }
    static_cast<void>(fixture.pass(1.0));
    require(cache.taken.size() == (waits ? 6u : 2u),
            waits ? "a waiting host-refused claim's need was not freed" : "an idle band pass freed past the margin");
  }
}

// A start the host refused, whose request waits to retry it, is admitted
// within two control passes (amend/M.md B3; design/M.md T6, starts held in
// the band to their wait limit): what its refusal asked counts in the
// deficit whatever the free() of that refusal decided, the deficit pass runs
// in the band or not, and the band ends once the headroom clears the margin
// by what the start asks, however the pass that freed it ended. In the band
// as on the M6 (the store a unit below its cap, H = 1 GiB, 1.1 GiB of
// headroom, a start of 300 MiB), over cache that frees at once and over
// cache that frees only once its copies land (the pass Pending), and with
// the start already within the headroom past the margin (1.2 GiB, a start
// of 103.5 MiB, a KV extent: no pass runs); and outside it as on the Studio
// (the store at its cap, H = 0, 1.5 GiB of headroom, a start of 600 MiB).
void testAHostRefusedStartIsAdmittedWithinTwoPasses() {
  struct Case final {
    bool band;
    uint64_t headroom, need;
    bool later;
    const char *what;
  };
  for (const Case &scenario :
       {Case{true, 1100 * kMiB, 300 * kMiB, false, "a start the host refused waited in the band after a pass met its deficit"},
        Case{true, 1100 * kMiB, 300 * kMiB, true,
             "a start the host refused waited in the band after its deficit landed"},
        Case{true, 1200 * kMiB, 108'527'616, false,
             "a start the host refused waited in the band although the headroom past the margin held it"},
        Case{false, 1536 * kMiB, 600 * kMiB, false, "no pass freed for a start the host refused outside the band"}}) {
    const bool band = scenario.band;
    const uint64_t headroom = scenario.headroom;
    const uint64_t need = scenario.need;
    Fixture fixture(host(headroom + kGiB + 2 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 1, band ? 2 : 1, 3, 1, 0.2, 60.0);
    fixture.ledger().addPool(experts);
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB, 0.1, scenario.later));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
    bool admitted = false;
    for (uint32_t pass = 0; pass < 2 && !admitted; ++pass) {
      // As the engine has it: the request waits to retry what the host
      // refused (Engine::syncClaims), its allocation is refused and the
      // ledger asked, then the control pass runs.
      fixture.ledger().setHostRefusedWaiting(start, true);
      const metal::AllocationResult result = fixture.allocate(start, need);
      admitted = static_cast<bool>(result);
      if (admitted)
        break;
      require(result.failure == metal::AllocationFailure::HostPressure, "the start was not refused by the host");
      static_cast<void>(fixture.governor->free(start, fixture.governor->need(start)));
      static_cast<void>(fixture.pass(1.0 + 1.5 * pass));
      cache.land();
    }
    admitted = admitted || static_cast<bool>(fixture.allocate(start, need));
    require(admitted, scenario.what);
  }
}

// A pool's growth beyond its credit leaves the host, besides the margin,
// what the starts it refused wait for (design/M.md R4: the deficit pass
// frees what a waiting start lacks): a pool whose next unit fits in the
// headroom past the margin does not take the room freed for the start, which
// is admitted at its retry. As on the M6 with the expert store one 117 MiB
// remainder chunk below its cap and a start of 300 MiB: in the band (1124 MiB
// of headroom, cache that frees once its copies land, the control pass the
// first to see them land) and outside it within one pass (1224 MiB, cache
// that frees at once). Once the start gives its memory back the store grows.
void testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor() {
  constexpr uint64_t kChunk = 117 * kMiB;
  constexpr uint64_t kNeed = 300 * kMiB;
  struct Case final {
    uint64_t headroom;
    bool later;
    const char *what;
  };
  for (const Case &scenario :
       {Case{1124 * kMiB, true, "the store grew in the band into what landed for a start the host refused"},
        Case{1224 * kMiB, false, "the store grew into what a pass freed for a start the host refused"}}) {
    Fixture fixture(host(scenario.headroom + kChunk + 2 * kGiB));
    FakeElasticPool experts("experts", fixture.memory, kChunk, 1, 2, 3, 1, 0.2, 60.0);
    fixture.ledger().addPool(experts);
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 16; ++key)
      cache.units.push_back(unit(key, 128 * kMiB, 0.1, scenario.later));
    fixture.memory.hold("cache", 2 * kGiB);
    fixture.ledger().addSource(cache);
    static_cast<void>(fixture.governor->evaluate());
    const ClaimId start = fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::Start));
    fixture.ledger().setHostRefusedWaiting(start, true);
    require(fixture.allocate(start, kNeed).failure == metal::AllocationFailure::HostPressure,
            "the start was not refused by the host");
    static_cast<void>(fixture.governor->free(start, fixture.governor->need(start)));
    static_cast<void>(fixture.pass(1.0));
    if (scenario.later) {
      cache.land();
      static_cast<void>(fixture.pass(2.5));
    }
    const bool admitted = static_cast<bool>(fixture.allocate(start, kNeed));
    require(admitted && experts.units() == 1, scenario.what);
    fixture.ledger().close(start);
    fixture.memory.release("kv", kNeed);
    static_cast<void>(fixture.pass(4.0));
    require(experts.units() == 2, "the store did not grow into what a start the host refused gave back");
  }
}

// AMENDMENTS Borrowed 1: /status reports a pool's price at its edge beside
// the uniform spread, and the rule keeps the uniform input. On a concave
// miss curve the ghost hits within a unit of the newest ghosts, and the hits
// on the unit the pool gives up first, are twice the span's average a unit
// (research/SUMMARY.md R1: about 2.2x on llama.cpp's tables). A claim choosing
// between a unit of the store and a cache unit priced between the two takes
// the store's unit, which the uniform price makes the cheaper.
void testTheMarginalPriceIsReportedBesideTheUniform() {
  Fixture fixture(host(32 * kGiB));
  // Four 1 GiB units resident of eight: the ghosts span 4 GiB. A miss, 2 ms.
  FakeElasticPool experts("experts", fixture.memory, kGiB, 1, 4, 8, 4, 2.0, 0.0);
  TestSource cache("cache", &fixture.memory);
  fixture.ledger().addSource(cache);
  fixture.ledger().addPool(experts);
  // 40 ghost hits a second over the span, 10 a unit; 20 within a unit of the
  // newest ghosts, and 20 on the unit given up first.
  for (double t = 0.0; t < 30.0; t += 0.5) {
    experts.served(0, 20, 0, 0);
    experts.nearGhostHits += 10;
    experts.tailHits += 10;
    fixture.ledger().observePools(t + 0.5, true);
  }
  const MemoryGovernorSnapshot snapshot = fixture.governor->snapshot();
  const MemoryAccountSnapshot &account = snapshot.accounts.back();
  const double uniform = 10 * 0.002 * kValueHorizonSeconds;
  require(account.name == "experts" && std::abs(account.valuePerGiB - uniform) < 1e-9 &&
              std::abs(account.growValuePerGiB - 2 * uniform) < 1e-9 &&
              std::abs(account.shrinkValuePerGiB - 2 * uniform) < 1e-9,
          "the store's price at its edge was not reported beside the uniform one: " +
              std::to_string(account.valuePerGiB) + " uniform, " + std::to_string(account.growValuePerGiB) +
              " to grow, " + std::to_string(account.shrinkValuePerGiB) + " to shrink");
  cache.units = {unit(1, kGiB, 1.5 * uniform)};
  fixture.memory.hold("cache", kGiB);
  const ClaimId claim =
      fixture.ledger().open(request(1, RequestPriority::Normal, ClaimClass::InService, kValueHorizonSeconds));
  require(fixture.ledger().free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              experts.shrinks == 1 && cache.taken.empty(),
          "the rule priced the store's unit at its edge, not by the uniform spread");
}

// The marginal price of a unit is its share of the bytes the counts count
// in. The store counts its near-ghost and tail hits in an elastic chunk's
// slots, and at its cap gives back its last chunk first, the rest of its
// slots (ExpertChunkLayout), as on the M6 a 117 MiB chunk beside 1023 MiB
// ones (amend/M.md D2): per GiB, the remainder's price at the edge is the
// chunk's, not the chunk's hits on fewer bytes.
void testTheMarginalPriceIsTheUnitsShareOfItsCounts() {
  Fixture fixture(host(32 * kGiB));
  ListedPool experts(kGiB, {1023 * kMiB, 117 * kMiB}, 8 * kGiB, 16 * kMiB, 2.0);
  experts.marginBytes = 1023 * kMiB;
  fixture.ledger().addPool(experts);
  for (double t = 0.0; t < 30.0; t += 0.5) {
    experts.nearGhostHits += 5;
    experts.tailHits += 5;
    fixture.ledger().observePools(t + 0.5, true);
  }
  const MemoryAccountSnapshot account = fixture.governor->snapshot().accounts.back();
  const double perGiB = 10 * 0.002 * kValueHorizonSeconds * double(kGiB) / double(1023 * kMiB);
  require(account.unitBytes == 117 * kMiB && std::abs(account.shrinkValuePerGiB - perGiB) < 1e-6 * perGiB &&
              std::abs(account.growValuePerGiB - perGiB) < 1e-6 * perGiB,
          "a remainder's price at the edge was " + std::to_string(account.shrinkValuePerGiB) + " to shrink and " +
              std::to_string(account.growValuePerGiB) + " to grow per GiB, not its share of the chunk's hits, " +
              std::to_string(perGiB));
}

// free() takes the cheapest unit per byte across sources first.
void testCheapestPerByteFirstAcrossSources() {
  MemoryLedger ledger;
  TestSource a("a"), b("b");
  a.units = {unit(1, kGiB, 2.0)};
  b.units = {unit(2, kGiB, 1.0), unit(3, kGiB, 3.0)};
  ledger.addSource(a);
  ledger.addSource(b);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = 2 * kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              b.taken == std::vector<uint64_t>{2} && a.taken == std::vector<uint64_t>{1},
          "free() did not take the two cheapest units per byte");
}

// A unit meets at most the whole need: one larger than the need is priced
// at its whole cost, and the dry run takes a smaller unit that meets the
// need for fewer seconds (amend/M.md B2, R2). On the M6's inputs
// (earlyfix.md): a 1023 MiB store chunk at 4.07 s beside an uncopied 8K
// chain at 3.2 s, for an extent of 108,527,616 bytes. An Optional claim
// worth 3.5 s, as a capture's KV is, gets the chain rather than stopping
// short at the chunk.
void testAUnitLargerThanTheNeedCostsItsWholeCost() {
  constexpr uint64_t extent = 108'527'616;
  for (const ClaimClass cls : {ClaimClass::InService, ClaimClass::Optional}) {
    MemoryLedger ledger;
    TestSource store("store"), cache("cache");
    store.units = {unit(1, 1023 * kMiB, 4.07)};
    cache.units = {unit(2, extent, 3.2)};
    ledger.addSource(store);
    ledger.addSource(cache);
    ClaimSpec spec = request(1, RequestPriority::Normal, cls);
    if (cls == ClaimClass::Optional) {
      spec.timing = NeedTiming::Immediate;
      spec.benefitSeconds = 3.5;
    }
    const ClaimId claim = ledger.open(spec);
    require(ledger.free(claim, {.bytes = extent}, 0, false, false).outcome == FreeOutcome::Met &&
                cache.taken == std::vector<uint64_t>{2} && store.taken.empty(),
            cls == ClaimClass::Optional
                ? "an optional claim stopped at a unit larger than its need that costs more than one that meets it"
                : "a unit larger than the need was priced by the bytes it frees past the need");
  }
}

// Inside a source its own order holds: a cheaper unit behind a dearer one
// waits for it.
void testSourceOrderIsKeptWithinASource() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 4.0), unit(2, kGiB, 1.0)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "free() skipped ahead inside a source's order");
}

// A source's units keyed by their place in its order, as the cache's
// reclaims are (steps from its state as it stands): taking one moves every
// unit after it up a place. Cumulative, taking one takes every unit before
// it too, as taking a reclaim takes the steps before it.
class PlacedSource final : public VictimSource {
public:
  [[nodiscard]] std::string_view name() const noexcept override { return "placed"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    const size_t place = after ? after->key + 1 : 0;
    if (place >= units.size())
      return std::nullopt;
    Victim victim = units[place].second;
    victim.key = place;
    return victim;
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    const size_t first = cumulative ? 0 : victim.key;
    uint64_t bytes = 0;
    for (size_t place = first; place <= victim.key; ++place) {
      taken.push_back(units.at(place).first);
      bytes += units[place].second.bytes;
    }
    units.erase(units.begin() + static_cast<std::ptrdiff_t>(first),
                units.begin() + static_cast<std::ptrdiff_t>(victim.key + 1));
    return {.nowBytes = bytes};
  }
  std::vector<std::pair<uint64_t, Victim>> units;  // the test's name for each unit, and the unit
  std::vector<uint64_t> taken;
  bool cumulative = false;
};

// free() takes the units its dry run chose from a source whose keys are
// places: each by its place as the order stands after the takes before it,
// past the units the run passed over, which stay.
void testUnitsKeyedByTheirPlaceAreTakenAsChosen() {
  MemoryLedger ledger;
  PlacedSource placed;
  TestSource other("other");
  Victim inUse = unit(0, kGiB, 0.5);
  inUse.cls = ReclaimClass::InUse;
  placed.units = {{1, unit(0, kGiB, 1.0)}, {2, inUse}, {3, unit(0, kGiB, 2.0)}, {4, unit(0, kGiB, 3.0)}};
  other.units = {unit(5, kGiB, 1.5)};
  ledger.addSource(placed);
  ledger.addSource(other);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = 4 * kGiB}, 0, false, false).outcome == FreeOutcome::Met,
          "the claim was not met");
  require(placed.taken == std::vector<uint64_t>{1, 3, 4} && other.taken == std::vector<uint64_t>{5} &&
              placed.units.size() == 1 && placed.units.front().first == 2,
          "free() did not take the units its dry run chose from a source keyed by place");
}

// A conversion's key is its lane, and free() takes no lane its dry run did
// not choose: when a lane it chose is no longer offered once the units
// before it were taken (their commitments filled the quota it was listed
// against), free() stops there with what it took, neither converting
// another lane nor throwing.
void testATakeConvertsOnlyTheLanesItsRunChose() {
  for (const bool another : {true, false}) {
    MemoryLedger ledger;
    QuotaSource lanes;
    lanes.quota = 12;
    lanes.lanes = {{1, 10}, {2, 8}};
    if (another)
      lanes.lanes.push_back({3, 2});
    ledger.addSource(lanes);
    const ClaimId claim = ledger.open(request(9, RequestPriority::Foreground));
    std::optional<FreeResult> freed;
    try {
      freed = ledger.free(claim, {.bytes = 2 * kGiB}, 0, false, false);
    } catch (const std::exception &error) {
      require(false, std::string("free() threw once a lane its dry run chose was no longer offered: ") + error.what());
    }
    require(lanes.converted == std::vector<uint64_t>{1} && freed->outcome == FreeOutcome::Pending &&
                freed->laterBytes == kGiB && ledger.claim(claim).converted == std::vector<uint64_t>{1},
            std::string("free() did not stop at the lane it chose that the quota no longer lists") +
                (another ? " (another lane fits the quota left)" : ""));
  }
}

// A state whose buffers the model's pool keeps, refilling those a lane
// starts from (QwenStateStorage::idleBytes), here every state's, meets none
// of a need of bytes alone. Between two units free() chooses from a source
// whose units are cumulative, as the cache's reclaims are, it goes with the
// later one, at its cost: free() takes what its dry run weighed, the units
// after them as weighed too, or nothing once that cost passes the benefit.
void testAStateThePoolKeepsGoesWithTheNextUnitOfItsSource() {
  for (const double benefit : {3.1, 3.3}) {
    MemoryLedger ledger;
    PlacedSource placed;
    placed.cumulative = true;
    const Victim state{.stateBuffers = 1, .cost = {.onceSeconds = 0.1}};
    placed.units = {{1, unit(0, kGiB, 1.0)}, {2, state}, {3, unit(0, kGiB, 1.0)}, {4, state}, {5, unit(0, kGiB, 1.0)}};
    ledger.addSource(placed);
    ledger.setBufferBytes([](uint32_t) { return uint64_t{0}; });
    ClaimSpec spec = request(1, RequestPriority::Normal);
    spec.benefitSeconds = benefit;
    const ClaimId claim = ledger.open(spec);
    FreeResult freed;
    try {
      freed = ledger.free(claim, {.bytes = 3 * kGiB}, 0, false, false);
    } catch (const std::logic_error &error) {
      require(false, std::string("free() took other units than its dry run weighed: ") + error.what());
    }
    require(benefit < 3.2 ? freed.outcome == FreeOutcome::Short && placed.taken.empty()
                          : freed.outcome == FreeOutcome::Met && freed.nowBytes == 3 * kGiB &&
                                placed.taken == std::vector<uint64_t>{1, 2, 3, 4, 5},
            benefit < 3.2 ? "free() took units whose cost with the states they take passed the benefit"
                          : "free() did not take the states with the units after them, as its dry run weighed");
  }
}

// At equal cost per byte: a unit that frees memory now, then the source
// registered first, the pools after the sources.
void testTiesGoToTheImmediateUnitThenTheSourceRegisteredFirst() {
  MemoryLedger ledger;
  TestSource first("first"), second("second");
  first.units = {unit(1, kGiB, 1.0, true), unit(2, kGiB, 1.0)};
  second.units = {unit(3, kGiB, 1.0)};
  ledger.addSource(first);
  ledger.addSource(second);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false, false));
  require(second.taken == std::vector<uint64_t>{3} && first.taken.empty(),
          "a tie went to the unit that lands later");
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false, false));
  require(first.taken == std::vector<uint64_t>{1}, "the unit left did not go");
  FakeMemory memory({});
  FakeElasticPool pool("experts", memory, kGiB, 0, 2, 2, 2, 0.0, 0.0);
  MemoryLedger tied;
  TestSource cache("cache");
  cache.units = {unit(4, kGiB, 0.0)};
  tied.addPool(pool);
  tied.addSource(cache);
  const ClaimId other = tied.open(request(2, RequestPriority::Normal));
  static_cast<void>(tied.free(other, {.bytes = kGiB}, 0, false, false));
  require(cache.taken == std::vector<uint64_t>{4} && pool.units() == 2, "a pool won a tie against a source");
}

// What each class may take (amend/M.md B2, B3): the bytes available() counts
// for it from units of distinct sizes name which.
void testClassBoundsEligibility() {
  FakeMemory memory({});
  FakeElasticPool pool("experts", memory, 32 * kMiB, 0, 1, 2, 1, 0.0, 0.0);
  MemoryLedger ledger;
  TestSource cache("cache"), engine("engine");
  Victim inUse = unit(2, 2 * kMiB, 0.0);
  inUse.cls = ReclaimClass::InUse;
  Victim certain = unit(3, 4 * kMiB, 0.0);
  certain.certainReuse = true;
  cache.units = {unit(1, kMiB, 0.0), inUse, certain};
  Victim conversion = unit(4, 8 * kMiB, 0.0);
  conversion.kind = VictimKind::Conversion;
  Victim passes = unit(5, 16 * kMiB, 0.0);
  passes.kind = VictimKind::PassCommand;
  engine.units = {conversion, passes};
  ledger.addSource(cache);
  ledger.addSource(engine);
  ledger.addPool(pool);
  const auto counted = [&](ClaimSpec spec, bool critical = false, bool deficit = false) {
    return ledger.available(spec, kGiB, 0, critical, deficit) / kMiB;
  };
  // 1 Ordinary, 2 InUse, 4 certain reuse, 8 a conversion, 16 the pass
  // command, 32 a pool unit.
  require(counted({.cls = ClaimClass::Floor}) == 1 + 4, "the floor took more than Ordinary units");
  require(counted({.cls = ClaimClass::InService}) == 1 + 4 + 8 + 32, "in-service work took the wrong units");
  require(counted({.cls = ClaimClass::InService, .reach = ReclaimClass::InUse, .supersedes = true}) == 63,
          "in-service work that reaches InUse and supersedes took the wrong units");
  require(counted({.cls = ClaimClass::Start, .supersedes = true}) == 1 + 4, "a start took more than Ordinary units");
  require(counted({.cls = ClaimClass::Optional, .reach = ReclaimClass::InUse, .supersedes = true}) == 1 + 4 + 32,
          "optional work converted, took InUse or the pass command");
  require(counted({.cls = ClaimClass::Growth}) == 1 + 32, "growth took certain reuse, a conversion or InUse");
  require(counted({.cls = ClaimClass::Host}) == 1, "the host took more than Ordinary cache without a deficit");
  require(counted({.cls = ClaimClass::Host}, false, true) == 1 + 8 + 32, "the host's deficit took the wrong units");
  require(counted({.cls = ClaimClass::Host, .reach = ReclaimClass::InUse}, true, true) == 1 + 2 + 4 + 8 + 32,
          "critical pressure did not give the host what is in use");
  ledger.setLanes(0, RequestPriority::Foreground);
  require(counted(request(1, RequestPriority::Background)) == 1 + 4 + 8,
          "a claim below a lane in service took a pool unit");
  require(counted(request(1, RequestPriority::Foreground)) == 1 + 4 + 8 + 32,
          "a claim at the lanes' priority was refused a pool unit");
}

// Short takes nothing (all or none).
void testShortTakesNothing() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 1.0), unit(2, kGiB, 1.0)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::InService));
  require(ledger.free(claim, {.bytes = 3 * kGiB}, 0, false, false).outcome == FreeOutcome::Short &&
              cache.taken.empty() && cache.units.size() == 2,
          "a short free() took units");
}

// A claim whose shortfall is only what waiting claims above it hold waits
// for them, taking nothing.
void testRoomHeldForHigherRanksMakesAWait() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1)};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(low, {.bytes = 512 * kMiB}, kGiB, false, false).outcome == FreeOutcome::Pending &&
              cache.taken.empty(),
          "a claim the room held above it fits took a victim instead of waiting");
}

// Landings first, then the cheapest victims, then owed copies: a need the
// landing bytes do not meet takes a victim before it waits for another
// lane's write-through (the base's escalate waited on owed copies first,
// Engine.cpp:2380). The victim meets it alone, so it is met.
void testLandingThenVictimsThenOwed() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.5)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 256 * kMiB, .owedBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "the claim waited on owed copies before it took a victim");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending &&
              ledger.claim(claim).waiting,
          "with no victim left the claim did not wait for the owed copies");
  ledger.setReturning([] { return Returning{.landingBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  cache.units = {unit(2, kGiB, 0.5)};
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending && cache.taken.size() == 1,
          "landings that meet the need did not come first");
}

// An Immediate claim does not wait for landings and takes nothing that
// lands later.
void testImmediateSkipsLandings() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1, true), unit(2, kGiB, 0.5)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t>) { return Returning{}; });
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate});
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{2},
          "an Immediate claim waited or took a unit that lands later");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Short,
          "an Immediate claim waited for landings");
}

// What lands, and the copies a claim's own lanes owe, count before victims
// also in part (amend/M.md B5): a claim that can wait takes the cheapest
// units for what they do not cover and waits for the rest. Another lane's
// owed copies count nothing here, nor what lands for an Immediate claim,
// nor the pages that land for a waiting claim ranked above; units that
// meet the need alone meet it.
void testLandingCountsBeforeVictimsInPart() {
  struct Case final {
    Returning all;
    Returning own;
    NeedTiming timing = NeedTiming::CanWait;
    FreeOutcome outcome = FreeOutcome::Pending;
    std::vector<uint64_t> taken;
  };
  const Case cases[] = {
      {.all = {.landingBytes = 512 * kMiB}, .taken = {1}},
      {.all = {.owedBytes = 512 * kMiB}, .own = {.owedBytes = 512 * kMiB}, .taken = {1}},
      {.all = {.owedBytes = 512 * kMiB}, .outcome = FreeOutcome::Met, .taken = {1, 2}},
      {.all = {.landingBytes = 512 * kMiB}, .timing = NeedTiming::Immediate, .outcome = FreeOutcome::Met,
       .taken = {1, 2}},
  };
  for (const Case &each : cases) {
    MemoryLedger ledger;
    TestSource cache("cache");
    cache.units = {unit(1, 512 * kMiB, 0.5), unit(2, 512 * kMiB, 0.6)};
    ledger.addSource(cache);
    ledger.setReturning([&] { return each.all; }, [&](std::span<const uint64_t> lanes) {
      return std::ranges::find(lanes, 1) != lanes.end() ? each.own : Returning{};
    });
    ClaimSpec spec = request(1, RequestPriority::Normal);
    spec.timing = each.timing;
    const ClaimId claim = ledger.open(spec);
    require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == each.outcome && cache.taken == each.taken &&
                ledger.claim(claim).waiting == (each.outcome == FreeOutcome::Pending),
            "the claim took victims for what lands, or did not wait for it after taking the rest");
  }
  // In pages: two land, a waiting claim ranked above them takes them first.
  for (const bool heldAbove : {false, true}) {
    MemoryLedger ledger;
    TestSource cache("cache");
    ledger.addSource(cache);
    ledger.setReturning([] { return Returning{.landingPages = {2, 0, 0}}; },
                        [](std::span<const uint64_t>) { return Returning{}; });
    if (heldAbove) {
      const ClaimId high = ledger.open(request(2, RequestPriority::Foreground));
      require(ledger.free(high, {.pages = 2}, 0, false, false).outcome == FreeOutcome::Pending,
              "the claim above did not wait for the pages that land");
    }
    for (uint64_t key : {1, 2})
      cache.units.push_back({.key = key, .pages = 2, .cost = {.onceSeconds = 0.5}});
    const ClaimId claim = ledger.open(request(3, RequestPriority::Background));
    const FreeOutcome outcome = ledger.free(claim, {.pages = 4}, 0, false, false).outcome;
    require(heldAbove ? outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1, 2}
                      : outcome == FreeOutcome::Pending && cache.taken == std::vector<uint64_t>{1},
            "the pages that land were not counted before victims, or counted although held above");
  }
}

// A model's pool of state buffers: those idle go one at a time, at no cost,
// as a unit sized only when taken (Model::releaseIdleBuffer).
class BufferPool final : public VictimSource {
public:
  explicit BufferPool(uint64_t bufferBytes) : bufferBytes_(bufferBytes) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "buffers"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    return after ? std::nullopt : std::optional<Victim>(Victim{.sized = false});
  }
  FreeResult take(const VictimQuery &, const Victim &) override {
    if (!idle)
      return {};
    --idle;
    return {FreeOutcome::Met, bufferBytes_};
  }
  uint32_t idle = 0;

private:
  uint64_t bufferBytes_;
};

// Cached states in the test's order, whose buffers return to the pool as
// they go.
class CachedStates final : public VictimSource {
public:
  explicit CachedStates(BufferPool &pool) : pool_(pool) {}
  [[nodiscard]] std::string_view name() const noexcept override { return "cache"; }
  [[nodiscard]] std::optional<Victim> peek(const VictimQuery &, const Victim *after) const override {
    auto unit = units.begin();
    if (after)
      unit = std::next(std::ranges::find(units, after->key, &Victim::key));
    return unit == units.end() ? std::nullopt : std::optional<Victim>(*unit);
  }
  FreeResult take(const VictimQuery &, const Victim &victim) override {
    const auto unit = std::ranges::find(units, victim.key, &Victim::key);
    taken.push_back(unit->key);
    pool_.idle += unit->stateBuffers;
    units.erase(unit);
    return {FreeOutcome::Met};
  }
  std::vector<Victim> units;
  std::vector<uint64_t> taken;

private:
  BufferPool &pool_;
};

// The buffers a state frees meet a need of bytes (design/M.md section 3.1,
// Victim::bytes): counted at what the pool releases of them, beyond the
// buffers of the states chosen before, and released once the states went;
// a need of buffers in place keeps them in the pool.
void testStateBuffersMeetANeedOfBytes() {
  MemoryLedger ledger;
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  for (uint64_t key = 1; key <= 3; ++key)
    cache.units.push_back({.key = key, .stateBuffers = 1, .cost = {.onceSeconds = 0.1}});
  ledger.addSource(cache);
  ledger.addSource(pool);
  // The first state's buffers release 64 MiB, each after it 32 MiB more:
  // two cover no 100 MiB, three do.
  ledger.setBufferBytes([](uint32_t states) { return states ? uint64_t{states + 1} * 32 * kMiB : 0; });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = 100 * kMiB}, 0, false, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 192 * kMiB &&
              cache.taken == std::vector<uint64_t>{1, 2, 3} && !pool.idle,
          "a need of bytes did not take the states whose buffers the pool releases, or left them in the pool");
  cache.units.push_back({.key = 4, .stateBuffers = 1, .cost = {.onceSeconds = 0.1}});
  const ClaimId lane = ledger.open(request(2, RequestPriority::Normal));
  const FreeResult inPlace = ledger.free(lane, {.stateBuffers = 1}, 0, false, false);
  require(inPlace.outcome == FreeOutcome::Met && !inPlace.nowBytes && cache.taken.back() == 4 && pool.idle == 1,
          "a need of buffers in place did not keep the buffers its state freed");
}

// A need the host refused is met in place first; failing that, in-service
// work is exempted from the host for its retries, a start falls short.
void testHostRefusedNeedIsMetInPlaceFirst() {
  Fixture fixture(host(1200 * kMiB));
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", kGiB);
  Victim leaf = unit(1, 0, 0.2);
  leaf.pages = 8;
  cache.units = {unit(2, kGiB, 0.1), leaf};
  fixture.ledger().addSource(cache);
  const ClaimId lane = fixture.ledger().open(request(1, RequestPriority::Normal));
  require(fixture.allocate(lane, 512 * kMiB).failure == metal::AllocationFailure::HostPressure,
          "the host did not refuse the lane");
  Need need = fixture.governor->need(lane, Need{.pages = 8});
  require(need.hostRefused && need.bytes == 512 * kMiB && need.pages == 8, "the need is not the refusal and pages");
  require(fixture.governor->free(lane, need).outcome == FreeOutcome::Met && cache.taken == std::vector<uint64_t>{1},
          "the host-refused need was not met in place by the leaf");
  require(!fixture.ledger().claim(lane).hostExempt, "a need met in place exempted the claim");
  need.pages = 0;
  require(fixture.governor->free(lane, need).outcome == FreeOutcome::Met && fixture.ledger().claim(lane).hostExempt,
          "in-service work nothing in place meets was not exempted");
  require(fixture.allocate(lane, 512 * kMiB) && fixture.ledger().claim(lane).hostExempt,
          "the exempt retry was refused, or its grant ended the exemption before the allocation did");
  fixture.ledger().setHostExempt(lane, false);
  const ClaimId start = fixture.ledger().open(request(2, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, 512 * kMiB), "the host admitted the start");
  require(fixture.governor->free(start, fixture.governor->need(start)).outcome == FreeOutcome::Short &&
              !fixture.ledger().claim(start).hostExempt,
          "a start was exempted from the host");
}

// The exemption covers the retry of the whole allocation, as the serving
// mark did (design/M.md R2): an attempt that takes several reservations,
// all or none (a hot block cache's floor of four extents, a history
// stream's buffers), passes the host with each of them, where the host's
// headroom keeps its margin beside one but not two.
void testAnExemptRetryPassesTheHostWithEachReservation() {
  constexpr uint64_t extent = 108'527'616;
  Fixture fixture(host(kHostMarginBytes + extent / 2));
  const ClaimId lane = fixture.ledger().open(request(1, RequestPriority::Normal));
  const auto attempt = [&] {
    std::vector<uint64_t> granted;
    for (int reservation = 0; reservation < 4; ++reservation) {
      if (!fixture.allocate(lane, extent)) {
        for (const uint64_t bytes : granted)
          fixture.memory.release("kv", bytes);
        return false;
      }
      granted.push_back(extent);
    }
    return true;
  };
  require(!attempt(), "the host admitted the floor beside its margin");
  require(fixture.governor->free(lane, fixture.governor->need(lane)).outcome == FreeOutcome::Met &&
              fixture.ledger().claim(lane).hostExempt,
          "in-service work nothing in place meets was not exempted");
  require(attempt(), "the exempt retry was refused the host after its first reservation");
}

// Units that cost nothing and are sized only when taken go first.
void testZeroCostUnsizedUnitsGoFirst() {
  MemoryLedger ledger;
  TestSource engine("engine"), cache("cache");
  Victim buffers = unit(1, 0, 0.0);
  buffers.sized = false;
  engine.units = {buffers};
  engine.unsizedBytes = 600 * kMiB;
  cache.units = {unit(2, kGiB, 0.0)};
  ledger.addSource(cache);
  ledger.addSource(engine);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = kGiB}, 512 * kMiB, false, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 600 * kMiB && engine.taken.size() == 1 &&
              cache.taken.empty(),
          "the unsized buffers did not go first and end the need");
}

// The dry run stops once its cost would pass the claim's benefit.
void testBenefitBoundsTheDryRun() {
  for (const double benefit : {1.0, 1.5}) {
    MemoryLedger ledger;
    TestSource cache("cache");
    cache.units = {unit(1, 512 * kMiB, 0.6), unit(2, 512 * kMiB, 0.6)};
    ledger.addSource(cache);
    const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .benefitSeconds = benefit});
    const FreeOutcome outcome = ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome;
    require(benefit < 1.2 ? outcome == FreeOutcome::Short && cache.taken.empty()
                          : outcome == FreeOutcome::Met && cache.taken.size() == 2,
            "the benefit did not bound what the claim took");
  }
}

void testAvailableTakesNothing() {
  FakeMemory memory({});
  FakeElasticPool pool("experts", memory, kGiB, 1, 3, 4, 3, 0.0, 0.0);
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1), unit(2, kGiB, 0.2, true)};
  ledger.addSource(cache);
  ledger.addPool(pool);
  require(ledger.available(request(1, RequestPriority::Normal), 10 * kGiB, 256 * kMiB, false, false) ==
              256 * kMiB + 4 * kGiB,
          "available() did not count the room, the cache and the pool's units above its floor");
  require(ledger.available(request(1, RequestPriority::Normal), kGiB, 256 * kMiB, false, false) == kGiB,
          "available() did not stop at enough");
  require(cache.taken.empty() && pool.units() == 3 && pool.shrinks == 0, "available() took something");
}

// available() is R2's steps 0-2 as a dry run (design/M.md R5): for a claim
// that can wait, what lands and the copies its own request owes count
// before any victim, as free() counts them; an Immediate claim counts
// neither.
void testAvailableCountsWhatLandsForAClaimThatCanWait() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 256 * kMiB, 0.1)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 512 * kMiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 1u) != lanes.end() ? Returning{.owedBytes = 128 * kMiB}
                                                                           : Returning{};
                      });
  const ClaimSpec start = request(1, RequestPriority::Normal, ClaimClass::Start);
  require(ledger.available(start, 2 * kGiB, 128 * kMiB, false, false) == 1024 * kMiB,
          "available() did not count what lands and the copies the request owes before its victims");
  ClaimSpec immediate = start;
  immediate.timing = NeedTiming::Immediate;
  require(ledger.available(immediate, 2 * kGiB, 128 * kMiB, false, false) == 384 * kMiB,
          "available() counted what lands for a claim that cannot wait");
  require(cache.taken.empty(), "available() took something");
}

// Free room beyond the lanes' projected growth goes to the pools at once,
// whatever their value, while no claim has taken their units (T3).
void testGrowthRetakesFreeRoomBeyondLaneDemand() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 8, 9, 2, 0.0, 0.0);
  fixture.memory.hold("weights", 3 * kGiB);
  fixture.ledger().addPool(experts);
  fixture.ledger().setLanes(kGiB, std::nullopt);
  static_cast<void>(fixture.pass(0.5));
  require(experts.units() == 4, "the pool did not take the free room beyond the lanes' demand");
  fixture.ledger().setLanes(0, std::nullopt);
  static_cast<void>(fixture.pass(1.0));
  require(experts.units() == 5, "the pool did not take the room the lanes no longer project");
}

// The room a lane is projected to grow into stays free: the pool neither
// grows into it nor gives it back as the lane takes it.
void testNoPingPongWhileALaneGrows() {
  Fixture fixture(host(32 * kGiB), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, 512 * kMiB, 2, 8, 9, 4, 0.25, 60.0);
  fixture.memory.hold("weights", 4 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  const ClaimId lane = ledger.open(request(1, RequestPriority::Normal));
  uint64_t projected = 2 * kGiB;
  for (int step = 0; step < 8; ++step) {
    ledger.setLanes(projected, RequestPriority::Normal);
    serve(ledger, experts, 400.0, step * 0.5, 0.5);
    static_cast<void>(fixture.governor->controlPass(step * 500.0));
    if (!fixture.allocate(lane, 256 * kMiB)) {
      static_cast<void>(fixture.governor->free(lane, fixture.governor->need(lane)));
      const metal::AllocationResult retried = fixture.allocate(lane, 256 * kMiB);
      require(retried, std::string("the lane was refused its growth: ") + metal::allocationFailureName(retried.failure) +
                           " at step " + std::to_string(step) + ", pool units " + std::to_string(experts.units()));
    }
    projected -= 256 * kMiB;
  }
  require(experts.grows == 0 && experts.shrinks == 0, "the pool cycled a unit against the growing lane");
}

// Displacement takes victims only while they cost less than the unit is
// worth beyond its cycle (refill and release).
void testDisplacementNeedsValueAboveCycle() {
  // 400 ghost hits a second at 0.25 ms: a 1 GiB unit is worth 6 s over the
  // horizon; its release takes 60 ms.
  for (const double victimCost : {5.0, 6.5}) {
    Fixture fixture(host(32 * kGiB), 6 * kGiB);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 4, 3, 0.25, 60.0);
    TestSource cache("cache", &fixture.memory);
    fixture.memory.hold("cache", kGiB);
    fixture.memory.hold("weights", 2 * kGiB);
    cache.units = {unit(1, kGiB, victimCost)};
    fixture.ledger().addSource(cache);
    fixture.ledger().addPool(experts);
    static_cast<void>(experts.release(1));
    static_cast<void>(experts.grow([](uint64_t, const std::function<void()> &allocate) {
      allocate();
      return metal::AllocationResult{};
    }));
    serve(fixture.ledger(), experts, 400.0, 0.0, 10.0);
    static_cast<void>(fixture.pass(10.5));
    require((experts.units() == 4) == (victimCost < 6.0 - 0.06),
            "displacement did not follow the unit's value beyond its cycle");
  }
}

// At constant inputs a claim that recurs every 10 passes moves the pool at
// most one unit down and one back each time.
void testNoFlipFlopAtConstantInputs() {
  Fixture fixture(host(32 * kGiB), 6 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 4, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("weights", 2 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  uint64_t recurrences = 0;
  std::optional<ClaimId> claim;
  for (int step = 0; step < 200; ++step) {
    const double seconds = step * 0.5;
    serve(ledger, experts, 400.0, seconds, 0.5);
    if (step % 10 == 0) {
      claim = ledger.open(request(1 + step, RequestPriority::Normal, ClaimClass::InService, 2.0));
      if (!fixture.allocate(*claim, kGiB)) {
        static_cast<void>(fixture.governor->free(*claim, fixture.governor->need(*claim)));
        require(fixture.allocate(*claim, kGiB), "the claim was refused");
      }
      ++recurrences;
    } else if (step % 10 == 5) {
      fixture.memory.release("kv", kGiB);
      ledger.close(*claim);
    }
    fixture.memory.advanceTo(seconds * 1000.0);
    static_cast<void>(fixture.governor->controlPass(seconds * 1000.0));
  }
  require(experts.shrinks <= recurrences && experts.grows <= recurrences,
          "the pool moved more than one unit each way per recurrence");
}

// macOS's critical pressure drops every pool to its floor and gives the host
// claim what it may take, InUse and certain reuse included.
void testCriticalDropsPoolsToFloors() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Critical));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 4, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 3 * kGiB);
  Victim inUse = unit(2, kGiB, 9.0);
  inUse.cls = ReclaimClass::InUse;
  Victim certain = unit(3, kGiB, 9.0);
  certain.certainReuse = true;
  cache.units = {unit(1, kGiB, 0.1), inUse, certain};
  fixture.ledger().addSource(cache);
  fixture.ledger().addPool(experts);
  static_cast<void>(fixture.pass(0.5));
  require(experts.units() == 2 && cache.units.empty(), "critical pressure left a pool above its floor or cache");
}

// Under macOS's critical pressure the host claim takes all it may at once,
// whatever is on its way back: 1 GiB of demotions in flight does not keep
// the pass from the 400 MiB of cache it finds, what is in use and what a
// request will reuse among it.
void testCriticalTakesWhatItFindsWhateverLands() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Critical));
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", 400 * kMiB);
  Victim inUse = unit(1, 200 * kMiB, 9.0);
  inUse.cls = ReclaimClass::InUse;
  Victim certain = unit(2, 200 * kMiB, 9.0);
  certain.certainReuse = true;
  cache.units = {inUse, certain};
  fixture.ledger().setReturning([] { return Returning{.landingBytes = kGiB, .transfers = 4}; },
                                [](std::span<const uint64_t>) { return Returning{}; });
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(0.5));
  require(cache.units.empty() && fixture.memory.held("cache") == 0,
          "a critical pass left the cache it found while more than it was on its way back");
}

// Under macOS's warning each settled pass frees at most the margin.
void testWarningFreesAtMostTheMarginPerSettledPass() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Warning));
  TestSource cache("cache", &fixture.memory);
  for (uint64_t key = 1; key <= 6; ++key)
    cache.units.push_back(unit(key, 512 * kMiB, 0.1));
  fixture.memory.hold("cache", 3 * kGiB);
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(1.0));
  require(cache.taken.size() == 2, "a warning pass did not free the margin");
  static_cast<void>(fixture.pass(1.5));
  require(cache.taken.size() == 2, "a pass before the host settled freed more");
  static_cast<void>(fixture.pass(2.0));
  require(cache.taken.size() == 4, "the next settled pass did not free the margin again");
}

// The host claim leaves units a request will reuse unless the pressure is
// critical.
void testHostClaimTakesNoCertainReuseUnlessCritical() {
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim certain = unit(1, kGiB, 0.0);
  certain.certainReuse = true;
  cache.units = {certain};
  ledger.addSource(cache);
  const ClaimSpec host{.cls = ClaimClass::Host, .holdSeconds = kValueHorizonSeconds};
  require(ledger.available(host, kGiB, 0, false, true) == 0, "the host took a unit a request reuses");
  require(ledger.available(host, kGiB, 0, true, true) == kGiB, "critical pressure left a unit a request reuses");
}

// The deficit counts what is landing: a band pass frees only the rest.
void testDeficitNetOfReturning() {
  // 512 MiB of headroom beside the cache's 800 MiB: a deficit of 512 MiB,
  // all of it landing (no pass), or 312 MiB of it beyond 200 MiB landing.
  for (const uint64_t landing : {512 * kMiB, 200 * kMiB}) {
    Fixture fixture(host(512 * kMiB + 800 * kMiB));
    TestSource cache("cache", &fixture.memory);
    for (uint64_t key = 1; key <= 8; ++key)
      cache.units.push_back(unit(key, 100 * kMiB, 0.1));
    fixture.memory.hold("cache", 800 * kMiB);
    fixture.ledger().addSource(cache);
    fixture.ledger().setReturning([&] { return Returning{.landingBytes = landing}; },
                                  [](std::span<const uint64_t>) { return Returning{}; });
    static_cast<void>(fixture.pass(1.0));
    require(cache.taken.size() == (landing == 512 * kMiB ? 0u : 4u),
            "the band pass did not free the deficit net of what is landing");
  }
}

// Missing telemetry holds elective work back but gives no cache back: no
// deficit is measured.
void testMissingTelemetryGivesNoCacheBack() {
  bool measured = true;
  metal::MetalBackend backend("");
  MemoryGovernor governor(backend, 8 * kGiB, 2 * kGiB, 2 * kGiB, [&]() -> std::optional<HostMemory> {
    return measured ? std::optional<HostMemory>(HostMemory{16 * kGiB, 16 * kGiB}) : std::nullopt;
  }, 0);
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1)};
  governor.ledger().addSource(cache);
  measured = false;
  static_cast<void>(governor.controlPass(1000.0));
  require(cache.taken.empty() && !governor.admits(ClaimClass::Start),
          "a pass without telemetry gave cache back or admitted a start");
}

// A host pass that waits on landings asks the transport to run the pass
// again at the next command-free point.
void testRerunWhileTheDeficitWaitsOnLandings() {
  Fixture fixture(host(8 * kGiB, MemoryPressure::Warning));
  TestSource cache("cache", &fixture.memory);
  cache.units = {unit(1, kGiB, 0.1, true)};
  fixture.memory.hold("cache", kGiB);
  fixture.ledger().addSource(cache);
  require(fixture.pass(1.0) && cache.taken.size() == 1, "a host pass that took a demotion did not ask to run again");
  require(!fixture.pass(3.0), "a pass with nothing returning asked to run again");
}

// macOS's warning alone takes cache, never a pool unit or a conversion,
// while the host lacks nothing past its margin; a measured deficit may take
// a pool unit by value (amend/M.md B3; kept behaviour, MemoryGovernor.cpp:
// 391-399).
void testWarningAloneTakesNoPoolUnit() {
  for (const bool deficit : {false, true}) {
    // The headroom beside the pool's 2 GiB and the cache's 1 GiB.
    const uint64_t headroom = deficit ? kGiB - 300 * kMiB : kGiB + 512 * kMiB;
    Fixture fixture(host(headroom + 3 * kGiB, MemoryPressure::Warning));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 1, 3, 4, 2, 0.0, 0.0);
    TestSource cache("cache", &fixture.memory), engine("engine");
    fixture.memory.hold("cache", kGiB);
    cache.units = {unit(1, 512 * kMiB, 3.0), unit(2, 512 * kMiB, 3.0)};
    Victim conversion = unit(3, kGiB, 2.0, true);
    conversion.kind = VictimKind::Conversion;
    engine.units = {conversion};
    fixture.ledger().addSource(cache);
    fixture.ledger().addSource(engine);
    fixture.ledger().addPool(experts);
    static_cast<void>(fixture.pass(1.0));
    if (!deficit)
      require(cache.taken.size() == 2 && experts.units() == 2 && engine.taken.empty(),
              "macOS's warning alone took a pool unit or converted a lane");
    else
      require(experts.units() == 1 && cache.taken.empty(),
              "a measured deficit did not take the cheapest unit, the pool's");
  }
}

// The store shrinks for a long prompt and grows back by displacing copied
// cache at the next pass after the prompt ends, the host's footprint
// unchanged, under persistent warning and in the band (amend/M.md B4; the
// base grew only with 1 GiB of headroom and no request waiting).
void storeRegrowsByDisplacement(MemoryPressure pressure) {
  // 800 MiB of headroom once the engine holds 17 GiB.
  Fixture fixture(host(800 * kMiB + 17 * kGiB, pressure), 16 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 4, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  // A long prompt 1 GiB short takes the cheapest unit, the store's (6 s
  // over its hold against 9 s of recompute for the cached unit).
  fixture.memory.hold("weights", 11 * kGiB);
  cache.units = {unit(1, kGiB, 9.0, true)};
  const ClaimId prompt = ledger.open(request(1, RequestPriority::Normal, ClaimClass::InService, 60.0));
  require(fixture.allocateFreeing(prompt, 2 * kGiB) && experts.units() == 3 && cache.taken.empty(),
          "the prompt did not take a store unit");
  // It ends: its KV and the cached unit, copied, are 3 GiB of cache.
  ledger.close(prompt);
  fixture.memory.release("kv", 2 * kGiB);
  fixture.memory.hold("cache", 3 * kGiB);
  cache.units = {unit(2, kGiB, 0.05), unit(3, kGiB, 0.05), unit(4, kGiB, 0.05)};
  static_cast<void>(fixture.pass(10.5));
  // The host pass frees its deficit; the displacement gives the store its
  // unit back for another, leaving the host's footprint where it was.
  require(experts.units() == 4 && cache.taken.size() == 2 && fixture.memory.total() == 16 * kGiB,
          pressure == MemoryPressure::Warning ? "under persistent warning the store did not grow back by displacement"
                                              : "in the band the store did not grow back by displacement");
}
void testStoreRegrowsByDisplacementUnderPersistentWarning() { storeRegrowsByDisplacement(MemoryPressure::Warning); }
void testStoreRegrowsByDisplacementInTheBand() { storeRegrowsByDisplacement(MemoryPressure::Normal); }

// Under macOS's warning, while the host lacks nothing past its margin, the
// store a unit short grows back before the host's pass, which may take only
// cache then, takes the cache it displaces (amend/M.md B4: one unit of
// copied cache is enough; the base's growExperts ran before the reclaim
// under the warning alone): it takes the one unit of copied cache there is
// for its own, the host's footprint unchanged.
void testTheStoreRegrowsBeforeTheWarningPassTakesItsCache() {
  // 1.5 GiB of headroom beside the engine's 7.5 GiB.
  Fixture fixture(host(1536 * kMiB + 7680 * kMiB, MemoryPressure::Warning), 8 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("cache", kGiB);
  fixture.memory.hold("weights", 3584 * kMiB);
  cache.units = {unit(1, kGiB, 0.05)};
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 400.0, 0.0, 10.0);
  static_cast<void>(fixture.pass(10.5));
  require(experts.units() == 4 && cache.taken == std::vector<uint64_t>{1} && fixture.memory.total() == 7680 * kMiB,
          "under the warning alone the host's pass took the cache the store grows back by");
}

// A pool's growth waits on nothing but its own victims (R3, amend/M.md B4):
// neither on a landing a waiting lane is owed (V's M-P3: the landing pays
// its owner first) nor on another lane's owed copies. A lane waits for its
// demotions to land with no room free beside a store worth growing: the
// pass reserves nothing, and the lane is granted the landing. With half a
// unit free and another lane owing copies, a start fits beside the store.
void testGrowthPendsOnlyOnItsOwnVictims() {
  {
    Fixture fixture(host(32 * kGiB), 8 * kGiB);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
    fixture.memory.hold("weights", 4 * kGiB);
    fixture.memory.hold("cache", kGiB);
    MemoryLedger &ledger = fixture.ledger();
    Returning returning{.landingBytes = kGiB};
    ledger.setReturning([&] { return returning; }, [](std::span<const uint64_t>) { return Returning{}; });
    ledger.addPool(experts);
    serve(ledger, experts, 400.0, 0.0, 10.0);
    const ClaimId lane = ledger.open(request(1, RequestPriority::Normal));
    require(!fixture.allocate(lane, kGiB) &&
                fixture.governor->free(lane, fixture.governor->need(lane)).outcome == FreeOutcome::Pending,
            "the lane did not wait for its demotions to land");
    static_cast<void>(fixture.pass(10.5));
    require(fixture.governor->snapshot().reservedBytes == 0 && experts.units() == 3,
            "the store's growth was reserved on the landing the lane waits for");
    fixture.memory.release("cache", kGiB);
    returning = {};
    require(fixture.allocate(lane, kGiB), "the lane was refused the landing it waited for");
  }
  {
    Fixture fixture(host(32 * kGiB), 8 * kGiB);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
    fixture.memory.hold("weights", 4608 * kMiB);
    MemoryLedger &ledger = fixture.ledger();
    ledger.setReturning([] { return Returning{.owedBytes = 4 * kGiB}; },
                        [](std::span<const uint64_t>) { return Returning{}; });
    ledger.addPool(experts);
    serve(ledger, experts, 400.0, 0.0, 10.0);
    static_cast<void>(fixture.pass(10.5));
    const ClaimId start = ledger.open(request(2, RequestPriority::Normal, ClaimClass::Start));
    require(fixture.governor->snapshot().reservedBytes == 0 && fixture.allocate(start, 256 * kMiB),
            "the store's growth was reserved on another lane's owed copies and refused a start the room fits");
  }
}

// Displacement sizes what it takes against the room as it stands (L9: all
// or none). Two pools in one pass: the hot block cache, worth more, grows
// 4 x 128 MiB into 896 MiB of free room; the store's 1 GiB unit then lacks
// 640 MiB, which its 216 MiB of cache cannot meet, so nothing goes (sized
// against the pass's 896 MiB it lacked 128 MiB, and the cache went for a
// growth the room refused). In the band without the host's consent, with
// 256 MiB free and the lanes projecting 512 MiB, the store frees the whole
// unit now and the 256 MiB the room lacks besides, and grows (freeing the
// unit alone, its growth was refused).
void testDisplacementTakesNothingItCannotGrowInto() {
  {
    Fixture fixture(host(32 * kGiB), 8 * kGiB);
    FakeElasticPool hot("hot_kv", fixture.memory, 128 * kMiB, 1, 5, 16, 1, 9.0, 0.0);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
    TestSource cache("cache", &fixture.memory);
    fixture.memory.hold("cache", 216 * kMiB);
    fixture.memory.hold("weights", 3880 * kMiB);
    cache.units = {unit(1, 108 * kMiB, 0.05), unit(2, 108 * kMiB, 0.05)};
    MemoryLedger &ledger = fixture.ledger();
    ledger.addSource(cache);
    ledger.addPool(hot);
    ledger.addPool(experts);
    for (double t = 0.0; t < 10.0; t += 0.5) {
      hot.served(0, 200, 0, 0);
      experts.served(0, 200, 0, 0);
      ledger.observePools(t + 0.5, true);
    }
    static_cast<void>(fixture.pass(10.5));
    require(hot.units() == 5 && experts.units() == 3 && cache.taken.empty(),
            "the store's displacement took cache for a unit the room the hot block cache left refused");
  }
  {
    // 1536 MiB of headroom beside the engine's 7936 MiB: the host admits
    // the unit only on its credit.
    Fixture fixture(host(1536 * kMiB + 7936 * kMiB), 8 * kGiB);
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 3, 0.25, 60.0);
    TestSource cache("cache", &fixture.memory);
    fixture.memory.hold("cache", 1280 * kMiB);
    fixture.memory.hold("weights", 3584 * kMiB);
    cache.units = {unit(1, kGiB, 0.05), unit(2, 256 * kMiB, 0.05)};
    MemoryLedger &ledger = fixture.ledger();
    ledger.addSource(cache);
    ledger.addPool(experts);
    ledger.setLanes(512 * kMiB, std::nullopt);
    serve(ledger, experts, 400.0, 0.0, 10.0);
    static_cast<void>(fixture.pass(10.5));
    require(experts.units() == 4 && cache.taken == std::vector<uint64_t>{1, 2},
            "without the host's consent the displacement did not free the unit and what the room lacked");
  }
}

// A claim lacks what admission asks of it: its need and the room claims
// ranked above it hold, beyond the room. With 512 MiB free and 1 GiB held
// above it, a 1 GiB claim lacks 1.5 GiB: 1 GiB of cache does not meet it,
// so it takes nothing (sized at its 1 GiB it took the cache for an
// allocation admission refused), available() counts 512 MiB, and with
// another 1 GiB of cache it takes both.
void testAClaimLacksTheRoomHeldAboveIt() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1)};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimSpec spec = request(2, RequestPriority::Background);
  const ClaimId low = ledger.open(spec);
  require(ledger.available(spec, kGiB, 512 * kMiB, false, false) == 512 * kMiB,
          "available() counted the room the claim above holds");
  require(ledger.free(low, {.bytes = kGiB}, 512 * kMiB, false, false).outcome == FreeOutcome::Short &&
              cache.taken.empty(),
          "the claim took cache for an allocation the room held above it refuses");
  cache.units.push_back(unit(2, kGiB, 0.1));
  require(ledger.free(low, {.bytes = kGiB}, 512 * kMiB, false, false).outcome == FreeOutcome::Met &&
              cache.taken.size() == 2,
          "the claim did not take what it and the room held above it lack");
}

// A need met in place asks admission for no bytes, so the room claims
// ranked above it hold is none of what it lacks (R2): below a claim waiting
// for 1 GiB with no room free, a lane's 8 pages are met by the leaf that
// leaves them, neither waiting on nothing landing nor, for an Immediate
// claim, taking cache bytes that leave no page.
void testANeedMetInPlaceLacksNoHeldBytes() {
  MemoryLedger ledger;
  TestSource cache("cache");
  Victim leaf = unit(1, 0, 0.2);
  leaf.pages = 8;
  cache.units = {leaf};
  ledger.addSource(cache);
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  waitForItsCopies(ledger, high, {.bytes = kGiB});
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(low, {.pages = 8}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1} && !ledger.claim(low).waiting,
          "a pages-only need waited on the room held above it instead of taking the leaf");
  leaf.key = 3;
  cache.units = {unit(2, kGiB, 0.1), leaf};
  ClaimSpec spec = request(3, RequestPriority::Background);
  spec.timing = NeedTiming::Immediate;
  const ClaimId now = ledger.open(spec);
  require(ledger.free(now, {.pages = 8}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1, 3},
          "an Immediate pages-only need took cache bytes that leave no page");
}

// Free room goes back to a pool claims keep taking units from only while a
// unit is worth its expected retake: with no ghost value and a claim every
// 20 s, at most one regrowth per three recurrences.
void testFreeRoomGrowthWaitsWhileClaimsRecur() {
  Fixture fixture(host(32 * kGiB), 6 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 4, 0.25, 60.0);
  fixture.memory.hold("weights", 2 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  uint64_t recurrences = 0;
  std::optional<ClaimId> claim;
  for (int step = 0; step < 480; ++step) {
    const double seconds = step * 0.5;
    if (step % 40 == 0) {
      claim = ledger.open(request(1 + step, RequestPriority::Normal));
      if (!fixture.allocate(*claim, kGiB)) {
        static_cast<void>(fixture.governor->free(*claim, fixture.governor->need(*claim)));
        require(fixture.allocate(*claim, kGiB), "the claim was refused");
      }
      ++recurrences;
    } else if (step % 40 == 20) {
      fixture.memory.release("kv", kGiB);
      ledger.close(*claim);
    }
    static_cast<void>(fixture.pass(seconds));
  }
  require(experts.grows * 3 <= recurrences, "free room went back to the pool while claims kept retaking it");
}

// A store that starts short of host memory grows to its cap within two
// passes once other applications free M + 1.5 units (amend/M.md A109; the
// base asked 2 GiB beyond the bytes, MemoryGovernor.cpp:208-215, and a
// cool-down).
void testShortStartRegrowsWhenTheHostFrees() {
  // Headroom 512 MiB beside the store's 2 GiB until 60 s, then M + 1.5 units.
  const uint64_t store = 2 * kGiB;
  Fixture fixture({.hostBytes = 64 * kGiB,
                   .otherBytes = 62 * kGiB - 512 * kMiB - store,
                   .script = test::lowHostAtStart(62 * kGiB - 512 * kMiB - store, 62 * kGiB - 2560 * kMiB - store,
                                                  60000.0)},
                  16 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 3, 4, 2, 0.25, 60.0);
  fixture.ledger().addPool(experts);
  for (int step = 0; step < 120; ++step)
    static_cast<void>(fixture.pass(step * 0.5));
  require(experts.units() == 2, "the store grew while the host was short");
  static_cast<void>(fixture.pass(60.0));
  static_cast<void>(fixture.pass(60.5));
  require(experts.units() == 3, "the store did not reach its cap within two passes of the host freeing");
}

// A lone converted lane waits for its own copies before it takes a store
// unit (amend/M.md B5; engine_test.cpp 10579/11078): its owed bytes count
// with the landings.
void testALoneConvertedLaneWaitsForItsOwnCopies() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 4, 3, 0.0, 0.0);
  MemoryLedger ledger;
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.owedBytes = 2 * kGiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 5) != lanes.end() ? Returning{.owedBytes = 2 * kGiB}
                                                                          : Returning{};
                      });
  const ClaimId lane = ledger.open(request(5, RequestPriority::Normal));
  require(ledger.free(lane, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending && experts.units() == 3,
          "a lone Spill lane took a store unit instead of waiting for its own copies");
}

// A claim that converted a lane waits for that lane's copies: its retries
// convert nothing else (the base converted another lane at each,
// Engine.cpp:2376-2384).
void testARetryAfterItsConversionTakesNothingElse() {
  MemoryLedger ledger;
  TestSource engine("engine");
  for (uint64_t id : {11, 12}) {
    Victim conversion = unit(id, kGiB, 1.0, true);
    conversion.kind = VictimKind::Conversion;
    engine.units.push_back(conversion);
  }
  ledger.addSource(engine);
  ledger.setReturning([&] { return Returning{}; },
                      [&](std::span<const uint64_t> lanes) {
                        const auto converted = [&](uint64_t id) { return std::ranges::count(engine.taken, id) != 0; };
                        return std::ranges::any_of(lanes, converted)
                                   ? Returning{.owedBytes = kGiB}
                                   : Returning{};
                      });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending &&
              engine.taken == std::vector<uint64_t>{11},
          "the claim did not convert one lane");
  for (int retry = 0; retry < 5; ++retry)
    require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending &&
                engine.taken.size() == 1,
            "a retry converted another lane while the first one's copies were owed");
}

// Another lane's owed copies do not hold a claim back from a store unit.
void testOneConversionDoesNotStallTheStore() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 4, 3, 0.0, 0.0);
  MemoryLedger ledger;
  ledger.addPool(experts);
  ledger.setReturning([] { return Returning{.owedBytes = 4 * kGiB}; },
                      [](std::span<const uint64_t> lanes) {
                        return std::ranges::find(lanes, 9) != lanes.end() ? Returning{.owedBytes = 4 * kGiB}
                                                                          : Returning{};
                      });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met && experts.units() == 2,
          "another lane's write-through held the claim back from the store");
}

// No pool unit for a claim below a lane in service: pools serve every lane
// (amend/M.md B7).
void testALowerPriorityClaimTakesNoPoolUnitWhileAHigherLaneServes() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 1, 3, 4, 3, 0.0, 0.0);
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 512 * kMiB, 1.0)};
  ledger.addSource(cache);
  ledger.addPool(experts);
  ledger.setLanes(0, RequestPriority::Foreground);
  const ClaimId background = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(background, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Short &&
              experts.units() == 3 && cache.taken.empty(),
          "a Background claim took a store unit while a Foreground lane served");
  ledger.setLanes(0, RequestPriority::Background);
  require(ledger.free(background, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met && experts.units() == 2,
          "a claim at the lanes' priority was refused a store unit");
}

// Copies ahead pay only after a free() reached an uncopied cached unit, and
// for the value horizon after it (amend/M.md B9): not for pending growth or
// short room alone (R7 as first written), from an Immediate claim's skip on
// (the base: only a reclaim to disk), and not past the horizon (the base:
// 300 s).
void testCopiesAheadOnlyAfterDemandReachedUncopiedCache() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.1, true), unit(2, kGiB, 0.2)};
  ledger.addSource(cache);
  ledger.observePools(100.0, true);
  require(!ledger.copyDemand(), "copies ahead were wanted before any free()");
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate});
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{2},
          "the Immediate claim did not skip the uncopied unit");
  require(ledger.copyDemand(), "an Immediate claim's skip of an uncopied unit asked no copies");
  ledger.observePools(100.0 + kValueHorizonSeconds, true);
  require(ledger.copyDemand(), "the demand ended within the horizon");
  ledger.observePools(100.5 + kValueHorizonSeconds, true);
  require(!ledger.copyDemand(), "copies ahead outlasted the horizon");
}

// The pool's value outlives an idle release and restore (amend/M.md B10):
// an idle interval does not dilute its rates, and the ghosts' span decays
// over the rates' time constant, not at once.
void testPoolValueSurvivesAnIdleReleaseAndRestore() {
  class Store final : public model::ElasticPool {
  public:
    [[nodiscard]] model::PoolState state() const override {
      return {.name = "experts", .floorBytes = kGiB, .currentBytes = 3 * kGiB, .capBytes = 3 * kGiB,
              .releaseUnitBytes = kGiB};
    }
    [[nodiscard]] model::PoolValue value() const override {
      return {.ghostHits = ghostHits, .ghostBytes = ghostBytes, .missMilliseconds = 0.25};
    }
    uint64_t release(uint64_t) override { return 0; }
    metal::AllocationResult grow(const metal::AllocationAdmission &) override { return {}; }
    uint64_t ghostHits = 0, ghostBytes = 2 * kGiB;
  } store;
  Fixture fixture(host(32 * kGiB));
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(store);
  double t = 0.0;
  for (; t < 120.0; t += 0.5) {
    store.ghostHits += 200;
    ledger.observePools(t, true);
  }
  const auto value = [&] { return fixture.governor->snapshot().accounts.back().valuePerGiB; };
  const double before = value();
  for (const double end = t + 60.0; t < end; t += 0.5)
    ledger.observePools(t, false);
  store.ghostBytes = 0;
  store.ghostHits += 200;
  ledger.observePools(t, true);
  require(before > 0.0 && std::abs(value() - before) <= 0.1 * before,
          "the pool's value after an idle release and restore is not within 10 % of before");
}

// A shortfall of X frees at most X and one unit of a pool (owner 21:05,
// borrowed item 3(c)), also when the unit that goes first is the smallest:
// a store whose last live chunk is 117 MiB, the rest of its size, before two
// of 1023 MiB gives the small chunk and one large one for 300 MiB, not every
// large chunk the dry run's 117 MiB units stood for.
void testAShortfallTakesAtMostOneUnitBeyondIt() {
  for (const uint64_t unitBytes : {kGiB, 512 * kMiB}) {
    FakeMemory memory({});
    FakeElasticPool experts("experts", memory, unitBytes, 0, 8, 9, 8, 0.0, 0.0);
    MemoryLedger ledger;
    ledger.addPool(experts);
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
    const uint64_t shortfall = 1331 * kMiB;
    const FreeResult freed = ledger.free(claim, {.bytes = shortfall}, 0, false, false);
    require(freed.outcome == FreeOutcome::Met && freed.nowBytes >= shortfall &&
                freed.nowBytes < shortfall + unitBytes,
            "a shortfall took more than itself and one unit");
  }
  ListedPool experts(kGiB, {1023 * kMiB, 1023 * kMiB, 117 * kMiB}, 8 * kGiB, 1382400, 0.1846);
  MemoryLedger ledger;
  ledger.addPool(experts);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  const FreeResult freed = ledger.free(claim, {.bytes = 300 * kMiB}, 0, false, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 1140 * kMiB &&
              experts.currentBytes() == kGiB + 1023 * kMiB,
          "a shortfall of 300 MiB took " + std::to_string(freed.nowBytes / kMiB) +
              " MiB of a store whose small chunk goes first, not it and one large chunk");
}

// A host reading that lags the release behind it takes nothing more: the
// next pass waits for the host to settle (owner 21:05, borrowed item 3(d)).
void testALaggingHostReadingTakesNoSecondRelease() {
  // 512 MiB of headroom beside the cache's 2 GiB; after the pass other
  // applications' bytes hide the 512 MiB it returns for 800 ms.
  Fixture fixture(host(2560 * kMiB, MemoryPressure::Normal, {at(1001.0, 2 * kGiB), at(1800.0, 2560 * kMiB)}));
  TestSource cache("cache", &fixture.memory);
  for (uint64_t key = 1; key <= 8; ++key)
    cache.units.push_back(unit(key, 256 * kMiB, 0.1));
  fixture.memory.hold("cache", 2 * kGiB);
  fixture.ledger().addSource(cache);
  static_cast<void>(fixture.pass(1.0));
  const size_t first = cache.taken.size();
  require(first == 2, "the band pass did not free its deficit");
  static_cast<void>(fixture.pass(1.5));
  require(cache.taken.size() == first, "a lagging host reading took a second release");
  static_cast<void>(fixture.pass(2.0));
  require(cache.taken.size() == first, "the settled reading took more than the deficit");
}

// A pool's floor bounds every take: a pool whose floor is its cap gives
// nothing, one at 0.85 of its cap down to that (amend/M.md B17).
void testPoolFloorBoundsEveryTake() {
  for (const uint32_t floor : {20u, 17u}) {
    FakeMemory memory({});
    FakeElasticPool experts("experts", memory, 256 * kMiB, floor, 20, 21, 20, 0.0, 0.0);
    MemoryLedger ledger;
    ledger.addPool(experts);
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
    static_cast<void>(ledger.free(claim, {.bytes = 2 * kGiB}, 0, false, false));
    static_cast<void>(ledger.free(claim, {.bytes = 512 * kMiB}, 0, false, false));
    require(experts.units() == (floor == 20 ? 20 : 18),
            "a claim took a pool below its floor or less than the need");
    static_cast<void>(ledger.free(claim, {.bytes = 256 * kMiB}, 0, false, false));
    require(experts.units() == floor, "the pool did not give units down to its floor");
  }
}

// One line per decision: the claimant, what it took, the price, the bytes
// and the outcome; a retry that decides the same again writes nothing
// (owner 21:05, borrowed item 4).
void testEachDecisionLogsOneLine() {
  MemoryLedger ledger;
  std::vector<std::string> lines;
  ledger.setDecisionLog([&](std::string_view line) { lines.emplace_back(line); });
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.25)};
  ledger.addSource(cache);
  const ClaimId claim = ledger.open(request(4, RequestPriority::Normal));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false, false));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false, false));
  static_cast<void>(ledger.free(claim, {.bytes = kGiB}, 0, false, false));
  require(lines.size() == 2 &&
              lines[0] == "Memory: in-service claim of request 4 needs 1024.0 MiB: met, took cache 1024.0 for 0.250 s "
                          "(1024.0 MiB now, 0.0 later)" &&
              lines[1] == "Memory: in-service claim of request 4 needs 1024.0 MiB: short (0.0 MiB now, 0.0 later)",
          "the decisions were not logged once each: " + (lines.empty() ? std::string() : lines[0]));
}

// HiSparse pricing on calibrated inputs (amend/M.md B8; owner 20:45 OQ1:
// the value outcome). A claim of a long prompt's hold, 200 s, lacks 1 GiB;
// the engine offers a decoding Resident lane's conversion with 1500 tokens
// left, priced by the spilled decode's term at DFlash2's 7.94 tokens a lane
// command, beside the store's units, priced by their ghost hits.
struct HiSparse final {
  HiSparse(uint64_t unitBytes, double ghostsPerSecond, double missMilliseconds, uint64_t laneBytes,
           double spillMillisecondsPerToken)
      : experts("experts", memory, unitBytes, 2, 6, 7, 6, missMilliseconds, 62.7) {
    rates = {.spilledReadMillisecondsPerLaneCommand = spillMillisecondsPerToken * 7.94,
             .decodeTokensPerLaneCommand = 7.94};
    ledger.setRates(rates);
    ledger.addSource(engine);
    ledger.addPool(experts);
    serve(ledger, experts, ghostsPerSecond, 0.0, 30.0);
    Victim lane = unit(3, laneBytes, 0.0, true);
    lane.kind = VictimKind::Conversion;
    lane.cost = value::conversion(value::spillDecodeSeconds(1500, rates));
    engine.units = {lane};
  }
  // What a claim of the prompt's hold lacking `bytes` takes: the
  // conversion, else the store gave a unit.
  bool converts(uint64_t bytes) {
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::InService, 200.0));
    static_cast<void>(ledger.free(claim, {.bytes = bytes}, 0, false, false));
    require(engine.taken.empty() != (experts.units() == 6), "the claim took both or neither");
    return !engine.taken.empty();
  }

  FakeMemory memory{{}};
  FakeElasticPool experts;
  MemoryLedger ledger;
  TestSource engine{"engine"};
  MemoryRates rates;
};

// M6 (kM6FlashNextQ2): a 1023 MiB chunk saves 23.5 MB a decode token (the
// profile's 26.2 MB for the 866 slots between floor and cap), 17 expert reads
// of 0.1846 ms (1,382,400 B at 7.49 GB/s), at mMNa's 21.6 tokens a second;
// converting a 128K lane frees 1.09 GiB (D9) at Spill's median +1.0 ms a
// token (D3), and at its worst +2.7: the lane converts, the store keeps its
// experts.
void testHiSparseOnM6ConvertsBeforeTheStore() {
  for (const double spill : {1.0, 2.7}) {
    HiSparse m6(1023 * kMiB, 21.6 * 17.0, 0.1846, 1116 * kMiB, spill);
    require(m6.converts(kGiB), "on the M6 the store gave a chunk before a 128K lane converted");
  }
}

// Studio 36 GB, decode alone: the store is worth +2-3 % a GiB at 5.9 ms a
// token (0.15 ms a token a GiB) at 21.6 tokens a second, 0.2222 ms an
// expert (1,382,400 B at 6.22 GB/s); a 64K lane frees 0.33 GiB (D9) at
// Spill's +2.0 ms a token: a 678 MiB chunk goes, the lane keeps decoding
// Resident. Without the decode term (the mutant) the lane would convert.
void testHiSparseOnStudio36GWeighsSpillDecode() {
  const uint64_t chunk = 678 * kMiB;
  const double ghosts = 21.6 * 0.15 / 0.2222 * double(chunk) / double(kGiB);
  HiSparse studio(chunk, ghosts, 0.2222, 338 * kMiB, 2.0);
  require(!studio.converts(300 * kMiB), "on a 36 GB Studio decoding the lane converted before a chunk went");
  HiSparse mutant(chunk, ghosts, 0.2222, 338 * kMiB, 0.0);
  require(mutant.converts(300 * kMiB), "the lane's conversion priced without the decode term did not go first");
}

// HiSparse off (35B, k35BStudio18G: no Spill, so the engine offers no
// conversion): copied cache goes first, then a store unit by value. A 1022
// MiB chunk saves 1.07 MB a decode token (0.56 expert reads of 1,911,603 B
// at 6.22 GB/s, 0.3073 ms each) at a lone lane's 253 tokens a second; a
// copied 256 MiB chain reads back in 54 ms at 5 GB/s, reused a quarter of
// the time.
void testHiSparseOffTakesCacheThenStoreAndNeverConverts() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, 1022 * kMiB, 2, 5, 6, 5, 0.3073, 62.7);
  MemoryLedger ledger;
  TestSource cache("cache");
  const MemoryRates rates{.tierReadBytesPerSecond = 5.0e9, .reuse = 0.25};
  ledger.setRates(rates);
  for (uint64_t key = 1; key <= 2; ++key) {
    Victim chain = unit(key, 256 * kMiB, 0.0);
    chain.cost = value::cached(8192, 256 * kMiB, true, false, rates);
    cache.units.push_back(chain);
  }
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 253.0 * 0.56, 0.0, 30.0);
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::InService, 200.0));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1, 2} && experts.units() == 4,
          "without Spill the claim did not take the cache first and then one store unit");
}

// The ledger values the store by the scans it reports (amend/M.md B8, A18):
// on the M6 (memory_value_test's testExpertValueCountsPrefillReads), with
// no decoder and no ghost, the store's scans read 3.2 GB a prefill-second
// over the 18,854 Flash-Next slots of 1,382,400 B it does not hold, at the
// profile's weight price; a claim of a long prompt's hold lacking an 8K
// copied chain takes the chain, read back at 5 GB/s at reuse 0.25, before
// the 1023 MiB chunk. A store whose scans read nothing costs about one miss
// and gives the chunk.
void testTheLedgerValuesTheStoreByItsScans() {
  constexpr uint64_t kSlotBytes = 1382400, kExtentBytes = 108527616;
  const MemoryRates rates{.weightReadSecondsPerByte = 0.2 / 5.0e9, .tierReadBytesPerSecond = 5.0e9, .reuse = 0.25};
  for (const double scanBytesPerSecond : {3.2e9, 0.0}) {
    ListedPool store(5722 * kSlotBytes - 1023 * kMiB, {1023 * kMiB}, 24576 * kSlotBytes, kSlotBytes, 0.1846);
    TestSource cache("cache");
    Victim chain = unit(1, kExtentBytes, 0.0);
    chain.cost = value::cached(8192, kExtentBytes, true, false, rates);
    cache.units = {chain};
    MemoryLedger ledger;
    ledger.setRates(rates);
    ledger.addSource(cache);
    ledger.addPool(store);
    // Prefill hits at the store's 0.288 share of the uses.
    const double hitsPerSecond = 3.2e9 / kSlotBytes * 0.288 / 0.712;
    for (double t = 0.0; t <= 30.0; t += 0.5) {
      ledger.observePools(t, true);
      store.hits += static_cast<uint64_t>(std::llround(hitsPerSecond * 0.5));
      store.scanReadBytes += static_cast<uint64_t>(std::llround(scanBytesPerSecond * 0.5));
    }
    const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::InService, 200.0));
    require(ledger.free(claim, {.bytes = kExtentBytes}, 0, false, false).outcome == FreeOutcome::Met,
            "the claim was not met");
    if (scanBytesPerSecond > 0.0)
      require(cache.taken == std::vector<uint64_t>{1} && store.currentBytes() == 5722 * kSlotBytes,
              "the store's scans did not keep its chunk before a copied chain");
    else
      require(cache.taken.empty() && store.currentBytes() == 5722 * kSlotBytes - 1023 * kMiB,
              "a store that scans nothing kept its chunk before a copied chain");
  }
}

// Displacement pays for the retake it expects, as growth into free room
// does (amend/V.md A30: no unit taken back within kValueHorizonSeconds of its
// regrow): a 1 GiB store unit worth about 0.12 s over the horizon (160 ghost
// hits a second at 0.25 ms over the 20 GiB its ghosts span) whose 256 keys
// of 4 MiB read back in 64 ms displaces cached KV worth 0.03 s while no
// claim takes the store's units (0.03 < 0.12 - 0.064). A claim of 1 GiB
// every 10 s takes a store unit (0.004 s over its 2 s hold) before the cache
// each time: a retake within the horizon is likely, which the unit does not
// pay for, and the cache stays.
void testDisplacementPaysForItsExpectedRetake() {
  Fixture fixture(host(32 * kGiB), 12 * kGiB);
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 10, 30, 10, 0.25, 0.0);
  experts.keyBytes = 4 * kMiB;
  TestSource cache("cache", &fixture.memory);
  fixture.memory.hold("weights", 2 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addSource(cache);
  ledger.addPool(experts);
  serve(ledger, experts, 160.0, 0.0, 10.0);
  std::optional<ClaimId> claim;
  std::optional<double> regrown;
  uint64_t recurrences = 0, retaken = 0, grows = 0, shrinks = 0;
  std::string history;
  for (int step = 0; step < 320; ++step) {
    const double seconds = 10.5 + step * 0.5;
    experts.served(0, 80, 0, 0);
    if (step % 20 == 0) {
      claim = ledger.open(request(1 + step, RequestPriority::Normal, ClaimClass::InService, 2.0));
      require(fixture.allocateFreeing(*claim, kGiB), "the claim was refused");
      ++recurrences;
    } else if (step % 20 == 10) {
      // It ends: its KV is cache.
      fixture.memory.release("kv", kGiB);
      ledger.close(*claim);
      cache.units.push_back(unit(100 + step, kGiB, 0.03));
      fixture.memory.hold("cache", kGiB);
    }
    static_cast<void>(fixture.pass(seconds));
    // A regrown unit taken back is the next unit the store gives after it.
    if (experts.shrinks != shrinks) {
      shrinks = experts.shrinks;
      if (regrown && seconds - *regrown < kValueHorizonSeconds)
        ++retaken;
      regrown.reset();
    }
    if (experts.grows != grows) {
      grows = experts.grows;
      regrown = seconds;
      history += " regrew at " + std::to_string(seconds) + " s to " + std::to_string(experts.units()) + " units;";
    }
  }
  require(grows * 3 <= recurrences && retaken == 0,
          "displacement regrew the store " + std::to_string(grows) + " times in " + std::to_string(recurrences) +
              " recurrences, " + std::to_string(retaken) + " of them taken back within the horizon:" + history);
}

// Growth counts what the host's counters do not show yet (smoke a: a store
// chunk's pages count once experts load into it, a second after it grew on
// the 122B and the M6): with the host's reading 1 s behind, a pass grows only
// the units the headroom keeps M beside, and none goes back once the reading
// shows them.
void testGrowthCountsWhatTheHostDoesNotShowYet() {
  // M + 1.5 GiB of headroom beside the store's 2 GiB.
  FakeMemory::Config config = host(2 * kGiB + kGiB + 1536 * kMiB);
  config.readingLagMilliseconds = 1000.0;
  Fixture fixture(std::move(config));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 6, 7, 2, 0.25, 60.0);
  fixture.ledger().addPool(experts);
  for (double t = 1.0; t <= 10.0; t += 0.5) {
    experts.served(0, 200, 0, 0);
    static_cast<void>(fixture.pass(t));
  }
  require(experts.units() == 3 && experts.shrinks == 0,
          "growth admitted against a reading that did not show it: " + std::to_string(experts.grows) + " grown, " +
              std::to_string(experts.shrinks) + " given back");
}

// Growth leaves the host what the lanes in service are projected to grow
// into, as it leaves them that room under the limit (smoke a: on the 122B a
// long prompt's KV grew into the host's margin after each unit the store
// grew back, and the host's pass took the unit back seconds later): with
// M + 1.5 GiB of headroom beside the store and lanes that project 1 GiB
// more, no 1 GiB unit grows; once they project nothing, one does.
void testGrowthLeavesTheHostWhatTheLanesGrowInto() {
  Fixture fixture(host(2 * kGiB + kGiB + 1536 * kMiB));
  FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 6, 7, 2, 0.25, 60.0);
  MemoryLedger &ledger = fixture.ledger();
  ledger.addPool(experts);
  ledger.setLanes(kGiB, RequestPriority::Normal);
  static_cast<void>(fixture.pass(1.0));
  require(experts.units() == 2, "the store grew into the host's margin the lanes are projected to grow into");
  ledger.setLanes(0, std::nullopt);
  static_cast<void>(fixture.pass(2.0));
  require(experts.units() == 3, "the store did not grow once the lanes projected nothing");
}

// A claim pays the release a pool's unit takes of the engine thread (the
// store's compaction, amend/M.md D4): an optional claim worth one state's
// read, 5 ms, takes the copied state that costs that, not a store chunk
// whose value is gone (about one miss) but whose release took 60 ms.
void testAClaimPaysForAPoolUnitsRelease() {
  FakeMemory memory({});
  FakeElasticPool experts("experts", memory, kGiB, 2, 4, 4, 4, 0.25, 60.0);
  static_cast<void>(experts.release(1));
  static_cast<void>(experts.grow([](uint64_t, const std::function<void()> &allocate) {
    allocate();
    return metal::AllocationResult{};
  }));
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, 64 * kMiB, 0.005)};
  ledger.addSource(cache);
  ledger.addPool(experts);
  const ClaimId claim =
      ledger.open({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate, .benefitSeconds = 0.005});
  require(ledger.free(claim, {.bytes = 64 * kMiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1} && experts.units() == 4,
          "an optional claim took a store chunk whose release costs more than its benefit");
}

// A need of bytes beside buffers no state can give (Model::statesToActivate
// with no draft ring idle) is met by the bytes of the states it evicts,
// which leave the model's pool (smoke a trace: the lane's activation was
// refused again with the evicted state's buffers still pooled).
void testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes() {
  MemoryLedger ledger;
  BufferPool pool(64 * kMiB);
  CachedStates cache(pool);
  cache.units.push_back({.key = 1, .stateBuffers = 1, .cost = {.onceSeconds = 0.1}});
  ledger.addSource(cache);
  ledger.addSource(pool);
  ledger.setBufferBytes([](uint32_t states) { return uint64_t{states} * 64 * kMiB; });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal, ClaimClass::Start));
  const FreeResult freed = ledger.free(
      claim, {.bytes = 48 * kMiB, .stateBuffers = std::numeric_limits<uint32_t>::max()}, 0, false, false);
  require(freed.outcome == FreeOutcome::Met && freed.nowBytes == 64 * kMiB &&
              cache.taken == std::vector<uint64_t>{1} && !pool.idle,
          "the evicted state's buffers stayed in the pool for a need no state meets in place");
}

// The pools grow before Ready by the host's rule at macOS's pressure then: a
// warning that began after the governor was built holds them at their
// floors; at normal pressure they grow to their cap.
void testStartGrowthFollowsThePressureNow() {
  for (const MemoryPressure pressure : {MemoryPressure::Warning, MemoryPressure::Normal}) {
    Fixture fixture(host(32 * kGiB, MemoryPressure::Normal, {at(100.0, 32 * kGiB, pressure)}));
    FakeElasticPool experts("experts", fixture.memory, kGiB, 2, 4, 5, 2, 0.25, 60.0);
    fixture.ledger().addPool(experts);
    fixture.memory.advanceTo(100.0);
    fixture.governor->growToTargets(100.0, fixture.memory.pressure());
    require(experts.units() == (pressure == MemoryPressure::Warning ? 2u : 4u),
            pressure == MemoryPressure::Warning ? "the start's growth went on under a warning that began after the "
                                                  "governor was built"
                                                : "the start's growth did not reach the cap at normal pressure");
  }
}

} // namespace

std::span<const test::TestCase> ledgerTests() {
  static constexpr test::TestCase kTests[] = {
      {"testAccountsSumToCharged", testAccountsSumToCharged},
      {"testSnapshotRendersEveryAccount", testSnapshotRendersEveryAccount},
      {"testWaitingClaimHoldsRoomForItsRank", testWaitingClaimHoldsRoomForItsRank},
      {"testShortClaimHoldsNothing", testShortClaimHoldsNothing},
      {"testNestedScopesRestore", testNestedScopesRestore},
      {"testHeldPagesFollowRank", testHeldPagesFollowRank},
      {"testWaitingBytesHoldPagesOfEachForm", testWaitingBytesHoldPagesOfEachForm},
      {"testAMetNeedKeepsTheWaitAShortOneEndsIt", testAMetNeedKeepsTheWaitAShortOneEndsIt},
      {"testPendingGrowthIsReservedAgainstOthers", testPendingGrowthIsReservedAgainstOthers},
      {"testPendingGrowthGrowsOnItsLandedVictims", testPendingGrowthGrowsOnItsLandedVictims},
      {"testPendingGrowthTakesWhatItsLandedVictimsLeave", testPendingGrowthTakesWhatItsLandedVictimsLeave},
      {"testPendingGrowthCreditsOnlyWhatItsTakeReleases", testPendingGrowthCreditsOnlyWhatItsTakeReleases},
      {"testAShortPendingGrowthHoldsNothing", testAShortPendingGrowthHoldsNothing},
      {"testDisplacementWithoutTheHostsConsentPendsOnUncopiedCache",
       testDisplacementWithoutTheHostsConsentPendsOnUncopiedCache},
      {"testTakenPendingGrowthIsCancelled", testTakenPendingGrowthIsCancelled},
      {"testOneHostRule", testOneHostRule},
      {"testBandWaivedWhenTheHostClaimFindsNothing", testBandWaivedWhenTheHostClaimFindsNothing},
      {"testEveryHostPassGivesBackIdleBuffers", testEveryHostPassGivesBackIdleBuffers},
      {"testDeficitCountsHostRefusedClaimsNotAnIdleMargin", testDeficitCountsHostRefusedClaimsNotAnIdleMargin},
      {"testAHostRefusedStartIsAdmittedWithinTwoPasses", testAHostRefusedStartIsAdmittedWithinTwoPasses},
      {"testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor", testGrowthLeavesTheHostWhatAStartItRefusedWaitsFor},
      {"testTheMarginalPriceIsReportedBesideTheUniform", testTheMarginalPriceIsReportedBesideTheUniform},
      {"testTheMarginalPriceIsTheUnitsShareOfItsCounts", testTheMarginalPriceIsTheUnitsShareOfItsCounts},
      {"testCheapestPerByteFirstAcrossSources", testCheapestPerByteFirstAcrossSources},
      {"testAUnitLargerThanTheNeedCostsItsWholeCost", testAUnitLargerThanTheNeedCostsItsWholeCost},
      {"testSourceOrderIsKeptWithinASource", testSourceOrderIsKeptWithinASource},
      {"testUnitsKeyedByTheirPlaceAreTakenAsChosen", testUnitsKeyedByTheirPlaceAreTakenAsChosen},
      {"testATakeConvertsOnlyTheLanesItsRunChose", testATakeConvertsOnlyTheLanesItsRunChose},
      {"testAStateThePoolKeepsGoesWithTheNextUnitOfItsSource", testAStateThePoolKeepsGoesWithTheNextUnitOfItsSource},
      {"testTiesGoToTheImmediateUnitThenTheSourceRegisteredFirst",
       testTiesGoToTheImmediateUnitThenTheSourceRegisteredFirst},
      {"testClassBoundsEligibility", testClassBoundsEligibility},
      {"testShortTakesNothing", testShortTakesNothing},
      {"testRoomHeldForHigherRanksMakesAWait", testRoomHeldForHigherRanksMakesAWait},
      {"testLandingThenVictimsThenOwed", testLandingThenVictimsThenOwed},
      {"testImmediateSkipsLandings", testImmediateSkipsLandings},
      {"testLandingCountsBeforeVictimsInPart", testLandingCountsBeforeVictimsInPart},
      {"testStateBuffersMeetANeedOfBytes", testStateBuffersMeetANeedOfBytes},
      {"testHostRefusedNeedIsMetInPlaceFirst", testHostRefusedNeedIsMetInPlaceFirst},
      {"testAnExemptRetryPassesTheHostWithEachReservation", testAnExemptRetryPassesTheHostWithEachReservation},
      {"testZeroCostUnsizedUnitsGoFirst", testZeroCostUnsizedUnitsGoFirst},
      {"testBenefitBoundsTheDryRun", testBenefitBoundsTheDryRun},
      {"testAvailableTakesNothing", testAvailableTakesNothing},
      {"testAvailableCountsWhatLandsForAClaimThatCanWait", testAvailableCountsWhatLandsForAClaimThatCanWait},
      {"testGrowthRetakesFreeRoomBeyondLaneDemand", testGrowthRetakesFreeRoomBeyondLaneDemand},
      {"testNoPingPongWhileALaneGrows", testNoPingPongWhileALaneGrows},
      {"testDisplacementNeedsValueAboveCycle", testDisplacementNeedsValueAboveCycle},
      {"testNoFlipFlopAtConstantInputs", testNoFlipFlopAtConstantInputs},
      {"testCriticalDropsPoolsToFloors", testCriticalDropsPoolsToFloors},
      {"testCriticalTakesWhatItFindsWhateverLands", testCriticalTakesWhatItFindsWhateverLands},
      {"testWarningFreesAtMostTheMarginPerSettledPass", testWarningFreesAtMostTheMarginPerSettledPass},
      {"testHostClaimTakesNoCertainReuseUnlessCritical", testHostClaimTakesNoCertainReuseUnlessCritical},
      {"testDeficitNetOfReturning", testDeficitNetOfReturning},
      {"testMissingTelemetryGivesNoCacheBack", testMissingTelemetryGivesNoCacheBack},
      {"testRerunWhileTheDeficitWaitsOnLandings", testRerunWhileTheDeficitWaitsOnLandings},
      {"testWarningAloneTakesNoPoolUnit", testWarningAloneTakesNoPoolUnit},
      {"testStoreRegrowsByDisplacementUnderPersistentWarning", testStoreRegrowsByDisplacementUnderPersistentWarning},
      {"testStoreRegrowsByDisplacementInTheBand", testStoreRegrowsByDisplacementInTheBand},
      {"testTheStoreRegrowsBeforeTheWarningPassTakesItsCache", testTheStoreRegrowsBeforeTheWarningPassTakesItsCache},
      {"testGrowthPendsOnlyOnItsOwnVictims", testGrowthPendsOnlyOnItsOwnVictims},
      {"testDisplacementTakesNothingItCannotGrowInto", testDisplacementTakesNothingItCannotGrowInto},
      {"testAClaimLacksTheRoomHeldAboveIt", testAClaimLacksTheRoomHeldAboveIt},
      {"testANeedMetInPlaceLacksNoHeldBytes", testANeedMetInPlaceLacksNoHeldBytes},
      {"testFreeRoomGrowthWaitsWhileClaimsRecur", testFreeRoomGrowthWaitsWhileClaimsRecur},
      {"testShortStartRegrowsWhenTheHostFrees", testShortStartRegrowsWhenTheHostFrees},
      {"testALoneConvertedLaneWaitsForItsOwnCopies", testALoneConvertedLaneWaitsForItsOwnCopies},
      {"testARetryAfterItsConversionTakesNothingElse", testARetryAfterItsConversionTakesNothingElse},
      {"testOneConversionDoesNotStallTheStore", testOneConversionDoesNotStallTheStore},
      {"testALowerPriorityClaimTakesNoPoolUnitWhileAHigherLaneServes",
       testALowerPriorityClaimTakesNoPoolUnitWhileAHigherLaneServes},
      {"testCopiesAheadOnlyAfterDemandReachedUncopiedCache", testCopiesAheadOnlyAfterDemandReachedUncopiedCache},
      {"testPoolValueSurvivesAnIdleReleaseAndRestore", testPoolValueSurvivesAnIdleReleaseAndRestore},
      {"testAShortfallTakesAtMostOneUnitBeyondIt", testAShortfallTakesAtMostOneUnitBeyondIt},
      {"testALaggingHostReadingTakesNoSecondRelease", testALaggingHostReadingTakesNoSecondRelease},
      {"testPoolFloorBoundsEveryTake", testPoolFloorBoundsEveryTake},
      {"testEachDecisionLogsOneLine", testEachDecisionLogsOneLine},
      {"testHiSparseOnM6ConvertsBeforeTheStore", testHiSparseOnM6ConvertsBeforeTheStore},
      {"testHiSparseOnStudio36GWeighsSpillDecode", testHiSparseOnStudio36GWeighsSpillDecode},
      {"testHiSparseOffTakesCacheThenStoreAndNeverConverts", testHiSparseOffTakesCacheThenStoreAndNeverConverts},
      {"testTheLedgerValuesTheStoreByItsScans", testTheLedgerValuesTheStoreByItsScans},
      {"testDisplacementPaysForItsExpectedRetake", testDisplacementPaysForItsExpectedRetake},
      {"testGrowthCountsWhatTheHostDoesNotShowYet", testGrowthCountsWhatTheHostDoesNotShowYet},
      {"testGrowthLeavesTheHostWhatTheLanesGrowInto", testGrowthLeavesTheHostWhatTheLanesGrowInto},
      {"testAClaimPaysForAPoolUnitsRelease", testAClaimPaysForAPoolUnitsRelease},
      {"testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes",
       testABytesNeedBesideBuffersNoStateGivesFreesTheirBytes},
      {"testStartGrowthFollowsThePressureNow", testStartGrowthFollowsThePressureNow},
  };
  return kTests;
}

// memory-ledger [RECORDED_LOADS_DIRECTORY] [--list] [--filter ...]
int main(int argc, char **argv) {
  if (argc > 1 && argv[1][0] != '-') {
    recordedLoadsDirectory() = argv[1];
    --argc;
    ++argv;
  }
  const test::TestGroup groups[] = {{"ledger", ledgerTests()}, {"value", valueTests()}};
  return test::runTests("memory-ledger", groups, argc, argv, [] { test::metalStatistics() = {}; });
}
