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
  // The units taken to land later free their memory.
  void land() {
    for (const Victim &unit : landing)
      if (memory_ && unit.bytes)
        memory_->release(name_, unit.bytes);
    landing.clear();
  }

  std::vector<Victim> units;
  std::vector<uint64_t> taken;
  std::vector<Victim> landing;
  uint64_t unsizedBytes = 0;
  mutable uint32_t peeks = 0;

private:
  std::string name_;
  FakeMemory *memory_ = nullptr;
};

Victim unit(uint64_t key, uint64_t bytes, double onceSeconds, bool later = false) {
  return {.key = key, .bytes = bytes, .later = later, .cost = {.onceSeconds = onceSeconds}};
}

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
      default:
        static_cast<void>(experts.grow(fixture.governor->allocationAdmission()));
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
                      [](uint64_t) { return Returning{}; });
  serve(ledger, experts, 400.0, 0.0, 10.0);
  const ClaimId waiting = ledger.open(request(7, RequestPriority::Normal));
  ledger.refused(waiting, {.bytes = 2 * kMiB});
  ledger.setWaiting(waiting, true);
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
// a lower priority, or at its priority a later submission.
void testWaitingClaimHoldsRoomForItsRank() {
  Fixture fixture(host(32 * kGiB), 10 * kGiB);
  fixture.memory.hold("weights", 9 * kGiB);
  MemoryLedger &ledger = fixture.ledger();
  const ClaimId high = ledger.open(request(1, RequestPriority::Foreground));
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(!fixture.allocate(high, 2 * kGiB), "2 GiB fit in 1 GiB of room");
  ledger.setWaiting(high, true);
  require(fixture.allocate(low, 512 * kMiB).failure == metal::AllocationFailure::EngineBudget,
          "a lower-ranked claim took room a waiting claim lacks");
  require(fixture.allocate(high, 512 * kMiB), "the waiting claim was refused its own room");
  const ClaimId earlier = ledger.open(request(3, RequestPriority::Normal));
  const ClaimId later = ledger.open(request(4, RequestPriority::Normal));
  ledger.setWaiting(high, false);
  require(!fixture.allocate(earlier, 2 * kGiB), "2 GiB fit in the room left");
  ledger.setWaiting(earlier, true);
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
  require(ledger.claim(inner).need.bytes == 2 * kGiB && ledger.claim(outer).need.bytes == 3 * kGiB,
          "a refusal was recorded on the wrong claim");
  ledger.setWaiting(outer, true);
  require(fixture.memory.allocate(fixture.governor->allocationAdmission(), "kv", 512 * kMiB),
          "an admission outside every scope was held back by a claim");
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
  static_cast<void>(ledger.free(high, {.pages = 6, .form = kv::PageForm::Index}, 0, false, false));
  static_cast<void>(ledger.free(mid, {.pages = 4}, 0, false, false));
  ledger.setWaiting(high, true);
  ledger.setWaiting(mid, true);
  const std::array<uint32_t, 3> forLow = ledger.heldPages(low);
  const std::array<uint32_t, 3> forMid = ledger.heldPages(mid);
  require(forLow[static_cast<size_t>(kv::PageForm::Index)] == 6 && forLow[static_cast<size_t>(kv::PageForm::Full)] == 4,
          "a lower-ranked claim is not held the pages of both claims above it");
  require(forMid[static_cast<size_t>(kv::PageForm::Index)] == 6 && forMid[static_cast<size_t>(kv::PageForm::Full)] == 0,
          "a claim is held its own pages or a lower claim's");
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
// unit after it up a place.
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
    taken.push_back(units.at(victim.key).first);
    units.erase(units.begin() + static_cast<std::ptrdiff_t>(victim.key));
    return {.nowBytes = victim.bytes};
  }
  std::vector<std::pair<uint64_t, Victim>> units;  // the test's name for each unit, and the unit
  std::vector<uint64_t> taken;
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
  ledger.refused(high, {.bytes = kGiB});
  ledger.setWaiting(high, true);
  const ClaimId low = ledger.open(request(2, RequestPriority::Background));
  require(ledger.free(low, {.bytes = 512 * kMiB}, kGiB, false, false).outcome == FreeOutcome::Pending &&
              cache.taken.empty(),
          "a claim the room held above it fits took a victim instead of waiting");
}

// Landings first, then the cheapest victims, then owed copies: a need the
// landing bytes do not meet takes a victim before it waits for another
// lane's write-through (the base's escalate waited on owed copies first,
// Engine.cpp:2380).
void testLandingThenVictimsThenOwed() {
  MemoryLedger ledger;
  TestSource cache("cache");
  cache.units = {unit(1, kGiB, 0.5)};
  ledger.addSource(cache);
  ledger.setReturning([] { return Returning{.landingBytes = 256 * kMiB, .owedBytes = 4 * kGiB}; },
                      [](uint64_t) { return Returning{}; });
  const ClaimId claim = ledger.open(request(1, RequestPriority::Normal));
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{1},
          "the claim waited on owed copies before it took a victim");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Pending &&
              ledger.claim(claim).waiting,
          "with no victim left the claim did not wait for the owed copies");
  ledger.setReturning([] { return Returning{.landingBytes = 2 * kGiB}; }, [](uint64_t) { return Returning{}; });
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
  ledger.setReturning([] { return Returning{.landingBytes = 4 * kGiB}; }, [](uint64_t) { return Returning{}; });
  const ClaimId claim = ledger.open({.cls = ClaimClass::Optional, .timing = NeedTiming::Immediate});
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Met &&
              cache.taken == std::vector<uint64_t>{2},
          "an Immediate claim waited or took a unit that lands later");
  require(ledger.free(claim, {.bytes = kGiB}, 0, false, false).outcome == FreeOutcome::Short,
          "an Immediate claim waited for landings");
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
  require(fixture.allocate(lane, 512 * kMiB) && !fixture.ledger().claim(lane).hostExempt,
          "the exempt retry was refused or kept its exemption");
  const ClaimId start = fixture.ledger().open(request(2, RequestPriority::Normal, ClaimClass::Start));
  require(!fixture.allocate(start, 512 * kMiB), "the host admitted the start");
  require(fixture.governor->free(start, fixture.governor->need(start)).outcome == FreeOutcome::Short &&
              !fixture.ledger().claim(start).hostExempt,
          "a start was exempted from the host");
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
                                  [](uint64_t) { return Returning{}; });
    static_cast<void>(fixture.pass(1.0));
    require(cache.taken.size() == (landing == 512 * kMiB ? 0u : 4u),
            "the band pass did not free the deficit net of what is landing");
  }
}

// Missing telemetry holds elective work back but gives no cache back: no
// deficit is measured (MemoryPressurePolicy kept the same rule).
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
                      [](uint64_t id) { return id == 5 ? Returning{.owedBytes = 2 * kGiB} : Returning{}; });
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
                      [&](uint64_t id) {
                        return std::ranges::count(engine.taken, id) ? Returning{.owedBytes = kGiB} : Returning{};
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
                      [](uint64_t id) { return id == 9 ? Returning{.owedBytes = 4 * kGiB} : Returning{}; });
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
// borrowed item 3(c)).
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

} // namespace

std::span<const test::TestCase> ledgerTests() {
  static constexpr test::TestCase kTests[] = {
      {"testAccountsSumToCharged", testAccountsSumToCharged},
      {"testSnapshotRendersEveryAccount", testSnapshotRendersEveryAccount},
      {"testWaitingClaimHoldsRoomForItsRank", testWaitingClaimHoldsRoomForItsRank},
      {"testShortClaimHoldsNothing", testShortClaimHoldsNothing},
      {"testNestedScopesRestore", testNestedScopesRestore},
      {"testHeldPagesFollowRank", testHeldPagesFollowRank},
      {"testPendingGrowthIsReservedAgainstOthers", testPendingGrowthIsReservedAgainstOthers},
      {"testTakenPendingGrowthIsCancelled", testTakenPendingGrowthIsCancelled},
      {"testOneHostRule", testOneHostRule},
      {"testBandWaivedWhenTheHostClaimFindsNothing", testBandWaivedWhenTheHostClaimFindsNothing},
      {"testDeficitCountsHostRefusedClaimsNotAnIdleMargin", testDeficitCountsHostRefusedClaimsNotAnIdleMargin},
      {"testCheapestPerByteFirstAcrossSources", testCheapestPerByteFirstAcrossSources},
      {"testSourceOrderIsKeptWithinASource", testSourceOrderIsKeptWithinASource},
      {"testUnitsKeyedByTheirPlaceAreTakenAsChosen", testUnitsKeyedByTheirPlaceAreTakenAsChosen},
      {"testTiesGoToTheImmediateUnitThenTheSourceRegisteredFirst",
       testTiesGoToTheImmediateUnitThenTheSourceRegisteredFirst},
      {"testClassBoundsEligibility", testClassBoundsEligibility},
      {"testShortTakesNothing", testShortTakesNothing},
      {"testRoomHeldForHigherRanksMakesAWait", testRoomHeldForHigherRanksMakesAWait},
      {"testLandingThenVictimsThenOwed", testLandingThenVictimsThenOwed},
      {"testImmediateSkipsLandings", testImmediateSkipsLandings},
      {"testHostRefusedNeedIsMetInPlaceFirst", testHostRefusedNeedIsMetInPlaceFirst},
      {"testZeroCostUnsizedUnitsGoFirst", testZeroCostUnsizedUnitsGoFirst},
      {"testBenefitBoundsTheDryRun", testBenefitBoundsTheDryRun},
      {"testAvailableTakesNothing", testAvailableTakesNothing},
      {"testGrowthRetakesFreeRoomBeyondLaneDemand", testGrowthRetakesFreeRoomBeyondLaneDemand},
      {"testNoPingPongWhileALaneGrows", testNoPingPongWhileALaneGrows},
      {"testDisplacementNeedsValueAboveCycle", testDisplacementNeedsValueAboveCycle},
      {"testNoFlipFlopAtConstantInputs", testNoFlipFlopAtConstantInputs},
      {"testCriticalDropsPoolsToFloors", testCriticalDropsPoolsToFloors},
      {"testWarningFreesAtMostTheMarginPerSettledPass", testWarningFreesAtMostTheMarginPerSettledPass},
      {"testHostClaimTakesNoCertainReuseUnlessCritical", testHostClaimTakesNoCertainReuseUnlessCritical},
      {"testDeficitNetOfReturning", testDeficitNetOfReturning},
      {"testMissingTelemetryGivesNoCacheBack", testMissingTelemetryGivesNoCacheBack},
      {"testRerunWhileTheDeficitWaitsOnLandings", testRerunWhileTheDeficitWaitsOnLandings},
      {"testWarningAloneTakesNoPoolUnit", testWarningAloneTakesNoPoolUnit},
      {"testStoreRegrowsByDisplacementUnderPersistentWarning", testStoreRegrowsByDisplacementUnderPersistentWarning},
      {"testStoreRegrowsByDisplacementInTheBand", testStoreRegrowsByDisplacementInTheBand},
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
