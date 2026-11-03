#include "engine/MemoryLedger.hpp"

#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryValue.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

constexpr double kGiB = double(1ULL << 30);
constexpr double kMiB = double(1ULL << 20);

uint64_t less(uint64_t a, uint64_t b) noexcept { return a > b ? a - b : 0; }

uint64_t plus(uint64_t a, uint64_t b) noexcept {
  return a <= std::numeric_limits<uint64_t>::max() - b ? a + b : std::numeric_limits<uint64_t>::max();
}

const char *className(ClaimClass cls) noexcept {
  switch (cls) {
  case ClaimClass::Floor: return "floor";
  case ClaimClass::InService: return "in-service";
  case ClaimClass::Start: return "start";
  case ClaimClass::Optional: return "optional";
  case ClaimClass::Growth: return "growth";
  case ClaimClass::Host: return "host";
  }
  return "host";
}

const char *outcomeName(FreeOutcome outcome) noexcept {
  switch (outcome) {
  case FreeOutcome::Met: return "met";
  case FreeOutcome::Pending: return "pending";
  case FreeOutcome::Short: return "short";
  }
  return "short";
}

// What a claim lacks, in each measure that can meet it: device bytes beyond
// the room it has, free pages of a form in kept extents, pooled state
// buffers.
struct Lacking final {
  uint64_t bytes = 0;
  uint32_t pages = 0;
  kv::PageForm form = kv::PageForm::Full;
  uint32_t buffers = 0;

  [[nodiscard]] bool none() const noexcept { return !bytes && !pages && !buffers; }
  // The share of the need a unit meets, in the measure it meets most of: at
  // most all of it, so that a unit larger than the need is priced at its
  // whole cost, not at its cost per byte (amend/M.md B2, R2).
  [[nodiscard]] double share(const Victim &victim) const noexcept {
    double share = 0.0;
    if (bytes)
      share = std::max(share, double(victim.bytes) / double(bytes));
    if (pages && victim.form == form)
      share = std::max(share, double(victim.pages) / double(pages));
    if (buffers)
      share = std::max(share, double(victim.stateBuffers) / double(buffers));
    return std::min(1.0, share);
  }
};

// The bytes of the keys a pool does not hold, over which its scan reads
// spread (value::poolUnit); none when it reports no whole size.
uint64_t missingBytes(const model::PoolState &state) noexcept { return less(state.wholeBytes, state.currentBytes); }

} // namespace

struct MemoryLedger::Account final {
  std::string name;
  AccountRole role = AccountRole::Fixed;
  std::function<uint64_t()> current;
  model::ElasticPool *pool = nullptr;
  // Growth the ledger decided that has not landed yet: held for this account
  // against every other until it grows, a claim cancels it or critical
  // pressure drops the pool to its floor; and what its victims free once
  // their transfers land, which it waits for.
  uint64_t pendingGrowth = 0;
  uint64_t pendingLaterBytes = 0;
  uint64_t grows = 0;
  uint64_t shrinks = 0;
  // The units claims took from it, which its rate differentiates.
  uint64_t taken = 0;
  value::PoolRate rate;
};

// One unit a dry run chose: a source's (by index, after the units of the
// source it passed over since the source's previous choice), or a pool's
// unit or pending growth (by account). A unit the run carries to a later
// one of its source is a choice of its own, before that one.
struct MemoryLedger::Choice final {
  size_t source = 0;
  uint32_t passed = 0;
  std::optional<AccountId> pool;
  bool pending = false;
  Victim victim;
};

struct MemoryLedger::DryRun final {
  std::vector<Choice> chosen;
  uint64_t bytes = 0;
  uint32_t pages = 0;
  uint32_t buffers = 0;
  // Of its bytes, those its units' state buffers give once they leave the
  // model's pool.
  uint64_t bufferBytes = 0;
  double cost = 0.0;
  bool covered = false;
  bool later = false;
  // It reached a cached unit that frees memory only once its copy lands.
  bool uncopied = false;
};

MemoryLedger::MemoryLedger() {
  accounts_.push_back(Account{.name = "fixed", .role = AccountRole::Fixed});
}

MemoryLedger::~MemoryLedger() = default;

AccountId MemoryLedger::addAccount(std::string name, AccountRole role, std::function<uint64_t()> currentBytes) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  if (role == AccountRole::Fixed || role == AccountRole::Pool || !currentBytes)
    throw std::invalid_argument("an account the ledger pulls needs its current bytes");
  accounts_.push_back(Account{.name = std::move(name), .role = role, .current = std::move(currentBytes)});
  return {static_cast<uint8_t>(accounts_.size() - 1)};
}

AccountId MemoryLedger::addPool(model::ElasticPool &pool) {
  if (accounts_.size() > std::numeric_limits<uint8_t>::max())
    throw std::logic_error("the memory ledger has no account id left");
  accounts_.push_back(Account{.name = std::string(pool.state().name), .role = AccountRole::Pool, .pool = &pool});
  return {static_cast<uint8_t>(accounts_.size() - 1)};
}

void MemoryLedger::addSource(VictimSource &source) {
  if (std::ranges::find(sources_, &source) == sources_.end())
    sources_.push_back(&source);
}

void MemoryLedger::removeSource(VictimSource &source) noexcept { std::erase(sources_, &source); }

void MemoryLedger::setReturning(std::function<Returning()> all,
                                std::function<Returning(std::span<const uint64_t>)> ofRequests) {
  returning_ = std::move(all);
  returningFor_ = std::move(ofRequests);
}

void MemoryLedger::setBufferBytes(std::function<uint64_t(uint32_t)> bytes) { bufferBytes_ = std::move(bytes); }

void MemoryLedger::setPageBytes(const std::array<uint64_t, 3> &bytes) noexcept { pageBytes_ = bytes; }

void MemoryLedger::setRates(const MemoryRates &rates) noexcept { rates_ = rates; }

void MemoryLedger::observePools(double nowSeconds, bool commandsRan) {
  nowSeconds_ = nowSeconds;
  for (Account &account : accounts_) {
    if (!account.pool)
      continue;
    account.rate.observe(account.pool->value(), account.taken, nowSeconds, commandsRan);
  }
}

void MemoryLedger::setLanes(uint64_t demandBytes, std::optional<RequestPriority> highest) noexcept {
  laneDemandBytes_ = demandBytes;
  lanePriority_ = highest;
}

void MemoryLedger::setDecisionLog(std::function<void(std::string_view)> log) { log_ = std::move(log); }

ClaimId MemoryLedger::open(const ClaimSpec &spec) {
  const ClaimId id{nextClaim_++};
  if (!nextClaim_)
    nextClaim_ = 1;
  claims_.emplace(id.value, ClaimState{.spec = spec, .openedSeconds = nowSeconds_});
  return id;
}

void MemoryLedger::update(ClaimId claim, const ClaimSpec &spec) { claims_.at(claim.value).spec = spec; }

void MemoryLedger::refused(ClaimId claim, const Need &need) {
  claims_.at(claim.value).refusal = {.bytes = need.bytes, .hostRefused = need.hostRefused};
}

void MemoryLedger::setWaiting(ClaimId claim, bool waiting) {
  ClaimState &state = claims_.at(claim.value);
  if (waiting && !state.waiting)
    state.waitingSinceSeconds = nowSeconds_;
  state.waiting = waiting;
}

void MemoryLedger::setHostRefusedWaiting(ClaimId claim, bool waiting) {
  claims_.at(claim.value).hostRefusedWaiting = waiting;
}

void MemoryLedger::setHostExempt(ClaimId claim, bool exempt) { claims_.at(claim.value).hostExempt = exempt; }

void MemoryLedger::setInDemand(ClaimId claim, bool inDemand) { claims_.at(claim.value).inDemand = inDemand; }

void MemoryLedger::close(ClaimId claim) noexcept { claims_.erase(claim.value); }

const ClaimState &MemoryLedger::claim(ClaimId claim) const { return claims_.at(claim.value); }

uint64_t MemoryLedger::heldForGrowth(std::optional<AccountId> growing, ClaimClass cls) const noexcept {
  uint64_t held = 0;
  for (size_t index = 0; index < accounts_.size(); ++index)
    if (!growing || growing->value != index)
      held = plus(held, accounts_[index].pendingGrowth);
  return cls == ClaimClass::Growth ? plus(held, laneDemandBytes_) : held;
}

uint64_t MemoryLedger::heldByClaims(const ClaimRank &rank, std::optional<uint32_t> self) const noexcept {
  uint64_t held = 0;
  for (const auto &[id, state] : claims_)
    if (state.waiting && (!self || id != *self) && state.spec.rank > rank)
      held = plus(held, state.need.bytes);
  return held;
}

uint64_t MemoryLedger::heldBytes(ClaimId claim, std::optional<AccountId> growing) const {
  const ClaimState &state = claims_.at(claim.value);
  return plus(heldForGrowth(growing, state.spec.cls), heldByClaims(state.spec.rank, claim.value));
}

std::array<uint32_t, 3> MemoryLedger::heldPages(ClaimId claim) const {
  const ClaimState &self = claims_.at(claim.value);
  std::array<uint32_t, 3> held{};
  for (const auto &[id, state] : claims_)
    if (state.waiting && id != claim.value && state.spec.rank > self.spec.rank)
      held[static_cast<size_t>(state.need.form)] += state.need.pages;
  // The bytes they lack, in each form's pages: a page handed out of a kept
  // extent's free pages takes room they wait for as an extent would (the
  // extents their landings complete hold free pages too). A form without
  // pages has none to hold.
  const uint64_t bytes = heldByClaims(self.spec.rank, claim.value);
  for (size_t form = 0; form < held.size(); ++form)
    if (pageBytes_[form]) {
      const uint64_t pages = bytes / pageBytes_[form] + (bytes % pageBytes_[form] != 0);
      held[form] = static_cast<uint32_t>(
          std::min<uint64_t>(held[form] + pages, std::numeric_limits<uint32_t>::max()));
    }
  return held;
}

bool MemoryLedger::claimWaiting() const noexcept {
  return std::ranges::any_of(claims_, [](const auto &entry) { return entry.second.waiting; });
}

uint64_t MemoryLedger::hostRefusedWaitingBytes() const noexcept {
  uint64_t bytes = 0;
  for (const auto &[id, state] : claims_)
    if ((state.waiting || state.hostRefusedWaiting) && state.refusal.hostRefused)
      bytes = plus(bytes, state.refusal.bytes);
  return bytes;
}

uint64_t MemoryLedger::pendingGrowthBytes() const noexcept {
  uint64_t bytes = 0;
  for (const Account &account : accounts_)
    bytes = plus(bytes, account.pendingGrowth);
  return bytes;
}

// A claim takes a pool's unit only when nothing cheaper meets its need, and
// while the engine's lanes may yet need it (its lane prefills into it, or
// another lane is in service beside it, whose growth and the requests that
// follow it take units as it did) its units over the time since it opened
// measure the rate from its first take, which the pool's own rate (PoolRate,
// over kRateTimeConstantSeconds) shows only as its time constant goes by,
// and which already counts those takes: the larger of the two. Smoke a, 35B
// at 18G (test-pressure-real): requests took store units 6, 13, 26, 40 and
// 45 s after Ready; the store's own rate stayed at 0.03-0.04 a second
// through the first 30 s, so the units it regrew 14 and 36 s after Ready went
// back 13 and 4 s later. A lane that decodes alone grows little: its takes
// count in the pool's rate only, and the store regrows right after its
// prefill (owner rule 12).
double MemoryLedger::takesPerSecond(size_t account) const noexcept {
  double inDemand = 0.0;
  for (const auto &[id, state] : claims_) {
    if (!state.inDemand)
      continue;
    const auto units = std::ranges::count(state.poolTakes, AccountId{static_cast<uint8_t>(account)});
    if (units)
      inDemand += nowSeconds_ > state.openedSeconds ? double(units) / (nowSeconds_ - state.openedSeconds)
                                                     : std::numeric_limits<double>::infinity();
  }
  return std::max(accounts_[account].rate.takesPerSecond(), inDemand);
}

uint64_t MemoryLedger::poolReleases() const noexcept {
  uint64_t units = 0;
  for (const Account &account : accounts_)
    units = plus(units, account.shrinks);
  return units;
}

uint64_t MemoryLedger::largestGrowthUnit() const {
  uint64_t unit = 0;
  for (const Account &account : accounts_)
    if (account.pool)
      unit = std::max(unit, account.pool->state().growUnitBytes);
  return unit;
}

Returning MemoryLedger::returning() const { return returning_ ? returning_() : Returning{}; }

bool MemoryLedger::copyDemand() const noexcept {
  return uncopiedReachedSeconds_ && nowSeconds_ - *uncopiedReachedSeconds_ <= kValueHorizonSeconds;
}

// The rule's dry run: the cheapest unit per share of the need first among
// the sources' heads and the pools' units, each source's own order kept, a
// tie going to a unit that frees memory now and then to the source
// registered first (the pools after the sources), until the need is met or
// the units' cost would pass the claim's benefit. A unit the claim's class
// may not take is skipped, not stopped at. A unit's state buffers count for
// a need of bytes as the bytes the model's pool releases of them beyond
// those of the units chosen before (setBufferBytes). A unit whose buffers
// the pool keeps (they refill those a lane starts from) meets none of a
// need of bytes alone: it goes with the next unit the run chooses from its
// source, its cost and buffers counted with that unit's, since the buffers
// after it count only beside its own and the cache weighs each of its units
// as a step from where the units before it leave the cache (R2 step 2: the
// same units in the same order). A unit that meets none of the need
// otherwise, or that an Immediate claim cannot take since it lands later,
// is skipped (VictimSource: taking a later unit leaves it where it was).
MemoryLedger::DryRun MemoryLedger::dryRun(const ClaimSpec &spec, const VictimQuery &query, uint64_t lackingBytes,
                                          uint32_t lackingPages, std::optional<AccountId> growing) const {
  const Lacking lack{.bytes = lackingBytes,
                     .pages = lackingPages,
                     .form = query.need.form,
                     .buffers = query.need.stateBuffers};
  const bool immediate = spec.timing == NeedTiming::Immediate;
  // What each class may take (amend/M.md B2, B3): InUse units only for
  // in-service work that reaches them or the host under critical pressure;
  // units a request will certainly reuse never for growth, and for the host
  // only under critical pressure; a conversion for in-service work or a host
  // short of memory; the command between passes for in-service work that
  // supersedes it.
  const auto mayTake = [&](const Victim &victim) {
    if (victim.cls == ReclaimClass::InUse &&
        !(spec.cls == ClaimClass::InService ? spec.reach == ReclaimClass::InUse
                                            : spec.cls == ClaimClass::Host && query.critical))
      return false;
    if (victim.certainReuse &&
        (spec.cls == ClaimClass::Growth || (spec.cls == ClaimClass::Host && !query.critical)))
      return false;
    switch (victim.kind) {
    case VictimKind::Memory: return true;
    case VictimKind::Conversion:
      return spec.cls == ClaimClass::InService ||
             (spec.cls == ClaimClass::Host && (query.deficit || query.critical));
    case VictimKind::PassCommand: return spec.cls == ClaimClass::InService && spec.supersedes;
    }
    return false;
  };
  // Pool units: above floors for in-service work, optional work (while they
  // cost less than its benefit), growth (another pool's) and a host short of
  // memory; for a start, only those of a pool whose units cost only their
  // rebuild, which nothing in service holds or reads (PoolState::
  // rebuildOnly; amend/M.md B2 as the owner amended it 11-03: the store and
  // the hot block cache stay out of its reach); none for a claim below a
  // lane in service. Pending growth only for claims that may take memory the
  // engine's limit holds for a pool.
  const bool belowALane = lanePriority_ && *lanePriority_ < spec.priority;
  const bool poolUnits = !belowALane && (spec.cls == ClaimClass::InService || spec.cls == ClaimClass::Optional ||
                                         spec.cls == ClaimClass::Growth ||
                                         (spec.cls == ClaimClass::Host && (query.deficit || query.critical)));
  const bool rebuiltUnits = poolUnits || (!belowALane && spec.cls == ClaimClass::Start);
  const bool pendingUnits = poolUnits && spec.cls != ClaimClass::Growth && spec.cls != ClaimClass::Host;

  struct Head final {
    std::optional<Victim> victim;
    size_t source = 0;
    uint32_t passed = 0;  // the source's units passed over since its last choice or carried unit
    // The units it carries to its next choice.
    std::vector<Choice> carried;
    std::optional<AccountId> pool;
    bool pending = false;
    // A pool's reading, and the bytes above its floor the run has taken.
    model::PoolState state;
    model::PoolValue counters;
    uint64_t taken = 0;
    bool pendingOffered = false;
  };
  std::vector<Head> heads;
  heads.reserve(sources_.size() + accounts_.size());
  const auto poolUnit = [&](Head &head) -> std::optional<Victim> {
    const Account &account = accounts_[head.pool->value];
    head.pending = false;
    if (pendingUnits && account.pendingGrowth && !head.pendingOffered) {
      head.pendingOffered = true;
      head.pending = true;
      const UnitCost cost = value::poolUnit(account.rate, head.counters, account.pendingGrowth,
                                            missingBytes(head.state), spec.holdSeconds, rates_);
      return Victim{.bytes = account.pendingGrowth, .cost = {.ratePerSecond = cost.ratePerSecond}};
    }
    const uint64_t above = less(head.state.currentBytes, head.state.floorBytes);
    if (!head.state.releaseUnitBytes || head.taken >= above)
      return std::nullopt;
    const uint64_t bytes = std::min(head.state.releaseUnitBytes, above - head.taken);
    // What the unit is worth over the claim's hold, and the release it takes
    // of the engine thread now.
    UnitCost cost =
        value::poolUnit(account.rate, head.counters, bytes, missingBytes(head.state), spec.holdSeconds, rates_);
    cost.onceSeconds += value::releaseSeconds(head.counters, bytes);
    return Victim{.key = head.taken / head.state.releaseUnitBytes, .bytes = bytes, .cost = cost};
  };
  for (size_t index = 0; index < sources_.size(); ++index)
    heads.push_back(Head{.victim = sources_[index]->peek(query, nullptr), .source = index});
  for (size_t index = 1; index < accounts_.size(); ++index) {
    const Account &account = accounts_[index];
    if (!account.pool || !rebuiltUnits || (growing && growing->value == index))
      continue;
    Head head{.pool = AccountId{static_cast<uint8_t>(index)},
              .state = account.pool->state(),
              .counters = account.pool->value()};
    if (!poolUnits && !head.state.rebuildOnly)
      continue;
    head.victim = poolUnit(head);
    heads.push_back(std::move(head));
  }
  const auto advance = [&](Head &head) {
    if (head.pool) {
      if (!head.pending)
        head.taken += head.victim->bytes;
      head.victim = poolUnit(head);
    } else {
      head.victim = sources_[head.source]->peek(query, &*head.victim);
    }
  };
  const auto pass = [&](Head &head) {
    ++head.passed;
    advance(head);
  };
  const auto carry = [&](Head &head) {
    head.carried.push_back(
        Choice{.source = head.source, .passed = std::exchange(head.passed, 0), .victim = *head.victim});
    advance(head);
  };
  // The head's unit with the units it carries, as one.
  const auto withCarried = [](const Head &head) {
    Victim unit = *head.victim;
    for (const Choice &carried : head.carried) {
      const Victim &each = carried.victim;
      unit.bytes = plus(unit.bytes, each.bytes);
      if (each.form == unit.form)
        unit.pages += each.pages;
      unit.stateBuffers += each.stateBuffers;
      unit.later = unit.later || each.later;
      unit.cost.onceSeconds += each.cost.onceSeconds;
      unit.cost.ratePerSecond += each.cost.ratePerSecond;
    }
    return unit;
  };

  DryRun run;
  const auto counted = [&](const Victim &victim) {
    Victim unit = victim;
    if (lack.bytes && victim.stateBuffers && bufferBytes_)
      unit.bytes = plus(unit.bytes, less(bufferBytes_(run.buffers + victim.stateBuffers), bufferBytes_(run.buffers)));
    return unit;
  };
  for (;;) {
    Head *best = nullptr;
    double bestPrice = 0.0;
    bool bestLater = false;
    for (Head &head : heads) {
      // Skip what this claim may not take, carry a unit whose buffers the
      // pool keeps, and skip what meets none of its need otherwise; keep a
      // unit that lands later in an Immediate claim's way, to note that the
      // run reached it.
      while (head.victim) {
        if (!mayTake(*head.victim))
          pass(head);
        else if (lack.share(counted(withCarried(head))) > 0.0)
          break;
        else if (lack.bytes && head.victim->stateBuffers && !(immediate && head.victim->later))
          carry(head);
        else
          pass(head);
      }
      if (!head.victim)
        continue;
      const Victim unit = counted(withCarried(head));
      const double price = unit.cost.at(spec.holdSeconds) / lack.share(unit);
      if (!best || price < bestPrice || (price == bestPrice && !unit.later && bestLater)) {
        best = &head;
        bestPrice = price;
        bestLater = unit.later;
      }
    }
    if (!best)
      break;
    const Victim units = withCarried(*best);
    const Victim victim = counted(units);
    if (victim.later && victim.kind == VictimKind::Memory)
      run.uncopied = true;
    if (victim.later && immediate) {
      pass(*best);
      continue;
    }
    const double cost = victim.cost.at(spec.holdSeconds);
    if (run.cost + cost > spec.benefitSeconds)
      break;
    run.cost += cost;
    run.bytes = plus(run.bytes, victim.bytes);
    run.bufferBytes = plus(run.bufferBytes, victim.bytes - units.bytes);
    if (victim.form == lack.form)
      run.pages += victim.pages;
    run.buffers += victim.stateBuffers;
    run.later = run.later || victim.later;
    run.chosen.insert(run.chosen.end(), best->carried.begin(), best->carried.end());
    best->carried.clear();
    run.chosen.push_back(Choice{
        .source = best->source, .passed = best->passed, .pool = best->pool, .pending = best->pending, .victim = victim});
    best->passed = 0;
    advance(*best);
    if ((lack.bytes && run.bytes >= lack.bytes) || (lack.pages && run.pages >= lack.pages) ||
        (lack.buffers && run.buffers >= lack.buffers)) {
      run.covered = true;
      break;
    }
  }
  return run;
}

// Takes what a covered dry run chose, in its order: a source's unit, each
// it carried before the one it went with, by its place in the source's
// order as it stands, past the units the run passed over, up to a unit no
// longer offered as chosen (VictimSource), a pool's units by releasing
// what the run's choices of them still lack, pending growth by cancelling
// it. The run sized a pool's units after its first at the first's bytes,
// and a later one may be larger (the store's last live chunk is the rest
// of a size, the chunk before it about 1 GiB): released choice by choice,
// each larger unit would go for a choice the one before already met.
FreeResult MemoryLedger::take(const DryRun &run, const VictimQuery &query, ClaimState *claim, bool &whole) {
  FreeResult result{.outcome = run.later ? FreeOutcome::Pending : FreeOutcome::Met};
  whole = true;
  // Each source's last unit the run passed over, which stays in its order.
  std::vector<std::optional<Victim>> passed(sources_.size());
  // Each pool's bytes the run chose so far, and those its releases gave.
  std::vector<uint64_t> chosen(accounts_.size()), released(accounts_.size());
  for (const Choice &choice : run.chosen) {
    if (choice.pool) {
      const size_t index = choice.pool->value;
      Account &account = accounts_[index];
      if (choice.pending) {
        ++account.taken;
        if (claim)
          claim->poolTakes.push_back(*choice.pool);
        account.pendingGrowth = account.pendingLaterBytes = 0;
        result.nowBytes = plus(result.nowBytes, choice.victim.bytes);
        continue;
      }
      chosen[index] = plus(chosen[index], choice.victim.bytes);
      if (released[index] >= chosen[index])
        continue;
      const uint64_t bytes = account.pool->release(chosen[index] - released[index]);
      if (bytes) {
        ++account.taken;
        ++account.shrinks;
        if (claim)
          claim->poolTakes.push_back(*choice.pool);
      }
      released[index] = plus(released[index], bytes);
      result.nowBytes = plus(result.nowBytes, bytes);
      continue;
    }
    VictimSource &source = *sources_[choice.source];
    std::optional<Victim> &before = passed[choice.source];
    std::optional<Victim> unit = source.peek(query, before ? &*before : nullptr);
    for (uint32_t over = 0; unit && over < choice.passed; ++over) {
      before = unit;
      unit = source.peek(query, &*before);
    }
    if (!unit || unit->kind != choice.victim.kind ||
        (unit->kind == VictimKind::Conversion && unit->key != choice.victim.key)) {
      whole = false;
      return result;
    }
    const FreeResult taken = source.take(query, *unit);
    result.nowBytes = plus(result.nowBytes, taken.nowBytes);
    result.laterBytes = plus(result.laterBytes, taken.laterBytes);
    if (unit->kind == VictimKind::Conversion && claim)
      claim->converted.push_back(unit->key);
  }
  return result;
}

FreeResult MemoryLedger::free(ClaimId claim, const Need &need, uint64_t freeBytes, bool critical, bool deficit) {
  return freeFor(claim.value, need, freeBytes, critical, deficit, std::nullopt, false);
}

uint64_t MemoryLedger::takeUnsized(const VictimQuery &query) {
  uint64_t bytes = 0;
  for (VictimSource *source : sources_) {
    for (std::optional<Victim> unit = source->peek(query, nullptr);
         unit && !unit->sized && unit->cost.at(query.claim->holdSeconds) == 0.0; unit = source->peek(query, nullptr)) {
      const FreeResult taken = source->take(query, *unit);
      bytes = plus(bytes, taken.nowBytes);
      if (!taken.nowBytes && !taken.laterBytes)
        break;
    }
  }
  return bytes;
}

// The one rule (DESIGN section 4.4; amend/M.md B2-B5): the units that cost
// nothing and cannot be sized go first; then a claim that can wait waits
// when what is landing, with its own lanes' owed copies, meets its need (or
// would, but for claims ranked above it); else the cheapest units that meet
// what those do not are taken, all or none, and it waits for the rest unless
// the units meet it alone; else it waits when every copy in flight or owed
// would meet it. A pool's growth waits on no landing but its own victims':
// what it waits for is held as pending growth (R3). A claim admission
// refused bytes lacks what admission asks beyond the room: its need and the
// room held against it; a need met in place lacks no bytes. A need the host
// refused counts only units that leave memory in place; the host's claim
// lacks its whole need, and growth the host admits only on its credit
// (`credited`) at least what free() leaves now: the room under the engine's
// limit returns nothing to the host. State buffers the units free meet a
// need of bytes once they leave the model's pool: the units that cost
// nothing go again (a need in place keeps them).
FreeResult MemoryLedger::freeFor(uint32_t id, const Need &need, uint64_t freeBytes, bool critical, bool deficit,
                                 std::optional<AccountId> growing, bool credited) {
  ClaimState &claim = claims_.at(id);
  const ClaimSpec &spec = claim.spec;
  const VictimQuery query{.claim = &spec, .need = need, .critical = critical, .rates = &rates_, .deficit = deficit};

  FreeResult result{.nowBytes = takeUnsized(query)};
  const bool host = spec.cls == ClaimClass::Host;
  const uint64_t room = host ? 0 : plus(freeBytes, result.nowBytes);
  // Pending growth and the lanes' projected growth hold room the claim
  // may take by value; waiting claims ranked above it hold room it waits
  // for.
  const uint64_t growth = host ? 0 : heldForGrowth(growing, spec.cls);
  const uint64_t held = host ? 0 : plus(growth, heldByClaims(spec.rank, id));
  const uint64_t forClaim = less(room, held);
  const uint64_t lackingBytes =
      need.hostRefused || !need.bytes
          ? 0
          : std::max(less(plus(need.bytes, held), room), credited ? less(need.bytes, result.nowBytes) : 0);
  const uint32_t lackingPages = need.pages;
  // Under macOS's critical pressure the host claim waits for nothing: it
  // takes all it finds at once, and what lands comes on top.
  const bool canWait = spec.timing == NeedTiming::CanWait && !need.hostRefused && spec.cls != ClaimClass::Growth &&
                       !(host && critical);

  // What a claim that can wait may wait for: what lands of every transfer
  // in flight and every owed copy, and the copies its own lanes owe (its
  // request's and those free() converted for it); the pages of its form
  // waiting claims ranked above it take first.
  Returning all, own;
  const size_t form = static_cast<size_t>(need.form);
  uint64_t heldPagesAbove = 0;
  if (canWait && (lackingBytes || lackingPages)) {
    all = returning();
    if (returningFor_) {
      std::vector<uint64_t> lanes = claim.converted;
      if (spec.requestId)
        lanes.push_back(spec.requestId);
      own = returningFor_(lanes);
    }
    heldPagesAbove = heldPages(ClaimId{id})[form];
  }
  const auto waits = [&](bool owedToo) {
    if (!canWait || (!lackingBytes && !lackingPages))
      return false;
    const uint64_t landing = plus(all.landingBytes, owedToo ? all.owedBytes : own.owedBytes);
    const uint64_t landingPages =
        uint64_t{all.landingPages[form]} + (owedToo ? all.owedPages[form] : own.owedPages[form]);
    if (lackingBytes && (plus(forClaim, landing) >= need.bytes || (!owedToo && plus(less(room, growth), all.landingBytes) >= need.bytes)))
      return true;
    return lackingPages && (less(landingPages, heldPagesAbove) >= lackingPages ||
                            (!owedToo && all.landingPages[form] >= lackingPages));
  };
  // What lands and its own lanes' owed copies count before victims also in
  // part (amend/M.md B5): the dry run covers only what they do not. The
  // room held above the claim is in what it lacks already; the host's pass
  // sizes its claim by what lands (MemoryGovernor::controlPass, hostPass).
  const uint64_t creditBytes = host ? 0 : plus(all.landingBytes, own.owedBytes);
  const uint64_t creditPages =
      host ? 0 : less(uint64_t{all.landingPages[form]} + own.owedPages[form], heldPagesAbove);

  std::optional<DryRun> run;
  if (!need.hostRefused && !lackingBytes && !lackingPages && !need.stateBuffers) {
    result.outcome = FreeOutcome::Met;
  } else if (waits(false)) {
    result.outcome = FreeOutcome::Pending;
  } else {
    run = dryRun(spec, query, less(lackingBytes, creditBytes),
                 static_cast<uint32_t>(less(lackingPages, creditPages)), growing);
    if (run->uncopied)
      uncopiedReachedSeconds_ = nowSeconds_;
    if (run->covered) {
      bool whole = true;
      FreeResult taken = take(*run, query, &claim, whole);
      // The buffers its states freed leave the model's pool unless they met
      // the need in place: a need of bytes beside buffers no state can give
      // (Model::statesToActivate with no ring idle) is met by their bytes.
      const bool metInPlace = need.stateBuffers && run->buffers >= need.stateBuffers;
      if (run->bufferBytes && !metInPlace)
        taken.nowBytes = plus(taken.nowBytes, takeUnsized(query));
      const bool alone = (lackingBytes && run->bytes >= lackingBytes) ||
                         (lackingPages && run->pages >= lackingPages) ||
                         (need.stateBuffers && run->buffers >= need.stateBuffers);
      // A take that stopped short waits for what it started to land, else
      // leaves what it freed to the claim's retry.
      const FreeOutcome outcome = !whole ? (taken.laterBytes ? FreeOutcome::Pending : FreeOutcome::Short)
                                  : alone ? taken.outcome
                                          : FreeOutcome::Pending;
      result = {outcome, plus(result.nowBytes, taken.nowBytes), taken.laterBytes};
    } else {
      result.outcome = waits(true) ? FreeOutcome::Pending : FreeOutcome::Short;
    }
  }
  // Pending: the claim waits, holding this need. Short: it waits for
  // nothing. Met: its wait stays as it was, since a need met for one of its
  // request's allocations does not end the request's wait for another.
  if (result.outcome == FreeOutcome::Pending) {
    if (!claim.waiting)
      claim.waitingSinceSeconds = nowSeconds_;
    claim.waiting = true;
    claim.need = need;
  } else if (result.outcome == FreeOutcome::Short) {
    claim.waiting = false;
  }
  // One line per decision: whatever takes units, and a change of a
  // request's outcome.
  if ((run && run->covered) || (spec.requestId && claim.outcome != result.outcome))
    log(spec, need, run && run->covered ? &*run : nullptr, result);
  claim.outcome = result.outcome;
  return result;
}

// R2's steps 0-2 as a dry run: the units that cost nothing and are sized
// only when taken count as their sources size them now (peek), and for a
// claim that can wait what lands and the copies its own request owes count
// before victims, as free() counts them.
uint64_t MemoryLedger::available(const ClaimSpec &spec, uint64_t enough, uint64_t freeBytes, bool critical,
                                 bool deficit, const RestoredChains *kept) const {
  const bool host = spec.cls == ClaimClass::Host;
  const uint64_t held =
      host ? 0 : plus(heldForGrowth(std::nullopt, spec.cls), heldByClaims(spec.rank, std::nullopt));
  uint64_t landing = 0;
  if (!host && spec.timing == NeedTiming::CanWait && spec.cls != ClaimClass::Growth) {
    landing = returning().landingBytes;
    if (returningFor_ && spec.requestId)
      landing = plus(landing, returningFor_(std::span<const uint64_t>(&spec.requestId, 1)).owedBytes);
  }
  const uint64_t room = host ? 0 : plus(freeBytes, landing);
  if (less(room, held) >= enough)
    return enough;
  const VictimQuery query{.claim = &spec, .need = {.bytes = enough}, .critical = critical, .rates = &rates_,
                          .deficit = deficit, .kept = kept};
  const uint64_t taken = dryRun(spec, query, less(plus(enough, held), room), 0, std::nullopt).bytes;
  return std::min(enough, less(plus(room, taken), held));
}

// R3 with amend/M.md B4, its growth rule A30's (owner 11-03 03:55: no unit
// regrown is taken back within the horizon). Pending growth grows once its
// bytes are free for it, taking what its victims left once they landed as a
// claim's retry takes it, without the host's consent on the host's credit
// of what that take releases (the host's footprint is then where it was),
// and is cancelled once that take finds it short. Free room beyond what the
// waiting claims, the lanes' projected growth and every pending growth hold
// goes to the pools, the one whose next unit is worth most first, while the
// host admits each unit and, once claims take a pool's units, while its
// regrowth gains: what it saves over the horizon if no claim takes it back
// within it, at the rate claims take the pool's units (takesPerSecond),
// beyond its cycle. At most one unit per pass then displaces other memory,
// outside critical pressure, when the victims cost less than its regrowth
// gains: with the host's consent the victims need only meet what the room
// lacks; without it they free a whole unit, which the growth then takes
// from the host's footprint again (its credit), and what the room lacks
// besides. Either way victims that free memory only once their transfers
// land make the unit the pool's pending growth. The room is the pass's less
// what the pass has grown.
bool MemoryLedger::grow(uint64_t freeBytes, const GrowthHost &host) {
  bool acted = false;
  uint64_t waitingBytes = 0;
  for (const auto &[id, state] : claims_)
    if (state.waiting)
      waitingBytes = plus(waitingBytes, state.need.bytes);
  uint64_t grownBytes = 0;
  const auto growUnit = [&](size_t index, uint64_t credit, std::string_view how) {
    Account &account = accounts_[index];
    const uint64_t bytes = account.pool->state().growUnitBytes;
    const ClaimId claim = open({.cls = ClaimClass::Growth, .holdSeconds = kValueHorizonSeconds});
    const metal::AllocationResult grown =
        host.grow(claim, AccountId{static_cast<uint8_t>(index)}, *account.pool, credit);
    close(claim);
    if (grown) {
      grownBytes = plus(grownBytes, bytes);
      ++account.grows;
      if (log_) {
        std::ostringstream line;
        line << std::fixed << std::setprecision(1) << "Memory: " << account.name << " grew " << double(bytes) / kMiB
             << " MiB " << how;
        log_(line.str());
      }
    }
    return static_cast<bool>(grown);
  };
  struct Candidate final {
    size_t account = 0;
    uint64_t unit = 0;
    double benefit = 0.0;
    double cycle = 0.0;
  };
  std::vector<Candidate> candidates;
  const auto candidate = [&](size_t index) {
    const Account &account = accounts_[index];
    const model::PoolState state = account.pool->state();
    const model::PoolValue counters = account.pool->value();
    return Candidate{index, state.growUnitBytes,
                     value::growthBenefit(account.rate, counters, state.growUnitBytes, missingBytes(state), rates_),
                     value::cycleSeconds(counters, state.growUnitBytes)};
  };
  for (size_t index = 1; index < accounts_.size(); ++index) {
    Account &account = accounts_[index];
    if (!account.pool)
      continue;
    if (account.pendingGrowth) {
      const uint64_t others = less(pendingGrowthBytes(), account.pendingGrowth);
      // Of its victims' bytes at most what is in flight now has not landed.
      const uint64_t landed = less(account.pendingLaterBytes, returning().landingBytes);
      const bool consents = host.admits(account.pendingGrowth);
      bool met = consents && less(less(freeBytes, grownBytes), plus(plus(others, waitingBytes), laneDemandBytes_)) >=
                                 account.pendingGrowth;
      uint64_t credit = 0;
      // Once they have landed, it takes what they left, as a claim's retry
      // does: the extents a demotion's landing empties, which only a take
      // releases, and the steps of its units still to go, within what its
      // regrowth gains; what that starts lands for a later pass. Without the host's consent it frees the whole unit now, its
      // credit, as a displacement does: a landing frees its page into a
      // kept extent, which leaves the host's footprint only as a take
      // releases it, and the host's pass or a claim may have taken it for
      // itself since. Short, the growth is no longer worth what it takes,
      // and holds nothing.
      if (!met && landed >= account.pendingLaterBytes) {
        const Candidate next = candidate(index);
        const ClaimId claim = open({.cls = ClaimClass::Growth,
                                    .timing = consents ? NeedTiming::CanWait : NeedTiming::Immediate,
                                    .holdSeconds = kValueHorizonSeconds,
                                    .benefitSeconds = std::max(
                                        0.0, value::regrowthSeconds(next.benefit, next.cycle, takesPerSecond(index)))});
        const FreeResult freed = freeFor(claim.value, {.bytes = account.pendingGrowth}, less(freeBytes, grownBytes),
                                         false, false, AccountId{static_cast<uint8_t>(index)}, !consents);
        close(claim);
        met = freed.outcome == FreeOutcome::Met;
        credit = consents ? 0 : freed.nowBytes;
        if (freed.outcome == FreeOutcome::Pending)
          account.pendingLaterBytes = freed.laterBytes;
        else if (freed.outcome == FreeOutcome::Short)
          account.pendingGrowth = account.pendingLaterBytes = 0;
      }
      if (met && growUnit(index, credit, "into its pending growth")) {
        account.pendingGrowth = account.pendingLaterBytes = 0;
        acted = true;
      }
      continue;
    }
    const Candidate next = candidate(index);
    if (next.unit)
      candidates.push_back(next);
  }
  std::ranges::stable_sort(candidates, std::ranges::greater{}, &Candidate::benefit);

  const auto room = [&] {
    return less(less(freeBytes, grownBytes), plus(plus(pendingGrowthBytes(), waitingBytes), laneDemandBytes_));
  };
  for (Candidate &next : candidates) {
    while (next.unit && room() >= next.unit && host.admits(next.unit) &&
           value::growsIntoFreeRoom(takesPerSecond(next.account), next.benefit, next.cycle) &&
           growUnit(next.account, 0, "into free room")) {
      acted = true;
      next = candidate(next.account);
    }
  }
  if (!host.displaces)
    return acted;
  for (const Candidate &next : candidates) {
    Account &account = accounts_[next.account];
    const double benefit = value::regrowthSeconds(next.benefit, next.cycle, takesPerSecond(next.account));
    if (!next.unit || account.pendingGrowth || benefit <= 0.0)
      continue;
    const bool consents = host.admits(next.unit);
    if (consents && room() >= next.unit)
      continue;
    const ClaimId claim = open({.cls = ClaimClass::Growth,
                                .timing = NeedTiming::CanWait,
                                .holdSeconds = kValueHorizonSeconds,
                                .benefitSeconds = benefit});
    const FreeResult freed = freeFor(claim.value, {.bytes = next.unit}, less(freeBytes, grownBytes), false, false,
                                     AccountId{static_cast<uint8_t>(next.account)}, !consents);
    close(claim);
    if (freed.outcome == FreeOutcome::Met)
      acted = growUnit(next.account, consents ? 0 : freed.nowBytes, "by displacement") || acted;
    else if (freed.outcome == FreeOutcome::Pending) {
      account.pendingGrowth = next.unit;
      account.pendingLaterBytes = freed.laterBytes;
      acted = true;
    }
    break;
  }
  return acted;
}

uint64_t MemoryLedger::dropToFloors() {
  uint64_t released = 0;
  for (Account &account : accounts_) {
    if (!account.pool)
      continue;
    account.pendingGrowth = account.pendingLaterBytes = 0;
    const uint64_t bytes = account.pool->release(std::numeric_limits<uint64_t>::max());
    if (bytes) {
      ++account.shrinks;
      ++account.taken;
    }
    released = plus(released, bytes);
  }
  return released;
}

void MemoryLedger::fill(MemoryGovernorSnapshot &snapshot, uint64_t chargedBytes) const {
  snapshot.accounts.clear();
  uint64_t others = 0;
  std::vector<MemoryAccountSnapshot> accounts;
  for (size_t index = 1; index < accounts_.size(); ++index) {
    const Account &account = accounts_[index];
    MemoryAccountSnapshot row{.name = account.name, .reservedBytes = account.pendingGrowth,
                              .grows = account.grows, .shrinks = account.shrinks};
    if (account.pool) {
      const model::PoolState state = account.pool->state();
      const model::PoolValue counters = account.pool->value();
      row.floorBytes = state.floorBytes;
      row.currentBytes = state.currentBytes;
      row.capBytes = state.capBytes;
      row.unitBytes = state.releaseUnitBytes ? state.releaseUnitBytes : state.growUnitBytes;
      if (row.unitBytes) {
        row.valuePerGiB = value::poolUnit(account.rate, counters, row.unitBytes,
                                          missingBytes(state), kValueHorizonSeconds, rates_)
                              .at(kValueHorizonSeconds) *
                          kGiB / double(row.unitBytes);
        const value::MarginalCost margin = value::marginal(account.rate, counters, row.unitBytes);
        row.growValuePerGiB = margin.growRatePerSecond * kValueHorizonSeconds * kGiB / double(row.unitBytes);
        row.shrinkValuePerGiB = margin.shrinkRatePerSecond * kValueHorizonSeconds * kGiB / double(row.unitBytes);
      }
    } else {
      row.currentBytes = account.current();
      row.capBytes = row.currentBytes;
    }
    others = plus(others, row.currentBytes);
    accounts.push_back(std::move(row));
  }
  const uint64_t fixed = less(chargedBytes, others);
  snapshot.accounts.push_back({.name = accounts_.front().name, .currentBytes = fixed, .capBytes = fixed});
  for (MemoryAccountSnapshot &row : accounts)
    snapshot.accounts.push_back(std::move(row));
  snapshot.reservedBytes = pendingGrowthBytes();

  const Returning back = returning();
  snapshot.owed = {.transfers = back.transfers,
                   .bytes = plus(back.landingBytes, back.owedBytes),
                   .oldestMilliseconds = back.oldestMilliseconds};
  snapshot.claims = {};
  std::optional<double> oldest;
  for (const auto &[id, state] : claims_) {
    if (!state.waiting)
      continue;
    ++snapshot.claims.waiting;
    snapshot.claims.bytes = plus(snapshot.claims.bytes, state.need.bytes);
    oldest = std::min(oldest.value_or(state.waitingSinceSeconds), state.waitingSinceSeconds);
  }
  if (oldest)
    snapshot.claims.oldestMilliseconds = std::max(0.0, nowSeconds_ - *oldest) * 1000.0;
  snapshot.owed.toClaimsBytes = std::min(snapshot.owed.bytes, snapshot.claims.bytes);
}

void MemoryLedger::log(const ClaimSpec &spec, const Need &need, const DryRun *run, const FreeResult &result) const {
  if (!log_)
    return;
  std::ostringstream line;
  line << std::fixed << std::setprecision(1) << "Memory: " << className(spec.cls) << " claim";
  if (spec.requestId)
    line << " of request " << spec.requestId;
  line << " needs " << double(need.bytes) / kMiB << " MiB";
  if (need.pages)
    line << ", " << need.pages << " pages";
  if (need.stateBuffers)
    line << ", " << need.stateBuffers << " state buffers";
  line << ": " << outcomeName(result.outcome);
  if (run) {
    line << ", took";
    for (const Choice &choice : run->chosen) {
      line << ' ';
      if (choice.pool)
        line << accounts_[choice.pool->value].name << (choice.pending ? " pending" : " unit");
      else
        line << sources_[choice.source]->name()
             << (choice.victim.kind == VictimKind::Conversion    ? " conversion"
                 : choice.victim.kind == VictimKind::PassCommand ? " pass command"
                 : choice.victim.cls == ReclaimClass::InUse      ? " in use"
                                                                 : "");
      line << ' ' << double(choice.victim.bytes) / kMiB;
    }
    line << std::setprecision(3) << " for " << run->cost << " s";
  }
  line << std::setprecision(1) << " (" << double(result.nowBytes) / kMiB << " MiB now, "
       << double(result.laterBytes) / kMiB << " later)";
  log_(line.str());
}

} // namespace splash::engine
