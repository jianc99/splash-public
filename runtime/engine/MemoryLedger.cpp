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
  // The share of the need a unit meets, in the measure it meets most of.
  [[nodiscard]] double share(const Victim &victim) const noexcept {
    double share = 0.0;
    if (bytes)
      share = std::max(share, double(victim.bytes) / double(bytes));
    if (pages && victim.form == form)
      share = std::max(share, double(victim.pages) / double(pages));
    if (buffers)
      share = std::max(share, double(victim.stateBuffers) / double(buffers));
    return share;
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
  // against every other.
  uint64_t pendingGrowth = 0;
  uint64_t grows = 0;
  uint64_t shrinks = 0;
  // The units claims took from it, which its rate differentiates.
  uint64_t taken = 0;
  value::PoolRate rate;
};

// One unit a dry run chose: a source's (by index, after the units of the
// source it passed over since the source's previous choice), or a pool's
// unit or pending growth (by account).
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

void MemoryLedger::addSource(VictimSource &source) { sources_.push_back(&source); }

void MemoryLedger::setReturning(std::function<Returning()> all, std::function<Returning(uint64_t)> ofRequest) {
  returning_ = std::move(all);
  returningFor_ = std::move(ofRequest);
}

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
  claims_.emplace(id.value, ClaimState{.spec = spec});
  return id;
}

void MemoryLedger::update(ClaimId claim, const ClaimSpec &spec) { claims_.at(claim.value).spec = spec; }

void MemoryLedger::refused(ClaimId claim, const Need &need) {
  claims_.at(claim.value).need = {.bytes = need.bytes, .hostRefused = need.hostRefused};
}

void MemoryLedger::setWaiting(ClaimId claim, bool waiting) {
  ClaimState &state = claims_.at(claim.value);
  if (waiting && !state.waiting)
    state.waitingSinceSeconds = nowSeconds_;
  state.waiting = waiting;
}

void MemoryLedger::setHostExempt(ClaimId claim, bool exempt) { claims_.at(claim.value).hostExempt = exempt; }

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
  return held;
}

bool MemoryLedger::claimWaiting() const noexcept {
  return std::ranges::any_of(claims_, [](const auto &entry) { return entry.second.waiting; });
}

uint64_t MemoryLedger::hostRefusedWaitingBytes() const noexcept {
  uint64_t bytes = 0;
  for (const auto &[id, state] : claims_)
    if (state.waiting && state.need.hostRefused)
      bytes = plus(bytes, state.need.bytes);
  return bytes;
}

uint64_t MemoryLedger::pendingGrowthBytes() const noexcept {
  uint64_t bytes = 0;
  for (const Account &account : accounts_)
    bytes = plus(bytes, account.pendingGrowth);
  return bytes;
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
// may not take is skipped, not stopped at.
MemoryLedger::DryRun MemoryLedger::dryRun(const ClaimSpec &spec, const VictimQuery &query, uint64_t lackingBytes,
                                          std::optional<AccountId> growing) const {
  const Lacking lack{.bytes = lackingBytes,
                     .pages = query.need.pages,
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
  // memory; none for a claim below a lane in service. Pending growth only
  // for claims that may take memory the engine's limit holds for a pool.
  const bool belowALane = lanePriority_ && *lanePriority_ < spec.priority;
  const bool poolUnits = !belowALane && (spec.cls == ClaimClass::InService || spec.cls == ClaimClass::Optional ||
                                         spec.cls == ClaimClass::Growth ||
                                         (spec.cls == ClaimClass::Host && (query.deficit || query.critical)));
  const bool pendingUnits = poolUnits && spec.cls != ClaimClass::Growth && spec.cls != ClaimClass::Host;

  struct Head final {
    std::optional<Victim> victim;
    size_t source = 0;
    uint32_t passed = 0;  // the source's units passed over since its last choice
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
    return Victim{.key = head.taken / head.state.releaseUnitBytes,
                  .bytes = bytes,
                  .cost = value::poolUnit(account.rate, head.counters, bytes, missingBytes(head.state),
                                          spec.holdSeconds, rates_)};
  };
  for (size_t index = 0; index < sources_.size(); ++index)
    heads.push_back(Head{.victim = sources_[index]->peek(query, nullptr), .source = index});
  for (size_t index = 1; index < accounts_.size(); ++index) {
    const Account &account = accounts_[index];
    if (!account.pool || !poolUnits || (growing && growing->value == index))
      continue;
    Head head{.pool = AccountId{static_cast<uint8_t>(index)},
              .state = account.pool->state(),
              .counters = account.pool->value()};
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

  DryRun run;
  for (;;) {
    Head *best = nullptr;
    double bestPrice = 0.0;
    for (Head &head : heads) {
      // Skip what this claim may not take or what meets none of its need;
      // keep a unit that lands later in an Immediate claim's way, to note
      // that the run reached it.
      while (head.victim && (!mayTake(*head.victim) || lack.share(*head.victim) == 0.0))
        pass(head);
      if (!head.victim)
        continue;
      const double price = head.victim->cost.at(spec.holdSeconds) / lack.share(*head.victim);
      if (!best || price < bestPrice || (price == bestPrice && !head.victim->later && best->victim->later)) {
        best = &head;
        bestPrice = price;
      }
    }
    if (!best)
      break;
    const Victim victim = *best->victim;
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
    if (victim.form == lack.form)
      run.pages += victim.pages;
    run.buffers += victim.stateBuffers;
    run.later = run.later || victim.later;
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

// Takes what a covered dry run chose, in its order: a source's unit by its
// place in the source's order as it stands, past the units the run passed
// over (VictimSource), a pool's unit by its release, pending growth by
// cancelling it.
FreeResult MemoryLedger::take(const DryRun &run, const VictimQuery &query, ClaimState *claim) {
  FreeResult result{.outcome = run.later ? FreeOutcome::Pending : FreeOutcome::Met};
  // Each source's last unit the run passed over, which stays in its order.
  std::vector<std::optional<Victim>> passed(sources_.size());
  for (const Choice &choice : run.chosen) {
    if (choice.pool) {
      Account &account = accounts_[choice.pool->value];
      ++account.taken;
      if (choice.pending) {
        account.pendingGrowth = 0;
        result.nowBytes = plus(result.nowBytes, choice.victim.bytes);
        continue;
      }
      const uint64_t released = account.pool->release(choice.victim.bytes);
      if (released)
        ++account.shrinks;
      result.nowBytes = plus(result.nowBytes, released);
      continue;
    }
    VictimSource &source = *sources_[choice.source];
    std::optional<Victim> &before = passed[choice.source];
    std::optional<Victim> unit = source.peek(query, before ? &*before : nullptr);
    for (uint32_t over = 0; unit && over < choice.passed; ++over) {
      before = unit;
      unit = source.peek(query, &*before);
    }
    if (!unit)
      throw std::logic_error(std::string(source.name()) + " no longer offers a unit its dry run chose");
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

// The one rule (DESIGN section 4.4; amend/M.md B2-B5): the units that cost
// nothing and cannot be sized go first; then a claim that can wait waits
// when what is landing, with its own lanes' owed copies, meets its need (or
// would, but for claims ranked above it); else the cheapest units that meet
// it are taken, all or none; else it waits when every copy in flight or owed
// would meet it. A pool's growth waits on no landing but its own victims':
// what it waits for is held as pending growth (R3). A claim admission
// refused bytes lacks what admission asks beyond the room: its need and the
// room held against it; a need met in place lacks no bytes. A need the host
// refused counts only units that leave memory in place; the host's claim
// lacks its whole need, and growth the host admits only on its credit
// (`credited`) at least what free() leaves now: the room under the engine's
// limit returns nothing to the host.
FreeResult MemoryLedger::freeFor(uint32_t id, const Need &need, uint64_t freeBytes, bool critical, bool deficit,
                                 std::optional<AccountId> growing, bool credited) {
  ClaimState &claim = claims_.at(id);
  const ClaimSpec &spec = claim.spec;
  claim.need = need;
  const VictimQuery query{.claim = &spec, .need = need, .critical = critical, .rates = &rates_, .deficit = deficit};

  FreeResult result;
  for (VictimSource *source : sources_) {
    for (std::optional<Victim> unit = source->peek(query, nullptr);
         unit && !unit->sized && unit->cost.at(spec.holdSeconds) == 0.0; unit = source->peek(query, nullptr)) {
      const FreeResult taken = source->take(query, *unit);
      result.nowBytes = plus(result.nowBytes, taken.nowBytes);
      if (!taken.nowBytes && !taken.laterBytes)
        break;
    }
  }
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
  const bool canWait = spec.timing == NeedTiming::CanWait && !need.hostRefused && spec.cls != ClaimClass::Growth;

  const auto waits = [&](bool owedToo) {
    if (!canWait || (!lackingBytes && !lackingPages))
      return false;
    const Returning all = returning();
    Returning own;
    if (returningFor_) {
      std::vector<uint64_t> lanes = claim.converted;
      if (spec.requestId)
        lanes.push_back(spec.requestId);
      for (uint64_t lane : lanes) {
        const Returning of = returningFor_(lane);
        own.owedBytes = plus(own.owedBytes, of.owedBytes);
        for (size_t form = 0; form < own.owedPages.size(); ++form)
          own.owedPages[form] += of.owedPages[form];
      }
    }
    const size_t form = static_cast<size_t>(need.form);
    const uint64_t landing = plus(all.landingBytes, owedToo ? all.owedBytes : own.owedBytes);
    const uint64_t landingPages =
        uint64_t{all.landingPages[form]} + (owedToo ? all.owedPages[form] : own.owedPages[form]);
    const uint64_t heldPagesAbove = heldPages(ClaimId{id})[form];
    if (lackingBytes && (plus(forClaim, landing) >= need.bytes || (!owedToo && plus(less(room, growth), all.landingBytes) >= need.bytes)))
      return true;
    return lackingPages && (less(landingPages, heldPagesAbove) >= lackingPages ||
                            (!owedToo && all.landingPages[form] >= lackingPages));
  };

  std::optional<DryRun> run;
  if (!need.hostRefused && !lackingBytes && !lackingPages && !need.stateBuffers) {
    result.outcome = FreeOutcome::Met;
  } else if (waits(false)) {
    result.outcome = FreeOutcome::Pending;
  } else {
    run = dryRun(spec, query, lackingBytes, growing);
    if (run->uncopied)
      uncopiedReachedSeconds_ = nowSeconds_;
    if (run->covered) {
      const FreeResult taken = take(*run, query, &claim);
      result = {taken.outcome, plus(result.nowBytes, taken.nowBytes), taken.laterBytes};
    } else {
      result.outcome = waits(true) ? FreeOutcome::Pending : FreeOutcome::Short;
    }
  }
  const bool waiting = result.outcome == FreeOutcome::Pending;
  if (waiting && !claim.waiting)
    claim.waitingSinceSeconds = nowSeconds_;
  claim.waiting = waiting;
  // One line per decision: whatever takes units, and a change of a
  // request's outcome.
  if ((run && run->covered) || (spec.requestId && claim.outcome != result.outcome))
    log(spec, need, run && run->covered ? &*run : nullptr, result);
  claim.outcome = result.outcome;
  return result;
}

uint64_t MemoryLedger::available(const ClaimSpec &spec, uint64_t enough, uint64_t freeBytes, bool critical,
                                 bool deficit) const {
  const uint64_t held = spec.cls == ClaimClass::Host
                            ? 0
                            : plus(heldForGrowth(std::nullopt, spec.cls), heldByClaims(spec.rank, std::nullopt));
  const uint64_t room = spec.cls == ClaimClass::Host ? 0 : freeBytes;
  if (less(room, held) >= enough)
    return enough;
  const VictimQuery query{.claim = &spec, .need = {.bytes = enough}, .critical = critical, .rates = &rates_,
                          .deficit = deficit};
  const uint64_t taken = dryRun(spec, query, less(plus(enough, held), room), std::nullopt).bytes;
  return std::min(enough, less(plus(room, taken), held));
}

// R3 with amend/M.md B4. Pending growth grows once its bytes are free for
// it. Free room beyond what the waiting claims, the lanes' projected growth
// and every pending growth hold goes to the pools, the one whose next unit
// is worth most first, while the host admits each unit and, once claims take
// a pool's units, while a unit is worth its expected retake. At most one
// unit per pass then displaces other memory, outside critical pressure,
// when the victims cost less than the unit is worth beyond its cycle: with
// the host's consent the victims need only meet what the room lacks, and
// what its own victims free later is the pool's pending growth; without it
// they free a whole unit now, which the growth then takes from the host's
// footprint again (its credit), and what the room lacks besides. The room
// is the pass's less what the pass has grown.
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
      if (less(less(freeBytes, grownBytes), plus(plus(others, waitingBytes), laneDemandBytes_)) >=
          account.pendingGrowth) {
        account.pendingGrowth = 0;
        acted = growUnit(index, 0, "into its pending growth") || acted;
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
    Account &account = accounts_[next.account];
    while (next.unit && room() >= next.unit && host.admits(next.unit) &&
           value::growsIntoFreeRoom(account.rate, next.benefit, next.cycle) && growUnit(next.account, 0, "into free room")) {
      acted = true;
      next = candidate(next.account);
    }
  }
  if (!host.displaces)
    return acted;
  for (const Candidate &next : candidates) {
    Account &account = accounts_[next.account];
    const double benefit = next.benefit - next.cycle;
    if (!next.unit || account.pendingGrowth || benefit <= 0.0)
      continue;
    const bool consents = host.admits(next.unit);
    if (consents && room() >= next.unit)
      continue;
    const ClaimId claim = open({.cls = ClaimClass::Growth,
                                .timing = consents ? NeedTiming::CanWait : NeedTiming::Immediate,
                                .holdSeconds = kValueHorizonSeconds,
                                .benefitSeconds = benefit});
    const FreeResult freed = freeFor(claim.value, {.bytes = next.unit}, less(freeBytes, grownBytes), false, false,
                                     AccountId{static_cast<uint8_t>(next.account)}, !consents);
    close(claim);
    if (freed.outcome == FreeOutcome::Met)
      acted = growUnit(next.account, consents ? 0 : freed.nowBytes, "by displacement") || acted;
    else if (freed.outcome == FreeOutcome::Pending) {
      account.pendingGrowth = next.unit;
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
    account.pendingGrowth = 0;
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
      if (row.unitBytes)
        row.valuePerGiB = value::poolUnit(account.rate, counters, row.unitBytes,
                                          missingBytes(state), kValueHorizonSeconds, rates_)
                              .at(kValueHorizonSeconds) *
                          kGiB / double(row.unitBytes);
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
