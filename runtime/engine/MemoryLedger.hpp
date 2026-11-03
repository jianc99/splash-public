#pragma once

#include "engine/ReclaimClass.hpp"
#include "engine/Types.hpp"
#include "model/ElasticPool.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace splash::engine {

struct MemoryGovernorSnapshot;
class RestoredChains;

// The memory layer's policy constants. The working-set margin and the host's
// reserves are not among them: they set the limit and gate the start
// (EngineMemoryPolicy).
// What elective allocation leaves free above the host's spare reserve.
inline constexpr uint64_t kHostMarginBytes = 1ULL << 30;
// How long the host's counters take to show what a pass returned.
inline constexpr double kHostSettleMilliseconds = 1000.0;
// The payback horizon of growth that displaces other memory, and how long
// memory a host pass returns stays returned: within the lifetime of a
// regrown expert-store chunk. Of the 40 chunks the Studio's recorded churn
// runs took back, 15 went again (after 1-114 s; 12 of them the 35B runs'
// 1-29 s, which the take rate prices) and 25 lived to the run's end:
// Kaplan-Meier survival 0.57 at 60 s and 0.53 from 114 s on, so the median
// lifetime lies past every retake recorded and 60 s stays within it.
inline constexpr double kValueHorizonSeconds = 60.0;
// The time constant of the measured rates: pools' ghost hits, scan reads and
// takes, the cache's reuse, the KV tier's throughput. amend/M.md's rule asks
// the smallest of 10, 30, 60 and 120 s at which every recorded run's decode
// misses and prefill reads, replayed through PoolRate at the control pass's
// 500 ms, change by less, at P95 over passes of steady load, than half the
// relative hysteresis a 1023 MiB store chunk's cycle (refill and compaction)
// gives it at the run's value. None meets it over every run: of the M6's 45,
// at 60 s 1.5.0's m150a (1.68x), m150b (1.06x), m150a2 (1.02x) and fn15b
// (1.11x) fail, at 120 s m150b (1.04x) and fn15b (1.17x); of the Studio's 63
// Flash-Next runs (valued at its 18G store's 0.0071 misses a token a slot:
// its 6681-slot runs against its 5672-slot ones), at 60 s s150Fa (1.23x)
// and the three 2-minute q runs (1.6-2.3x), at 120 s the q runs; its five
// 35B runs have no pinned-store pair to value them. 60 s is the smallest at
// which every 15-minute run of the trees 1.6.0 builds on passes (at 30 s
// mMNa and mSl fail, 1.05x): the exclusion is open (merge report).
// memory_value_test replays mMNa and pFa.
inline constexpr double kRateTimeConstantSeconds = 60.0;

struct AccountId final {
  uint8_t value = 0;
  friend bool operator==(AccountId, AccountId) = default;
};
inline constexpr AccountId kFixedAccount{0};
// For status only: nothing branches on an account's role or name.
enum class AccountRole : uint8_t { Fixed, Kv, States, Pool };

// Why an allocation is made: decides the host's rule and what free() may take.
enum class ClaimClass : uint8_t { Floor, InService, Start, Optional, Growth, Host };
// Whether the claimant can wait for memory a transfer returns.
enum class NeedTiming : uint8_t { CanWait, Immediate };

// Who memory goes to first, the larger first: Engine::yieldsBefore inverted,
// then the earlier submission (Engine::laneToYield's tie).
struct ClaimRank final {
  uint8_t priority = 0;          // 2 - RequestPriority: Foreground 2, Background 0
  bool decoding = false;         // at equal priority a decoding lane outranks a prefill
  uint64_t completedTokens = 0;  // then more completed work
  uint64_t earliness = 0;        // then earlier submission: UINT64_MAX - Scheduler::submissionOrder
  auto operator<=>(const ClaimRank &) const = default;
};

// Everything the caller knows about why it allocates, passed down explicitly.
struct ClaimSpec final {
  ClaimClass cls = ClaimClass::InService;
  uint64_t requestId = 0;                       // 0: the engine's own, a pool's growth, the host
  ClaimRank rank;
  ReclaimClass reach = ReclaimClass::Ordinary;  // InUse: may take what unfinished requests use
  NeedTiming timing = NeedTiming::CanWait;
  // May take the command between its passes, as the scheduler's decision
  // that the claimant's work supersedes it says.
  bool supersedes = false;
  double holdSeconds = 0.0;                     // how long the claimant keeps what it gets
  // The seconds of engine work the units taken for it may cost in all.
  double benefitSeconds = std::numeric_limits<double>::infinity();
  // The tier that decides what it may take from lanes: no conversion of a
  // lane of a higher priority, and no pool unit while such a lane is in
  // service (pools serve every lane). The rank still orders held room. A
  // claim that is no lane's (the host, a pool's growth) takes Foreground.
  RequestPriority priority = RequestPriority::Foreground;
};

struct ClaimId final {
  uint32_t value = 0;
  explicit operator bool() const noexcept { return value != 0; }
  friend bool operator==(ClaimId, ClaimId) = default;
};

// What a refused claim lacks. bytes and hostRefused come from the admission
// refused inside its scope; pages and stateBuffers are set by the caller for
// needs met in place (TokenAdmission::additionalPages,
// Model::statesToActivate).
struct Need final {
  uint64_t bytes = 0;
  bool hostRefused = false;
  uint32_t pages = 0;
  kv::PageForm form = kv::PageForm::Full;
  uint32_t stateBuffers = 0;
};

enum class FreeOutcome : uint8_t { Met, Pending, Short };
struct FreeResult final {
  FreeOutcome outcome = FreeOutcome::Short;
  uint64_t nowBytes = 0;     // left memory now (extents, chunks, buffers)
  uint64_t laterBytes = 0;   // leave memory when transfers it started land
};

// Seconds of engine work a unit costs if taken now and kept away for a hold
// of that many seconds.
struct UnitCost final {
  double onceSeconds = 0.0;
  double ratePerSecond = 0.0;
  [[nodiscard]] double at(double holdSeconds) const noexcept { return onceSeconds + ratePerSecond * holdSeconds; }
};

// The rates units are priced at, set once per control pass (setRates).
struct MemoryRates final {
  double prefillSecondsPerRow = 0.0;      // Scheduler::prefillMilliseconds of a 2048-row command, per row
  // Scheduler::weightReadMillisecondsPerByte, per second: what a byte the
  // expert store misses in a prefill costs.
  double weightReadSecondsPerByte = 0.0;
  // The engine's decayed mean of a decode command's hot-block read time
  // (CommandSample::spilledKvReadMilliseconds) per lane with Index pages;
  // from the measured 7.0-11.5 ms (DESIGN section 0 Q5) until its first
  // sample.
  double spilledReadMillisecondsPerLaneCommand = 9.0;
  double decodeTokensPerLaneCommand = 0.0;  // Scheduler::decodeTokensPerLaneCommand
  // Model::loadReads's SSD rate until the KV tier has read, then its read
  // throughput (KvTier::throughput), over kRateTimeConstantSeconds.
  double tierReadBytesPerSecond = 0.0;
  // The probes' hit share over kRateTimeConstantSeconds; 0 before probes.
  double reuse = 0.0;
};

// What taking a unit gives up besides its bytes, which the claim's class
// decides whether it may: memory nothing needs now (a cached unit, a
// buffer), a Resident lane's KV in memory (its conversion to Spill), or the
// command between its passes.
enum class VictimKind : uint8_t { Memory, Conversion, PassCommand };

// One unit a source would give up, in the source's own order.
struct Victim final {
  uint64_t key = 0;             // the source's handle (block, state block, request id, pool unit)
  uint64_t bytes = 0;           // device bytes it frees (an extent it empties, a chunk, buffers)
  uint32_t pages = 0;           // free KV pages of `form` it leaves in kept extents (in-place needs)
  kv::PageForm form = kv::PageForm::Full;
  uint32_t stateBuffers = 0;    // pooled state buffers it frees (in-place needs; bytes, setBufferBytes)
  bool later = false;           // frees on a landing (a demotion, a state write, a conversion's copies)
  bool certainReuse = false;    // in use, the newest resume point, a chain a queued request restores
  // false: a unit that costs nothing, taken before any dry run, whose
  // bytes are at least what peek gives (available() counts them)
  bool sized = true;
  ReclaimClass cls = ReclaimClass::Ordinary;
  UnitCost cost;
  VictimKind kind = VictimKind::Memory;
};
struct VictimQuery final {
  const ClaimSpec *claim = nullptr;
  Need need;
  bool critical = false;        // macOS critical: InUse and certain-reuse units for Host claims too
  const MemoryRates *rates = nullptr;
  // The host lacks memory beyond its margin: a Host claim may take pool
  // units and conversions, not only cache.
  bool deficit = false;
  // Chains a walk that may take what is of certain reuse keeps all the
  // same: the one a start restores, which its placement leaves out of the
  // need it asks about (Engine::placement).
  const RestoredChains *kept = nullptr;
};

// A source of the units free() takes, on the engine thread between commands.
// peek changes nothing, and may understate what later units free but never
// overstate it: free() takes nothing unless what it peeked covers the need.
// What a later unit frees includes what goes only once another of its
// transfers lands, as the writes that landing lets start: take starts what
// can start now, and the claim, waiting, takes the rest on its retries as
// landings allow. take takes exactly the unit the latest peek returned. A
// key may name its unit by no more than its place in the source's order
// (the cache's reclaims are steps from its state as it stands), but a
// conversion's is
// its lane's request id, which free() records: free() takes the units its
// dry run chose in the source's order, each found again by peeking from the
// head as it stands then, past as many units as the run passed over before
// it. So taking a unit leaves the units before it where they were and those
// after it in their order; a unit found again that is not the one chosen (of
// another kind, or another lane's conversion) ends the take there, with what
// it took, and the claim's retry weighs the rest afresh. Units that cost
// nothing and are sized only when taken come first in a source's order.
class VictimSource {
public:
  virtual ~VictimSource() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual std::optional<Victim> peek(const VictimQuery &query, const Victim *after) const = 0;
  virtual FreeResult take(const VictimQuery &query, const Victim &victim) = 0;
};

// What transfers free when they land (Cache::returning). Landing: in flight
// now. Owed: copies Spill requests owe and have not started. Restores free
// nothing and are not here.
struct Returning final {
  // By kv::PageForm: demotions, and Spill landings as far as free Index
  // pages take them.
  std::array<uint32_t, 3> landingPages{};
  // Full: owed copies of pages that spill once copied, all of them (the
  // Full extents they empty fund their Index pages).
  std::array<uint32_t, 3> owedPages{};
  // The extents the landing pages complete (compaction), and the state
  // write's buffers.
  uint64_t landingBytes = 0;
  uint64_t owedBytes = 0;  // the extents the owed pages complete
  uint32_t transfers = 0;
  double oldestMilliseconds = 0.0;
};

// What the ledger knows of an open claim.
struct ClaimState final {
  ClaimSpec spec;
  // What the latest refusal in its scope recorded (refused): the need of
  // the allocation it attempts now.
  Need refusal;
  // What its latest free() that left it waiting was given, the in-place
  // part with it: a waiting claim holds it until its wait ends, once the
  // request is admitted or its lane runs, a free() finds it short, or its
  // resource wait ends (Engine::syncClaims). A need met for one of the
  // request's allocations does not end its wait for another.
  Need need;
  bool waiting = false;
  double waitingSinceSeconds = 0.0;
  // Its request waits to retry an allocation the host refused
  // (Engine::syncClaims): what that refusal asked counts in the host's
  // deficit (amend/M.md B3) until the wait ends. It holds no room: the
  // host, not the engine's limit, refused it.
  bool hostRefusedWaiting = false;
  // The host refused it after in-place units could not meet it: the
  // retries of that allocation pass the host with each of their
  // reservations (a Floor or InService claim), until the allocation ends
  // (Engine::allocate), as the serving mark did (design/M.md R2).
  bool hostExempt = false;
  // The lanes free() converted for it, whose copies it waits for as its own.
  std::vector<uint64_t> converted;
  // What its latest free() decided.
  std::optional<FreeOutcome> outcome;
  // When it opened on the control clock, the pools whose units free() took
  // for it, one entry a unit, and whether the engine's lanes may yet need
  // what it took: its lane prefills, growing into it, or shares the engine
  // with another lane in service (Engine::syncClaims). While they may, the
  // pools' units go at its rate since it opened (MemoryLedger::grow).
  double openedSeconds = 0.0;
  std::vector<AccountId> poolTakes;
  bool inDemand = false;
};

// The host's side of one growth pass, from the governor: whether the host
// admits a pool's growth of so many bytes on its own (the host's rule for
// the Growth class at the pass's sample), whether growth may displace other
// memory (not under macOS's critical pressure), and the growth itself: one
// unit of the pool under the governor's admission inside
// Scope{claim, its account, creditBytes}.
struct GrowthHost final {
  std::function<bool(uint64_t bytes)> admits;
  bool displaces = false;
  std::function<metal::AllocationResult(ClaimId claim, AccountId account, model::ElasticPool &pool,
                                        uint64_t creditBytes)>
      grow;
};

// The books and the one rule. No Metal, no host: the governor supplies the
// charged bytes and the host's decisions.
class MemoryLedger final {
public:
  MemoryLedger();
  ~MemoryLedger();
  MemoryLedger(const MemoryLedger &) = delete;
  MemoryLedger &operator=(const MemoryLedger &) = delete;

  // Registration (startup; stable for the process). An account's current
  // bytes are pulled from its owner. Units tie, at equal cost per byte, to
  // the source registered first and then to the pools.
  AccountId addAccount(std::string name, AccountRole role, std::function<uint64_t()> currentBytes);
  AccountId addPool(model::ElasticPool &pool);   // account + victim source; name = PoolState::name
  // The cache, the engine's lane victims: each source once, whoever
  // registers it again (two engines over one cache), until it is removed.
  void addSource(VictimSource &source);
  void removeSource(VictimSource &source) noexcept;
  // What transfers return when they land: all of them (Cache::returning),
  // and those of some requests' own pages, the extents their owed copies
  // complete beside every landing (Cache::returningFor).
  void setReturning(std::function<Returning()> all,
                    std::function<Returning(std::span<const uint64_t> requestIds)> ofRequests);
  // The bytes the model's pool releases of `states` more state buffers
  // (Model::idleBufferBytes beyond its value at none): what the buffers a
  // unit frees give a need of bytes once they leave the pool.
  void setBufferBytes(std::function<uint64_t(uint32_t states)> bytes);
  // The bytes of a KV page of each kv::PageForm (Cache::bytesPerPage), in
  // which the bytes waiting claims hold are held as pages (heldPages).
  void setPageBytes(const std::array<uint64_t, 3> &bytes) noexcept;
  void setRates(const MemoryRates &rates) noexcept;
  // Differentiates each pool's counters, and the units claims took from it,
  // over the control clock, only across intervals in which the engine ran a
  // command (an idle interval says nothing of demand); also the clock the
  // ledger dates its claims and copy demand by.
  void observePools(double nowSeconds, bool commandsRan);
  // The lanes in service: their projected growth, which pools' growth leaves
  // free, and the highest priority among them.
  void setLanes(uint64_t demandBytes, std::optional<RequestPriority> highest) noexcept;
  [[nodiscard]] uint64_t laneDemandBytes() const noexcept { return laneDemandBytes_; }
  // Where the ledger writes one line per decision: the claimant, what it
  // took, the price, the bytes and the outcome.
  void setDecisionLog(std::function<void(std::string_view)> log);

  // Claims.
  [[nodiscard]] ClaimId open(const ClaimSpec &spec);
  void update(ClaimId claim, const ClaimSpec &spec);
  void refused(ClaimId claim, const Need &need);
  void setWaiting(ClaimId claim, bool waiting);
  void setHostRefusedWaiting(ClaimId claim, bool waiting);
  void setHostExempt(ClaimId claim, bool exempt);
  void setInDemand(ClaimId claim, bool inDemand);
  void close(ClaimId claim) noexcept;
  [[nodiscard]] const ClaimState &claim(ClaimId claim) const;
  // Every open claim by its id, for the checks of the ledger's relations.
  [[nodiscard]] const std::map<uint32_t, ClaimState> &claims() const noexcept { return claims_; }
  // What admission holds against an allocation in `claim`'s scope: other
  // accounts' pending growth, waiting claims ranked above it, and for a
  // Growth claim the lanes' projected growth.
  [[nodiscard]] uint64_t heldBytes(ClaimId claim, std::optional<AccountId> growing) const;
  // The free KV pages of each form held against a page handed out in
  // `claim`'s scope: the pages waiting claims ranked above it lack, and the
  // bytes they lack in that form's pages, the room a hand-out from free
  // pages in kept extents would take from what they wait for.
  [[nodiscard]] std::array<uint32_t, 3> heldPages(ClaimId claim) const;
  // The waiting claims, and the bytes the latest refusals of the claims that
  // wait, or whose requests wait to retry what the host refused, asked of
  // the host.
  [[nodiscard]] bool claimWaiting() const noexcept;
  [[nodiscard]] uint64_t hostRefusedWaitingBytes() const noexcept;
  // The units the pools gave back since the engine started.
  [[nodiscard]] uint64_t poolReleases() const noexcept;
  // The largest unit a pool grows by next: the host's band reaches past its
  // margin by it.
  [[nodiscard]] uint64_t largestGrowthUnit() const;
  [[nodiscard]] Returning returning() const;
  // Whether, within the value horizon, a free() reached a cached unit that
  // frees memory only once its copy lands (took it so, or skipped it for an
  // Immediate claim): a temporary tier's copies ahead then pay.
  [[nodiscard]] bool copyDemand() const noexcept;
  // The rates units are priced at now, which a claim's benefit is priced at
  // too.
  [[nodiscard]] const MemoryRates &rates() const noexcept { return rates_; }

  // The rule: what is taken for a claim, what is there for one, what grows.
  // `critical`: macOS's critical pressure; `deficit`: the host lacks memory
  // past its margin; `kept`: the chains a walk for what is there keeps
  // (VictimQuery::kept).
  [[nodiscard]] FreeResult free(ClaimId claim, const Need &need, uint64_t freeBytes, bool critical, bool deficit);
  [[nodiscard]] uint64_t available(const ClaimSpec &spec, uint64_t enough, uint64_t freeBytes, bool critical,
                                   bool deficit, const RestoredChains *kept = nullptr) const;
  // One growth pass. True when a pool grew or started freeing for its
  // pending growth.
  bool grow(uint64_t freeBytes, const GrowthHost &host);
  uint64_t dropToFloors();                       // critical pressure: every pool to its floor

  // The accounts, reservations, owed bytes and claims.
  void fill(MemoryGovernorSnapshot &snapshot, uint64_t chargedBytes) const;

private:
  struct Account;
  struct Choice;
  struct DryRun;

  // Other accounts' pending growth, and for a Growth claim the lanes'
  // projected growth; the needs of waiting claims ranked above `rank`.
  [[nodiscard]] uint64_t heldForGrowth(std::optional<AccountId> growing, ClaimClass cls) const noexcept;
  [[nodiscard]] uint64_t heldByClaims(const ClaimRank &rank, std::optional<uint32_t> self) const noexcept;
  [[nodiscard]] uint64_t pendingGrowthBytes() const noexcept;
  // The units of the pool `account` claims take per second: its measured
  // rate, or, if more, that of the claims in demand that took its units,
  // each its units since it opened.
  [[nodiscard]] double takesPerSecond(size_t account) const noexcept;
  [[nodiscard]] DryRun dryRun(const ClaimSpec &spec, const VictimQuery &query, uint64_t lackingBytes,
                              uint32_t lackingPages, std::optional<AccountId> growing) const;
  // `whole`: false when a unit the run chose was no longer offered as
  // chosen, where the take stopped.
  [[nodiscard]] FreeResult take(const DryRun &run, const VictimQuery &query, ClaimState *claim, bool &whole);
  // Takes each source's units that cost nothing and are sized only when
  // taken, until one frees nothing; the bytes they freed.
  uint64_t takeUnsized(const VictimQuery &query);
  // `credited`: the host admits the claim only on what free() leaves now (a
  // pool's growth without its consent).
  [[nodiscard]] FreeResult freeFor(uint32_t claim, const Need &need, uint64_t freeBytes, bool critical,
                                   bool deficit, std::optional<AccountId> growing, bool credited);
  void log(const ClaimSpec &spec, const Need &need, const DryRun *run, const FreeResult &result) const;

  std::vector<Account> accounts_;
  std::vector<VictimSource *> sources_;
  std::map<uint32_t, ClaimState> claims_;
  uint32_t nextClaim_ = 1;
  std::function<Returning()> returning_;
  std::function<Returning(std::span<const uint64_t>)> returningFor_;
  std::function<uint64_t(uint32_t)> bufferBytes_;
  std::array<uint64_t, 3> pageBytes_{};
  MemoryRates rates_;
  uint64_t laneDemandBytes_ = 0;
  std::optional<RequestPriority> lanePriority_;
  std::function<void(std::string_view)> log_;
  double nowSeconds_ = 0.0;
  std::optional<double> uncopiedReachedSeconds_;
};

} // namespace splash::engine
