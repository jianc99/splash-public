#pragma once

#include "engine/ReclaimClass.hpp"
#include "model/ElasticPool.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <compare>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace splash::engine {

struct MemoryGovernorSnapshot;

// The memory layer's policy constants. The working-set margin and the host's
// reserves are not among them: they set the limit and gate the start
// (EngineMemoryPolicy).
// What elective allocation leaves free above the host's spare reserve.
inline constexpr uint64_t kHostMarginBytes = 1ULL << 30;
// How long the host's counters take to show what a pass returned.
inline constexpr double kHostSettleMilliseconds = 1000.0;
// The payback horizon of growth that displaces other memory, and how long
// memory a host pass returns stays returned.
inline constexpr double kValueHorizonSeconds = 60.0;
// The time constant of the measured rates: pools' ghost hits, the cache's
// reuse, the KV tier's throughput.
inline constexpr double kRateTimeConstantSeconds = 30.0;

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
  double prefillSecondsPerRow = 0.0;     // Scheduler::prefillMilliseconds of a 2048-row command, per row
  double commandInterceptSeconds = 0.0;  // Scheduler::prefillMilliseconds of a one-row command
  // Priors, each replaced by its first measurement.
  double tierReadBytesPerSecond = 4.0e9; // KvTier::throughput
  double tierWriteBytesPerSecond = 2.0e9;
  double reuse = 0.25;                   // the probes' hit share (EngineSnapshot::cacheHits, coldMisses)
};

// One unit a source would give up, in the source's own order.
struct Victim final {
  uint64_t key = 0;             // the source's handle (block, state block, request id, pool unit)
  uint64_t bytes = 0;           // device bytes it frees (an extent it empties, a chunk, buffers)
  uint32_t pages = 0;           // free KV pages of `form` it leaves in kept extents (in-place needs)
  kv::PageForm form = kv::PageForm::Full;
  uint32_t stateBuffers = 0;    // pooled state buffers it frees (in-place needs)
  bool later = false;           // frees on a landing (a demotion, a state write, a conversion's copies)
  bool certainReuse = false;    // in use, the newest resume point, a chain a queued request restores
  bool sized = true;            // false: a unit that costs nothing and whose size is known only when taken
  ReclaimClass cls = ReclaimClass::Ordinary;
  UnitCost cost;
};
struct VictimQuery final {
  const ClaimSpec *claim = nullptr;
  Need need;
  bool critical = false;        // macOS critical: InUse and certain-reuse units for Host claims too
  const MemoryRates *rates = nullptr;
};

// A source of the units free() takes, on the engine thread between commands.
// peek changes nothing, and may understate what later units free but never
// overstate it: free() takes nothing unless what it peeked covers the need.
// take takes exactly the unit the latest peek returned.
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

// The books and the one rule. No Metal, no host: the governor supplies the
// charged bytes and the host's decisions.
class MemoryLedger final {
public:
  // Registration (startup; stable for the process). An account's current
  // bytes are pulled from its owner.
  AccountId addAccount(std::string name, AccountRole role, std::function<uint64_t()> currentBytes);
  AccountId addPool(model::ElasticPool &pool);   // account + victim source; name = PoolState::name
  void addSource(VictimSource &source);          // the cache, the engine's lane victims
  void setReturning(std::function<Returning()> returning);
  void setRates(const MemoryRates &rates) noexcept;
  void observePools(double nowSeconds);          // differentiates each pool's PoolValue over the control clock
  void setLaneDemand(uint64_t bytes) noexcept;   // the lanes' projected growth

  // Claims.
  [[nodiscard]] ClaimId open(const ClaimSpec &spec);
  void update(ClaimId claim, const ClaimSpec &spec);
  void refused(ClaimId claim, const Need &need);
  void setWaiting(ClaimId claim, bool waiting);
  void close(ClaimId claim) noexcept;
  // What admission holds against an allocation in `claim`'s scope: other
  // accounts' pending growth, waiting claims ranked above it, and for a
  // Growth claim the lanes' projected growth.
  [[nodiscard]] uint64_t heldBytes(ClaimId claim, std::optional<AccountId> growing) const;
  [[nodiscard]] std::array<uint32_t, 3> heldPages(ClaimId claim) const;

  // The rule: what is taken for a claim, what is there for one, what grows.
  [[nodiscard]] FreeResult free(ClaimId claim, const Need &need, uint64_t freeBytes, bool critical);
  [[nodiscard]] uint64_t available(const ClaimSpec &spec, uint64_t enough, uint64_t freeBytes) const;
  // One growth pass; `admitGrowth` is the governor's admission inside the
  // pool's growth scope. True when a pool grew or started freeing for its
  // pending growth.
  bool grow(uint64_t freeBytes,
            const std::function<metal::AllocationResult(AccountId, model::ElasticPool &)> &admitGrowth);
  uint64_t dropToFloors();                       // critical pressure: every pool to its floor

  // The accounts, reservations, owed bytes and claims.
  void fill(MemoryGovernorSnapshot &snapshot, uint64_t chargedBytes) const;
};

} // namespace splash::engine
