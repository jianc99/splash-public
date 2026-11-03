#pragma once

// A pool's regrowth under amend/V.md A30 (owner 11-03 03:55) and the
// ledger's growth rule (amend/M.md R3), watched at each control pass of a
// serving loop: it does not oscillate, a unit it grew back taken back within
// kValueHorizonSeconds again within that of the last such take back
// (coordinator 11-03 09:20, as matrix.py's store line judges it: an isolated
// take back, by a claim of any class, the host's too, is counted, not failed);
// and it is never left short of its
// cap for more than two settled passes while its next unit would grow by the
// rule's own terms: its regrowth gains at the rate claims take its units
// (MemoryAccountSnapshot::regrowthSeconds; into free room also while no claim
// has taken any), and either the free room beyond the lanes' projected
// growth, the waiting claims and every pending growth holds the unit while
// no claim waits for memory or to retry what the host refused, or
// what it would displace beside that room costs less than its regrowth gains
// (MemoryLedger::available), with the host's consent. A store that never
// regrows into memory it could take (#645) fails the second; one held short
// by the memory its lanes grow into, by cache worth more than its regrowth,
// or by claims that keep taking its units, does not.

#include "engine/MemoryGovernor.hpp"
#include "engine/MemoryLedger.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace splash::test {

class RegrowthWatch final {
public:
  explicit RegrowthWatch(std::string pool) : pool_(std::move(pool)) {}

  // At `nowMilliseconds`, the pool at `units` units; `controlPasses` the
  // loop's count of control passes, of which only a new one between commands
  // is judged.
  void observe(engine::MemoryGovernor &governor, uint32_t units, double nowMilliseconds, uint64_t controlPasses,
               bool commandInFlight) {
    if (units_ && units > *units_) {
      grownAt_ = nowMilliseconds;
      ++grows;
      // A unit grew: the store is not stuck, whatever it lacks still.
      since_.reset();
      count_ = 0;
    } else if (units_ && units < *units_ && grownAt_ &&
               nowMilliseconds - *grownAt_ <= 1000.0 * engine::kValueHorizonSeconds) {
      ++retaken;
      if (takenBackAt_ && nowMilliseconds - *takenBackAt_ <= 1000.0 * engine::kValueHorizonSeconds)
        ++oscillations;
      takenBackAt_ = nowMilliseconds;
      history += " regrew at " + seconds(*grownAt_) + ", taken back at " + seconds(nowMilliseconds) + ";";
    }
    units_ = units;
    if (commandInFlight || controlPasses == passes_)
      return;
    passes_ = controlPasses;
    const std::optional<std::string> could = wouldGrow(governor);
    if (!could) {
      since_.reset();
      count_ = 0;
      return;
    }
    if (!since_)
      since_ = nowMilliseconds;
    if (++count_ > 2 && nowMilliseconds - *since_ >= 2.0 * engine::kHostSettleMilliseconds && stuck.empty())
      stuck = "the " + pool_ + " pool stayed short of its cap from " + seconds(*since_) + " to " +
              seconds(nowMilliseconds) + " while its next unit would grow: " + *could;
  }

  // Why the pool's next unit would grow now by the rule's terms, if it would.
  [[nodiscard]] std::optional<std::string> wouldGrow(engine::MemoryGovernor &governor) const {
    const engine::MemoryGovernorSnapshot memory = governor.snapshot();
    engine::MemoryLedger &ledger = governor.ledger();
    const engine::MemoryAccountSnapshot *pool = nullptr;
    for (const engine::MemoryAccountSnapshot &account : memory.accounts)
      if (account.name == pool_)
        pool = &account;
    if (!pool || pool->currentBytes >= pool->capBytes || !pool->unitBytes || pool->reservedBytes)
      return std::nullopt;
    const uint64_t unit = pool->unitBytes, lanes = ledger.laneDemandBytes();
    const uint64_t held = memory.reservedBytes + memory.claims.bytes + lanes;
    const uint64_t room = memory.headroomBytes > held ? memory.headroomBytes - held : 0;
    const bool consents = memory.hostGrowthAllowed &&
                          memory.hostHeadroomBytes >= unit + lanes + ledger.hostRefusedWaitingBytes() + engine::kHostMarginBytes;
    if (!consents)
      return std::nullopt;
    const std::string terms = "regrowth " + std::to_string(pool->regrowthSeconds) + " s at " +
                              std::to_string(pool->takesPerSecond) + " takes a second, " +
                              std::to_string(room >> 20) + " MiB of room beyond " + std::to_string(held >> 20) +
                              " MiB held, a unit of " + std::to_string(unit >> 20) + " MiB";
    if (room >= unit) {
      const bool waiting = memory.claims.waiting || ledger.hostRefusedWaitingBytes();
      return !waiting && (pool->takesPerSecond == 0.0 || pool->regrowthSeconds >= 0.0)
                 ? std::optional(terms + " (free room)")
                 : std::nullopt;
    }
    if (pool->regrowthSeconds <= 0.0)
      return std::nullopt;
    const engine::ClaimSpec growth{.cls = engine::ClaimClass::Growth,
                                   .timing = engine::NeedTiming::CanWait,
                                   .holdSeconds = engine::kValueHorizonSeconds,
                                   .benefitSeconds = pool->regrowthSeconds};
    if (ledger.available(growth, unit, memory.headroomBytes, false, false) < unit)
      return std::nullopt;
    return terms + " (what it displaces costs less than its regrowth gains)";
  }

  // Units grown back, those taken back within the horizon of the last, and
  // those taken back so within the horizon of the take back before.
  uint32_t grows = 0, retaken = 0, oscillations = 0;
  std::string history, stuck;

private:
  static std::string seconds(double milliseconds) { return std::to_string(milliseconds / 1000.0) + " s"; }

  std::string pool_;
  std::optional<uint32_t> units_;
  std::optional<double> grownAt_, takenBackAt_, since_;
  uint64_t passes_ = 0;
  uint32_t count_ = 0;
};

} // namespace splash::test
