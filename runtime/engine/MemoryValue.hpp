#pragma once

#include "engine/MemoryLedger.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

// What memory is worth, in seconds of engine work: the prices the ledger
// compares units by, from measured inputs.
namespace splash::engine::value {

// A pool's measured rates: EWMAs (kRateTimeConstantSeconds) of its ghost
// hits, hits and scan reads per second, of the bytes its ghosts span, and of
// the units claims take from it per second. Only intervals in which the
// engine ran a command update them (an idle interval says nothing of
// demand); the first such interval sets them. They outlive the pool's own
// resets: an idle release and restore clear its ghosts but not these.
class PoolRate final {
public:
  // `taken`, the units claims took from it, is monotonic like the pool's
  // counters.
  void observe(const model::PoolValue &value, uint64_t taken, double nowSeconds, bool commandsRan) noexcept {
    const Counters now{value.hits, value.ghostHits, value.scanReadBytes, taken};
    if (last_ && commandsRan && nowSeconds > lastSeconds_) {
      const double seconds = nowSeconds - lastSeconds_;
      const double weight = rates_ ? 1.0 - std::exp(-seconds / kRateTimeConstantSeconds) : 1.0;
      const auto mix = [&](double &rate, uint64_t after, uint64_t before) {
        rate += weight * (double(after - before) / seconds - rate);
      };
      Rates rates = rates_.value_or(Rates{});
      mix(rates.hits, now.hits, last_->hits);
      mix(rates.ghostHits, now.ghostHits, last_->ghostHits);
      mix(rates.scanReadBytes, now.scanReadBytes, last_->scanReadBytes);
      mix(rates.takes, now.taken, last_->taken);
      rates.ghostBytes += weight * (double(value.ghostBytes) - rates.ghostBytes);
      rates_ = rates;
    }
    last_ = now;
    lastSeconds_ = nowSeconds;
  }
  [[nodiscard]] double hitsPerSecond() const noexcept { return rates_ ? rates_->hits : 0.0; }
  [[nodiscard]] double ghostHitsPerSecond() const noexcept { return rates_ ? rates_->ghostHits : 0.0; }
  [[nodiscard]] double ghostBytes() const noexcept { return rates_ ? rates_->ghostBytes : 0.0; }
  [[nodiscard]] double scanReadBytesPerSecond() const noexcept { return rates_ ? rates_->scanReadBytes : 0.0; }
  [[nodiscard]] double takesPerSecond() const noexcept { return rates_ ? rates_->takes : 0.0; }

private:
  struct Counters final {
    uint64_t hits = 0, ghostHits = 0, scanReadBytes = 0, taken = 0;
  };
  struct Rates final {
    double hits = 0.0, ghostHits = 0.0, scanReadBytes = 0.0, takes = 0.0, ghostBytes = 0.0;
  };
  std::optional<Counters> last_;
  double lastSeconds_ = 0.0;
  std::optional<Rates> rates_;
};

// A pool's unit of `unitBytes`, kept away for `holdSeconds`. Decode: a ghost
// hit says a pool larger by the ghosts' span would have served it, so the
// unit carries its share of that span of the ghost rate, capped at the span
// measured. Prefill: scans leave no ghosts; the scan reads spread over the
// keys not resident (`missingBytes`, uniform: the measured marginal hit slope
// is 0.67-1.05 per unit of resident share, amend/S.md S2), each byte priced at
// the scheduler's weight-read price. Without ghosts and scans a unit costs at
// most one miss, if the pool is used within the hold.
[[nodiscard]] inline UnitCost poolUnit(const PoolRate &rate, const model::PoolValue &value, uint64_t unitBytes,
                                       uint64_t missingBytes, double holdSeconds, const MemoryRates &rates) noexcept {
  const double missSeconds = value.missMilliseconds / 1000.0;
  const double span = rate.ghostBytes();
  const double unit = double(unitBytes);
  UnitCost cost;
  if (span > 0.0)
    cost.ratePerSecond = rate.ghostHitsPerSecond() * missSeconds * std::min(unit, span) / span;
  else
    cost.onceSeconds = missSeconds * std::min(1.0, rate.hitsPerSecond() * holdSeconds);
  if (missingBytes)
    cost.ratePerSecond += rate.scanReadBytesPerSecond() * rates.weightReadSecondsPerByte * unit / double(missingBytes);
  return cost;
}
// What growth that displaces other memory costs beyond what it takes: the
// unit read back key by key once it comes back (PoolValue::keyBytes a miss;
// none priced for a pool that reports no key), and given back again at the
// pool's measured release time per byte (the store's compactions).
[[nodiscard]] inline double cycleSeconds(const model::PoolValue &value, uint64_t unitBytes) noexcept {
  const double unit = double(unitBytes);
  const double refill = value.keyBytes ? unit / double(value.keyBytes) * value.missMilliseconds / 1000.0 : 0.0;
  const double release =
      value.releasedBytes ? unit * value.releaseMilliseconds / double(value.releasedBytes) / 1000.0 : 0.0;
  return refill + release;
}
// What a pool's next unit saves over kValueHorizonSeconds.
[[nodiscard]] inline double growthBenefit(const PoolRate &rate, const model::PoolValue &value, uint64_t unitBytes,
                                          uint64_t missingBytes, const MemoryRates &rates) noexcept {
  return poolUnit(rate, value, unitBytes, missingBytes, kValueHorizonSeconds, rates).at(kValueHorizonSeconds);
}
// Whether a unit grown into free room pays for being taken back within the
// horizon: claims that take the pool's units at the measured rate take one
// within it with probability 1 - exp(-rate * horizon) (Poisson), each time
// costing its cycle. A pool no claim has taken from grows unconditionally.
[[nodiscard]] inline bool growsIntoFreeRoom(const PoolRate &rate, double benefitSeconds, double cycle) noexcept {
  const double takes = rate.takesPerSecond();
  return takes == 0.0 || benefitSeconds >= cycle * (1.0 - std::exp(-takes * kValueHorizonSeconds));
}
// A cache unit: its tokens computed again, or its copy read back, as often
// as it is reused.
[[nodiscard]] inline UnitCost cached(uint64_t tokens, uint64_t bytes, bool copied, bool certainReuse,
                                     const MemoryRates &rates) noexcept {
  const double rebuild = copied && rates.tierReadBytesPerSecond > 0.0 ? double(bytes) / rates.tierReadBytesPerSecond
                                                                      : double(tokens) * rates.prefillSecondsPerRow;
  return {.onceSeconds = (certainReuse ? 1.0 : rates.reuse) * rebuild};
}
// A Spill lane's decode beside a Resident one: each decode command reads its
// hot blocks once per lane with Index pages, and a lane commits at least one
// token per command (DESIGN section 0 Q5).
[[nodiscard]] inline double spillDecodeSeconds(uint64_t projectedOutputTokens, const MemoryRates &rates) noexcept {
  return double(projectedOutputTokens) / std::max(1.0, rates.decodeTokensPerLaneCommand) *
         rates.spilledReadMillisecondsPerLaneCommand / 1000.0;
}
// A Resident lane's conversion to Spill: its spill penalty, once (the
// copies it writes cost no engine time: a claim waits for them).
[[nodiscard]] inline UnitCost conversion(double spillPenaltySeconds) noexcept {
  return {.onceSeconds = spillPenaltySeconds};
}

} // namespace splash::engine::value
