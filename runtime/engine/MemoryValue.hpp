#pragma once

#include "engine/MemoryLedger.hpp"

#include <cstdint>

// What memory is worth, in seconds of engine work: the prices the ledger
// compares units by, from measured inputs.
namespace splash::engine::value {

// A pool's measured rates: an EWMA (kRateTimeConstantSeconds) of
// d(ghostHits)/dt and d(hits)/dt over the control clock.
class PoolRate final {
public:
  void observe(const model::PoolValue &value, double nowSeconds) noexcept;
  [[nodiscard]] double ghostHitsPerSecond() const noexcept;
  [[nodiscard]] double hitsPerSecond() const noexcept;
};
// A pool's unit of `unitBytes`, kept away for `holdSeconds`.
[[nodiscard]] UnitCost poolUnit(const PoolRate &rate, const model::PoolValue &value, uint64_t unitBytes,
                                double holdSeconds) noexcept;
// Giving a unit of the pool back, at its measured release time per byte
// (releaseMilliseconds over releasedBytes), and reading it back once it comes
// back: the cost of growth that displaces other memory beyond what it takes.
[[nodiscard]] double cycleSeconds(const model::PoolValue &value, uint64_t unitBytes,
                                  const MemoryRates &rates) noexcept;
// What a pool's next unit saves over kValueHorizonSeconds.
[[nodiscard]] double growthBenefit(const PoolRate &rate, const model::PoolValue &value, uint64_t unitBytes) noexcept;
// A cache unit: its tokens computed again, or its copy read back, as often
// as it is reused.
[[nodiscard]] UnitCost cached(uint64_t tokens, uint64_t bytes, bool copied, bool certainReuse,
                              const MemoryRates &rates) noexcept;
// A Resident lane's conversion to Spill: its Full pages written, and the
// slower prefill of its spilled history.
[[nodiscard]] UnitCost conversion(uint64_t heldFullBytes, double spillPenaltySeconds,
                                  const MemoryRates &rates) noexcept;

} // namespace splash::engine::value
