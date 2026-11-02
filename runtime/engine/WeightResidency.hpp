#pragma once

#include "metal/MetalBackend.hpp"
#include "model/WeightMemory.hpp"

#include <cstdint>
#include <limits>
#include <optional>

namespace splash::engine {

struct WeightsSnapshot;

// What the engine gives back while idle and takes again before it runs the
// next request (EngineConfig::weights).
struct WeightResidencyConfig final {
  // The weights and what goes with them (ReleasableMemory).
  model::WeightMemory *weights = nullptr;
  // Admits the memory they take back (MemoryGovernor::allocationAdmission).
  metal::AllocationAdmission admit;
  // --idle-release: infinite with `off`, when only a release request frees
  // them.
  double idleReleaseMilliseconds = std::numeric_limits<double>::infinity();
};

// The weights' release and restore, on the engine's clock.
class WeightResidency final {
public:
  // Throws std::invalid_argument without the weights or their admission.
  explicit WeightResidency(WeightResidencyConfig config);
  // Neither released nor being restored.
  [[nodiscard]] bool held() const noexcept;
  struct Step final {
    enum class Kind : uint8_t { Progress, Restored, Refused } kind;
    metal::AllocationFailure failure = metal::AllocationFailure::None;
  };
  // A request waits: takes back the next part (WeightMemory::restore); the
  // first call starts the restore.
  [[nodiscard]] Step restore(double nowMilliseconds);
  // No request waits any longer: gives back what came back, and counts a
  // failure if the restore was refused.
  void abandon(double nowMilliseconds);
  // engineIdle: Engine::idle(). idleOnly: the idle release, after
  // idleReleaseMilliseconds without a request, else a release request. True
  // when it released.
  [[nodiscard]] bool release(double nowMilliseconds, bool engineIdle, bool idleOnly);
  // The idle period begins: at Ready, when the engine's last request left,
  // and when the weights were restored.
  void idleFrom(double nowMilliseconds) noexcept;
  [[nodiscard]] WeightsSnapshot snapshot() const noexcept;

private:
  WeightResidencyConfig config_;
  double idleSince_ = 0.0;
  std::optional<double> restoreStarted_, refusedSince_;
  uint64_t restores_ = 0, failures_ = 0;
};

} // namespace splash::engine
