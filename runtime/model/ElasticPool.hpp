#pragma once

#include "metal/MetalBackend.hpp"

#include <cstdint>
#include <string_view>

namespace splash::model {

struct PoolState final {
  std::string_view name;          // a literal, for /status and logs; nothing branches on it
  uint64_t floorBytes = 0;        // never released below now: lanes' floors, a held staging, the store's floor,
                                  // entries in use
  uint64_t currentBytes = 0;      // allocated now
  uint64_t capBytes = 0;          // the bound grow() approaches; == currentBytes for a pool that grows only as it
                                  // is used
  uint64_t releaseUnitBytes = 0;  // what release(1) frees now; 0 at the floor
  uint64_t growUnitBytes = 0;     // what grow() takes next; 0 when the engine cannot grow it now (at its cap,
                                  // released, a pool that grows only as it is used)
};

// The pool's own measurements, monotonic since it was built; the consumer differentiates them over its own
// clock. Inputs to the prices of the pool's units (engine/MemoryValue.hpp), which this header does not define.
struct PoolValue final {
  uint64_t hits = 0;              // uses the pool served from memory
  uint64_t ghostHits = 0;         // misses a pool larger by ghostBytes would have served (ARC ghosts);
                                  // 0 without ghosts
  uint64_t ghostBytes = 0;        // bytes the ghosts span now (ghost keys * bytes per key); 0 without ghosts
  double missMilliseconds = 0.0;  // measured mean cost of one miss: an SSD read, or a rebuild (an image encode);
                                  // 0 when a miss costs only an allocation
  uint64_t releasedBytes = 0;     // bytes release() gave back
  double releaseMilliseconds = 0.0;  // engine-thread time release() took to give them back (the store's
                                     // compactions)
};

// A memory of the model's that follows the engine's memory. The model owns the mechanism; the engine alone
// decides when a unit goes or comes.
// Threads: state() and value() on the engine thread at any time, a command in flight included: they read
// only what the mechanism publishes and never wait behind a load. release() and grow() on the engine thread
// with no command in flight. A failure inside a mechanism throws to the engine's failure boundary.
class ElasticPool {
public:
  virtual ~ElasticPool() = default;
  [[nodiscard]] virtual PoolState state() const = 0;
  [[nodiscard]] virtual PoolValue value() const = 0;
  // Whole units, the fewest whose bytes reach `bytes`, never below the floor; the bytes given back, 0 at the
  // floor. 1 gives one unit; UINT64_MAX everything above the floor (the store in one compaction per size).
  [[nodiscard]] virtual uint64_t release(uint64_t bytes) = 0;
  // One unit under `admit`; a refusal changes nothing and is returned. Granted with nothing taken when
  // growUnitBytes is 0.
  [[nodiscard]] virtual metal::AllocationResult grow(const metal::AllocationAdmission &admit) = 0;
};

} // namespace splash::model
