#include "model/SpilledKvReads.hpp"

#include <stdexcept>

namespace splash::model {

// The null one's, until the runtime and target integration builds active
// ones over the hot block cache and the history stream.

bool SpilledKvReads::active() const noexcept { return false; }

ops::AttentionBlocks SpilledKvReads::verify(metal::CommandGraph &, uint32_t, const ops::AttentionBlocks &) {
  throw std::logic_error("a command whose lanes hold no Index page resolves no block");
}

PrefillKv SpilledKvReads::prefill(metal::CommandGraph &, uint32_t, uint32_t, uint32_t, uint32_t,
                                  const PrefillKv &pool) {
  return pool;
}

void SpilledKvReads::layerEnd(metal::CommandGraph &, uint32_t) {}

} // namespace splash::model
