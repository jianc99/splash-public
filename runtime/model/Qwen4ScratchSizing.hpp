#pragma once

#include "metal/abi/BlockIndexer.h"

#include <cstdint>

namespace splash::model {

// A verify can compute seven scratch tokens past the logical window. The
// select writes through its last complete block plus the bitmap tile slack,
// inclusive. Round that physical bound to bitmap words; the score stride
// covers the same blocks. Logical request admission still uses the window.
constexpr uint32_t qwen4IndexerMaskWords(uint32_t contextTokens) {
  const uint32_t physicalEnd = contextTokens + SPLASH_SPECULATIVE_SCRATCH_TOKENS;
  return (physicalEnd / BLOCK_INDEXER_BLOCK + BLOCK_INDEXER_TILE_SLACK + 32) / 32;
}

} // namespace splash::model
