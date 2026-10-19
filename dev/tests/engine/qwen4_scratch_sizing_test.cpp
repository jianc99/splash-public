#include "model/Qwen4ScratchSizing.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>

int main() {
  uint32_t failures = 0, cases = 0;
  const auto window = [&](uint32_t context) {
    const uint32_t capacity = splash::model::qwen4IndexerMaskWords(context) * 32;
    for (uint32_t remaining = 2; remaining <= 8; ++remaining) {
      const uint32_t first = context - remaining;
      const uint32_t complete = (first + SPLASH_TARGET_VERIFY_ROWS) / BLOCK_INDEXER_BLOCK;
      const uint32_t tail = (first + SPLASH_TARGET_VERIFY_ROWS - 1) / BLOCK_INDEXER_BLOCK;
      ++cases;
      if (capacity <= complete || capacity <= tail + BLOCK_INDEXER_TILE_SLACK) {
        ++failures;
        std::cerr << "window=" << context << " remaining=" << remaining << " first=" << first
                  << " physical_rows=" << SPLASH_TARGET_VERIFY_ROWS << " capacity=" << capacity
                  << " complete=" << complete << " last_bitmap_bit=" << tail + BLOCK_INDEXER_TILE_SLACK << '\n';
      }
    }
  };
  constexpr double factor = 1.9999771118164062;
  static_assert(uint32_t(262144 * factor) == 524282);
  window(uint32_t(262144 * factor));
  for (uint32_t residue = 0; residue < 128; ++residue) {
    const double fractionalFactor = double(524416 + residue) / 262144;
    window(uint32_t(262144 * fractionalFactor));
  }
  window(262144);
  window(1010000);
  for (const uint32_t context : {262144U, 1010000U}) {
    const uint32_t original = (context / BLOCK_INDEXER_BLOCK + BLOCK_INDEXER_TILE_SLACK + 31) / 32;
    if (splash::model::qwen4IndexerMaskWords(context) != original) ++failures;
  }
  std::cout << "QSA scratch sizing: " << cases << " cases, " << failures << " failures\n";
  return failures ? 1 : 0;
}
