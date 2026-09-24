#pragma once

// The out-projection input tables a producer can write for its consumer,
// prepared by the consumer's own kernel: the reference the Metal tests hold
// the producers that write the table themselves to, byte for byte.

#include "metal/CommandGraph.hpp"
#include "ops/Linear.hpp"

#include <cstdint>
#include <stdexcept>

namespace splash::test {

// Prepares `lanes` verify blocks of the plain bf16 rows in `input`, `width`
// wide, as the `layout` table and sums, with the dispatch the projections
// issue when no producer wrote the table.
inline void addReferencePreparation(metal::CommandGraph &graph, ops::LinearInput layout,
                                    const metal::MetalBuffer &input,
                                    const metal::MetalBuffer &table,
                                    const metal::MetalBuffer &sums, uint32_t width,
                                    uint32_t lanes) {
  if (layout == ops::LinearInput::Plain)
    throw std::invalid_argument("a plain input has no table to prepare");
  graph.add(layout == ops::LinearInput::Table16 ? "decode_linear_gguf_prepare"
                                                : "decode_linear_q4_prepare",
            {input, table, sums}, width, {width / 32, lanes, 1}, {128, 1, 1});
}

} // namespace splash::test
