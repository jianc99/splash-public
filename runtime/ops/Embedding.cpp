#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/Gguf.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

const char *embeddingPipeline(uint32_t hiddenSize) {
  switch (hiddenSize) {
  case 5120:
    return "embedding_q4_h5120";
  case 2048:
    return "embedding_q4_h2048";
  default:
    throw std::invalid_argument("unsupported compiled Q4 embedding shape");
  }
}

} // namespace

void Embedding::add(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    const EmbeddingWeights &table, metal::MetalBuffer output,
                    uint32_t rows) {
  if (!rows || !table.outputSize || !table.inputSize)
    throw std::invalid_argument("invalid Q4 embedding shape");
  // Both gathers read `rows` token ids and write `rows` bf16 rows of the table's width.
  if (tokens.sizeBytes() < uint64_t{rows} * sizeof(uint32_t) ||
      output.sizeBytes() < uint64_t{rows} * table.inputSize * sizeof(uint16_t))
    throw std::invalid_argument("embedding buffers are smaller than the gathered rows");
  if (table.layout() == WeightLayout::Block32) {
    const NativeRows &native = table.blocks();
    const GgufEmbedParams params{rows, table.outputSize, table.inputSize};
    graph.add(std::string("gguf_embed_") + native.name(),
              {std::move(tokens), native.rows, std::move(output)}, params,
              {(rows * table.inputSize + 255) / 256, 1, 1}, {256, 1, 1});
    return;
  }
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize};
  const AffineWeights &affine = table.affine();
  graph.add(embeddingPipeline(table.inputSize),
            {std::move(tokens), affine.weights, affine.scales, affine.biases, std::move(output)},
            params, {hiddenGroups, 1, 1});
}

} // namespace splash::ops
