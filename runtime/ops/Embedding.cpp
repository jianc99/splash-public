#include "ops/Embedding.hpp"

#include "metal/abi/Embedding.h"
#include "metal/abi/Gguf.h"

#include <stdexcept>
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
  if (!table.isAffine()) {
    const GgufEmbedParams params{rows, table.outputSize, table.inputSize};
    const uint32_t type = table.nativeRows().type;
    const char *kernel = type == 12 ? "gguf_embed_q4k"
                         : type == 14 ? "gguf_embed_q6k"
                         : type == 8 ? "gguf_embed_q80" : nullptr;
    if (!kernel) throw std::invalid_argument("unsupported GGUF embedding type");
    graph.add(kernel, {std::move(tokens), table.nativeRows().plane0, std::move(output)},
              params, {(rows * table.inputSize + 255) / 256, 1, 1}, {256, 1, 1});
    return;
  }
  const uint32_t hiddenGroups = (table.inputSize + 127) / 128;
  const Q4EmbeddingParams params{rows, table.outputSize};
  graph.add(embeddingPipeline(table.inputSize),
            {std::move(tokens), table.affine().weights, table.affine().scales, table.affine().biases,
             std::move(output)},
            params, {hiddenGroups, 1, 1});
}

} // namespace splash::ops
