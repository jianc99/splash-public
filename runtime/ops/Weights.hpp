#pragma once

#include "metal/MetalBackend.hpp"

#include <compare>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace splash::ops {

// Physical layout, independent of the checkpoint container and compute tile.
enum class WeightLayout : uint8_t { Affine64, Block32 };

struct ProjectionShape final {
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  WeightLayout layout = WeightLayout::Affine64;
  auto operator<=>(const ProjectionShape &) const = default;
};

struct AffineWeights final {
  metal::MetalBuffer weights;
  metal::MetalBuffer scales;
  metal::MetalBuffer biases;
};

// A prepared block-quantized or F32 tensor occupying a projection's output
// columns [columnOffset, columnOffset + outputSize). The packing ABI is in
// metal/abi/QuantFormat.h; the source container is not part of this view.
struct QuantizedSegment final {
  metal::MetalBuffer plane0;
  metal::MetalBuffer plane1;
  metal::MetalBuffer meta;
  uint32_t type = 0;
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
  uint32_t p0 = 0;
  uint32_t p1 = 0;
  uint32_t metaBytes = 0;
  uint32_t metaGroups = 0;
  uint32_t columnOffset = 0;
  uint32_t formatId = 0;    // GGUF_FMT_* (metal/abi/QuantFormat.h)
  const char *format = "";
  // A float tensor the GGUF keeps unquantized (F32, as llama.cpp keeps the
  // MoE router): plane0 holds its [outputSize][inputSize] floats, multiplied
  // unrounded in fp32 (kernels/shared/gguf_float.metal).
  [[nodiscard]] bool isFloat() const noexcept;
};

struct BlockWeights final {
  std::vector<QuantizedSegment> segments;
};

// Immutable weights in one of the two layouts: Affine holds the Affine64
// form, Block the Block32 form, and layout() names the one held. Projections,
// token tables and MoE blocks share this pattern; the accessor of the layout
// not held throws std::bad_variant_access. A default value holds empty affine
// weights, which a reader replaces.
template <class Affine, class Block>
class LayoutWeights {
public:
  LayoutWeights() = default;
  LayoutWeights(Affine weights) : storage_(std::move(weights)) {}
  LayoutWeights(Block weights) : storage_(std::move(weights)) {}

  [[nodiscard]] WeightLayout layout() const noexcept {
    return std::visit([](const auto &weights) { return layoutOf(weights); }, storage_);
  }
  [[nodiscard]] const Affine &affine() const { return std::get<Affine>(storage_); }
  [[nodiscard]] const Block &blocks() const { return std::get<Block>(storage_); }

private:
  // One layout per alternative: an alternative without one does not compile.
  static constexpr WeightLayout layoutOf(const Affine &) noexcept { return WeightLayout::Affine64; }
  static constexpr WeightLayout layoutOf(const Block &) noexcept { return WeightLayout::Block32; }

  std::variant<Affine, Block> storage_;
};

// A projection of outputSize x inputSize. A block projection holds at least
// one segment; its segments tile the leading output columns (LinearGguf.cpp).
class Projection final : public LayoutWeights<AffineWeights, BlockWeights> {
public:
  Projection() = default;
  Projection(uint32_t output, uint32_t input, AffineWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {}
  Projection(uint32_t output, uint32_t input, BlockWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {
    if (blocks().segments.empty()) throw std::invalid_argument("block projection has no segments");
  }

  [[nodiscard]] ProjectionShape shape() const noexcept {
    return {outputSize, inputSize, layout()};
  }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

// A token table of outputSize rows of inputSize values, which Embedding
// gathers: affine Q4 rows, or one segment of native GGUF rows (block_q4_K,
// block_q6_K or block_q8_0). It is intentionally a separate type: no table
// may be bound as a projection.
class EmbeddingWeights final : public LayoutWeights<AffineWeights, QuantizedSegment> {
public:
  EmbeddingWeights() = default;
  EmbeddingWeights(uint32_t output, uint32_t input, AffineWeights weights)
      : LayoutWeights(std::move(weights)), outputSize(output), inputSize(input) {}
  EmbeddingWeights(uint32_t output, uint32_t input, QuantizedSegment rows)
      : LayoutWeights(std::move(rows)), outputSize(output), inputSize(input) {
    if (blocks().outputSize != output || blocks().inputSize != input)
      throw std::invalid_argument("native embedding rows do not match the table");
  }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
};

} // namespace splash::ops
