#pragma once

#include "metal/MetalBackend.hpp"

#include <compare>
#include <cstdint>
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

class Projection final {
public:
  Projection() = default;
  Projection(metal::MetalBuffer weights, metal::MetalBuffer scales,
             metal::MetalBuffer biases, uint32_t output, uint32_t input)
      : outputSize(output), inputSize(input),
        storage_(AffineWeights{std::move(weights), std::move(scales), std::move(biases)}) {}
  Projection(uint32_t output, uint32_t input, BlockWeights weights)
      : outputSize(output), inputSize(input), storage_(std::move(weights)) {}

  [[nodiscard]] WeightLayout layout() const noexcept {
    return std::holds_alternative<AffineWeights>(storage_) ? WeightLayout::Affine64
         : WeightLayout::Block32;
  }
  [[nodiscard]] ProjectionShape shape() const noexcept {
    return {outputSize, inputSize, layout()};
  }
  [[nodiscard]] AffineWeights &affine() { return std::get<AffineWeights>(storage_); }
  [[nodiscard]] const AffineWeights &affine() const { return std::get<AffineWeights>(storage_); }
  [[nodiscard]] const std::vector<QuantizedSegment> &segments() const {
    return std::get<BlockWeights>(storage_).segments;
  }

  uint32_t outputSize = 0;
  uint32_t inputSize = 0;

private:
  std::variant<AffineWeights, BlockWeights> storage_;
};

// Gather consumes native rows. This is intentionally a separate type: neither
// affine row tables nor GGUF blocks may be bound as a prepared projection.
class EmbeddingWeights final {
public:
  EmbeddingWeights() = default;
  explicit EmbeddingWeights(QuantizedSegment rows)
      : outputSize(rows.outputSize), inputSize(rows.inputSize), storage_(std::move(rows)) {}
  [[nodiscard]] bool isAffine() const noexcept { return std::holds_alternative<AffineWeights>(storage_); }
  [[nodiscard]] AffineWeights &affine() { return std::get<AffineWeights>(storage_); }
  [[nodiscard]] const AffineWeights &affine() const { return std::get<AffineWeights>(storage_); }
  [[nodiscard]] const QuantizedSegment &nativeRows() const { return std::get<QuantizedSegment>(storage_); }
  uint32_t outputSize = 0;
  uint32_t inputSize = 0;
private:
  std::variant<AffineWeights, QuantizedSegment> storage_;
};

} // namespace splash::ops
