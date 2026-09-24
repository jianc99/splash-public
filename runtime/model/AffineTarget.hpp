#pragma once

#include "model/PreparedWeights.hpp"
#include "model/WeightStore.hpp"

namespace splash::model {

struct Qwen3_8Layout;
struct Qwen3_6MoeLayout;

// Native MLX affine source -> the existing packed target ABI. Both this adapter
// and the block-quantized adapter publish through PreparedWeights and serve
// through WeightFile; neither changes inference kernels. Before the first
// image is written, the disk check budgets every missing image together with
// alsoPrepared, the model's other prepared files.
class AffineTargetLoader final {
public:
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_8Layout &layout, PreparationCheck check = {},
                     std::span<const PreparedWeight> alsoPrepared = {});
  AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                     const Qwen3_6MoeLayout &layout, PreparationCheck check = {},
                     std::span<const PreparedWeight> alsoPrepared = {});
  ~AffineTargetLoader();
  [[nodiscard]] WeightFile layer(uint32_t index, bool fullAttention);
  [[nodiscard]] WeightFile head(uint32_t layers);
  [[nodiscard]] WeightFile embedding(uint32_t vocabulary, uint32_t hidden);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] uint64_t preparedAffineBytes(const Qwen3_8Layout &layout);
[[nodiscard]] uint64_t preparedAffineBytes(const Qwen3_6MoeLayout &layout);

} // namespace splash::model
