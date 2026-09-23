#pragma once

#include "model/PreparedWeights.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace splash::model {

struct SourceTensor final {
  const WeightSource *file = nullptr;
  std::string dtype;
  std::vector<uint64_t> shape;
  uint64_t offset = 0;
  uint64_t bytes = 0;
  void read(uint64_t at, std::span<uint8_t> destination) const;
};

// MLX affine checkpoint configuration and safetensors index. Tensor data is
// read in bounded slices without loading the MLX runtime or allocating tensors.
class AffineCheckpoint final {
public:
  explicit AffineCheckpoint(const std::filesystem::path &directory,
                        const PreparationCheck &check = {});
  ~AffineCheckpoint();
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  void requireConfigString(std::string_view key, std::string_view expected) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  [[nodiscard]] const std::string &digest() const noexcept;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
