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
  uint64_t offset = 0;     // in the file
  uint64_t bytes = 0;
  uint64_t dataOffset = 0; // where the file's tensor data starts
  void read(uint64_t at, std::span<uint8_t> destination) const;
  // Records the tensor as an input of a prepared file: its bytes in its
  // file's tensor data, dtype and shape.
  void identify(WeightIdentity &identity) const;
};

// Checkpoint configuration and safetensors index. Opening parses only
// metadata; tensor data is hashed when a prepared file's identity first
// needs it and read in bounded slices, without loading the MLX runtime or
// allocating tensors.
class SafetensorsCheckpoint final {
public:
  explicit SafetensorsCheckpoint(const std::filesystem::path &directory,
                        const PreparationCheck &check = {});
  ~SafetensorsCheckpoint();
  [[nodiscard]] const SourceTensor *find(std::string_view name) const noexcept;
  [[nodiscard]] const SourceTensor &require(std::string_view name) const;
  void requireQuantization(std::string_view projection, uint32_t bits) const;
  void requireConfigNumber(std::string_view key, double expected) const;
  void requireConfigString(std::string_view key, std::string_view expected) const;
  void requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
