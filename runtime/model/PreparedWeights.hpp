#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace splash::model {

// Shared by source adapters. Preparation owns bounded buffers; inference only
// opens completed, immutable artifacts through WeightFile. The admission reserve
// includes source metadata and small staging buffers; it is not a model copy.
inline constexpr uint64_t kWeightPreparationWorkspaceBytes = 64 * 1024 * 1024;

enum class TargetSource : uint8_t { Packed, Affine, Gguf };

using PreparationCheck = std::function<void()>;

void readWeightBytes(int descriptor, uint64_t offset, std::span<uint8_t> bytes);
void writeWeightBytes(int descriptor, uint64_t offset, std::span<const uint8_t> bytes);
[[nodiscard]] std::string weightDigest(std::span<const uint8_t> bytes);
[[nodiscard]] std::string weightFileDigest(int descriptor, const PreparationCheck &check = {});

class WeightSource final {
public:
  explicit WeightSource(const std::filesystem::path &path, const PreparationCheck &check = {});
  ~WeightSource();
  WeightSource(const WeightSource &) = delete;
  WeightSource &operator=(const WeightSource &) = delete;
  [[nodiscard]] int descriptor() const noexcept;
  [[nodiscard]] const std::string &digest() const noexcept;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PreparedWeights final {
public:
  explicit PreparedWeights(std::filesystem::path root = {});
  // key is a SHA-256 of source content, layout ABI and transformation parameters.
  // The writer receives an empty, pre-sized file. Errors never publish it.
  [[nodiscard]] std::filesystem::path prepare(
      std::string_view key, uint64_t bytes,
      const std::function<void(int)> &write,
      const PreparationCheck &check = {}) const;

private:
  std::filesystem::path root_;
};

} // namespace splash::model
