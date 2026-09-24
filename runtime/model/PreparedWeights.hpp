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

struct PreparedWeight {
  std::string key;
  uint64_t bytes;
  std::string name{};
  std::string source{};
};

// Leave room for the OS and other applications; this is a disk reserve, not
// a promise that concurrent system activity can never exhaust the volume.
inline constexpr uint64_t kWeightCacheDiskReserve = uint64_t{2} << 30;
void requireWeightDiskSpace(uint64_t available, uint64_t required);

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
  [[nodiscard]] const std::filesystem::path &path() const noexcept;
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
  // Check the entire missing model before writing its first artifact. Completed
  // layers remain reusable after an interruption; they are not partial files.
  void requireSpace(std::span<const PreparedWeight> weights,
                    const PreparationCheck &check = {}) const;
  // key is a SHA-256 of source content, layout ABI and transformation parameters.
  // The writer receives an empty, preallocated file. Writer failures never
  // publish partial data. check applies to every load; prepareCheck admits
  // the additional conversion workspace only on a cache miss.
  [[nodiscard]] std::filesystem::path prepare(
      const PreparedWeight &weight,
      const std::function<void(int)> &write,
      const PreparationCheck &check = {},
      const PreparationCheck &prepareCheck = {}) const;

private:
  std::filesystem::path root_;
};

} // namespace splash::model
