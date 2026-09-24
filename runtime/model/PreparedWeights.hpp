#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

namespace splash::model {

// Shared by source adapters. Preparation owns bounded buffers; inference only
// opens completed, immutable artifacts through WeightFile. The admission reserve
// includes source metadata and small staging buffers; it is not a model copy.
inline constexpr uint64_t kWeightPreparationWorkspaceBytes = 64 * 1024 * 1024;
// The staging of one conversion step, its input and output together. Every
// adapter sizes its chunks to it, whatever the tensor, layer or expert count.
inline constexpr uint64_t kWeightPreparationStagingBytes = kWeightPreparationWorkspaceBytes / 2;

enum class TargetSource : uint8_t { Packed, Affine, Gguf };

using PreparationCheck = std::function<void()>;

// A prepared file: its cache key and size, the component it is (such as
// target/layer-0.bin), the digest of the source data it is written from and
// where that source is. An entry of the same component and inputs under
// another key is an earlier preparation, which publishing this one removes.
struct PreparedWeight {
  std::string key;
  uint64_t bytes;
  std::string component{};
  std::string inputs{};
  std::string source{};
};

// Leave room for the OS and other applications; this is a disk reserve, not
// a promise that concurrent system activity can never exhaust the volume.
inline constexpr uint64_t kWeightCacheDiskReserve = uint64_t{2} << 30;
void requireWeightDiskSpace(uint64_t available, uint64_t required);

void readWeightBytes(int descriptor, uint64_t offset, std::span<uint8_t> bytes);
void writeWeightBytes(int descriptor, uint64_t offset, std::span<const uint8_t> bytes);
// Copies bytes [from, from + bytes) of source to destination at `to` through
// staging; check runs before each piece.
void copyWeightBytes(int source, uint64_t from, int destination, uint64_t to, uint64_t bytes,
                     std::span<uint8_t> staging, const PreparationCheck &check = {});
[[nodiscard]] std::string weightDigest(std::span<const uint8_t> bytes);
[[nodiscard]] std::string weightDigest(std::string_view text);

// A source file, opened once; checkUnchanged throws when it was modified or
// replaced since. The digest of its tensor data is computed on first use,
// after the caller has validated the metadata, and remembered for this file
// identity, so a warm start does not read the file again.
class WeightSource final {
public:
  explicit WeightSource(const std::filesystem::path &path, PreparationCheck check = {});
  ~WeightSource();
  WeightSource(const WeightSource &) = delete;
  WeightSource &operator=(const WeightSource &) = delete;
  [[nodiscard]] const std::filesystem::path &path() const noexcept;
  [[nodiscard]] int descriptor() const noexcept;
  // SHA-256 of bytes [dataOffset, end), the file's tensor data: editing only
  // its metadata keeps the identity of every tensor.
  [[nodiscard]] const std::string &digest(uint64_t dataOffset) const;
  void checkUnchanged() const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// What a prepared file's key is the SHA-256 of: the adapter's preparation
// identity, its plan as one record per line, and every source tensor it
// reads, by the digest of its file's tensor data, its offset there, size,
// type and shape.
class WeightIdentity final {
public:
  explicit WeightIdentity(std::string_view preparation) { text_ << preparation << '\n'; }
  template <class... Fields> WeightIdentity &record(const Fields &...fields) {
    ((text_ << fields << ' '), ...);
    text_ << '\n';
    return *this;
  }
  // Bytes [offset, offset + bytes) of source, whose tensor data starts at
  // dataOffset.
  WeightIdentity &input(const WeightSource &source, uint64_t dataOffset, uint64_t offset, uint64_t bytes,
                        std::string_view type, std::span<const uint64_t> shape);
  [[nodiscard]] PreparedWeight weight(uint64_t bytes, std::string component, std::string source) const;
private:
  std::ostringstream text_;
  std::set<std::string> digests_;
};

// The callbacks of one load. check runs throughout (cancellation, memory
// pressure), also while waiting for the converter lock or hashing;
// admitConversion admits the conversion workspace on a cache miss, before
// anything is allocated and again before each chunk; unchanged throws when a
// source changed after its tensor data was hashed.
struct PreparationGuards {
  PreparationCheck check{};
  PreparationCheck admitConversion{};
  PreparationCheck unchanged{};
};

// Writes a prepared file into its empty, preallocated destination; admit
// runs before each chunk of conversion work.
using WeightWriter = std::function<void(int destination, const PreparationCheck &admit)>;

// The cache is SPLASH_WEIGHT_CACHE, or ~/Library/Caches/Splash/weights.
class PreparedWeights final {
public:
  PreparedWeights();
  // Check the entire missing model before writing its first artifact. Completed
  // layers remain reusable after an interruption; they are not partial files.
  void requireSpace(std::span<const PreparedWeight> weights,
                    const PreparationCheck &check = {}) const;
  // The complete file of weight: reused, or written now under the converter
  // lock and published atomically; writer failures never publish partial
  // data. The sources are checked unchanged before, after writing and before
  // the path is returned, so no file of a modified source is published or
  // used.
  [[nodiscard]] std::filesystem::path prepare(const PreparedWeight &weight, const WeightWriter &write,
                                              const PreparationGuards &guards = {}) const;

private:
  std::filesystem::path root_;
};

} // namespace splash::model
