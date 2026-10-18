#include "model/KvCopyRead.hpp"

#include <cstdint>
#include <cstdlib>
#include <new>
#include <utility>

namespace splash::model {
namespace {

constexpr uint64_t kAlignment = kv::SlotFormat::kReadAlignmentBytes;

constexpr uint64_t alignUp(uint64_t value) noexcept { return (value + kAlignment - 1) / kAlignment * kAlignment; }

// The thread's buffer, grown to the largest read it made.
struct ReadBuffer final {
  std::byte *bytes = nullptr;
  uint64_t size = 0;

  ReadBuffer() = default;
  ReadBuffer(const ReadBuffer &) = delete;
  ReadBuffer &operator=(const ReadBuffer &) = delete;
  ~ReadBuffer() { std::free(bytes); }

  std::span<std::byte> hold(uint64_t wanted) {
    if (size < wanted) {
      std::free(std::exchange(bytes, nullptr));
      size = 0;
      bytes = static_cast<std::byte *>(std::aligned_alloc(kAlignment, wanted));
      if (!bytes) throw std::bad_alloc();
      size = wanted;
    }
    return {bytes, size};
  }
};

} // namespace

std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range) {
  thread_local ReadBuffer buffer;
  // The aligned range around the bytes, and a unit more: an adopted copy
  // reads the whole segments a range touches (SlotFile::readAt), and a range
  // of segments (the index, a block's record, a layer's) leaves at most the
  // room after a record, less than a unit, outside (kv::SlotFormat).
  return copy.readAt(range.offset, range.bytes,
                     buffer.hold(alignUp(range.offset % kAlignment + range.bytes) + kAlignment));
}

} // namespace splash::model
