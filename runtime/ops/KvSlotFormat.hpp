#pragma once

#include "ops/PageStorage.hpp"
#include "ops/PagedKv.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace splash::kv {

// Bytes of a page's tier slot (model::SlotFile): an offset from the slot's
// start and a length.
struct SlotRange final {
  uint64_t offset = 0;
  uint64_t bytes = 0;

  bool operator==(const SlotRange &) const = default;
};

// How a page sits in its tier slot, and the only definition of it.
//
// Version 0, for a layout that cannot spill (Layout::spillable): the page's
// spans in PageStorage::spans order, one segment.
//
// Version 1, for a spillable layout: the page's index, layer by layer
// (indexBytesPerLayerPage each), then, from base = the index rounded up to
// kReadAlignmentBytes, one record per layer and block of
// SPLASH_ATTENTION_BLOCK_TOKENS tokens, record (layer, block) at base +
// (8 layer + block) pitch. A record holds the block's tokens of each tensor
// slab of the layer, tensor-major and head by head: keys, key scales, values,
// value scales (BF16 has no scales), so a slot is a permutation of the page's
// bytes. The pitch is the smallest alignUp(recordBytes, 2^j), j <= 12, at
// which every record lies within the fewest aligned units it can, so a block
// is one aligned read (int8 Flash-Next: 4,160 B records, one 8 KiB read
// each). Its segments, which a persistent file checksums, are the index
// region, then each record.
class SlotFormat final {
public:
  // The alignment of a direct read (F_NOCACHE).
  static constexpr uint64_t kReadAlignmentBytes = 4096;
  // A version-1 page's blocks per layer: a location's places.
  static constexpr uint32_t kBlocksPerPage = 1u << SPLASH_ATTENTION_LOCATION_BLOCK_BITS;

  // The format of the layout's pages: version 1 when it is spillable.
  explicit SlotFormat(const Layout &layout);

  [[nodiscard]] uint32_t version() const noexcept;
  [[nodiscard]] const Layout &layout() const noexcept { return layout_; }
  // The bytes a slot stores (model::SlotFile::slotBytesFor rounds them up).
  [[nodiscard]] uint64_t payloadBytes() const noexcept;
  // The byte lengths of the payload's consecutive segments from its start
  // (model::SlotFile::Options::segments): one for version 0.
  [[nodiscard]] std::vector<uint64_t> segments() const;

  // Version 1 only; std::logic_error for version 0, std::out_of_range for a
  // layer or block past the page. The page's index; one record and its
  // pitch; one layer's records, which a prefill's history reads together.
  [[nodiscard]] SlotRange indexRange() const;
  [[nodiscard]] uint64_t recordBytes() const;
  [[nodiscard]] uint64_t recordPitch() const;
  [[nodiscard]] SlotRange recordRange(uint32_t layer, uint32_t block) const;
  [[nodiscard]] SlotRange layerRange(uint32_t layer) const;

  // The page's memory in slot order, which a write gathers and a whole
  // restore scatters into (model::SlotFile::write and read): its spans for
  // version 0, its index and record pieces for version 1.
  [[nodiscard]] std::vector<std::span<std::byte>> transferSpans(const PageStorage &pages, uint32_t page) const;
  // Version 1 only. What indexRange() read, into a page's index
  // (PageStorage::indexSpans).
  void scatterIndex(std::span<const std::byte> index, std::span<const std::span<std::byte>> indexSpans) const;
  // What recordRange(layer, block) read, into block `place` of a page's
  // layer (PageStorage::layerSpans of any storage with the layout's heads,
  // dimension and format: the pool, the hot block cache, staging).
  void scatterRecord(std::span<const std::byte> record, const std::array<std::span<std::byte>, 4> &layer,
                     uint32_t place) const;
  // What layerRange(layer) read, into a page's layer as scatterRecord does
  // each of its blocks at its own place.
  void scatterLayer(std::span<const std::byte> records, const std::array<std::span<std::byte>, 4> &layer) const;

private:
  Layout layout_;
};

} // namespace splash::kv
