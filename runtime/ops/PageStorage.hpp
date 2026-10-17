#pragma once

#include "metal/MetalBackend.hpp"
#include "metal/abi/KvExtent.h"
#include "ops/PagedKv.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace splash::kv {

// Storage for paged KV in extents of extentPages pages each, which KvPool
// alone allocates and releases: its runway when it is built, then an extent
// when it needs one of its pages. An extent is an ordinary shared Metal buffer
// of its K/V and, for a layout with a QSA index, a second one of its pages'
// index, so that the K/V of pages whose copy a sparse KV store keeps can go
// while their index stays (spill, releaseSpilledKv).
// An extent whose pages are all free stays allocated until a reclaim pass
// between commands releases it (Engine::reclaimMemory); the cleanup after
// startup warmup keeps one. Inside an extent every attention layer has a
// region that holds the keys of all its pages, then their key scales, values
// and value scales (abi/KvExtent.h). Kernels reach a page through its entry
// in a request's GPU page table, so extents are allocated
// metal::BufferReach::Addressed: the command queue's residency set keeps them
// resident for every command buffer. The host reaches the same bytes through
// the page's spans(). Active requests may overwrite slots at or beyond their
// logical commit index. Cached KV blocks reference only fully committed
// pages, which are immutable while shared.
class PageStorage final : public ExtentStorage {
public:
  // pageCount must be a whole number of extents of extentPages pages, a
  // whole number of the layout's alignment units (Layout::extentPagesFor).
  PageStorage(metal::MetalBackend &backend,
              metal::AllocationAdmission admitAllocation,
              Layout layout,
              uint32_t pageCount,
              uint32_t extentPages);

  PageStorage(const PageStorage &) = delete;
  PageStorage &operator=(const PageStorage &) = delete;

  [[nodiscard]] uint32_t pageCount() const noexcept override {
    return pageCount_;
  }
  [[nodiscard]] uint64_t bytesPerPage() const noexcept override {
    return layout_.bytesPerModelPage();
  }
  [[nodiscard]] Layout layout() const noexcept { return layout_; }
  [[nodiscard]] uint32_t extentPages() const noexcept override {
    return extentPages_;
  }
  [[nodiscard]] uint64_t extentBytes() const noexcept {
    return uint64_t{extentPages_} * layout_.bytesPerModelPage();
  }
  // The bytes of the allocated extents, which KvPool's record of its extents
  // must match, and of the buffers that hold memory now: less by the K/V
  // buffers releaseSpilledKv gave back.
  [[nodiscard]] uint64_t allocatedExtentBytes() const noexcept;
  [[nodiscard]] uint64_t actualAllocatedBytes() const noexcept;
  [[nodiscard]] bool isAllocated(uint32_t page) const;
  // std::out_of_range for an extent past the pool.
  [[nodiscard]] metal::AllocationResult allocateExtent(uint32_t extent) override;
  // The caller must prove that no active, prefix, reserved, or in-flight
  // reference remains anywhere in this extent. std::out_of_range for an
  // extent past the pool. KvPool never releases an extent that holds a page
  // a request or the cache holds; GPU page tables rely on it (Runtime's
  // PageTableBinding).
  void releaseExtent(uint32_t extent) override;
  // Copies every tensor of each page in every layer (spans()).
  // std::out_of_range for a page past the pool.
  void copyPages(std::span<const PageCopy> copies) override;
  // Zeroes every tensor of each page in every layer (spans()).
  // std::out_of_range for a page past the pool.
  void clearPages(std::span<const uint32_t> pages) override;
  // Where each attention layer's region sits in every extent, by layer.
  [[nodiscard]] std::span<const SplashKvLayer> layers() const noexcept {
    return layers_;
  }
  // Where a layer's QSA index (Layout::indexDimension), the pooled keys of
  // every page, sits in every extent's index buffer, in bytes.
  // std::out_of_range without an index or for a layer past the layout.
  [[nodiscard]] uint32_t indexOffset(uint32_t layer) const;
  // A page's K/V has a copy a sparse KV store keeps (model::SparseKvStore):
  // releaseSpilledKv may give its extent's K/V buffer back once every page of
  // the extent has spilled and no transfer pins one; its index stays.
  // Restoring (unspill) takes the buffer back if it went, through the
  // allocation admission, and the caller writes the page's K/V again.
  void spill(uint32_t page);
  [[nodiscard]] bool spilled(uint32_t page) const;
  [[nodiscard]] bool kvResident(uint32_t page) const;
  [[nodiscard]] metal::AllocationResult unspill(uint32_t page);
  // The K/V buffers it gave back, in bytes; none while a command is in
  // flight.
  uint64_t releaseSpilledKv();
  // Moves on whenever a K/V buffer goes or comes back at another address:
  // GPU page tables written before then name stale entries.
  [[nodiscard]] uint64_t kvRevision() const noexcept { return kvRevision_; }
  [[nodiscard]] bool movable(uint32_t extent) const override { return !spilledPages_.at(extent); }
  // A transfer (engine::KvPageTier) reads or writes the page's bytes while
  // it is pinned.
  void pin(uint32_t page);
  void unpin(uint32_t page);
  [[nodiscard]] metal::AllocationResult prepareAcquired(uint32_t page) override;

  // Writes the entries kernels reach `pages` by, from index `first` on, to a
  // CPU-visible GPU page table, which must hold all of them; entries before
  // `first` are left as they are. Throws std::logic_error for a table that is
  // not CPU-visible or too small and for a page whose extent is not allocated
  // (a GPU table holds only pages of allocated extents), std::invalid_argument
  // for `first` past the pages.
  void writeEntries(std::span<const uint32_t> pages, uint32_t first,
                    const metal::MetalBuffer &table) const;
  // The same for an index table (ops::QsaIndex): entries of the extents'
  // index buffers. A page whose K/V went has a null K/V entry, which nothing
  // reads: its store serves its blocks.
  void writeIndexEntries(std::span<const uint32_t> pages, uint32_t first,
                         const metal::MetalBuffer &table) const;
  // The page's memory as the host reaches it, and the only code that names
  // it: the page's bytes of each tensor in every layer's region, layer by
  // layer as keys, key scales, values and value scales, where
  // splash_kv_offset places them for the kernels, then each layer's QSA
  // pooled keys (indexOffset). BF16 pages have no scale bytes. Throws
  // std::logic_error for a page whose extent is not allocated or whose K/V
  // buffer went.
  [[nodiscard]] std::vector<std::span<std::byte>> spans(uint32_t page) const;

private:
  [[nodiscard]] size_t extentIndex(uint32_t page) const;
  [[nodiscard]] SplashKvPage entry(uint32_t page) const;
  [[nodiscard]] uint64_t kvBytes() const noexcept {
    return uint64_t{extentPages_} * layout_.attentionLayers * layout_.bytesPerLayerPage();
  }
  [[nodiscard]] uint64_t indexBytes() const noexcept { return extentBytes() - kvBytes(); }
  [[nodiscard]] metal::MetalBuffer allocateBuffer(uint64_t bytes, const std::string &name) const;

  metal::MetalBackend &backend_;
  metal::AllocationAdmission admitAllocation_;
  Layout layout_;
  uint32_t pageCount_ = 0;
  uint32_t extentPages_ = 0;
  std::vector<SplashKvLayer> layers_;
  // Each extent's K/V buffer and index buffer, empty while it is not
  // allocated (the K/V one also once releaseSpilledKv gave it back), and
  // their GPU addresses, zero then.
  std::vector<metal::MetalBuffer> extents_;
  std::vector<metal::MetalBuffer> indexes_;
  std::vector<uint64_t> extentAddresses_;
  std::vector<uint64_t> indexAddresses_;
  std::vector<bool> allocated_;
  // Each page's spill, each extent's spilled pages, and each page's pins.
  std::vector<bool> spilled_;
  std::vector<uint32_t> spilledPages_;
  std::vector<uint32_t> pins_;
  uint64_t kvRevision_ = 0;
};

} // namespace splash::kv
