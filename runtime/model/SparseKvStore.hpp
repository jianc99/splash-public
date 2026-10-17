#pragma once

// HiSparse for block-sparse attention (--sparse-kv-memory): a decoding
// request's committed KV pages are written to a file on the SSD, and its
// attention reads the blocks it selects through a hot cache of a lane's own.
// After a layer's selection and block lists (ops::AttentionBlocks), a host
// step resolves every entry of each lane's list: a block of a page that is
// not in the file yet is read from the request's page, as without the store;
// any other from its slot in the cache, where a block the cache does not hold
// is read from the file first (uncached reads, in parallel), into the slot of
// the least recently used block the step does not select. The entries'
// locations tell the attention where (metal/abi/PagedAttention.h).
//
// The cache is KV pages of the pool's layout (kv::PageStorage), every layer's
// region of a page holding 8 block slots of that layer, so the attention
// reads a cached block as it reads one of a page table. A lane's cache holds
// at least as many slots per layer as its rows select blocks, so a step
// always places every block it selects. The file keeps each layer's block
// records in page order, a page's 8 records in a 4 KiB-aligned run: a
// block's keys, values and scales are one record, read with one read.
//
// Only committed pages enter the file: a command writes only at or past the
// lane's committed tokens, so a page wholly before them never changes. The
// store writes them when a prefill or verify command is prepared, while no
// command runs, so the pages it copies are the request's own (the pool moves
// pages only between commands); a prefilled prompt's pages are in the file
// when its decode starts.
//
// Once a decoding request attends sparsely in every verify, the pages in its
// file spill (kv::PageStorage::spill): an extent all of whose pages spilled
// gives its K/V memory back between commands (release), and its QSA index,
// in buffers of its own, stays. A page's copy lives as long as the page
// spills, past its request: the cache keeps the request's blocks, and a
// command that reads a spilled page through a page table (a prefill, a
// verify of a page not in its lane's file, a lane without a cache) first
// reads it back from its copy (restore), taking its extent's memory again.

#include "metal/MetalBackend.hpp"
#include "ops/PageStorage.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace splash::model {

struct SparseKvStoreOptions final {
  // Each lane's cache, rounded down to whole extents of the pool's pages.
  uint64_t memoryBytes = 0;
  // State lanes, each with its cache and file.
  uint32_t lanes = 0;
  uint32_t maximumContextTokens = 0;
  // The most blocks a lane's rows select in a step: its rows times their top
  // blocks. A cache with fewer slots per layer is refused.
  uint32_t selectedBlocks = 0;
  // A decoding request's pages spill once it committed this many tokens:
  // every verify of it then attends through its lists.
  uint64_t spillFromTokens = 0;
  // Where the files go; each is unlinked once open.
  std::filesystem::path directory;
};

// What the store did, since it started.
struct SparseKvStats final {
  uint64_t resolves = 0;
  double resolveSeconds = 0.0;
  uint64_t entries = 0;
  uint64_t paged = 0;
  uint64_t hits = 0;
  uint64_t reads = 0;
  uint64_t readBytes = 0;
  double readSeconds = 0.0;
  uint64_t writtenPages = 0;
  uint64_t writtenBytes = 0;
  double writeSeconds = 0.0;
  uint64_t spilledPages = 0;
  uint64_t restoredPages = 0;
  double restoreSeconds = 0.0;
  uint64_t releasedBytes = 0;
};

class SparseKvStore final {
public:
  // The cache's pages come from `admission`, the pool's: they count as KV.
  // A lane's are allocated when its request first reaches spillFromTokens
  // and go once the lane is vacated; a lane the budget refuses them reads
  // every block from its requests' pages. Throws
  // std::invalid_argument for a layout without token-major values or a cache
  // smaller than one extent or than options.selectedBlocks slots, and
  // std::system_error when a file cannot be made.
  SparseKvStore(metal::MetalBackend &backend, metal::AllocationAdmission admission, kv::PageStorage &pages,
                SparseKvStoreOptions options);
  ~SparseKvStore();
  SparseKvStore(const SparseKvStore &) = delete;
  SparseKvStore &operator=(const SparseKvStore &) = delete;

  // Before a command of `stateLane`'s request `requestId`, whose first
  // `committedTokens` tokens are committed in the pages `pageTable` names,
  // and before its page table is written: restores the spilled pages the
  // command reads through the page table (every one but a verify's in the
  // lane's file), writes the pages wholly committed since the last call to
  // the lane's file, and for a verify past spillFromTokens spills the pages
  // in it. Another request on the lane starts a file of its own. Throws
  // metal::MetalAllocationError when the budget refuses a restore.
  void prepare(uint32_t stateLane, uint64_t requestId, uint64_t committedTokens, std::span<const uint32_t> pageTable,
               bool verify);
  // Between commands, before the next is prepared: gives back the K/V memory
  // of the extents whose pages all spilled and no transfer pins
  // (kv::PageStorage::releaseSpilledKv) and the caches of vacated lanes, and
  // drops the copies of pages that stopped spilling; nothing while a command
  // is in flight. Returns the bytes it gave back.
  uint64_t release();
  // The lane's request let go of it (it ended or was suspended): the lane's
  // cache goes at the next release unless a request uses the lane first.
  void vacate(uint32_t stateLane);

  // A layer's host step: the list of batch lane l, `counts[l]` entries from
  // lists + l * stride, belongs to state lane stateLanes[l]; writes each
  // entry's location to the same place of `locations`. Throws when a read
  // fails; the command then fails with it.
  void resolve(uint32_t layer, std::span<const uint32_t> stateLanes, const uint32_t *lists, const uint32_t *counts,
               uint32_t stride, uint32_t *locations);

  // The page table the locations index (ops::AttentionBlocks::locationPages).
  [[nodiscard]] const metal::MetalBuffer &locationPages() const noexcept { return locationPages_; }
  // A cache page as the host reaches it (kv::PageStorage::spans): page
  // location >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS of the location pages.
  [[nodiscard]] std::vector<std::span<std::byte>> cacheSpans(uint32_t page) const { return cache_->spans(page); }
  // The cache's bytes allocated now, and a lane's.
  [[nodiscard]] uint64_t cacheBytes() const noexcept;
  [[nodiscard]] uint64_t laneCacheBytes() const noexcept { return uint64_t{lanePages_} * pages_.bytesPerPage(); }
  [[nodiscard]] SparseKvStats stats() const noexcept { return stats_; }

private:
  struct Lane;
  struct File;
  class Readers;
  // Where a spilled page's copy is: a file and the page's place in it.
  struct Copy final {
    std::shared_ptr<File> file;
    uint32_t page = 0;
  };
  struct Miss final {
    uint32_t lane;
    uint32_t block;
    uint32_t slot;
  };

  [[nodiscard]] uint64_t recordOffset(uint32_t layer, uint32_t block) const noexcept;
  void allocate(uint32_t stateLane);
  void report(const Lane &lane) const noexcept;
  void reset(Lane &lane, uint64_t requestId);
  void forgetBlocks(Lane &lane);
  [[nodiscard]] std::shared_ptr<File> openFile() const;
  void write(Lane &lane, std::span<const uint32_t> pageTable, uint32_t from, uint32_t to);
  void restore(std::span<const uint32_t> pages);
  void read(uint32_t layer, std::span<const Miss> misses);

  metal::MetalBackend &backend_;
  kv::PageStorage &pages_;
  kv::Layout layout_;
  std::filesystem::path directory_;
  uint64_t spillFromTokens_ = 0;
  uint32_t lanes_ = 0;
  uint32_t maximumBlocks_ = 0;
  // Each lane's cache pages, slots per layer (8 a page) and the bytes of a
  // record, of the 4 KiB-aligned run of a page's records, and of each tensor
  // a record holds of one head (keys, then key scales, values, value scales).
  uint32_t lanePages_ = 0;
  uint32_t slots_ = 0;
  uint32_t recordBytes_ = 0;
  uint32_t pageRunBytes_ = 0;
  std::array<uint32_t, 4> headBytes_{};
  std::unique_ptr<kv::PageStorage> cache_;
  // Each cache page's tensors of each layer (kv::PageStorage::spans), keys,
  // key scales, values and value scales, null where the format has none.
  std::vector<std::byte *> cacheTensors_;
  std::unique_ptr<Readers> readers_;
  metal::MetalBuffer locationPages_;
  std::vector<std::unique_ptr<Lane>> laneState_;
  // Each pool page's copy while it spills.
  std::vector<Copy> copies_;
  std::vector<Miss> misses_;
  SparseKvStats stats_;
};

} // namespace splash::model
