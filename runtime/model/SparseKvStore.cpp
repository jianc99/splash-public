#include "model/SparseKvStore.hpp"

#include "AwakeClock.hpp"
#include "StderrLine.hpp"

#include "metal/abi/PagedAttention.h"
#include "model/WeightImages.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>

namespace splash::model {
namespace {

constexpr uint32_t kPageTokens = kv::kPageTokens;
constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
constexpr uint32_t kPageBlocks = kPageTokens / kBlockTokens;
constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();
// Reads in flight: the SSD's random reads of a few KiB peak near 32 (about
// 230K a second on an M5 Pro, 4.5 us each in a batch of a thousand), which
// libdispatch's workers, near the CPU count, do not keep up (7-8 us).
constexpr unsigned kReaders = 32;
// A write gathers up to this many pages of a layer into one run of the file.
constexpr uint32_t kWritePages = 32;
constexpr unsigned kWriters = 8;
constexpr uint64_t kAlignment = 4096;

constexpr uint64_t alignUp(uint64_t bytes) noexcept { return (bytes + kAlignment - 1) / kAlignment * kAlignment; }

double secondsSince(AwakeClock::time_point start) { return millisecondsSince(start) / 1e3; }

struct AlignedBytes final {
  explicit AlignedBytes(uint64_t bytes) : bytes(static_cast<std::byte *>(std::aligned_alloc(kAlignment, alignUp(bytes)))) {
    if (!this->bytes) throw std::bad_alloc();
  }
  ~AlignedBytes() { std::free(bytes); }
  AlignedBytes(const AlignedBytes &) = delete;
  AlignedBytes &operator=(const AlignedBytes &) = delete;
  std::byte *bytes;
};

void readFully(int file, std::byte *to, uint64_t bytes, uint64_t offset) {
  while (bytes) {
    const ssize_t count = ::pread(file, to, bytes, static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), "sparse KV read");
    to += count;
    bytes -= static_cast<uint64_t>(count);
    offset += static_cast<uint64_t>(count);
  }
}

void writeFully(int file, const std::byte *from, uint64_t bytes, uint64_t offset) {
  while (bytes) {
    const ssize_t count = ::pwrite(file, from, bytes, static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), "sparse KV write");
    from += count;
    bytes -= static_cast<uint64_t>(count);
    offset += static_cast<uint64_t>(count);
  }
}

} // namespace

// Threads that each take a batch's reads until none is left, then wait for
// the next batch, each with a staging buffer of its own.
class SparseKvStore::Readers final {
public:
  Readers(unsigned count, uint64_t stagingBytes) {
    for (unsigned index = 0; index < count; ++index)
      threads_.emplace_back([this, stagingBytes] { serve(stagingBytes); });
  }
  ~Readers() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread &thread : threads_) thread.join();
  }
  // Runs task(index, staging) for each index below count; rethrows the first
  // failure once every reader stopped.
  void run(size_t count, const std::function<void(size_t, std::byte *)> &task) {
    std::unique_lock lock(mutex_);
    task_ = &task;
    count_ = count;
    next_ = 0;
    failure_ = nullptr;
    busy_ = static_cast<unsigned>(threads_.size());
    ++generation_;
    start_.notify_all();
    done_.wait(lock, [this] { return !busy_; });
    task_ = nullptr;
    if (failure_) std::rethrow_exception(failure_);
  }

private:
  void serve(uint64_t stagingBytes) {
    AlignedBytes staging(stagingBytes);
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock lock(mutex_);
      start_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      lock.unlock();
      for (size_t index; !failed_ && (index = next_++) < count_;) {
        try {
          (*task_)(index, staging.bytes);
        } catch (...) {
          std::lock_guard failure(mutex_);
          if (!failure_) failure_ = std::current_exception();
          failed_ = true;
        }
      }
      lock.lock();
      if (!--busy_) {
        failed_ = false;
        done_.notify_one();
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  uint64_t generation_ = 0;
  unsigned busy_ = 0;
  bool stop_ = false;
  const std::function<void(size_t, std::byte *)> *task_ = nullptr;
  size_t count_ = 0;
  std::atomic<size_t> next_ = 0;
  std::atomic<bool> failed_ = false;
  std::exception_ptr failure_;
  std::vector<std::thread> threads_;
};

// A lane's request, its file, and per layer its cache's directory: each
// block's slot, each slot's block, and the slots' recency, most recent first,
// with the resolve that last used each.
struct SparseKvStore::Lane final {
  uint64_t request = 0;
  int file = -1;
  // Whether the lane's cache pages are allocated, or the budget refused
  // them: its requests then read every block from their pages.
  bool cached = false;
  bool refused = false;
  // The request's share of the store's work: its resolves and reads, and
  // the time of the reads and writes it took part in.
  SparseKvStats stats;
  // The request's pages [0, filePages) are in the file.
  uint32_t filePages = 0;
  uint64_t resolves = 0;
  std::vector<uint32_t> slotOf;
  std::vector<uint32_t> blockIn;
  std::vector<uint32_t> newer;
  std::vector<uint32_t> older;
  std::vector<uint64_t> used;
  std::vector<uint32_t> newest;
  std::vector<uint32_t> oldest;

  ~Lane() {
    if (file >= 0) ::close(file);
  }
  // Makes slot `slot` of the layer's slots from `base` the most recent.
  void touch(uint32_t layer, uint32_t base, uint32_t slot) {
    const uint32_t s = base + slot;
    if (newest[layer] == slot) return;
    if (newer[s] != kNone) older[base + newer[s]] = older[s];
    if (older[s] != kNone) newer[base + older[s]] = newer[s];
    if (oldest[layer] == slot) oldest[layer] = newer[s];
    newer[s] = kNone;
    older[s] = newest[layer];
    if (newest[layer] != kNone) newer[base + newest[layer]] = slot;
    newest[layer] = slot;
    if (oldest[layer] == kNone) oldest[layer] = slot;
  }
};

SparseKvStore::SparseKvStore(metal::MetalBackend &backend, metal::AllocationAdmission admission,
                             const kv::PageStorage &pages, SparseKvStoreOptions options)
    : pages_(pages), layout_(pages.layout()), lanes_(options.lanes),
      maximumBlocks_((options.maximumContextTokens + kPageTokens - 1) / kPageTokens * kPageBlocks) {
  if (!layout_.valuesTokenMajor)
    throw std::invalid_argument("a sparse KV store needs token-major values, whose blocks are contiguous");
  const uint64_t extentBytes = pages_.extentBytes();
  const uint64_t laneExtents = options.memoryBytes / extentBytes;
  lanePages_ = static_cast<uint32_t>(laneExtents * pages_.extentPages());
  slots_ = lanePages_ * kPageBlocks;
  if (!lanes_ || !maximumBlocks_ || !laneExtents || slots_ < options.selectedBlocks)
    throw std::invalid_argument("--sparse-kv-memory holds fewer blocks a layer than a step selects: give each lane at "
                                "least " +
                                std::to_string((options.selectedBlocks / (pages_.extentPages() * kPageBlocks) + 1) *
                                               extentBytes / (1 << 20)) +
                                " MiB");
  // A record holds each tensor's run of the block's tokens of every head.
  const uint32_t element = layout_.format == kv::Format::Int8 ? 1 : 2;
  const uint32_t scale = layout_.format == kv::Format::Int8 ? sizeof(float) : 0;
  headBytes_ = {kBlockTokens * layout_.headDimension * element, kBlockTokens * scale,
                kBlockTokens * layout_.headDimension * element, kBlockTokens * scale};
  for (const uint32_t bytes : headBytes_) recordBytes_ += layout_.kvHeads * bytes;
  pageRunBytes_ = static_cast<uint32_t>(alignUp(uint64_t{kPageBlocks} * recordBytes_));

  cache_ = std::make_unique<kv::PageStorage>(backend, std::move(admission), layout_, lanes_ * lanePages_,
                                             pages_.extentPages());
  cacheTensors_.assign(uint64_t{lanes_} * lanePages_ * layout_.attentionLayers * 4, nullptr);
  readers_ = std::make_unique<Readers>(kReaders, alignUp(recordBytes_) + kAlignment);
  locationPages_ = backend.allocateBuffer(uint64_t{lanes_} * lanePages_ * sizeof(SplashKvPage),
                                          metal::BufferStorage::Shared, "sparse-kv-location-pages");

  const uint32_t layers = layout_.attentionLayers;
  for (uint32_t index = 0; index < lanes_; ++index) {
    auto lane = std::make_unique<Lane>();
    std::string name = (options.directory / "splash-sparse-kv-XXXXXX").string();
    lane->file = ::mkstemp(name.data());
    if (lane->file < 0) throw std::system_error(errno, std::generic_category(), "sparse KV file " + name);
    ::unlink(name.c_str());
    if (::fcntl(lane->file, F_NOCACHE, 1) < 0)
      throw std::system_error(errno, std::generic_category(), "sparse KV file " + name);
    lane->slotOf.assign(uint64_t{layers} * maximumBlocks_, kNone);
    lane->blockIn.assign(uint64_t{layers} * slots_, kNone);
    lane->newer.resize(uint64_t{layers} * slots_);
    lane->older.resize(uint64_t{layers} * slots_);
    lane->used.assign(uint64_t{layers} * slots_, 0);
    lane->newest.resize(layers);
    lane->oldest.resize(layers);
    laneState_.push_back(std::move(lane));
    reset(*laneState_.back(), 0);
  }
}

SparseKvStore::~SparseKvStore() {
  for (const std::unique_ptr<Lane> &lane : laneState_) report(*lane);
}

uint64_t SparseKvStore::cacheBytes() const noexcept { return cache_->actualAllocatedBytes(); }

// Allocates a lane's cache pages, the first time a request uses the lane,
// and points the location pages at them; with the budget refusing one, the
// lane keeps none and reads every block from its requests' pages.
void SparseKvStore::allocate(uint32_t stateLane) {
  Lane &lane = *laneState_[stateLane];
  const uint32_t extents = lanePages_ / pages_.extentPages(), first = stateLane * extents;
  for (uint32_t extent = first; extent < first + extents; ++extent) {
    if (cache_->allocateExtent(extent)) continue;
    while (extent-- > first) cache_->releaseExtent(extent);
    lane.refused = true;
    logLine("Warning · Sparse KV: no memory for lane ", stateLane, "'s cache; its requests read their pages.");
    return;
  }
  std::vector<uint32_t> pages((stateLane + 1) * lanePages_);
  for (uint32_t page = 0; page < pages.size(); ++page) pages[page] = page;
  cache_->writeEntries(pages, stateLane * lanePages_, locationPages_);
  const uint32_t tensorsPerLayer = layout_.format == kv::Format::Int8 ? 4 : 2;
  for (uint32_t page = stateLane * lanePages_; page < pages.size(); ++page) {
    const std::vector<std::span<std::byte>> tensors = cache_->spans(page);
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
      for (uint32_t tensor = 0, at = 0; tensor < 4; ++tensor)
        if (headBytes_[tensor])
          cacheTensors_[(uint64_t{page} * layout_.attentionLayers + layer) * 4 + tensor] =
              tensors[layer * tensorsPerLayer + at++].data();
  }
  lane.cached = true;
}

uint64_t SparseKvStore::recordOffset(uint32_t layer, uint32_t block) const noexcept {
  const uint64_t page = uint64_t{layer} * (maximumBlocks_ / kPageBlocks) + block / kPageBlocks;
  return page * pageRunBytes_ + uint64_t{block % kPageBlocks} * recordBytes_;
}

// What the request the lane served did, if it resolved anything.
void SparseKvStore::report(const Lane &lane) const noexcept {
  if (const SparseKvStats &done = lane.stats; done.resolves) {
    const uint64_t cached = done.entries - done.paged;
    logLine("Sparse KV · ", done.resolves / layout_.attentionLayers, " steps · ", done.entries / done.resolves,
            " blocks a layer, ", done.paged * 100 / std::max<uint64_t>(1, done.entries), "% in pages · ",
            done.hits * 100 / std::max<uint64_t>(1, cached), "% of the rest cached · ", done.reads, " reads (",
            done.readBytes >> 20, " MiB) in ", static_cast<uint64_t>(done.readSeconds * 1e3), " ms, ",
            static_cast<uint64_t>(done.resolveSeconds * 1e3), " ms resolving in all · ",
            done.writtenPages, " pages written in ", static_cast<uint64_t>(done.writeSeconds * 1e3), " ms");
  }
}

// Forgets the lane's blocks and file, after reporting them; every slot is
// free, the oldest first.
void SparseKvStore::reset(Lane &lane, uint64_t requestId) {
  report(lane);
  lane.stats = {};
  lane.request = requestId;
  lane.filePages = 0;
  if (::ftruncate(lane.file, 0) < 0) throw std::system_error(errno, std::generic_category(), "sparse KV file");
  std::ranges::fill(lane.slotOf, kNone);
  std::ranges::fill(lane.blockIn, kNone);
  std::ranges::fill(lane.used, 0);
  for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
    const uint64_t base = uint64_t{layer} * slots_;
    for (uint32_t slot = 0; slot < slots_; ++slot) {
      lane.newer[base + slot] = slot ? slot - 1 : kNone;
      lane.older[base + slot] = slot + 1 < slots_ ? slot + 1 : kNone;
    }
    lane.newest[layer] = 0;
    lane.oldest[layer] = slots_ - 1;
  }
}

void SparseKvStore::prepare(uint32_t stateLane, uint64_t requestId, uint64_t committedTokens,
                            std::span<const uint32_t> pageTable) {
  if (stateLane >= lanes_) throw std::out_of_range("sparse KV lane");
  Lane &lane = *laneState_[stateLane];
  if (lane.request != requestId) reset(lane, requestId);
  if (!lane.cached && !lane.refused) allocate(stateLane);
  if (lane.refused) return;
  const uint64_t complete = std::min<uint64_t>(committedTokens / kPageTokens, pageTable.size());
  if (complete > maximumBlocks_ / kPageBlocks) throw std::out_of_range("sparse KV request past its context");
  if (complete > lane.filePages) write(lane, pageTable, lane.filePages, static_cast<uint32_t>(complete));
}

// Copies each page's blocks into the file, a layer's run of up to
// kWritePages pages at once.
void SparseKvStore::write(Lane &lane, std::span<const uint32_t> pageTable, uint32_t from, uint32_t to) {
  const auto start = AwakeClock::now();
  const uint32_t layers = layout_.attentionLayers;
  const uint32_t tensorsPerLayer = layout_.format == kv::Format::Int8 ? 4 : 2;
  const uint32_t runs = (to - from + kWritePages - 1) / kWritePages;
  std::vector<std::vector<std::span<std::byte>>> spans(to - from);
  for (uint32_t page = from; page < to; ++page) spans[page - from] = pages_.spans(pageTable[page]);
  std::vector<std::unique_ptr<AlignedBytes>> staging(kWriters);
  parallelFor(
      uint64_t{runs} * layers,
      [&](size_t task, unsigned worker) {
        if (!staging[worker]) staging[worker] = std::make_unique<AlignedBytes>(uint64_t{kWritePages} * pageRunBytes_);
        const uint32_t layer = static_cast<uint32_t>(task % layers);
        const uint32_t first = from + static_cast<uint32_t>(task / layers) * kWritePages;
        const uint32_t count = std::min(kWritePages, to - first);
        std::byte *run = staging[worker]->bytes;
        std::memset(run, 0, uint64_t{count} * pageRunBytes_);
        for (uint32_t page = 0; page < count; ++page) {
          const std::vector<std::span<std::byte>> &tensors = spans[first + page - from];
          for (uint32_t place = 0; place < kPageBlocks; ++place) {
            std::byte *record = run + uint64_t{page} * pageRunBytes_ + uint64_t{place} * recordBytes_;
            for (uint32_t tensor = 0, at = 0; tensor < 4; ++tensor) {
              if (!headBytes_[tensor]) continue;
              const std::span<std::byte> source = tensors[layer * tensorsPerLayer + at++];
              const uint64_t perHead = uint64_t{kPageTokens} * (headBytes_[tensor] / kBlockTokens);
              for (uint32_t head = 0; head < layout_.kvHeads; ++head) {
                std::memcpy(record, source.data() + head * perHead + place * headBytes_[tensor], headBytes_[tensor]);
                record += headBytes_[tensor];
              }
            }
          }
        }
        writeFully(lane.file, run, uint64_t{count} * pageRunBytes_, recordOffset(layer, first * kPageBlocks));
      },
      kWriters);
  lane.filePages = to;
  for (SparseKvStats *stats : {&stats_, &lane.stats}) {
    stats->writtenPages += to - from;
    stats->writtenBytes += uint64_t{to - from} * layers * pageRunBytes_;
    stats->writeSeconds += secondsSince(start);
  }
}

void SparseKvStore::resolve(uint32_t layer, std::span<const uint32_t> stateLanes, const uint32_t *lists,
                            const uint32_t *counts, uint32_t stride, uint32_t *locations) {
  if (layer >= layout_.attentionLayers) throw std::out_of_range("sparse KV layer");
  const auto start = AwakeClock::now();
  ++stats_.resolves;
  misses_.clear();
  const uint64_t base = uint64_t{layer} * slots_;
  for (uint32_t index = 0; index < stateLanes.size(); ++index) {
    if (stateLanes[index] >= lanes_) throw std::out_of_range("sparse KV lane");
    Lane &lane = *laneState_[stateLanes[index]];
    const uint64_t stamp = ++lane.resolves;
    const uint32_t inFile = lane.filePages * kPageBlocks;
    const uint32_t count = std::min(counts[index], stride);
    const uint32_t *entries = lists + uint64_t{index} * stride;
    uint32_t *out = locations + uint64_t{index} * stride;
    const uint32_t first = stateLanes[index] * slots_;
    ++lane.stats.resolves;
    stats_.entries += count;
    lane.stats.entries += count;
    for (uint32_t entry = 0; entry < count; ++entry) {
      const uint32_t block = entries[entry] >> SPLASH_ATTENTION_BLOCK_ROW_BITS;
      if (block >= inFile) {
        out[entry] = SPLASH_ATTENTION_PAGED;
        ++stats_.paged;
        ++lane.stats.paged;
        continue;
      }
      uint32_t &slotOf = lane.slotOf[uint64_t{layer} * maximumBlocks_ + block];
      uint32_t slot = slotOf;
      if (slot != kNone) {
        ++stats_.hits;
        ++lane.stats.hits;
      } else {
        // The oldest slot, which no entry of this step holds while the cache
        // has at least as many slots as a step selects blocks.
        slot = lane.oldest[layer];
        if (lane.used[base + slot] == stamp) throw std::logic_error("sparse KV cache smaller than its selection");
        if (lane.blockIn[base + slot] != kNone) lane.slotOf[uint64_t{layer} * maximumBlocks_ + lane.blockIn[base + slot]] = kNone;
        lane.blockIn[base + slot] = block;
        slotOf = slot;
        misses_.push_back({stateLanes[index], block, slot});
      }
      lane.used[base + slot] = stamp;
      lane.touch(layer, static_cast<uint32_t>(base), slot);
      out[entry] = first + slot;
    }
  }
  if (!misses_.empty()) read(layer, misses_);
  const double seconds = secondsSince(start);
  stats_.resolveSeconds += seconds;
  for (const uint32_t stateLane : stateLanes) laneState_[stateLane]->stats.resolveSeconds += seconds;
}

// Reads each missed block's record into its slot: the 4 KiB-aligned run that
// holds it, then each tensor's run of every head into the slot's page.
void SparseKvStore::read(uint32_t layer, std::span<const Miss> misses) {
  const auto start = AwakeClock::now();
  std::atomic<uint64_t> bytes = 0;
  readers_->run(misses.size(), [&](size_t index, std::byte *staging) {
    const Miss &miss = misses[index];
    const uint64_t offset = recordOffset(layer, miss.block);
    const uint64_t begin = offset / kAlignment * kAlignment;
    const uint64_t size = alignUp(offset + recordBytes_) - begin;
    readFully(laneState_[miss.lane]->file, staging, size, begin);
    bytes += size;
    const std::byte *record = staging + (offset - begin);
    const uint32_t page = miss.lane * lanePages_ + miss.slot / kPageBlocks;
    const uint32_t place = miss.slot % kPageBlocks;
    std::byte *const *tensors = cacheTensors_.data() + (uint64_t{page} * layout_.attentionLayers + layer) * 4;
    for (uint32_t tensor = 0; tensor < 4; ++tensor) {
      if (!headBytes_[tensor]) continue;
      const uint64_t perHead = uint64_t{kPageTokens} * (headBytes_[tensor] / kBlockTokens);
      for (uint32_t head = 0; head < layout_.kvHeads; ++head) {
        std::memcpy(tensors[tensor] + head * perHead + place * headBytes_[tensor], record, headBytes_[tensor]);
        record += headBytes_[tensor];
      }
    }
  });
  const double seconds = secondsSince(start);
  stats_.reads += misses.size();
  stats_.readBytes += bytes;
  stats_.readSeconds += seconds;
  // Each lane's reads, and the time of the reads it took part in.
  uint32_t last = kNone;
  for (const Miss &miss : misses) {
    SparseKvStats &stats = laneState_[miss.lane]->stats;
    ++stats.reads;
    stats.readBytes += alignUp(recordOffset(layer, miss.block) % kAlignment + recordBytes_);
    if (miss.lane != last) stats.readSeconds += seconds;
    last = miss.lane;
  }
}

} // namespace splash::model
