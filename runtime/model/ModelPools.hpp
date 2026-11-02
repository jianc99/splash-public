#pragma once

// The model's memories that follow the engine's (model::ElasticPool), over
// the mechanisms that hold them: the store of streamed experts (ExpertStore)
// and the hot block cache of spilled K/V (KvBlockCache). The runtime's own,
// the history's staging, the vision arena and the image embeddings, are
// Runtime::Impl's (Runtime::pools). An adapter only calls its mechanism's
// functions: none changes how the mechanism holds or gives back its memory.

#include "AwakeClock.hpp"
#include "model/ElasticPool.hpp"
#include "model/ExpertStore.hpp"
#include "model/KvBlockCache.hpp"

#include <algorithm>
#include <cstdint>

namespace splash::model {

// What a pool's release() gave back and the engine thread's time it took
// (PoolValue::releasedBytes, releaseMilliseconds), for a mechanism that does
// not count them itself.
class ReleaseCount final {
public:
  // Runs `release`, which returns the bytes it gave back.
  template <class Release> uint64_t operator()(Release release) {
    const AwakeClock::time_point started = AwakeClock::now();
    const uint64_t released = release();
    bytes_ += released;
    milliseconds_ += millisecondsSince(started);
    return released;
  }
  [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] double milliseconds() const noexcept { return milliseconds_; }

private:
  uint64_t bytes_ = 0;
  double milliseconds_ = 0.0;
};

// Gives units back with `releaseOne`, which returns a unit's bytes or zero
// once it has none to give, until their bytes reach `bytes`; returns them.
template <class ReleaseOne> uint64_t releaseUnits(uint64_t bytes, ReleaseOne releaseOne) {
  uint64_t released = 0;
  while (released < bytes) {
    const uint64_t unit = releaseOne();
    if (!unit) break;
    released += unit;
  }
  return released;
}

// The store of streamed experts. Its floor chunks, staging and tables are
// its floor; a unit is an elastic chunk, the last live one of the size with
// the most above its floor going first (ExpertStore::releaseAboveFloor), all
// of them in one compaction per size when the bytes asked reach them
// (releaseToFloor), and the next one of the size with the least above its
// floor coming back (growTowardCap). Released while idle, it holds nothing;
// whole, it would hold every block's experts in slots of the live chunks'
// mean bytes beside its staging and tables. Its value: the routed experts it
// held, decode's ghost hits and the bytes its ghosts would take, the
// host-step work a decode load took per expert it read and a slot's mean
// bytes, which a miss reads, and the bytes prefill loads read, whose misses
// leave no ghosts (SlotRecycler recycles a long chunk's loads first); the
// store counts what its shrinks gave back and their compactions' time
// itself (ExpertStoreStats).
class ExpertStorePool final : public ElasticPool {
public:
  explicit ExpertStorePool(ExpertStore &store) noexcept : store_(store) {}

  [[nodiscard]] PoolState state() const override {
    const ExpertStoreStats stats = store_.stats();
    return {.name = "experts",
            .floorBytes = stats.liveBytes ? stats.floorBytes : 0,
            .currentBytes = stats.liveBytes,
            .capBytes = stats.capBytes,
            .releaseUnitBytes = stats.releaseUnitBytes,
            .growUnitBytes = stats.growUnitBytes,
            .wholeBytes = stats.liveBytes - stats.arenaBytes + stats.experts * slotBytes(stats)};
  }
  [[nodiscard]] PoolValue value() const override {
    const ExpertStoreStats stats = store_.stats();
    uint64_t hits = 0;
    for (const ExpertStoreStats::Use &use : stats.uses) hits += use.routed - use.read;
    const ExpertStoreStats::Use &prefill = stats.uses[0], &decode = stats.uses[1];
    return {.hits = hits,
            .ghostHits = stats.ghostHits,
            .ghostBytes = stats.ghostBytes,
            .missMilliseconds = decode.read ? decode.work.totalMilliseconds / double(decode.read) : 0.0,
            .releasedBytes = stats.releasedBytes,
            .releaseMilliseconds = stats.releaseMilliseconds,
            .keyBytes = slotBytes(stats),
            .scanReadBytes = prefill.readBytes};
  }
  uint64_t release(uint64_t bytes) override {
    const ExpertStoreStats stats = store_.stats();
    if (stats.liveBytes && bytes >= stats.liveBytes - stats.floorBytes) return store_.releaseToFloor();
    return releaseUnits(bytes, [this] { return store_.releaseAboveFloor(); });
  }
  metal::AllocationResult grow(const metal::AllocationAdmission &admit) override {
    return store_.growTowardCap(admit);
  }

private:
  // A slot's mean bytes over the live chunks; 0 while released.
  static uint64_t slotBytes(const ExpertStoreStats &stats) noexcept {
    return stats.slots ? stats.arenaBytes / stats.slots : 0;
  }

  ExpertStore &store_;
};

// The hot block cache of spilled K/V. The floors of the lanes that decode
// over Index pages are its floor; a unit is an extent, given back above the
// floors (KvBlockCache::releaseAboveFloor) and taken toward the planned
// target while a lane holds a floor (growTowardTarget), which its floors may
// take it past. Its value: its resolves' hits and ghost hits, the bytes its
// ghosts would take, and the read time of a block it missed.
class HotBlocksPool final : public ElasticPool {
public:
  explicit HotBlocksPool(KvBlockCache &cache) noexcept : cache_(cache) {}

  [[nodiscard]] PoolState state() const override {
    const KvBlockCacheStats stats = cache_.stats();
    const uint64_t extent = cache_.extentBytes();
    return {.name = "hot_kv",
            .floorBytes = stats.floorBytes,
            .currentBytes = stats.bytes,
            .capBytes = std::max(stats.targetBytes, stats.bytes),
            .releaseUnitBytes = stats.bytes > stats.floorBytes ? extent : 0,
            .growUnitBytes = stats.lanes && stats.bytes < stats.targetBytes ? extent : 0};
  }
  [[nodiscard]] PoolValue value() const override {
    const KvBlockCacheStats stats = cache_.stats();
    return {.hits = stats.reads.hits,
            .ghostHits = stats.reads.ghostHits,
            .ghostBytes = stats.ghostBytes,
            .missMilliseconds = stats.reads.misses ? stats.read.totalMilliseconds / double(stats.reads.misses) : 0.0,
            .releasedBytes = released_.bytes(),
            .releaseMilliseconds = released_.milliseconds()};
  }
  uint64_t release(uint64_t bytes) override {
    return released_([&] { return releaseUnits(bytes, [this] { return cache_.releaseAboveFloor(); }); });
  }
  metal::AllocationResult grow(const metal::AllocationAdmission &admit) override {
    return cache_.growTowardTarget(admit);
  }

private:
  KvBlockCache &cache_;
  ReleaseCount released_;
};

} // namespace splash::model
