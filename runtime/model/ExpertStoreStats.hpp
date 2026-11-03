#pragma once

// What a store of streamed routed experts reports (ExpertStore::stats,
// /status), apart from the store, whose header brings Metal and the MoE
// operators with it.

#include "model/LatencyHistogram.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace splash::model {

// The loads' host-step work against the experts each read: the least-squares
// line through every load's (reads, work), as its count, means and centred
// sums. Its slope is the work one read more adds with the store's readers
// overlapping, the marginal read; its intercept, a load's own work.
struct ReadWork final {
  uint64_t loads = 0;
  double reads = 0.0, milliseconds = 0.0;   // means
  double readSpread = 0.0, coSpread = 0.0;  // sum (r - mean r)^2 and sum (r - mean r)(w - mean w)

  void add(double read, double work) noexcept {
    ++loads;
    const double off = read - reads;
    reads += off / double(loads);
    milliseconds += (work - milliseconds) / double(loads);
    readSpread += off * (read - reads);
    coSpread += off * (work - milliseconds);
  }
  // The marginal read in ms, never below zero; none until loads of different
  // reads identify it.
  [[nodiscard]] std::optional<double> marginalMilliseconds() const noexcept {
    if (readSpread <= 0.0) return std::nullopt;
    return std::max(0.0, coSpread / readSpread);
  }
};

struct ExpertStoreStats final {
  // The store at its cap, at its floor and as it is, as ExpertStoreOptions
  // counts them (its staging and tables too), its chunks at the cap and
  // those live. The store as it is, the live chunks, their bytes and their
  // slots are zero while the store is released.
  uint64_t capBytes = 0;
  uint64_t floorBytes = 0;
  uint64_t liveBytes = 0;
  uint64_t capChunks = 0;
  uint64_t chunks = 0;
  uint64_t arenaBytes = 0; // the live chunks of every size
  uint64_t stagingBytes = 0;
  uint64_t slots = 0;    // the live chunks'
  uint64_t experts = 0;  // every block's routed experts
  uint64_t resident = 0; // slots holding an expert
  // The chunks given back while memory was short, and taken back after.
  uint64_t shrinks = 0;
  uint64_t regrows = 0;
  // Their bytes given back, and the engine thread's time giving them back
  // took, mostly the compaction of their experts below them.
  uint64_t releasedBytes = 0;
  double releaseMilliseconds = 0.0;
  // The bytes releaseAboveFloor() would give back next and growTowardCap()
  // would take: zero at the floor and at the cap, while released and after
  // a failed load.
  uint64_t releaseUnitBytes = 0;
  uint64_t growUnitBytes = 0;
  // The bytes of every chunk above the floors, in the order
  // releaseAboveFloor() gives them back (the first is releaseUnitBytes).
  std::vector<uint64_t> releaseUnits;
  // The claims of experts the streamed sizes' recycling gave up lately
  // (SlotRecycler::ghostHits), and the slots those ghosts would take now.
  uint64_t ghostHits = 0;
  uint64_t ghostBytes = 0;
  // The hits one elastic chunk decides (SlotRecycler::nearGhostHits,
  // tailHits): ghost hits a chunk more would have served, and hits on the
  // experts a chunk less would have given up first; and the bytes of the
  // slots they count in, an elastic chunk's of each streamed size.
  uint64_t nearGhostHits = 0;
  uint64_t tailHits = 0;
  uint64_t marginBytes = 0;
  // Per ops::MoePhase, at its value (uses[0] prefill, uses[1] decode): the
  // blocks loaded, the distinct experts they routed, those not resident and
  // of them those read ahead, the experts read ahead whole (routed or not),
  // the bytes read from the sources (ahead or not, routed or not), and the
  // round trips of the loads' host steps, one per round (metal::
  // HostStepTiming: the GPU's work before a step to the runner taking it up,
  // its work, its end to the GPU's work after it), and the line of each
  // load's work against what it read.
  struct Use final {
    uint64_t loads = 0;
    uint64_t routed = 0;
    uint64_t read = 0;
    uint64_t prefetched = 0;
    uint64_t readAhead = 0;
    uint64_t readBytes = 0;
    LatencyStats pickup, work, resume;
    ReadWork readWork;

    // What a miss costs the loads: the marginal read (DESIGN section 0,
    // owner 11-03 00:35: what a read adds with the readers overlapping, not
    // the mean work a read), the mean until loads identify it.
    [[nodiscard]] double missMilliseconds() const noexcept {
      return readWork.marginalMilliseconds().value_or(read ? work.totalMilliseconds / double(read) : 0.0);
    }
    // A load's own work besides its reads at that price: the line's at no
    // read through the loads' means (about 0 while the mean prices them).
    [[nodiscard]] double loadMilliseconds() const noexcept {
      return readWork.milliseconds - missMilliseconds() * readWork.reads;
    }
  };
  Use uses[2];
  // Why a load failed, which fails every later load too; empty while none
  // has.
  std::string failure;
};

} // namespace splash::model
