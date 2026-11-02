#pragma once

// What a store of streamed routed experts reports (ExpertStore::stats,
// /status), apart from the store, whose header brings Metal and the MoE
// operators with it.

#include "model/LatencyHistogram.hpp"

#include <cstdint>
#include <string>

namespace splash::model {

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
  // The claims of experts the streamed sizes' recycling gave up lately
  // (SlotRecycler::ghostHits), and the slots those ghosts would take now.
  uint64_t ghostHits = 0;
  uint64_t ghostBytes = 0;
  // The hits one elastic chunk decides (SlotRecycler::nearGhostHits,
  // tailHits): ghost hits a chunk more would have served, and hits on the
  // experts a chunk less would have given up first.
  uint64_t nearGhostHits = 0;
  uint64_t tailHits = 0;
  // Per ops::MoePhase, at its value (uses[0] prefill, uses[1] decode): the
  // blocks loaded, the distinct experts they routed, those not resident and
  // of them those read ahead, the experts read ahead whole (routed or not),
  // the bytes read from the sources (ahead or not, routed or not), and the
  // round trips of the loads' host steps, one per round (metal::
  // HostStepTiming: the GPU's work before a step to the runner taking it up,
  // its work, its end to the GPU's work after it).
  struct Use final {
    uint64_t loads = 0;
    uint64_t routed = 0;
    uint64_t read = 0;
    uint64_t prefetched = 0;
    uint64_t readAhead = 0;
    uint64_t readBytes = 0;
    LatencyStats pickup, work, resume;
  };
  Use uses[2];
  // Why a load failed, which fails every later load too; empty while none
  // has.
  std::string failure;
};

} // namespace splash::model
