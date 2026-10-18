#pragma once

// PROTOTYPE (night/dflash-verify-len, measurement only): how many of each
// lane's SPLASH_TARGET_VERIFY_ROWS verify rows the decode cycle being encoded
// verifies. Runtime::decodeAsync sets the cycle's rule before it encodes a
// cycle: the draft selector (draft_select_dflash) writes each lane's rows into
// the cycle's verify-rows buffer by that rule, the MoE grouping routes the
// rows below them alone, acceptance retains at most that many tokens, and the
// expert store's read-ahead prediction counts those rows. Unless a probe
// control file says otherwise (SPLASH_PROTO_VERIFY_CTL), every lane verifies
// all its rows and the MoE steps route every row, as without the probe.

#include "metal/MetalBackend.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <cstdint>

namespace splash::ops {

using VerifyRows = std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH>;

// SelectorBatchParams' verify fields (metal/abi/Sampling.h).
struct VerifyRule final {
  float threshold = 0.0F;
  uint32_t cumulative = 0;
  uint32_t extra = 0;
  uint32_t minimum = 2;
  VerifyRows cap{SPLASH_TARGET_VERIFY_ROWS, SPLASH_TARGET_VERIFY_ROWS, SPLASH_TARGET_VERIFY_ROWS,
                 SPLASH_TARGET_VERIFY_ROWS};
  // Whether a lane may verify fewer than all its rows: the MoE grouping then
  // reads the verify-rows buffer.
  [[nodiscard]] bool limits() const noexcept {
    if (threshold > 0.0F) return true;
    for (const uint32_t rows : cap)
      if (rows < SPLASH_TARGET_VERIFY_ROWS) return true;
    return false;
  }
};

struct CycleVerify final {
  VerifyRule rule;
  // The cycle's verify rows, one uint32 per lane (shared storage), which the
  // selector writes; empty outside a decode cycle's encoding.
  metal::MetalBuffer rows;
};

inline CycleVerify &cycleVerify() noexcept {
  static thread_local CycleVerify cycle;
  return cycle;
}

// Whether decode row `row` (lane row / SPLASH_TARGET_VERIFY_ROWS) is past its
// lane's verified rows.
inline bool unverifiedRow(const VerifyRows &rows, uint64_t row) noexcept {
  const uint64_t lane = row / SPLASH_TARGET_VERIFY_ROWS;
  return lane < rows.size() && row % SPLASH_TARGET_VERIFY_ROWS >= rows[lane];
}

} // namespace splash::ops
