#pragma once

// PROTOTYPE (night/dflash-verify-len, measurement only): how many of each
// lane's SPLASH_TARGET_VERIFY_ROWS verify rows the decode cycle being encoded
// verifies. Runtime::decodeAsync sets it before it encodes a cycle; the MoE
// grouping (rows past a lane's count route no expert) and the expert store's
// read-ahead prediction read it while that cycle encodes. Every lane verifies
// all its rows unless a probe control file says otherwise
// (SPLASH_PROTO_VERIFY_CTL), so a build that never sets it is unchanged.

#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <cstdint>

namespace splash::ops {

using VerifyRows = std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH>;

inline VerifyRows &cycleVerifyRows() noexcept {
  static thread_local VerifyRows rows{SPLASH_TARGET_VERIFY_ROWS, SPLASH_TARGET_VERIFY_ROWS,
                                      SPLASH_TARGET_VERIFY_ROWS, SPLASH_TARGET_VERIFY_ROWS};
  return rows;
}

// Whether decode row `row` (lane row / SPLASH_TARGET_VERIFY_ROWS) is past its
// lane's verified rows.
inline bool unverifiedRow(const VerifyRows &rows, uint64_t row) noexcept {
  const uint64_t lane = row / SPLASH_TARGET_VERIFY_ROWS;
  return lane < rows.size() && row % SPLASH_TARGET_VERIFY_ROWS >= rows[lane];
}

} // namespace splash::ops
