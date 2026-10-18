#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Sampling.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

struct DraftSelectorWorkspace final {
  uint64_t partialIdsBytes = 0;
  uint64_t partialValuesBytes = 0;
  uint64_t candidatesBytes = 0;
  uint64_t unaryBytes = 0;
  uint64_t proposalProbabilitiesBytes = 0;
};

struct DraftSelectorBuffers final {
  // fp32 [rows][vocabulary].
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // Each lane's verify rows, a uint32 per lane (VerifyRowsPolicy).
  metal::MetalBuffer verifyRows;
};

// How many rows of its block each lane's verify verifies, which the selector
// writes (SelectorBatchParams in metal/abi/Sampling.h): the anchor and each
// proposal the lane reaches with a probability of at least `threshold` (the
// product of the selector probabilities of the proposals before it), and at
// least `minimum` rows; every row with a threshold of 0. A lane's rows never
// exceed its maximum (DraftSelector::add).
struct VerifyRowsPolicy final {
  float threshold = 0.0F;
  uint32_t minimum = 2;
  bool operator==(const VerifyRowsPolicy &) const = default;
};

// The draft's selector codebooks, which score the edge from a proposal
// position's predecessor candidate to each of its candidates.
struct DraftCodebooks final {
  metal::MetalBuffer predecessor;
  metal::MetalBuffer successor;
};

// The DFlash draft's proposal policy (draft_select_* in
// metal/kernels/decode/sampling.metal): each lane keeps the
// SPLASH_DRAFT_CANDIDATES most likely draft tokens of every proposal
// position, scores each candidate with its edge from the previous position's
// choice, and walks the SPLASH_DRAFT_PROPOSAL_TOKENS positions greedily or,
// for a sampling lane, drawing at its temperature; it writes each lane's
// verify rows by `verify`, at most maximumRows[lane] of them.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies, VerifyRowsPolicy verify,
           std::span<const uint32_t> maximumRows) const;

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
