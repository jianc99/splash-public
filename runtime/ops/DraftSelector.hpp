#pragma once

#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "metal/abi/Sampling.h"
#include "ops/Sampling.hpp"
#include "ops/Weights.hpp"

#include <array>
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
  // fp32 [rows][the columns of the draft's vocabulary] (DraftVocabulary).
  metal::MetalBuffer logits;
  metal::MetalBuffer partialIds;
  metal::MetalBuffer partialValues;
  metal::MetalBuffer candidates;
  metal::MetalBuffer unary;
  metal::MetalBuffer selectorHidden;
  metal::MetalBuffer uniforms;
  metal::MetalBuffer proposedTokens;
  metal::MetalBuffer proposalProbabilities;
  // fp32 [lanes][SPLASH_DRAFT_PROPOSAL_TOKENS]: each lane's confidence at each
  // proposal position, the largest probability among the position's
  // candidates (VerifyRowsParams in metal/abi/Sampling.h).
  metal::MetalBuffer confidence;
};

// What a decode cycle's verify rows follow (DraftSelector::addVerifyRows,
// VerifyRowsParams): the acceptance model logit(p) = slope *
// logit(confidence) + intercepts[position], and costs[R], the cost of a cycle
// whose lanes verify R rows in all, for R from the lane count to eight rows a
// lane, in any unit. With everyRow, each lane verifies its maximum instead.
struct VerifyRowsModel final {
  static constexpr uint32_t kCosts = SPLASH_DRAFT_QUERY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH + 1;
  float slope = 1.0F;
  std::array<float, SPLASH_DRAFT_PROPOSAL_TOKENS> intercepts{};
  std::array<float, kCosts> costs{};
  bool everyRow = true;
};

// The tokens a draft proposes from, which its logits rows hold in order: the
// vocabulary's first `leading`, then its tokens from `trailing` to its end.
// The default holds every token.
struct DraftVocabulary final {
  uint32_t leading = 0;
  uint32_t trailing = 0;

  // The columns of its logits rows in a vocabulary of `vocabulary` tokens.
  [[nodiscard]] constexpr uint32_t columns(uint32_t vocabulary) const noexcept {
    return leading + (vocabulary - trailing);
  }
  [[nodiscard]] constexpr bool holds(uint32_t token) const noexcept { return token < leading || token >= trailing; }
  // The token of logits column `column`, as the selector reads it
  // (draft_select_top16_sharded, SelectorBatchParams).
  [[nodiscard]] constexpr uint32_t token(uint32_t column) const noexcept {
    return column < leading ? column : column - leading + trailing;
  }
  // The rows of a vocabulary head of `vocabulary` rows that write its logits
  // columns, in column order (Projection::rows): the head's view over it.
  [[nodiscard]] constexpr std::array<Projection::RowRange, 2> rows(uint32_t vocabulary) const noexcept {
    return {{{0, leading}, {trailing, vocabulary - trailing}}};
  }
  bool operator==(const DraftVocabulary &) const = default;
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
// confidence at each position as it walks.
class DraftSelector final {
public:
  explicit DraftSelector(uint32_t vocabulary);

  // Exact scratch/output bytes for that many proposal positions.
  [[nodiscard]] static DraftSelectorWorkspace workspace(uint32_t positions);

  // Proposals from `vocabulary`'s tokens, whose logits rows buffers.logits
  // holds.
  void add(metal::CommandGraph &graph, const DraftSelectorBuffers &buffers,
           const DraftCodebooks &codebooks, DraftVocabulary vocabulary, std::span<const uint32_t> anchors,
           std::span<const SamplingPolicy> policies) const;
  // After add: each lane's verify rows (a uint32 per lane in `verifyRows`) by
  // `model` from its confidences, at most maximumRows[lane] of them.
  static void addVerifyRows(metal::CommandGraph &graph, const metal::MetalBuffer &confidence,
                            const metal::MetalBuffer &verifyRows, const VerifyRowsModel &model,
                            std::span<const uint32_t> maximumRows);

private:
  uint32_t vocabulary_ = 0;
};

} // namespace splash::ops
