#pragma once

// Qwen3.8-Flash-Next's QSA indexer, which picks the blocks its
// full-attention layers attend (kernels/shared/qwen4_qsa.metal). Its
// projections run through ops::Linear; model/Qwen4ExpTarget.cpp composes
// them.

#include "metal/CommandGraph.hpp"
#include "metal/abi/Qwen4.h"
#include "ops/Normalization.hpp"

#include <cstdint>
#include <span>

namespace splash::ops {

// QSA of a dispatch's lanes (a verify step's, or a prefill sequence as one):
// `rows` query rows of each lane, lane l's from logical position
// firstPositions[l] and its KV through pageTables[l], which attend sparsely
// past topBlocks complete blocks; bitmaps of maskWords words and scores of
// scoreStride per row, which cover each row's complete blocks and its tail.
struct QsaRows final {
  uint32_t rows = 0;
  std::span<const uint32_t> firstPositions;
  std::span<const metal::MetalBuffer> pageTables;
  uint32_t topBlocks = 0;
  uint32_t maskWords = 0;
  uint32_t scoreStride = 0;
};

// The QSA blocks a dispatch's lanes complete, which the pool pools: lane l's
// counts[l] Qwen4QsaBlock records from record l * laneRecords of `records`,
// its KV through pageTables[l].
struct QsaBlocks final {
  metal::MetalBuffer records;
  std::span<const uint32_t> counts;
  uint32_t laneRecords = 0;
  std::span<const metal::MetalBuffer> pageTables;
};

// Where a layer's QSA index tensors sit in every KV extent, in bytes
// (kv::PageStorage::indexPlacement). Page tables hold SplashKvPage entries.
struct QsaIndex final {
  uint32_t keysOffset = 0;
  uint32_t pooledOffset = 0;
};

struct Qwen4 final {
  // The complete blocks of QWEN4_QSA_BLOCK tokens before the last of `rows`
  // rows from `firstPosition`, its own included.
  [[nodiscard]] static constexpr uint64_t qsaCompleteBlocks(uint64_t firstPosition, uint32_t rows) noexcept {
    return (firstPosition + rows) / QWEN4_QSA_BLOCK;
  }
  // Whether any row of `rows` from `firstPosition` attends sparsely: has
  // more than topBlocks complete blocks.
  [[nodiscard]] static bool qsaSparse(uint32_t firstPosition, uint32_t rows, uint32_t topBlocks) noexcept {
    return qsaCompleteBlocks(firstPosition, rows) > topBlocks;
  }
  // The indexer's query heads, normalized and rotated (bf16 rows of
  // QWEN4_QSA_HEADS heads), and the raw keys stored at the rows' slots.
  static void addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                            const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                            QsaIndex index, metal::MetalBuffer queries, const QsaRows &rows);
  // The pooled keys of the lanes' blocks.
  static void addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                         metal::MetalBuffer inverseFrequencies, QsaIndex index);
  // The rows' scores of their complete blocks' pooled keys and, from them,
  // their bitmaps. A dispatch of one lane scores its rows in batches of the
  // score scratch's scoreRows rows; one of several lanes scores every lane's
  // rows at once, which the scratch must hold.
  static void addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, QsaIndex index,
                           metal::MetalBuffer scores, uint32_t scoreRows, metal::MetalBuffer mask,
                           const QsaRows &rows);
};

} // namespace splash::ops
