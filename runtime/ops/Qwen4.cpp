#include "ops/Qwen4.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Qwen4.h"
#include "metal/abi/RoPE.h"
#include "ops/BufferExtent.hpp"
#include "ops/LaneBindings.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace splash::ops {

namespace {

// The parameters every QSA dispatch of `rows`' lanes shares.
Qwen4QsaParams qsaParams(const QsaRows &rows, QsaIndex index) {
  if (!rows.rows || rows.firstPositions.empty() || rows.firstPositions.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      rows.pageTables.size() != rows.firstPositions.size())
    throw std::invalid_argument("invalid QSA rows");
  Qwen4QsaParams params{.lanes = static_cast<uint32_t>(rows.firstPositions.size()),
                        .rows = rows.rows,
                        .first_position = {},
                        .counts = {},
                        .lane_records = 0,
                        .row_offset = 0,
                        .top_blocks = rows.topBlocks,
                        .mask_words = rows.maskWords,
                        .score_stride = rows.scoreStride,
                        .keys_offset = index.keysOffset,
                        .pooled_offset = index.pooledOffset};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  return params;
}

} // namespace

void Qwen4::addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                          const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                          QsaIndex index, metal::MetalBuffer queries, const QsaRows &rows) {
  const Qwen4QsaParams params = qsaParams(rows, index);
  if (!queryNorm.float32) throw std::invalid_argument("the QSA query norm is not F32");
  const uint64_t laneRows = uint64_t{params.lanes} * rows.rows;
  requireBytes(query, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(float), "QSA query");
  requireBytes(key, laneRows * QWEN4_QSA_DIMENSION * sizeof(float), "QSA key");
  requireBytes(queryNorm.buffer, QWEN4_QSA_DIMENSION * sizeof(float), "QSA query norm");
  requireBytes(ropeCos, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE cosine");
  requireBytes(ropeSin, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE sine");
  requireBytes(queries, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(uint16_t), "QSA queries");
  const auto tables = laneBindings(rows.pageTables);
  graph.add("qwen4_qsa_project",
            {std::move(query), std::move(key), queryNorm.buffer, std::move(ropeCos), std::move(ropeSin), tables[0],
             tables[1], tables[2], tables[3], std::move(queries)},
            params, {rows.rows, QWEN4_QSA_HEADS + 1, params.lanes}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                       metal::MetalBuffer inverseFrequencies, QsaIndex index) {
  if (blocks.counts.empty() || blocks.counts.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      blocks.pageTables.size() != blocks.counts.size() || (blocks.counts.size() > 1 && !blocks.laneRecords))
    throw std::invalid_argument("invalid QSA blocks");
  Qwen4QsaParams params{.lanes = static_cast<uint32_t>(blocks.counts.size()),
                        .rows = 0,
                        .first_position = {},
                        .counts = {},
                        .lane_records = blocks.laneRecords,
                        .row_offset = 0,
                        .top_blocks = 0,
                        .mask_words = 0,
                        .score_stride = 0,
                        .keys_offset = index.keysOffset,
                        .pooled_offset = index.pooledOffset};
  uint32_t most = 0;
  for (size_t lane = 0; lane < blocks.counts.size(); ++lane) {
    const uint32_t count = blocks.counts[lane];
    if (count && blocks.counts.size() > 1 && count > blocks.laneRecords)
      throw std::invalid_argument("a QSA lane's blocks overrun the next lane's");
    requireBytes(blocks.records, (uint64_t{blocks.laneRecords} * lane + count) * sizeof(Qwen4QsaBlock),
                 "QSA block");
    params.counts[lane] = count;
    most = std::max(most, count);
  }
  if (!most) return;
  const auto tables = laneBindings(blocks.pageTables);
  graph.add("qwen4_qsa_pool",
            {blocks.records, keyNorm.buffer, std::move(inverseFrequencies), tables[0], tables[1], tables[2],
             tables[3]},
            params, {most, params.lanes, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, QsaIndex index,
                         metal::MetalBuffer scores, uint32_t scoreRows, metal::MetalBuffer mask,
                         const QsaRows &rows) {
  Qwen4QsaParams params = qsaParams(rows, index);
  if (!scoreRows || !rows.topBlocks || !rows.maskWords ||
      (params.lanes > 1 && scoreRows < uint64_t{params.lanes} * rows.rows))
    throw std::invalid_argument("invalid QSA selection");
  const auto tables = laneBindings(rows.pageTables);
  uint32_t blocks = 0;
  for (const uint32_t first : rows.firstPositions)
    blocks = std::max(blocks, static_cast<uint32_t>(qsaCompleteBlocks(first, rows.rows)));
  // A row's scores hold each complete block before it, and its bitmap those
  // and its tail block: the kernels write no further.
  if (rows.scoreStride < blocks || uint64_t{rows.maskWords} * 32 <= blocks)
    throw std::invalid_argument("QSA scores or bitmaps narrower than the rows' history");
  const uint64_t laneRows = uint64_t{params.lanes} * rows.rows;
  requireBytes(queries, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(uint16_t), "QSA queries");
  requireBytes(scores, (params.lanes > 1 ? laneRows : std::min(scoreRows, rows.rows)) * uint64_t{rows.scoreStride} *
                           sizeof(float),
               "QSA score");
  requireBytes(mask, laneRows * rows.maskWords * sizeof(uint32_t), "QSA bitmap");
  // One lane's rows in batches of the scratch's rows; several lanes' at once.
  const uint32_t batchRows = params.lanes > 1 ? rows.rows : scoreRows;
  for (uint32_t offset = 0; offset < rows.rows; offset += batchRows) {
    params.row_offset = offset;
    params.rows = std::min(batchRows, rows.rows - offset);
    graph.add("qwen4_qsa_score", {queries, tables[0], tables[1], tables[2], tables[3], scores}, params,
              {(blocks + QWEN4_QSA_SCORE_BLOCKS - 1) / QWEN4_QSA_SCORE_BLOCKS,
               (params.rows + QWEN4_QSA_SCORE_ROWS - 1) / QWEN4_QSA_SCORE_ROWS, params.lanes},
              {QWEN4_QSA_SCORE_THREADS, 1, 1});
    graph.add("qwen4_qsa_select", {scores, mask}, params, {params.rows, params.lanes, 1},
              {QWEN4_QSA_SELECT_THREADS, 1, 1});
  }
}

} // namespace splash::ops
