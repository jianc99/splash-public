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

constexpr uint64_t kQsaKeyBytes = QWEN4_QSA_DIMENSION * sizeof(uint16_t);
constexpr uint64_t kQsaPendingBytes = QWEN4_QSA_PENDING * kQsaKeyBytes;

// The parameters every QSA dispatch of `rows`' lanes shares.
Qwen4QsaParams qsaParams(const QsaRows &rows) {
  if (!rows.rows || rows.firstPositions.empty() || rows.firstPositions.size() > SPLASH_MAXIMUM_BATCH_WIDTH ||
      rows.index.tables.size() != rows.firstPositions.size())
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
                        .index_offset = rows.index.offset};
  std::ranges::copy(rows.firstPositions, std::begin(params.first_position));
  return params;
}

} // namespace

void Qwen4::addQsaProject(metal::CommandGraph &graph, metal::MetalBuffer query, metal::MetalBuffer key,
                          const NormWeights &queryNorm, metal::MetalBuffer ropeCos, metal::MetalBuffer ropeSin,
                          metal::MetalBuffer queries, metal::MetalBuffer keys, const QsaRows &rows) {
  const Qwen4QsaParams params = qsaParams(rows);
  if (!queryNorm.float32) throw std::invalid_argument("the QSA query norm is not F32");
  const uint64_t laneRows = uint64_t{params.lanes} * rows.rows;
  requireBytes(query, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(float), "QSA query");
  requireBytes(key, laneRows * QWEN4_QSA_DIMENSION * sizeof(float), "QSA key");
  requireBytes(queryNorm.buffer, QWEN4_QSA_DIMENSION * sizeof(float), "QSA query norm");
  requireBytes(ropeCos, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE cosine");
  requireBytes(ropeSin, laneRows * SPLASH_TARGET_ROPE_PAIRS * sizeof(float), "QSA RoPE sine");
  requireBytes(queries, laneRows * QWEN4_QSA_HEADS * QWEN4_QSA_DIMENSION * sizeof(uint16_t), "QSA queries");
  requireBytes(keys, laneRows * kQsaKeyBytes, "QSA keys");
  graph.add("qwen4_qsa_project",
            {std::move(query), std::move(key), queryNorm.buffer, std::move(ropeCos), std::move(ropeSin),
             std::move(queries), std::move(keys)},
            params, {rows.rows, QWEN4_QSA_HEADS + 1, params.lanes}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaPool(metal::CommandGraph &graph, const QsaBlocks &blocks, const NormWeights &keyNorm,
                       metal::MetalBuffer inverseFrequencies, const QsaRows &rows) {
  Qwen4QsaParams params = qsaParams(rows);
  if (blocks.counts.size() != params.lanes || blocks.pending.size() != params.lanes ||
      (params.lanes > 1 && !blocks.laneRecords))
    throw std::invalid_argument("invalid QSA blocks");
  params.lane_records = blocks.laneRecords;
  uint32_t most = 0;
  for (uint32_t lane = 0; lane < params.lanes; ++lane) {
    const uint32_t count = blocks.counts[lane];
    if (count && params.lanes > 1 && count > blocks.laneRecords)
      throw std::invalid_argument("a QSA lane's blocks overrun the next lane's");
    // A lane's rows complete at most the blocks their last row ends, the
    // first of them reaching back into its pending keys.
    if (count > qsaCompleteBlocks(rows.firstPositions[lane], rows.rows) - rows.firstPositions[lane] / QWEN4_QSA_BLOCK)
      throw std::invalid_argument("a QSA lane's blocks are not its rows'");
    requireBytes(blocks.records, (uint64_t{blocks.laneRecords} * lane + count) * sizeof(Qwen4QsaBlock),
                 "QSA block");
    requireBytes(blocks.pending[lane], kQsaPendingBytes, "QSA pending keys");
    params.counts[lane] = count;
    most = std::max(most, count);
  }
  if (!most) return;
  requireBytes(blocks.keys, uint64_t{params.lanes} * rows.rows * kQsaKeyBytes, "QSA keys");
  const auto pending = laneBindings(blocks.pending);
  const auto tables = laneBindings(rows.index.tables);
  graph.add("qwen4_qsa_pool",
            {blocks.records, blocks.keys, pending[0], pending[1], pending[2], pending[3], keyNorm.buffer,
             std::move(inverseFrequencies), tables[0], tables[1], tables[2], tables[3]},
            params, {most, params.lanes, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaSelect(metal::CommandGraph &graph, metal::MetalBuffer queries, metal::MetalBuffer scores,
                         uint32_t scoreRows, metal::MetalBuffer mask, const QsaRows &rows) {
  Qwen4QsaParams params = qsaParams(rows);
  if (!scoreRows || !rows.topBlocks || !rows.maskWords ||
      (params.lanes > 1 && scoreRows < uint64_t{params.lanes} * rows.rows))
    throw std::invalid_argument("invalid QSA selection");
  const auto tables = laneBindings(rows.index.tables);
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

void Qwen4::addQsaPending(metal::CommandGraph &graph, metal::MetalBuffer keys, metal::MetalBuffer pendingIn,
                          metal::MetalBuffer pendingOut, uint32_t firstPosition, uint32_t retained) {
  requireBytes(keys, uint64_t{retained} * kQsaKeyBytes, "QSA keys");
  requireBytes(pendingIn, kQsaPendingBytes, "QSA pending keys");
  requireBytes(pendingOut, kQsaPendingBytes, "QSA pending keys");
  const Qwen4QsaPendingParams params{
      .lanes = 1, .rows = retained, .layer_rows = 0, .first_position = {firstPosition}, .retained = {retained}};
  graph.add("qwen4_qsa_pending", {std::move(keys), std::move(pendingIn), std::move(pendingOut)}, params, {1, 1, 1},
            {QWEN4_QSA_DIMENSION, 1, 1});
}

void Qwen4::addQsaCommit(metal::CommandGraph &graph, metal::MetalBuffer keys, uint32_t layers, uint32_t layerRows,
                         uint32_t laneRows, std::span<const uint32_t> firstPositions, metal::MetalBuffer retained,
                         std::span<const metal::MetalBuffer> pendingIn,
                         std::span<const metal::MetalBuffer> pendingOut) {
  const auto lanes = static_cast<uint32_t>(firstPositions.size());
  if (!layers || !laneRows || !lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH || pendingIn.size() != lanes ||
      pendingOut.size() != lanes || uint64_t{lanes} * laneRows > layerRows)
    throw std::invalid_argument("invalid QSA commit");
  requireBytes(keys, uint64_t{layers} * layerRows * kQsaKeyBytes, "QSA keys");
  requireBytes(retained, uint64_t{lanes} * sizeof(uint32_t), "QSA retained rows");
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    requireBytes(pendingIn[lane], layers * kQsaPendingBytes, "QSA pending keys");
    requireBytes(pendingOut[lane], layers * kQsaPendingBytes, "QSA pending keys");
  }
  Qwen4QsaPendingParams params{
      .lanes = lanes, .rows = laneRows, .layer_rows = layerRows, .first_position = {}, .retained = {}};
  std::ranges::copy(firstPositions, std::begin(params.first_position));
  const auto in = laneBindings(pendingIn), out = laneBindings(pendingOut);
  graph.add("qwen4_qsa_commit",
            {std::move(keys), std::move(retained), in[0], in[1], in[2], in[3], out[0], out[1], out[2], out[3]},
            params, {layers, lanes, 1}, {QWEN4_QSA_DIMENSION, 1, 1});
}

} // namespace splash::ops
