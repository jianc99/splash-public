#pragma once

// Parameters of Qwen3.8-Flash-Next's QSA indexer (kernels/shared/qwen4_qsa.metal,
// ops/Qwen4.cpp), whose block bitmaps are the block masks of its
// full-attention layers' sparse attention.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

// QSA (Qwen Sparse Attention) of a full-attention layer: the indexer's
// QWEN4_QSA_HEADS query heads and one key head of QWEN4_QSA_DIMENSION values;
// blocks of QWEN4_QSA_BLOCK tokens, those of the sparse attention's block
// masks (metal/abi/PagedAttention.h); `top_blocks` blocks per query row past
// which the row attends sparsely. Each KV page also holds, per layer, the raw
// indexer keys of its SPLASH_TARGET_KV_BLOCK_TOKENS tokens (bf16 [token]
// [dimension]) and the pooled keys of its blocks (bf16 [block][dimension]) in
// its extent's index regions (ops/PageStorage indexPlacement), which kernels
// reach through the page's table entry and the regions' byte offsets.
#define QWEN4_QSA_DIMENSION 128u
#define QWEN4_QSA_HEADS 4u
#define QWEN4_QSA_BLOCK SPLASH_ATTENTION_BLOCK_TOKENS
// The score kernel's tiles: QWEN4_QSA_SCORE_ROWS rows by
// QWEN4_QSA_SCORE_BLOCKS blocks, a threadgroup of QWEN4_QSA_SCORE_THREADS
// threads each.
#define QWEN4_QSA_SCORE_ROWS 8u
#define QWEN4_QSA_SCORE_BLOCKS 32u
#define QWEN4_QSA_SCORE_THREADS 256u
// The select kernel's threads, a threadgroup per row.
#define QWEN4_QSA_SELECT_THREADS 1024u
// A QSA dispatch serves `lanes` lanes, each through its own page table: a
// verify step's lanes, or a prefill sequence as one. Lane l's `rows` rows
// follow the rows of the lanes before it in the queries, the scores and the
// bitmaps, the first at logical position first_position[l] + row_offset. A
// dispatch of one lane selects a long sequence's rows in batches of the score
// scratch's rows: the batch from row `row_offset` of its queries and bitmaps,
// scored into the scratch's first rows. The pool pools lane l's counts[l]
// blocks, its records lane_records records apart.
struct Qwen4QsaParams {
  uint32_t lanes;
  uint32_t rows;
  uint32_t first_position[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t counts[SPLASH_MAXIMUM_BATCH_WIDTH];
  uint32_t lane_records;
  uint32_t row_offset;
  uint32_t top_blocks;
  uint32_t mask_words;    // 32-bit words of a row's block bitmap
  uint32_t score_stride;  // scores per row of the score scratch
  uint32_t keys_offset;   // the layer's raw-key region in every extent, in bytes
  uint32_t pooled_offset; // the layer's pooled-key region in every extent, in bytes
};

static_assert(sizeof(Qwen4QsaParams) == 36 + 8 * SPLASH_MAXIMUM_BATCH_WIDTH,
              "Qwen4 QSA parameters are the same size on both sides");

// One block to pool: its first token's logical position and RoPE position
// (t, h, w; text repeats one), to which its pooled key is rotated.
struct Qwen4QsaBlock {
  uint32_t position;
  uint32_t rope[3];
};

static_assert(sizeof(Qwen4QsaBlock) == 16,
              "Qwen4 QSA blocks are 16 bytes on both sides");
