#pragma once

// GGUF quantized projection parameters shared by host dispatch code and Metal
// kernels (kernels/shared/gguf_linear.metal).
#include "metal/abi/QuantFormat.h"

// Prefill tiles (pf kernels): the grid covers whole 128-row tiles of the chunk; the simdgroups of the last tile
// whose rows start past `rows` skip their matmuls.
struct GgufPrefillParams {
  uint32_t output_size; // columns of this segment
  uint32_t input_size;  // K
  uint32_t rows;        // rows of the chunk
  uint32_t out_stride;  // row stride of the destination (0 = output_size)
  uint32_t out_offset;  // first destination column of this segment
};
static_assert(sizeof(GgufPrefillParams) == 20, "GGUF prefill parameters are 20 bytes on both sides");

// Decode tiles of both families (kernels/decode/linear_gguf_sgmatrix.metal on
// Apple9, kernels/shared/gguf_linear.metal elsewhere): one tensor per dispatch,
// over the dispatch's 64-column tiles (grid.x) and K partitions (grid.y).
struct GgufDecodeParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions; 1 = no cross-threadgroup reduction
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column of the tensor
};
static_assert(sizeof(GgufDecodeParams) == 16, "GGUF decode parameters are 16 bytes on both sides");

// Decode of a fused projection: up to three column segments of any formats in
// one dispatch, tiles in segment order.
struct GgufDecodeFusedParams {
  uint32_t input_size;  // K
  uint32_t splits;      // K partitions of every segment
  uint32_t out_stride;  // columns of a destination row
  uint32_t cols[3];     // columns per segment; 0 past the last
  uint32_t fmt[3];      // GGUF_FMT_* per segment
  uint32_t offset[3];   // first destination column per segment
};
static_assert(sizeof(GgufDecodeFusedParams) == 48, "GGUF fused decode parameters are 48 bytes on both sides");

struct GgufEmbedParams {
  uint32_t rows;
  uint32_t vocabulary;
  uint32_t hidden;
};
static_assert(sizeof(GgufEmbedParams) == 12, "GGUF embedding parameters are 12 bytes on both sides");

// fp32 projection of a GGUF float tensor (kernels/shared/gguf_float.metal):
// out[r][out_offset + n] = sum_k x[r][k] * W[n][k] for rows r < rows, W as
// stored ([output_size][input_size] floats of ggml type GGUF_TYPE_F32).
#define GGUF_TYPE_F32 0u
struct GgufFloatParams {
  uint32_t rows;
  uint32_t input_size;  // K, a multiple of 8
  uint32_t output_size; // N, a multiple of 8
  uint32_t out_stride;  // columns of a destination row
  uint32_t out_offset;  // first destination column
};
static_assert(sizeof(GgufFloatParams) == 20, "GGUF float parameters are 20 bytes on both sides");

#define GGUF_EPILOGUE_NONE 0u
#define GGUF_EPILOGUE_RESIDUAL 1u
#define GGUF_EPILOGUE_UP_WITH_GATE 2u
