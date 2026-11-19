#pragma once

// Parameter layout of the dense attention (kernels/shared/dense_attention.metal)
// and its host dispatch (ops/DenseAttention.hpp).
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

// Query rows of one head a dense_attention threadgroup takes, its threads
// (four simdgroups whose softmax gives each row four lanes; a static_assert
// in shared/dense_attention.metal ties the two together), and the keys of
// one key tile, which a threadgroup reads whole.
#define SPLASH_DENSE_ATTENTION_ROWS 32u
#define SPLASH_DENSE_ATTENTION_THREADS 128u
#define SPLASH_DENSE_ATTENTION_KEYS 128u

// The attention of `rows` rows, each over every one of their keys and
// values. Element (head, row, dimension) of a tensor lies head * its head
// stride + row * its row stride + dimension elements from its buffer's
// start: the head strides here, the row strides the kernel's function
// constants.
struct DenseAttentionParams {
  uint32_t rows;
  float scale;
  uint32_t query_head_stride;
  uint32_t key_head_stride;
  uint32_t value_head_stride;
  uint32_t output_head_stride;
};

static_assert(sizeof(DenseAttentionParams) == 24,
              "dense attention parameters are 24 bytes on both sides");
