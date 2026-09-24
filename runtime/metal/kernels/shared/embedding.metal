#include "metal/abi/Gguf.h"
#include "metal/abi/KernelABI.h"

template <uint Hidden>
inline void q4_embedding_impl(device const uint *tokens,
                              device const uchar *weights,
                              device const bfloat *scales,
                              device const bfloat *biases,
                              device bfloat *output,
                              constant Q4EmbeddingParams &params, uint index,
                              uint grid_size) {
  constexpr uint QuantGroups = Hidden / 64;
  uint elements = params.rows * Hidden;
  for (uint element = index; element < elements; element += grid_size) {
    uint row = element / Hidden;
    uint dim = element % Hidden;
    uint raw_token = tokens[row];
    // Runtime validates every token. Keep the bounds guard local to the
    // storage table so a malformed direct operator call cannot read past it.
    uint token = raw_token < params.vocabulary_size ? raw_token : 0;
    uchar packed = weights[ulong(token) * (Hidden / 2) + dim / 2];
    float quantized = float((packed >> ((dim & 1) * 4)) & 15);
    ulong parameter = ulong(token) * QuantGroups + dim / 64;
    output[element] =
        bfloat(quantized * float(scales[parameter]) + float(biases[parameter]));
  }
}

#define Q4_EMBEDDING_ENTRY(Name, Hidden)                                      \
  kernel void Name(                                                          \
      device const uint *tokens [[buffer(0)]],                               \
      device const uchar *weights [[buffer(1)]],                             \
      device const bfloat *scales [[buffer(2)]],                             \
      device const bfloat *biases [[buffer(3)]],                             \
      device bfloat *output [[buffer(4)]],                                   \
      constant Q4EmbeddingParams &params [[buffer(5)]],                      \
      uint index [[thread_position_in_grid]],                                \
      uint grid_size [[threads_per_grid]]) {                                 \
    q4_embedding_impl<Hidden>(tokens, weights, scales, biases, output,       \
                              params, index, grid_size);                     \
  }

Q4_EMBEDDING_ENTRY(embedding_q4_h5120, 5120)
Q4_EMBEDDING_ENTRY(embedding_q4_h2048, 2048)
#undef Q4_EMBEDDING_ENTRY

// Token gathers from native GGUF rows (ops/Embedding.cpp): a row is
// hidden / Weights blocks of Bytes bytes, each laid out as the format's ggml
// block_* at the byte offsets its struct names. Every format's value keeps the
// source order of its float operations (reassociate(off)), as the GGUF GEMMs do.

// The little-endian half at byte `at` of a block.
inline half gguf_half(device const uchar *block, uint at) {
  return as_type<half>(ushort(block[at] | (block[at + 1] << 8)));
}

// block_q4_K: half d | half dmin | uchar scales[12] | uchar qs[128], eight
// 32-weight groups with 6-bit scales and mins.
struct GgufEmbedQ4K {
  enum : uint { Weights = 256, Bytes = 144, D = 0, DMin = 2, Scales = 4, Codes = 16 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint j = (dim % Weights) / 32, l = dim % 32;
    const half d = gguf_half(block, D), dmin = gguf_half(block, DMin);
    device const uchar *sc = block + Scales; uchar m, s;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; } else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    const uchar q = (block[Codes + (j / 2) * 32 + l] >> ((j % 2) * 4)) & 15;
    return bfloat(float(d) * float(s) * float(q) - float(dmin) * float(m));
  }
};
// block_q6_K: uchar ql[128] | uchar qh[64] | int8 scales[16] | half d; codes
// are 6-bit with zero point 32.
struct GgufEmbedQ6K {
  enum : uint { Weights = 256, Bytes = 210, Low = 0, High = 128, Scales = 192, D = 208, Zero = 32 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const uint l = dim % Weights, n = l / 128, r = l % 128, quarter = r / 32, pos = r % 32;
    const uchar lo = (block[Low + n * 64 + (quarter & 1) * 32 + pos] >> ((quarter >> 1) * 4)) & 15;
    const uchar hi = (block[High + n * 32 + pos] >> (2 * quarter)) & 3;
    const char sc = as_type<char>(block[Scales + n * 8 + 2 * quarter + pos / 16]);
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(sc) * float(int(lo | (hi << 4)) - int(Zero)));
  }
};
// block_q8_0: half d | int8 qs[32].
struct GgufEmbedQ80 {
  enum : uint { Weights = 32, Bytes = 34, D = 0, Codes = 2 };
  __attribute__((always_inline)) static bfloat value(device const uchar *block, uint dim) {
#pragma clang fp reassociate(off)
    const half d = gguf_half(block, D);
    return bfloat(float(d) * float(as_type<char>(block[Codes + dim % Weights])));
  }
};

// Inlined, with each format's value, so every gather stays one function (the
// compiler otherwise keeps Q6_K's as a call).
template <class F>
__attribute__((always_inline)) inline void gguf_embedding(device const uint *tokens, device const uchar *table,
                                                         device bfloat *output, constant GgufEmbedParams &p,
                                                         uint index) {
  const uint elements = p.rows * p.hidden; if (index >= elements) return;
  const uint row = index / p.hidden, dim = index % p.hidden;
  uint token = tokens[row]; token = token < p.vocabulary ? token : 0;
  device const uchar *block = table + (ulong(token) * (p.hidden / F::Weights) + dim / F::Weights) * F::Bytes;
  output[index] = F::value(block, dim);
}

#define GGUF_EMBEDDING_ENTRY(Name, F) \
  kernel void Name(device const uint *tokens [[buffer(0)]], device const uchar *table [[buffer(1)]], \
                   device bfloat *output [[buffer(2)]], constant GgufEmbedParams &p [[buffer(3)]], \
                   uint index [[thread_position_in_grid]]) { \
    gguf_embedding<F>(tokens, table, output, p, index); \
  }
GGUF_EMBEDDING_ENTRY(gguf_embed_q4k, GgufEmbedQ4K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q6k, GgufEmbedQ6K)
GGUF_EMBEDDING_ENTRY(gguf_embed_q80, GgufEmbedQ80)
#undef GGUF_EMBEDDING_ENTRY
