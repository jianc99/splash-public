// GGUF quantized GEMMs (K-quants, i-quants, Q8_0) for Apple9 and Apple10.
// Decode weights with FP32 group coefficients, then round once to the half tile,
// matching llama.cpp Metal dequantize.h / mul_mm.metal. Keep activations BF16.
// Weight planes and meta in the MDGG0001 layout (metal/abi/QuantFormat.h), decoded by kernels/common/quant_formats.h.
// Activations bf16 [rows][K]; weights staged as fp16 in threadgroup memory; fp32 accumulation; bf16 output.
// Keep the source order of float operations, which Metal's default fast math lets the compiler reassociate. Set
// before the includes, so it also holds for the shared format and reduction code compiled here.
#pragma clang fp reassociate(off)
#include "metal/abi/Gguf.h"
#include "metal/kernels/common/gguf_staged.h"
#include "metal/kernels/common/gguf_tile.h"
#include "metal/kernels/common/moe_expert_slab.h"
#include "metal/kernels/common/split_reduce.h"

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

// ---------------- decode tiles: each simdgroup stages its own Cols x KS sub-tile privately and runs matmul2d alone.
// MPP computes 16-row fragments, so a tile holds 8, 16 or 32 rows: a 3-lane step runs the 32-row tile over the storage
// of four lanes (LinearPlan::storageRows) and the padding lane's rows are computed and discarded. Rows are independent,
// so every active row is the bits of any other tile height (gguf-projection full); on a 16-core M5 Pro the 32-row tile
// at three lanes costs what it costs at four, 3-15% less than a 16-row plus an 8-row matmul per stage (0.207 vs 0.218 ms,
// Q4_K 12288 x 5120, DRAM-cold; 20-core: 0.173 vs 0.203).

// The destination of a Rows-row tile, zeroed by the caller: returning an initialized cooperative tensor loses its
// initial values on Apple9 in runtime-format kernels (also with shader validation).
template <typename TA, ushort Rows, ushort Cols, ushort KS>
inline auto gguf_make_acc(device TA *input, uint input_size, threadgroup half *stage) {
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  auto a0 = a.template slice<KS, Rows>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0);
  return operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
}
template <class Acc> inline void gguf_zero(thread Acc &acc) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
}
// fn(row, column, value) for every element of a tile's destination.
template <class Acc, class Fn> inline void gguf_elements(thread Acc &acc, Fn fn) {
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    const auto index = acc.get_multidimensional_index(i);
    fn(uint(index[1]), uint(index[0]), float(acc[i]));
  }
}

// The decode tile loop without the store: dequantize one KS-input step of the Cols columns into the stage, then run
// the tile's matmul2d on it, over steps [step_begin, step_end) of K.
template <class F, typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void sg_accum(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint output_origin,
                     threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint step_begin, uint step_end, thread Acc &acc) {
  constexpr ushort GPS = KS / 32, Items = Cols * GPS, IPT = (Items + 31) / 32;
  auto a = tensor(input, dextents<int, 2>{int(input_size), Rows}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(Rows, Cols, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, units = groups / F::MetaGroups;
  const uint tile = output_origin / QUANT_TILE_ROWS, tile_offset = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(tile) * groups * QUANT_TILE_ROWS + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * QUANT_TILE_ROWS + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * QUANT_TILE_ROWS + tile_offset) * F::MetaBytes;
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + (Buffers > 1 ? KS * Cols : 0), dextents<int, 2>{KS, Cols}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, Cols>(0, 0), b1 = bt1.slice<KS, Cols>(0, 0);
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
  const uint unit0 = (step_begin * GPS) / F::MetaGroups;
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = simd_lane + it * 32; const bool live = item < Items;
    const uint col = live ? item % Cols : 0, gi = live ? item / Cols : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(step_begin + pf) * GPS + gi;
      if (live && step_begin + pf < step_end) packed[pf][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + (ulong(unit0) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit0;
  }
  for (uint step = step_begin; step < step_end; ++step) {
    threadgroup half *buf = stage + (Buffers > 1 ? (step & 1) * (KS * Cols) : 0);
    if constexpr (Buffers == 1) simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort it = 0; it < IPT; ++it) {
      const uint item = simd_lane + it * 32; if (item >= Items) break;
      const uint col = item % Cols, gi = item / Cols, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
      if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit; }
      dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
    if (step + Prefetch < step_end) {
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = simd_lane + it * 32; if (item >= Items) break;
        const uint col = item % Cols, gi = item / Cols; const ulong g = ulong(step + Prefetch) * GPS + gi;
        packed[Prefetch - 1][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
      }
    }
    auto a_slice = a.template slice<KS, Rows>(step * KS, 0);
    if (Buffers > 1 && (step & 1)) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);   // the stage may be reused by a following accumulate
}

// A tile's sums over every K partition, handed to store(row, column, sum). One partition stores its own; more publish
// fp32 partials [split][Rows][destination column] (kernels/common/split_reduce.h) and the last arriving partition adds
// them in split order. `column0` is the simdgroup's first destination column, `counter` its threadgroup's (one per 64
// destination columns: the segments of a projection never share one).
template <ushort Rows, class Acc, class Store>
inline void gguf_store_sums(thread Acc &acc, uint splits, uint split, device coherent(device) float *partials,
                            device atomic_uint *counter, uint stride, uint column0, uint thread_index,
                            threadgroup uint *arrival, Store store) {
  if (splits == 1) { gguf_elements(acc, store); return; }
  const auto at = [&](uint s, uint row, uint column) { return (ulong(s) * Rows + row) * stride + column0 + column; };
  gguf_elements(acc, [&](uint row, uint column, float v) { partials[at(split, row, column)] = v; });
  if (!split_arrive_last(counter, splits, thread_index, arrival)) return;
  gguf_elements(acc, [&](uint row, uint column, float v) {
    store(row, column, split_sum(v, split, splits, [&](uint s) { return partials[at(s, row, column)]; }));
  });
  split_release(counter, thread_index);
}

// ---------------- pf: prefill with a shared B stage (TileN x KS, all threads dequantize), each simdgroup owns RowsPerSG
// rows. `rows` counts the chunk's rows from the tile's first: simdgroups past them (the last tile of a chunk that is not
// a multiple of the tile) still stage but skip their matmuls and stores, so a chunk costs its rows rounded up to
// RowsPerSG rather than to the tile (a 33-row Q4_K 17408 x 5120 chunk: 1.7x faster on M5 and M3 than a 128-row tile).
template <class F, typename TA, ushort RowsPerSG, ushort Simdgroups, ushort TileN, ushort KS, ushort Prefetch, GgufEpilogue Ep = EpNone>
inline void pf_tile(device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                    uint output_size, uint input_size, uint output_origin, uint rows, threadgroup half *stage,
                    threadgroup half2 *tl, uint simd_lane, uint simd_group, uint out_stride = 0, uint out_offset = 0,
                    device bfloat *aux = nullptr) {
  if (out_stride == 0) out_stride = output_size;
  const bool owns_rows = simd_group * RowsPerSG < rows;   // uniform per simdgroup
  constexpr ushort Threads = Simdgroups * 32, GPS = KS / 32, Items = TileN * GPS, IPT = (Items + Threads - 1) / Threads;
  auto a = tensor(input + ulong(simd_group) * RowsPerSG * input_size, dextents<int, 2>{int(input_size), RowsPerSG}, array<int, 2>{1, int(input_size)});
  constexpr auto descriptor = matmul2d_descriptor(RowsPerSG, TileN, KS, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<1>> operation;
  const uint groups = input_size / 32, steps = groups / GPS, units = groups / F::MetaGroups;
  const uint tile = output_origin / QUANT_TILE_ROWS, tile_offset = output_origin % QUANT_TILE_ROWS;
  device uchar *tw0 = w0 + (ulong(tile) * groups * QUANT_TILE_ROWS + tile_offset) * F::P0;
  device uchar *tw1 = w1 + (ulong(tile) * groups * QUANT_TILE_ROWS + tile_offset) * F::P1;
  device uchar *tmeta = meta + (ulong(tile) * units * QUANT_TILE_ROWS + tile_offset) * F::MetaBytes;
  auto a0 = a.template slice<KS, RowsPerSG>(0, 0);
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt0(stage, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  tensor<threadgroup half, dextents<int, 2>, tensor_inline> bt1(stage + KS * TileN, dextents<int, 2>{KS, TileN}, array<int, 2>{1, KS});
  auto b0 = bt0.slice<KS, TileN>(0, 0), b1 = bt1.slice<KS, TileN>(0, 0);
  auto acc = operation.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
  const uint thread_index = simd_group * 32 + simd_lane;
  typename F::Payload packed[Prefetch][IPT]; typename F::Meta hdr[IPT]; uint hdr_unit[IPT];
#pragma unroll
  for (ushort it = 0; it < IPT; ++it) {
    const uint item = thread_index + it * Threads; const bool live = item < Items;
    const uint col = live ? item % TileN : 0, gi = live ? item / TileN : 0;
#pragma unroll
    for (ushort pf = 0; pf < Prefetch; ++pf) {
      const ulong g = ulong(pf) * GPS + gi;
      if (live && pf < steps) packed[pf][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
    }
    hdr[it] = F::loadMeta(tmeta + col * F::MetaBytes); hdr_unit[it] = 0;
  }
  // The step loop with or without this simdgroup's matmuls, one copy each: a full tile runs the loop unchanged (a
  // branch around the matmul inside the loop cost 1-3% at full tiles on the M3 Max).
  const auto run_steps = [&](auto with_matmuls) {
    for (uint step = 0; step < steps; ++step) {
      threadgroup half *buf = stage + (step & 1) * (KS * TileN);
#pragma unroll
      for (ushort it = 0; it < IPT; ++it) {
        const uint item = thread_index + it * Threads; if (item >= Items) break;
        const uint col = item % TileN, gi = item / TileN, g = step * GPS + gi, unit = g / F::MetaGroups; const ushort j = g % F::MetaGroups;
        if (unit != hdr_unit[it]) { hdr[it] = F::loadMeta(tmeta + (ulong(unit) * QUANT_TILE_ROWS + col) * F::MetaBytes); hdr_unit[it] = unit; }
        dequant32<F>(packed[0][it], hdr[it], j, tl, buf + col * KS + gi * 32);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (ushort pf = 0; pf + 1 < Prefetch; ++pf)
#pragma unroll
        for (ushort it = 0; it < IPT; ++it) packed[pf][it] = packed[pf + 1][it];
      if (step + Prefetch < steps) {
#pragma unroll
        for (ushort it = 0; it < IPT; ++it) {
          const uint item = thread_index + it * Threads; if (item >= Items) break;
          const uint col = item % TileN, gi = item / TileN; const ulong g = ulong(step + Prefetch) * GPS + gi;
          packed[Prefetch - 1][it] = F::load(tw0 + (g * QUANT_TILE_ROWS + col) * F::P0, tw1 + (g * QUANT_TILE_ROWS + col) * F::P1);
        }
      }
      if constexpr (decltype(with_matmuls)::value) {
        auto a_slice = a.template slice<KS, RowsPerSG>(step * KS, 0);
        if (step & 1) operation.run(a_slice, b1, acc); else operation.run(a_slice, b0, acc);
      }
    }
  };
  if (!owns_rows) { run_steps(false_type{}); return; }
  run_steps(true_type{});
#pragma unroll
  for (ushort i = 0; i < acc.get_capacity(); ++i) {
    if (!acc.is_valid_element(i)) continue;
    auto index = acc.get_multidimensional_index(i);
    const ulong o = (ulong(simd_group) * RowsPerSG + index[1]) * out_stride + out_offset + output_origin + index[0];
    output[o] = gguf_epilogue<Ep>(acc[i], aux, o);
  }
}

#define TGLUT_INIT(F)                                                                                     \
  threadgroup half2 tl[F::Kind == QuantCodebook ? 256 : 1];                                                \
  if constexpr (F::Kind == QuantCodebook) quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, Threads);
#define ABUF(TA, P) device TA *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], constant P &p [[buffer(5)]]
#define ABUFE(P) device bfloat *input [[buffer(0)]], device uchar *w0 [[buffer(1)]], device uchar *w1 [[buffer(2)]], \
                 device uchar *meta [[buffer(3)]], device bfloat *output [[buffer(4)]], device bfloat *aux [[buffer(5)]], constant P &p [[buffer(6)]]
#define IDS uint simd_lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]]
#define PF_ROWS(R, S) const uint first = group.x * (R * S), rows = p.rows > first ? p.rows - first : 0
#define PFE_K(F, f, EP, ep, R, S, N, KS, P)                                                               \
  kernel void gguf_prefill_##f##_##ep(ABUFE(GgufPrefillParams), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N]; const uint rs = p.out_stride ? p.out_stride : p.output_size;      \
    PF_ROWS(R, S);                                                                                        \
    pf_tile<F, bfloat, R, S, N, KS, P, EP>(input + ulong(first) * p.input_size, w0, w1, meta, output + ulong(first) * rs, \
                                   p.output_size, p.input_size, group.y * N, rows, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset, aux + ulong(first) * rs); }
#define PF_K(F, f, TA, ta, R, S, N, KS, P)                                                                \
  kernel void gguf_prefill_##f##_##ta(ABUF(TA, GgufPrefillParams), uint2 group [[threadgroup_position_in_grid]], IDS) { \
    constexpr ushort Threads = S * 32; TGLUT_INIT(F)                                                      \
    threadgroup half stage[2 * KS * N];                                                                   \
    PF_ROWS(R, S);                                                                                        \
    pf_tile<F, TA, R, S, N, KS, P>(input + ulong(first) * p.input_size, w0, w1, meta, output + ulong(first) * (p.out_stride ? p.out_stride : p.output_size), \
                                   p.output_size, p.input_size, group.y * N, rows, stage, tl, simd_lane, simd_group, p.out_stride, p.out_offset); }
// runtime dequantizer selection (uniform per threadgroup)
template <typename TA, ushort Rows, ushort Cols, ushort KS, ushort Buffers, ushort Prefetch, class Acc>
inline void gguf_accum_any(uint fmt, device TA *input, device uchar *w0, device uchar *w1, device uchar *meta, uint input_size, uint origin,
                           threadgroup half *stage, threadgroup half2 *tl, uint simd_lane, uint sb, uint se, thread Acc &acc) {
  quant_format_switch(fmt, [&](auto format) {
    sg_accum<decltype(format), TA, Rows, Cols, KS, Buffers, Prefetch>(input, w0, w1, meta, input_size, origin, stage, tl,
                                                                     simd_lane, sb, se, acc);
  });
}

// ---------------- decode dispatches over (64-column tiles, K partitions): two simdgroups of 32 columns per
// threadgroup, every request lane in its tile, `splits` partitions of K (grid.y; kernels/common/split_reduce.h).
// Gate/up runs as a gate pass (a) into the gate scratch and an up pass (g) whose epilogue applies silu(gate) to the bf16
// up value, as the Apple9 register kernels do.
template <class F, ushort Rows, GgufEpilogue Ep>
inline void gguf_decode_tile(device bfloat *input, device uchar *w0, device uchar *w1, device uchar *meta, device bfloat *output,
                             device coherent(device) float *partials, device atomic_uint *counters, device bfloat *aux,
                             constant GgufDecodeParams &p, uint2 group, uint simd_lane, uint simd_group,
                             threadgroup half *stage, threadgroup half2 *tl, threadgroup uint *arrival) {
  const uint per = p.input_size / 32 / p.splits, origin = group.x * 64 + simd_group * 32, column0 = p.out_offset + origin;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  auto acc = gguf_make_acc<bfloat, Rows, 32, 32>(input, p.input_size, my);
  gguf_zero(acc);
  sg_accum<F, bfloat, Rows, 32, 32, 2, 1>(input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, group.y * per,
                                          (group.y + 1) * per, acc);
  gguf_store_sums<Rows>(acc, p.splits, group.y, partials, counters + p.out_offset / 64 + group.x, p.out_stride, column0,
                        simd_group * 32 + simd_lane, arrival, [&](uint row, uint column, float v) {
    // gguf_epilogue inline: calling it here reorders the lambda's captures.
    const ulong o = ulong(row) * p.out_stride + column0 + column;
    if constexpr (Ep == EpResidual) v += float(aux[o]);
    if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * gguf_silu(float(aux[o]));
    output[o] = bfloat(v);
  });
}
#define SEGBUF(i, w0, w1, m) device uchar *w0 [[buffer(i)]], device uchar *w1 [[buffer(i + 1)]], device uchar *m [[buffer(i + 2)]]
#define GGUF_DECODE_K(F, f, R, EP, ep)                                                                               \
  kernel void gguf_decode_##f##_m##R##_##ep(device bfloat *input [[buffer(0)]], SEGBUF(1, w0, w1, meta), device bfloat *output [[buffer(4)]], \
                                            device coherent(device) float *partials [[buffer(5)]], device atomic_uint *counters [[buffer(6)]], \
                                            device bfloat *aux [[buffer(7)]], constant GgufDecodeParams &p [[buffer(8)]], \
                                            uint2 group [[threadgroup_position_in_grid]], IDS) {                      \
    constexpr ushort Threads = 64; TGLUT_INIT(F) threadgroup half stage[2 * 2 * 32 * 32]; threadgroup uint arrival;  \
    gguf_decode_tile<F, R, EP>(input, w0, w1, meta, output, partials, counters, aux, p, group, simd_lane, simd_group, stage, tl, &arrival); }
#define GGUF_DECODE_ROWS(F, f, EP, ep) GGUF_DECODE_K(F, f, 8, EP, ep) GGUF_DECODE_K(F, f, 16, EP, ep) GGUF_DECODE_K(F, f, 32, EP, ep)
#define GGUF_DECODE_SET(F, f) GGUF_DECODE_ROWS(F, f, EpNone, a) GGUF_DECODE_ROWS(F, f, EpResidual, r) GGUF_DECODE_ROWS(F, f, EpUpWithGate, g)
QUANT_FORMATS(GGUF_DECODE_SET)

// Fused projections (qkv|z|ab, q|k|v): up to three column segments of any formats in one dispatch, so the small
// segments do not run as dispatches of their own. The threadgroup's tile picks its segment, and the segment's format
// picks the decode; every segment takes the same K splits.
#define GGUF_DECODE_FUSED_K(R)                                                                                     \
  kernel void gguf_decode_fused_m##R(device bfloat *input [[buffer(0)]], SEGBUF(1, w0a, w1a, ma), SEGBUF(4, w0b, w1b, mb), \
                                     SEGBUF(7, w0c, w1c, mc), device bfloat *output [[buffer(10)]],          \
                                     device coherent(device) float *partials [[buffer(11)]], device atomic_uint *counters [[buffer(12)]], \
                                     constant GgufDecodeFusedParams &p [[buffer(13)]], uint2 group [[threadgroup_position_in_grid]], IDS) { \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256]; threadgroup uint arrival;           \
    quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, 64);                                               \
    const uint t0 = p.cols[0] / 64, t1 = t0 + p.cols[1] / 64, s = group.x < t0 ? 0 : group.x < t1 ? 1 : 2;  \
    device uchar *w0 = s == 0 ? w0a : s == 1 ? w0b : w0c;                                                   \
    device uchar *w1 = s == 0 ? w1a : s == 1 ? w1b : w1c;                                                   \
    device uchar *meta = s == 0 ? ma : s == 1 ? mb : mc;                                                    \
    const uint local = group.x - (s == 0 ? 0 : s == 1 ? t0 : t1), per = p.input_size / 32 / p.splits;       \
    const uint origin = local * 64 + simd_group * 32, column0 = p.offset[s] + origin;                       \
    threadgroup half *my = stage + simd_group * (2 * 32 * 32);                                              \
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(input, p.input_size, my);                                   \
    gguf_zero(acc);                                                                                         \
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(p.fmt[s], input, w0, w1, meta, p.input_size, origin, my, tl, simd_lane, \
                                            group.y * per, (group.y + 1) * per, acc);                      \
    gguf_store_sums<R>(acc, p.splits, group.y, partials, counters + p.offset[s] / 64 + local, p.out_stride, column0, \
                       simd_group * 32 + simd_lane, &arrival, [&](uint row, uint column, float v) {        \
      output[ulong(row) * p.out_stride + column0 + column] = bfloat(v);                                     \
    });                                                                                                     \
  }
GGUF_DECODE_FUSED_K(8) GGUF_DECODE_FUSED_K(16) GGUF_DECODE_FUSED_K(32)


#define PROD_SET(F, f) \
  PF_K(F, f, bfloat, a, 32, 4, 64, 64, 1) PFE_K(F, f, EpResidual, r, 32, 4, 64, 64, 1) PFE_K(F, f, EpUpWithGate, g, 32, 4, 64, 64, 1)
QUANT_FORMATS(PROD_SET)

// ---------------- MoE experts (ops/MoE.cpp; kernels/shared/moe.metal groups the rows): threadgroup (x, y) computes
// 64 columns of grouped tile y with the weights of the tile's expert (moe_gguf_segment), in the format the tile picks
// at run time: on a 16-core M5 Pro one run-time-format dispatch over two segments is within -11..+8% of a dispatch
// per format (time-sg at 23040x2048 Q4_K and 92160x512 Q5_K, one to four lanes). aux is the gate of the up pass.
// Two simdgroups each stream their own 32 columns through the decode tile, grid (N / 64, tiles), on 8-row tiles
// (decode steps, short prefill chunks) or 32-row tiles (longer chunks, ops::moeGgufPrefillTile), where a tile runs the
// 16- or 32-row matmul that holds its live rows: an expert's last tile is mostly partial. On the 35B's real prefill
// routes (wikitext, chat, code; the three passes of a layer on a 16-core M5 Pro) 32-row tiles take 2.42-2.78 ms at
// 512 rows and 6.70-6.83 ms at 2048 rows against 3.44-3.86 and 8.20-8.23 for 64-row tiles sharing one 64-column stage
// over four 16-row simdgroups, and 1.29-1.71 ms against 1.59-2.60 for 8-row tiles at 128-256 rows.
template <ushort Rows, GgufEpilogue Ep>
inline void moe_gguf_expert_tile(device bfloat *input, device const MoeTileDescriptor *tiles, device const uint *tile_count,
                                 device uchar *w0, device uchar *w1, device uchar *meta, device uchar *sw0, device uchar *sw1,
                                 device uchar *smeta, device bfloat *output, device bfloat *aux,
                                 constant MoeGgufExpertParams &p, uint2 group, uint simd_lane, uint simd_group,
                                 threadgroup half *stage, threadgroup half2 *tl) {
  if (group.y >= *tile_count) return;
  const MoeTileDescriptor tile = tiles[group.y];
  const MoeGgufSegment s = moe_gguf_segment(tile.expert, p, w0, w1, meta, sw0, sw1, smeta);
  device bfloat *x = input + ulong(group.y) * Rows * p.input_size;
  const ulong out = ulong(group.y) * Rows * p.output_size;
  const uint origin = group.x * 64 + simd_group * 32;
  threadgroup half *my = stage + simd_group * (2 * 32 * 32);
  quant_iq4_pair_table(tl, simd_group * 32 + simd_lane, 64);
  const auto run = [&](auto rows) {
    constexpr ushort R = decltype(rows)::value;
    auto acc = gguf_make_acc<bfloat, R, 32, 32>(x, p.input_size, my);
    gguf_zero(acc);
    gguf_accum_any<bfloat, R, 32, 32, 2, 1>(s.format, x, s.w0, s.w1, s.meta, p.input_size, origin, my, tl, simd_lane, 0,
                                            p.input_size / 32, acc);
    gguf_elements(acc, [&](uint row, uint column, float v) {
      const ulong o = out + ulong(row) * p.output_size + origin + column;   // gguf_epilogue inline, as above
      if constexpr (Ep == EpUpWithGate) v = float(bfloat(v)) * gguf_silu(float(aux[o]));
      output[o] = bfloat(v);
    });
  };
  if constexpr (Rows == 8) run(integral_constant<ushort, 8>{});
  else if (tile.rows <= 16) run(integral_constant<ushort, 16>{});
  else run(integral_constant<ushort, 32>{});
}
#define MOE_GGUF_K(Name, Rows, Ep)                                                                                      \
  kernel void Name(device bfloat *input [[buffer(0)]], device const MoeTileDescriptor *tiles [[buffer(1)]],            \
                   device const uint *tile_count [[buffer(2)]], SEGBUF(3, w0, w1, meta), SEGBUF(6, sw0, sw1, smeta),     \
                   device bfloat *output [[buffer(9)]], device bfloat *aux [[buffer(10)]],                               \
                   constant MoeGgufExpertParams &p [[buffer(11)]], uint2 group [[threadgroup_position_in_grid]], IDS) {  \
    threadgroup half stage[2 * 2 * 32 * 32]; threadgroup half2 tl[256];                                                \
    moe_gguf_expert_tile<Rows, Ep>(input, tiles, tile_count, w0, w1, meta, sw0, sw1, smeta, output, aux, p, group,     \
                                   simd_lane, simd_group, stage, tl);                                                  \
  }
MOE_GGUF_K(moe_expert_gguf_m8, 8, EpNone)
MOE_GGUF_K(moe_expert_gguf_m8_up, 8, EpUpWithGate)
MOE_GGUF_K(moe_expert_gguf_m32, 32, EpNone)
MOE_GGUF_K(moe_expert_gguf_m32_up, 32, EpUpWithGate)
#undef MOE_GGUF_K
