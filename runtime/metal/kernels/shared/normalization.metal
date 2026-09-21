#include "metal/abi/KernelABI.h"
#include "metal/kernels/common/rms_inverse.h"
#include "metal/kernels/common/q4_sgmatrix.h"

kernel void norm_rms(device const bfloat *input [[buffer(0)]],
                        device const bfloat *weight [[buffer(1)]],
                        device bfloat *output [[buffer(2)]],
                        constant uint &width [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  float inverse_rms = rms_inverse(input + row * width, width, reductions,
                                  thread_index, lane, simd_group);
  for (uint column = thread_index; column < width; column += 256) {
    output[row * width + column] =
        bfloat(float(input[row * width + column]) * inverse_rms *
               float(weight[column]));
  }
}

// Keep the ordinary output for non-Q4 consumers, and emit the matrix operand
// layout from the same rounded bfloat values. No additional dispatch is needed.
kernel void norm_rms_q4_decode(device const bfloat *input [[buffer(0)]],
    device const bfloat *weight [[buffer(1)]], device bfloat *output [[buffer(2)]],
    device bfloat *table [[buffer(3)]], device float *sums [[buffer(4)]],
    constant uint &width [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float reductions[8];
  const float inverse = rms_inverse(input + row * width, width, reductions, tid, lane, sg);
  for (uint g = sg; g < width / 64; g += 8) {
    const uint k = g * 64 + lane * 2;
    const bfloat a = bfloat(float(input[row * width + k]) * inverse * float(weight[k]));
    const bfloat b = bfloat(float(input[row * width + k + 1]) * inverse * float(weight[k + 1]));
    output[row * width + k] = a;
    output[row * width + k + 1] = b;
    q4sg::write_input(table + ulong(row / 8) * width * 8,
                       sums + ulong(row / 8) * width / 8, g, row % 8, lane, a, b);
  }
}
