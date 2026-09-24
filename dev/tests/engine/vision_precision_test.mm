#include "metal/CommandGraph.hpp"
#include "metal/abi/Vision.h"

#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace splash;
uint16_t bf(float value) {
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  return uint16_t((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}
float fp(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 2)
        throw std::runtime_error("usage: vision-precision METALLIB");
      metal::MetalBackend backend(argv[1]);
      auto alloc = [&](uint64_t bytes) {
        auto buffer = backend.allocateBuffer(
            bytes, metal::BufferStorage::Shared, "vision-test");
        std::memset(buffer.contents(), 0, bytes);
        return buffer;
      };
      constexpr uint32_t m = 64, n = 128, k = 128;
      auto x = alloc(m * k * 2), w = alloc(n * k * 4), bias = alloc(n * 4),
           y = alloc(m * n * 2), residual = alloc(m * n * 2);
      std::fill_n(static_cast<uint16_t *>(x.contents()), m * k, bf(1));
      // Values lost by BF16, F16 or TF32 rounding before matmul.
      auto *weights = static_cast<float *>(w.contents());
      for (uint32_t i = 0; i < n * k; ++i)
        weights[i] = i % 2 ? -1.f : 1.f + 1.f / 65536;
      metal::CommandGraph graph;
      graph.add("vision_gemm_m64n128_f32", {x, w, bias, y, residual},
                VisionGemmParams{n, k}, {1, 1, 1});
      (void)backend.submitCommand(graph.dispatches());
      for (uint32_t i = 0; i < m * n; ++i)
        if (static_cast<uint16_t *>(y.contents())[i] != bf(k / 2.f / 65536))
          throw std::runtime_error(
              "F32 vision GEMM lost sub-BF16 weight precision: " +
              std::to_string(fp(static_cast<uint16_t *>(y.contents())[i])));
      // F32 bias must join accumulation before output rounding, not be rounded
      // first.
      std::memset(w.contents(), 0, n * k * 4);
      std::fill_n(static_cast<float *>(bias.contents()), n, 1.f + 1.f / 65536);
      std::fill_n(static_cast<uint16_t *>(residual.contents()), m * n, bf(-1));
      metal::CommandGraph biased;
      biased.add("vision_gemm_m64n128_residual_f32_bias",
                 {x, w, bias, y, residual}, VisionGemmParams{n, k}, {1, 1, 1});
      (void)backend.submitCommand(biased.dispatches());
      for (uint32_t i = 0; i < m * n; ++i)
        if (static_cast<uint16_t *>(y.contents())[i] != bf(1.f / 65536))
          throw std::runtime_error("F32 vision bias lost precision");
      constexpr uint32_t width = 256;
      auto normInput = alloc(width * 2), normWeight = alloc(width * 4);
      auto normBias = alloc(width * 4), normOutput = alloc(width * 2);
      for (uint32_t i = 0; i < width; ++i) {
        static_cast<uint16_t *>(normInput.contents())[i] = bf(i % 2 ? -1 : 1);
        static_cast<float *>(normWeight.contents())[i] = 1.f + 1.f / 65536;
        static_cast<float *>(normBias.contents())[i] = -1.f;
      }
      metal::CommandGraph norm;
      norm.add("vision_layer_norm_f32",
               {normInput, normWeight, normBias, normOutput},
               VisionNormParams{width}, {1, 1, 1});
      (void)backend.submitCommand(norm.dispatches());
      for (uint32_t i = 0; i < width; ++i) {
        const float expected = fp(bf(
            (i % 2 ? -1.f : 1.f) / std::sqrt(1.f + 1e-6f) * (1.f + 1.f / 65536) -
            1.f));
        if (std::abs(fp(static_cast<uint16_t *>(normOutput.contents())[i]) -
                     expected) > 1e-6f)
          throw std::runtime_error("F32 vision LayerNorm lost precision");
      }
      constexpr uint32_t hidden = 1152, tokens = 16;
      auto table = alloc(48 * 48 * hidden * 4),
           positions = alloc(tokens * hidden * 2);
      auto cos = alloc(tokens * 72 * 4), sin = alloc(tokens * 72 * 4);
      for (uint32_t row = 0; row < 48; ++row)
        for (uint32_t col = 0; col < 48; ++col)
          std::fill_n(static_cast<float *>(table.contents()) +
                          (row * 48 + col) * hidden,
                      hidden, col == 15 ? 2.f + 1.f / 65536 : -1.f);
      metal::CommandGraph positional;
      positional.add("vision_prepare_positions_f32",
                     {table, positions, cos, sin}, VisionGridParams{4, 4},
                     {tokens, 1, 1});
      (void)backend.submitCommand(positional.dispatches());
      // Token 1 is (row=0, col=1). Its bilinear taps cancel the integer part.
      const float fraction = 47.f / 3.f - 15.f;
      const float expectedPosition =
          fp(bf((1.f - fraction) * (2.f + 1.f / 65536) - fraction));
      for (uint32_t i = 0; i < hidden; ++i)
        if (std::abs(
                fp(static_cast<uint16_t *>(positions.contents())[hidden + i]) -
                expectedPosition) > 1e-6f)
          throw std::runtime_error(
              "F32 vision position interpolation lost precision");
      std::cout << "F32 vision GEMM, bias, LayerNorm and positions preserve "
                   "sub-BF16 values PASS\n";
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
