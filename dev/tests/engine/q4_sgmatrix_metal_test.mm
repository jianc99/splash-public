#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"
#include "tuning/LinearNumerics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splash;
using namespace splash::ops;
using namespace splash::ops::tuning;
namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
struct Guarded {
  metal::MetalBuffer backing, view;
  uint64_t size;
  Guarded(metal::MetalBackend &backend, uint64_t bytes) : size(bytes) {
    backing = backend.allocateBuffer(bytes + 256);
    std::memset(backing.contents(), 0xa5, bytes + 256);
    view = backend.view(backing, 0, bytes);
  }
  void check() const {
    const auto *p = static_cast<const uint8_t *>(backing.contents());
    for (uint64_t i = size; i < size + 256; ++i) require(p[i] == 0xa5, "out-of-view write");
  }
};
uint32_t hash(uint32_t v) { v ^= v >> 16; v *= 0x7feb352d; v ^= v >> 15; return v * 0x846ca68b; }
Q4Projection weights(metal::MetalBackend &backend, LinearMatrix shape, uint32_t seed, bool zero) {
  const uint64_t params = uint64_t(shape.outputSize) * shape.inputSize / 64;
  Q4Projection p{backend.allocateBuffer(params * 32), backend.allocateBuffer(params * 2),
                 backend.allocateBuffer(params * 2), shape.outputSize, shape.inputSize};
  auto *q = static_cast<uint8_t *>(p.weights.contents());
  auto *s = static_cast<uint16_t *>(p.scales.contents());
  auto *b = static_cast<uint16_t *>(p.biases.contents());
  for (uint64_t i = 0; i < params * 32; ++i) q[i] = zero ? 0 : hash(uint32_t(i) + seed);
  for (uint64_t i = 0; i < params; ++i) {
    s[i] = floatToBf16((int(hash(uint32_t(i) + seed + 7) % 17) - 8) / 2048.0f);
    b[i] = zero ? 0 : floatToBf16(-float((hash(uint32_t(i) + seed + 11) % 16)) * bf16ToFloat(s[i]));
  }
  return p;
}
struct Reference { double value, error; };
Reference reference(const Q4Projection &p, const uint16_t *input, uint32_t row, uint32_t col, uint32_t splits) {
  const auto *q = static_cast<const uint8_t *>(p.weights.contents());
  const auto *sc = static_cast<const uint16_t *>(p.scales.contents());
  const auto *bi = static_cast<const uint16_t *>(p.biases.contents());
  double value = 0, magnitude = 0, quantMagnitude = 0;
  const uint32_t groups = p.inputSize / 64;
  for (uint32_t g = 0; g < groups; ++g) {
    const uint64_t at = (uint64_t(col / 256) * groups + g) * 256 + col % 256;
    double dot = 0, sum = 0, absolute = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const double x = bf16ToFloat(input[uint64_t(row) * p.inputSize + g * 64 + k]);
      const uint32_t nibble = (q[at * 32 + k / 2] >> (4 * (k % 2))) & 15;
      dot += x * nibble; sum += x; absolute += std::abs(x);
    }
    const double scale = bf16ToFloat(sc[at]), bias = bf16ToFloat(bi[at]);
    value += scale * dot + bias * sum;
    quantMagnitude += absolute * std::abs(scale);
    magnitude += absolute * (15 * std::abs(scale) + std::abs(bias));
  }
  // gamma_n bounds a chain of n fp32 roundings. Within a 64-element group,
  // operands are <=143|x|; budget 64 dot steps plus 8 for the row sum and
  // offset correction. Across groups budget two affine FMAs and S additions.
  // Absolute operand magnitudes make this valid even under cancellation.
  constexpr double u = 0x1p-24;
  const auto gamma = [&](double n) { return n * u / (1 - n * u); };
  const double error = gamma(72) * 143 * quantMagnitude + gamma(2 * groups + splits) * magnitude;
  return {double(bf16ToFloat(floatToBf16(float(value)))),
          error + ulpBf16(float(value))};
}
void runCase(metal::MetalBackend &backend, uint32_t n, uint32_t k, uint32_t splits,
             LinearEpilogue epilogue, uint32_t fixture) {
  const LinearWorkload workload{{n, k}, 8, LinearPhase::Decode, epilogue};
  const auto plan = Q4Linear::plan(workload,
      {LinearTile::Simdgroup, n / (epilogue == LinearEpilogue::GateUp ? 32 : 64), LinearSimdgroups::Four, splits});
  const auto size = plan.scratchSize();
  Guarded input(backend, 16ULL * k), output(backend, 16ULL * n), residual(backend, 16ULL * n);
  Guarded table(backend, size.input), sums(backend, size.sums), partials(backend, size.partials), counters(backend, size.counters);
  LinearScratch scratch{table.view, sums.view, partials.view, counters.view};
  std::memset(counters.view.contents(), 0, size.counters);
  auto *x = static_cast<uint16_t *>(input.view.contents());
  auto *r = static_cast<uint16_t *>(residual.view.contents());
  for (uint32_t i = 0; i < 8 * k; ++i) {
    float v = float(int(hash(i + 37) % 257) - 128) / 32;
    if (fixture == 1) v *= 262144; // bf16 values above half's range.
    if (fixture == 2) v = (i % 2 ? -1 : 1) * 131072.0f + float(i % 7) * 1024;
    if (fixture == 3) v = float(int(i % 5) - 2) * 524288;
    x[i] = floatToBf16(v);
  }
  for (uint32_t i = 0; i < 8 * n; ++i) r[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  const auto p = weights(backend, {n,k}, 31, fixture == 3);
  const auto gate = weights(backend, {n,k}, 177, fixture == 3);
  Q4Linear linear(backend.capabilities());
  LinearBuffers b{input.view, output.view, {}, epilogue == LinearEpilogue::Residual ? residual.view : metal::MetalBuffer{}, {}, {}, scratch};
  metal::CommandGraph graph;
  // Reuse one workspace repeatedly in a single command to expose incomplete
  // publication, stale counters and dependencies between consecutive dispatches.
  for (uint32_t repeat = 0; repeat < 3; ++repeat)
    linear.add(graph, b, p, plan, epilogue == LinearEpilogue::GateUp ? &gate : nullptr);
  (void)backend.submitCommand(graph.dispatches());
  const auto *prepared = static_cast<const uint16_t *>(table.view.contents());
  const auto *rowSums = static_cast<const float *>(sums.view.contents());
  for (uint32_t g=0;g<k/64;++g) for (uint32_t row=0;row<8;++row) {
    float sum=0;
    for (uint32_t z=0;z<64;++z) {
      const uint32_t j=(z%16/8)*4+z%4, kp=2*(z/16)+(z/4)%2;
      const uint32_t at=g*512+(((j/4)*8+kp)*4+row/2)*8+(j%4)*2+row%2;
      require(prepared[at]==x[row*k+g*64+z],"prepared input layout mismatch");
      sum+=bf16ToFloat(x[row*k+g*64+z]);
    }
    require(sum==rowSums[g*8+row],"prepared sum mismatch");
  }
  const auto *actual = static_cast<const uint16_t *>(output.view.contents());
  const std::vector<uint16_t> first(actual, actual + 8 * n);
  (void)backend.submitCommand(graph.dispatches());
  require(std::memcmp(first.data(), actual, 16ULL * n) == 0, "nondeterministic split reduction");
  const auto *counts = static_cast<const uint32_t *>(counters.view.contents());
  for (uint64_t i = 0; i < size.counters / 4; ++i) require(counts[i] == 0, "counter not reset");
  for (uint32_t row = 0; row < 8; ++row) {
    for (uint32_t col : {0U, 7U, 8U, 31U, 32U, 63U, 64U, 127U, 128U, 255U, n-1}) {
      auto ref = reference(p, x, row, col, splits);
      if (epilogue == LinearEpilogue::Residual) ref.value += bf16ToFloat(r[row * n + col]);
      if (epilogue == LinearEpilogue::GateUp) {
        const auto g = reference(gate, x, row, col, splits);
        const double activation = g.value / (1 + std::exp(-g.value));
        ref.error = 1.1 * g.error * (std::abs(ref.value) + ref.error) + std::abs(activation) * ref.error;
        ref.value *= activation;
      }
      const double expected = bf16ToFloat(floatToBf16(float(ref.value)));
      const double value = bf16ToFloat(actual[row * n + col]);
      const double bound = ref.error + ulpBf16(float(expected)) + ulpBf16(float(value));
      if (!std::isfinite(value) || std::abs(value - expected) > bound) {
        std::cerr << "N=" << n << " K=" << k << " S=" << splits << " epilogue=" << int(epilogue)
                  << " fixture=" << fixture << " row=" << row << " col=" << col
                  << " actual=" << value << " reference=" << expected << " bound=" << bound << '\n';
        throw std::runtime_error("simdgroup result exceeds independent fp64 error bound");
      }
      if (fixture == 3) require(value == expected, "zero-weight offset cancellation is not exact");
    }
  }
  for (auto *guard : {&input,&output,&residual,&table,&sums,&partials,&counters}) guard->check();
}
void fusedNorm(metal::MetalBackend &backend, uint32_t k) {
  auto input=backend.allocateBuffer(k*16), weight=backend.allocateBuffer(k*2);
  auto output=backend.allocateBuffer(k*16), fused=backend.allocateBuffer(k*16);
  auto a=backend.allocateBuffer(k*16), b=backend.allocateBuffer(k*16);
  auto sa=backend.allocateBuffer(k/2), sb=backend.allocateBuffer(k/2);
  auto *x=static_cast<uint16_t *>(input.contents()), *w=static_cast<uint16_t *>(weight.contents());
  for (uint32_t i=0;i<k*8;++i) x[i]=floatToBf16(float(int(hash(i)%257)-128)*8192);
  for (uint32_t i=0;i<k;++i) w[i]=floatToBf16(float(int(i%17)-8)/4);
  metal::CommandGraph graph;
  Normalization::addRms(graph,input,weight,output,k,8);
  graph.add("decode_linear_q4_prepare",{output,a,sa},k,{k/32,1,1},{128,1,1});
  Normalization::addRms(graph,input,weight,fused,k,8,{b,sb,{},{}});
  (void)backend.submitCommand(graph.dispatches());
  require(!std::memcmp(output.contents(),fused.contents(),k*16),"fused norm changed bf16 output");
  require(!std::memcmp(a.contents(),b.contents(),k*16),"fused operand permutation mismatch");
  require(!std::memcmp(sa.contents(),sb.contents(),k/2),"fused input sums mismatch");
}
void fusedAttentionGate(metal::MetalBackend &backend, uint32_t heads, uint32_t kvHeads) {
  const uint32_t width = heads * 256, packedWidth = 2 * width + 2 * kvHeads * 256;
  auto packed = backend.allocateBuffer(uint64_t{packedWidth} * 16);
  auto attention = backend.allocateBuffer(uint64_t{width} * 32 * 2);
  for (auto buffer : {packed, attention}) {
    auto *data = static_cast<uint16_t *>(buffer.contents());
    for (uint64_t i = 0; i < buffer.sizeBytes() / 2; ++i)
      data[i] = floatToBf16(float(int(hash(uint32_t(i)) % 257) - 128) / 16);
  }
  Guarded output(backend, width * 16), fused(backend, width * 16);
  Guarded a(backend, width * 16), b(backend, width * 16);
  Guarded sa(backend, width / 2), sb(backend, width / 2);
  metal::CommandGraph graph;
  PagedAttention::addVerifyGate(graph, packed, attention, output.view, 8, 32, 32,
                                heads, {1, kvHeads, 256}, 1);
  graph.add("decode_linear_q4_prepare", {output.view, a.view, sa.view}, width,
            {width / 32, 1, 1}, {128, 1, 1});
  PagedAttention::addVerifyGate(graph, packed, attention, fused.view, 8, 32, 32,
                                heads, {1, kvHeads, 256}, 1, {b.view, sb.view, {}, {}});
  (void)backend.submitCommand(graph.dispatches());
  require(!std::memcmp(output.view.contents(), fused.view.contents(), width * 16),
          "fused attention gate output");
  require(!std::memcmp(a.view.contents(), b.view.contents(), width * 16),
          "fused attention gate table");
  require(!std::memcmp(sa.view.contents(), sb.view.contents(), width / 2),
          "fused attention gate sums");
  for (auto *buffer : {&output, &fused, &a, &b, &sa, &sb}) buffer->check();
}


}
int main(int argc,char **argv) {
  try {
    require(argc==2,"usage: q4-sgmatrix <production.metallib>");
    metal::MetalBackend backend(argv[1]);
    fusedAttentionGate(backend, 24, 4);
    fusedAttentionGate(backend, 16, 2);
    uint32_t cases=0;
    for (auto [n,k] : std::array<std::array<uint32_t,2>,4>{{{256,256},{768,768},{512,5120},{512,17408}}})
      for (uint32_t splits : {1U,2U,4U,8U}) {
        if ((k/64)%splits) continue;
        for (auto e : {LinearEpilogue::None,LinearEpilogue::Residual,LinearEpilogue::GateUp})
          for (uint32_t fixture=0;fixture<4;++fixture) { runCase(backend,n,k,splits,e,fixture); ++cases; }
      }
    for (uint32_t width : {64U, 320U, 2048U, 5120U, 17408U}) fusedNorm(backend, width);
    std::cout << "Q4 simdgroup: PASS cases=" << cases << " (fp64, range, cancellation, guards, repeated dispatch, fused norm)\n";
  } catch (const std::exception &e) { std::cerr << "Q4 simdgroup: FAIL: " << e.what() << '\n'; return 1; }
}
