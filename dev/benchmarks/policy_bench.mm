// DRAM-cold GPU time of ops::Linear plans: the measurements behind the kernel selection policy
// (runtime/ops/KernelPolicy.hpp, Linear.cpp, LinearGguf.cpp) and the gate for changing it.
//
//   policy-bench <metallib> [--suite affine|prefill|gguf|bandwidth] [options]
//
// Suites
//   affine     affine Q4 decode at 8/16/24/32 rows: the device's plan, Split128 at every K split, and the sequential
//              tiles (N128/N256, four or eight simdgroups, paired or not) at the persistent group counts --groups
//              names. A three-lane step can also run the 32-row plans over four lanes of storage: timing runs
//              measure those as the 32-row variants, and --check runs them at 24 rows over NaN padding rows (@32).
//   prefill    affine prefill chunks at --prefill-rows: the device's prefill plan and the other prefill tiles (a
//              gate/up pair runs its gate and up-with-gate passes), the input-sums dispatch where the model pays one
//              (the mixer output and draft context projections), and beside them the decode plans of 8-32 rows.
//   gguf       GGUF projections per format (--formats): the staged and register decode tiles at every K split,
//              one to four lanes; three lanes of the staged tile run its 32-row tile over four lanes of storage.
//   bandwidth  stream-read bandwidth of a 1 GiB buffer over threadgroups per core: the device's B.
//
// Shapes (--shapes): 27b and 35b are the MLX 4-bit targets with their DFlash drafts, each shape with its count
// of projections per decode step (for gguf, the targets' dense projections at one format each); lm_head the
// vocabulary projections into fp32; grid the fitting grid of N x K plain projections. --emulate C,... adds every
// selected production shape at other core counts by width, N x cores / C rounded to 256 columns, so its grid per
// core is that of a C-core GPU (the host keeps its bandwidth per core; tiles per core are reported).
// --filter A,B keeps shapes whose label contains any of them.
//
// Timing: a shape's weights are a ring of copies of at least --ring-mib (384) MiB; before timing every copy is
// streamed once, then each sample is one command of back-to-back dispatches on successive copies (about
// --target-ms of GPU time), the shape's variants interleaved with a rotating start (--order reverse reverses
// them). Reported per variant: the median, minimum and maximum over --samples samples of the GPU time per
// dispatch. Take >= 9 samples over >= 4 runs alternating --order, and medians across runs
// (dev/benchmarks/policy_summary.py). To gate a policy change, alternate runs of the two builds and pass the
// candidate's to policy_summary.py --new; plan-census lists the plans the builds differ in.
//
// --check runs every variant once on buffers of exactly its plans' sizes (the production scratch formulas), each
// followed by a guard band, with NaN split partials and NaN padding rows, twice on the same scratch: counters must
// return to zero, the two outputs must be equal bit for bit and every output of the active rows must lie within
// its fp64 bound. Run it under MTL_SHADER_VALIDATION=1 with the options of the timing runs before any of them.
// A timing run first repeats that check without the fp64 comparison (--no-self-check skips it).
//
// --csv FILE appends one row per variant (header when the file is new); stdout gets a readable table. --list
// prints the variants and their kernels without running them. --baseline FILE takes another build's --list output
// of the same options and times that build's plan of every affine decode shape and rows beside this build's
// (baseline = 1 in the CSV), so the plans of two builds compare within every run.
#include "../tests/engine/GgufFormatReference.hpp"
#include "metal/CommandGraph.hpp"
#include "metal/MetalBackend.hpp"
#include "ops/Linear.hpp"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace splash;
using namespace splash::ops;
using splash::metal::CommandGraph;
using splash::metal::MetalBackend;
using splash::metal::MetalBuffer;
using splash::ops::tuning::bf16ToFloat;
using splash::ops::tuning::floatToBf16;
using splash::ops::tuning::ulpBf16;

namespace {

constexpr uint32_t kLaneRows = 8, kMaximumRows = 32;
constexpr uint64_t kGuardBytes = 4096;
constexpr uint8_t kGuardByte = 0x5a;
constexpr uint16_t kNaN = 0x7FC0;         // bf16 quiet NaN: padding rows of an input
constexpr uint32_t kNaN32 = 0x7FC00000u;  // fp32 quiet NaN: split partials before a run
constexpr uint64_t kCopyAlignment = 16384;

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

std::vector<std::string> splitList(const std::string &text, char separator = ',') {
  std::vector<std::string> out;
  std::stringstream stream(text);
  for (std::string item; std::getline(stream, item, separator);)
    if (!item.empty()) out.push_back(item);
  return out;
}
std::vector<uint32_t> numbers(const std::string &text) {
  std::vector<uint32_t> out;
  for (const std::string &item : splitList(text)) out.push_back(uint32_t(std::stoul(item)));
  return out;
}
uint64_t roundUp(uint64_t value, uint64_t quantum) { return (value + quantum - 1) / quantum * quantum; }

struct Options {
  std::string suite = "affine";
  std::vector<std::string> shapeSets{"27b", "35b"};
  std::vector<std::string> filters;
  std::vector<uint32_t> emulate;
  std::vector<uint32_t> rows{8, 16, 24, 32};
  std::vector<uint32_t> prefillRows{1, 8, 16, 24, 32, 33, 48, 64};
  // Persistent group counts of the sequential tiles: "full" (one tile per threadgroup), "policy" (the device
  // plan's count where it runs that tile), "half" (two tiles per threadgroup) or a multiple of the core count.
  std::vector<std::string> groups{"full", "policy"};
  std::vector<std::string> formats;
  std::vector<std::string> tiles;  // keep only these tile names (plus the device's plan)
  uint32_t samples = 9;
  double targetMs = 1.5;
  uint64_t ringBytes = 384ull << 20;
  bool check = false, reverse = false, legacy = false, selfCheck = true, list = false;
  uint32_t run = 0;
  std::string csv;
  // Another build's plan label per (shape, rows) (--baseline).
  std::map<std::pair<std::string, uint32_t>, std::string> baseline;
};

// ---------------------------------------------------------------- shapes

struct Shape {
  std::string label, model, kind;
  uint32_t n = 0, k = 0;
  LinearEpilogue epilogue = LinearEpilogue::None;
  uint32_t count = 0;     // projections of this shape per decode step (0: not a production shape)
  uint32_t emulated = 0;  // the core count this width emulates (0: native)
  FloatOutput destination = FloatOutput::BFloat16;
  std::string format;     // gguf: the format of its single segment
};

const char *epilogueName(LinearEpilogue e) {
  switch (e) {
    case LinearEpilogue::None: return "none";
    case LinearEpilogue::Residual: return "residual";
    case LinearEpilogue::GateUp: return "gate_up";
    case LinearEpilogue::UpWithGate: return "up_with_gate";
  }
  return "?";
}

Shape shape(const std::string &model, const std::string &kind, uint32_t n, uint32_t k, LinearEpilogue epilogue,
            uint32_t count) {
  Shape s;
  s.label = model + "." + kind;
  s.model = model;
  s.kind = kind;
  s.n = n;
  s.k = k;
  s.epilogue = epilogue;
  s.count = count;
  return s;
}

// The MLX 4-bit targets and their DFlash drafts: every decode projection kind with its count per decode step
// (a draft decode and its context commit per step). The 27B draft's gate/up has the target's shape (64 + 5).
std::vector<Shape> productionShapes(const std::string &model) {
  const auto N = LinearEpilogue::None, R = LinearEpilogue::Residual, G = LinearEpilogue::GateUp;
  if (model == "27b")
    return {shape(model, "gdn_in", 16640, 5120, N, 48),     shape(model, "attn_in", 14336, 5120, N, 16),
            shape(model, "mixer_out", 5120, 6144, R, 64),   shape(model, "gate_up", 17408, 5120, G, 69),
            shape(model, "down", 5120, 17408, R, 64),       shape(model, "d_down", 5120, 17408, N, 5),
            shape(model, "d_context", 5120, 25600, N, 1),   shape(model, "d_qkv", 6144, 5120, N, 10),
            shape(model, "d_out", 5120, 4096, N, 5),        shape(model, "d_dynamic", 1280, 5120, N, 10),
            shape(model, "d_selector", 256, 5120, N, 1)};
  if (model == "35b")
    return {shape(model, "gdn_in", 12544, 2048, N, 30),     shape(model, "attn_in", 9216, 2048, N, 10),
            shape(model, "mixer_out", 2048, 4096, R, 40),   shape(model, "d_qkv", 6144, 2048, N, 12),
            shape(model, "d_out", 2048, 4096, N, 6),        shape(model, "d_gate_up", 6144, 2048, G, 6),
            shape(model, "d_down", 2048, 6144, N, 6),       shape(model, "d_dynamic", 512, 2048, N, 12),
            shape(model, "d_context", 2048, 16384, N, 1),   shape(model, "d_selector", 256, 2048, N, 1)};
  if (model == "lm_head") {
    Shape a = shape("27b", "lm_head", 248320, 5120, N, 2), b = shape("35b", "lm_head", 248320, 2048, N, 2);
    a.destination = b.destination = FloatOutput::Float32;
    return {a, b};
  }
  if (model == "grid") {
    std::vector<Shape> out;
    for (const uint32_t k : {2048u, 4096u, 6144u, 8192u, 16384u, 25600u})
      for (const uint32_t n : {256u, 512u, 1024u, 1536u, 2048u, 2560u, 3072u, 4096u, 5120u, 6144u, 8192u, 12288u, 16384u})
        out.push_back(shape("grid", "n" + std::to_string(n) + ".k" + std::to_string(k), n, k, N, 0));
    return out;
  }
  fail("unknown shape set " + model);
}

// The GGUF targets' dense decode projections (the fused qkv|z and q|k|v rows as one segment of their width).
std::vector<Shape> ggufShapes(const std::string &model) {
  const auto N = LinearEpilogue::None, R = LinearEpilogue::Residual, G = LinearEpilogue::GateUp;
  if (model == "27b")
    return {shape(model, "gdn_in", 16384, 5120, N, 48),     shape(model, "attn_in", 14336, 5120, N, 16),
            shape(model, "mixer_out", 5120, 6144, R, 64),   shape(model, "gate_up", 17408, 5120, G, 64),
            shape(model, "down", 5120, 17408, R, 64)};
  if (model == "35b")
    return {shape(model, "gdn_in", 12288, 2048, N, 30),     shape(model, "attn_in", 9216, 2048, N, 10),
            shape(model, "mixer_out", 2048, 4096, R, 40)};
  if (model == "lm_head") {
    Shape a = shape("27b", "lm_head", 248320, 5120, N, 2), b = shape("35b", "lm_head", 248320, 2048, N, 2);
    a.destination = b.destination = FloatOutput::Float32;
    return {a, b};
  }
  fail("unknown gguf shape set " + model);
}
// The dominant formats of the models' dense projections (UD-Q4_K_M, UD-IQ3_XXS; UD-Q4_K_M, UD-Q2_K_XL).
std::vector<std::string> defaultFormats(const std::string &model) {
  if (model == "27b") return {"q4k", "q5k", "iq4xs", "q6k", "iq3xxs", "iq3s", "iq2s"};
  if (model == "35b") return {"q80", "q5k", "q6k"};
  return {"q6k", "q4k"};
}

// The selected shapes, their width emulations, and the filter.
std::vector<Shape> selectShapes(const Options &o, uint32_t cores) {
  std::vector<Shape> base, out;
  const bool gguf = o.suite == "gguf";
  for (const std::string &set : o.shapeSets) {
    for (Shape s : gguf ? ggufShapes(set) : productionShapes(set)) {
      if (!gguf) {
        base.push_back(s);
        continue;
      }
      for (const std::string &format : o.formats.empty() ? defaultFormats(set) : o.formats) {
        Shape f = s;
        f.format = format;
        f.label = s.label + "." + format;
        base.push_back(f);
      }
    }
  }
  for (const Shape &s : base) {
    out.push_back(s);
    if (s.model == "grid" || s.kind == "lm_head") continue;
    for (const uint32_t c : o.emulate) {
      if (c == cores) continue;
      const double width = double(s.n) * cores / c;
      const uint32_t n = uint32_t(std::max(1.0, std::round(width / 256))) * 256;
      // Rounding to whole 256-column tiles must keep the grid per core within 10%.
      if (std::fabs(n / width - 1) > 0.10) continue;
      Shape e = s;
      e.n = n;
      e.emulated = c;
      e.label = "x" + std::to_string(c) + "." + s.label;
      out.push_back(e);
    }
  }
  if (o.filters.empty()) return out;
  std::vector<Shape> kept;
  for (const Shape &s : out)
    for (const std::string &f : o.filters)
      if (s.label.find(f) != std::string::npos) {
        kept.push_back(s);
        break;
      }
  return kept;
}

// ---------------------------------------------------------------- buffers

// A buffer of `bytes` followed by a guard band.
struct Guarded {
  MetalBuffer backing, view;
  uint64_t bytes = 0;
  Guarded() = default;
  Guarded(MetalBackend &backend, uint64_t size, uint8_t fill) : bytes(size) {
    backing = backend.allocateBuffer(std::max<uint64_t>(size, 4) + kGuardBytes);
    std::memset(backing.contents(), fill, std::max<uint64_t>(size, 4));
    std::memset(static_cast<uint8_t *>(backing.contents()) + std::max<uint64_t>(size, 4), kGuardByte, kGuardBytes);
    view = backend.view(backing, 0, std::max<uint64_t>(size, 4));
  }
  [[nodiscard]] bool intact() const {
    if (!backing) return true;
    const auto *guard = static_cast<const uint8_t *>(backing.contents()) + std::max<uint64_t>(bytes, 4);
    return std::all_of(guard, guard + kGuardBytes, [](uint8_t b) { return b == kGuardByte; });
  }
  // The view when the plans need the buffer, else nothing.
  [[nodiscard]] MetalBuffer used() const { return bytes ? view : MetalBuffer{}; }
};

// Everything the variants of one shape bind besides the weights. Check runs size them per variant, timing runs
// to the largest variant.
struct Sizes {
  uint64_t input = 0, output = 0, residual = 0, gateScratch = 0, sums = 0, downSums = 0;
  LinearScratchSize scratch;
  void include(const Sizes &o) {
    input = std::max(input, o.input);
    output = std::max(output, o.output);
    residual = std::max(residual, o.residual);
    gateScratch = std::max(gateScratch, o.gateScratch);
    sums = std::max(sums, o.sums);
    downSums = std::max(downSums, o.downSums);
    scratch.include(o.scratch);
  }
};
struct Buffers {
  Guarded input, output, residual, gateScratch, sums, downSums;
  Guarded scratchInput, scratchSums, partials, counters, rotated;
  Buffers(MetalBackend &backend, const Sizes &s, uint8_t outputFill)
      : input(backend, s.input, 0), output(backend, s.output, outputFill), residual(backend, s.residual, 0),
        gateScratch(backend, s.gateScratch, 0xff), sums(backend, s.sums, 0), downSums(backend, s.downSums, 0xff),
        scratchInput(backend, s.scratch.input, 0), scratchSums(backend, s.scratch.sums, 0),
        partials(backend, s.scratch.partials, 0), counters(backend, s.scratch.counters, 0),
        rotated(backend, s.scratch.rotated, 0) {}
  [[nodiscard]] bool intact() const {
    for (const Guarded *g : {&input, &output, &residual, &gateScratch, &sums, &downSums, &scratchInput, &scratchSums,
                             &partials, &counters, &rotated})
      if (!g->intact()) return false;
    return true;
  }
  [[nodiscard]] LinearScratch scratch() const {
    return {scratchInput.used(), scratchSums.used(), partials.used(), counters.used(), rotated.used()};
  }
  [[nodiscard]] bool countersZero() const {
    const auto *c = static_cast<const uint32_t *>(counters.view.contents());
    return std::all_of(c, c + counters.bytes / 4, [](uint32_t v) { return v == 0; });
  }
};

// One contiguous allocation per weight plane holding `copies` copies of a generated image, each copy a view.
struct Ring {
  std::vector<MetalBuffer> planes;       // backing allocations, one per plane
  std::vector<uint64_t> stride;          // bytes between copies, per plane
  uint32_t copies = 0;
  [[nodiscard]] MetalBuffer copy(const MetalBackend &backend, size_t plane, uint32_t c, uint64_t bytes) const {
    return backend.view(planes[plane], stride[plane] * c, bytes);
  }
};
Ring makeRing(MetalBackend &backend, const std::vector<const std::vector<uint8_t> *> &images, uint32_t copies) {
  Ring ring;
  ring.copies = copies;
  for (const auto *image : images) {
    const uint64_t stride = roundUp(std::max<uint64_t>(image->size(), 16), kCopyAlignment);
    MetalBuffer buffer = backend.allocateBuffer(stride * copies);
    auto *bytes = static_cast<uint8_t *>(buffer.contents());
    for (uint32_t c = 0; c < copies; ++c) std::memcpy(bytes + stride * c, image->data(), image->size());
    ring.planes.push_back(buffer);
    ring.stride.push_back(stride);
  }
  return ring;
}
uint32_t ringCopies(uint64_t bytesPerCopy, uint64_t ringBytes) {
  return uint32_t(std::clamp<uint64_t>((ringBytes + bytesPerCopy - 1) / bytesPerCopy, 2, 4096));
}

// ---------------------------------------------------------------- affine weights and fp64 reference

struct AffineImage {
  std::vector<uint8_t> weights, scales, biases;  // StorageN = 256 order, bf16 scales and biases
  uint32_t n = 0, k = 0;
};
AffineImage affineImage(uint32_t n, uint32_t k, uint32_t seed) {
  AffineImage a;
  a.n = n;
  a.k = k;
  const uint64_t parameters = uint64_t(n) * (k / 64);
  a.weights.resize(parameters * 32);
  a.scales.resize(parameters * 2);
  a.biases.resize(parameters * 2);
  std::mt19937_64 random(seed);
  auto *w = reinterpret_cast<uint64_t *>(a.weights.data());
  for (uint64_t i = 0; i < a.weights.size() / 8; ++i) w[i] = random();
  std::uniform_real_distribution<float> parameter(-0.02f, 0.02f);
  auto *s = reinterpret_cast<uint16_t *>(a.scales.data());
  auto *b = reinterpret_cast<uint16_t *>(a.biases.data());
  for (uint64_t i = 0; i < parameters; ++i) {
    s[i] = floatToBf16(parameter(random));
    b[i] = floatToBf16(parameter(random));
  }
  return a;
}
// The index of (column n, quant group g) in StorageN = 256 order (kernels/common/q4_mpp_tiles.h).
inline uint64_t affineParameter(uint32_t n, uint32_t g, uint32_t groups) {
  return (uint64_t(n / 256) * groups + g) * 256 + n % 256;
}

// fp64 projections of rows x (bf16, `rows` x K) through an affine image: out[r][n], plus the per-row bound of the
// fp32 kernels' rounding (the magnitudes LinearNumerics' simdgroup bound uses, which covers any association of
// the per-group products, the scale and bias FMAs and up to eight split additions).
struct Reference {
  std::vector<double> value;  // rows x n
  std::vector<double> bound;  // per row
};
Reference affineReference(const AffineImage &a, const std::vector<uint16_t> &x, uint32_t rows) {
  const uint32_t n = a.n, k = a.k, groups = k / 64;
  Reference ref;
  ref.value.assign(uint64_t(rows) * n, 0);
  std::vector<double> xs(uint64_t(rows) * k), xsum(uint64_t(rows) * groups), xabs(uint64_t(rows) * groups);
  for (uint64_t i = 0; i < xs.size(); ++i) xs[i] = bf16ToFloat(x[i]);
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t g = 0; g < groups; ++g)
      for (uint32_t j = 0; j < 64; ++j) {
        xsum[uint64_t(r) * groups + g] += xs[uint64_t(r) * k + g * 64 + j];
        xabs[uint64_t(r) * groups + g] += std::fabs(xs[uint64_t(r) * k + g * 64 + j]);
      }
  const auto *scales = reinterpret_cast<const uint16_t *>(a.scales.data());
  const auto *biases = reinterpret_cast<const uint16_t *>(a.biases.data());
  const uint8_t *weights = a.weights.data();
  const double *xp = xs.data(), *sums = xsum.data();
  double *value = ref.value.data();
  const uint32_t chunk = 64;
  // Blocks capture pointers only (a captured std::vector would be copied).
  dispatch_apply((n + chunk - 1) / chunk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
    double q[64];
    const uint32_t first = uint32_t(block) * chunk, last = std::min(n, first + chunk);
    for (uint32_t col = first; col < last; ++col)
      for (uint32_t g = 0; g < groups; ++g) {
        const uint64_t parameter = affineParameter(col, g, groups);
        const uint8_t *packed = weights + parameter * 32;
        for (uint32_t j = 0; j < 32; ++j) {
          q[2 * j] = packed[j] & 15;
          q[2 * j + 1] = packed[j] >> 4;
        }
        const double scale = bf16ToFloat(scales[parameter]), bias = bf16ToFloat(biases[parameter]);
        for (uint32_t r = 0; r < rows; ++r) {
          const double *xr = xp + uint64_t(r) * k + g * 64;
          double dot = 0;
          for (uint32_t j = 0; j < 64; ++j) dot += q[j] * xr[j];
          value[uint64_t(r) * n + col] += scale * dot + bias * sums[uint64_t(r) * groups + g];
        }
      }
  });
  std::vector<double> maxScale(groups), maxBias(groups);
  for (uint32_t col = 0; col < n; ++col)
    for (uint32_t g = 0; g < groups; ++g) {
      const uint64_t parameter = affineParameter(col, g, groups);
      maxScale[g] = std::max(maxScale[g], double(std::fabs(bf16ToFloat(scales[parameter]))));
      maxBias[g] = std::max(maxBias[g], double(std::fabs(bf16ToFloat(biases[parameter]))));
    }
  constexpr double u = 0x1p-24;
  const auto gamma = [&](double m) { return m * u / (1 - m * u); };
  ref.bound.assign(rows, 0);
  for (uint32_t r = 0; r < rows; ++r) {
    double quant = 0, affine = 0;
    for (uint32_t g = 0; g < groups; ++g) {
      const double absolute = xabs[uint64_t(r) * groups + g];
      quant += absolute * maxScale[g];
      affine += absolute * (15 * maxScale[g] + maxBias[g]);
    }
    ref.bound[r] = gamma(72) * 143 * quant + gamma(2.0 * groups + 8) * affine;
  }
  return ref;
}

// The largest |actual - fp64| an output may show: one bf16 spacing of the rounded value plus the fp32 bound, and
// what the epilogue propagates (LinearNumerics' split tolerance, with fp64 references); an fp32 output only the
// bound and one fp32 spacing.
double tolerance(LinearEpilogue epilogue, FloatOutput destination, double value, double bound, double residual,
                 double gate, double up) {
  if (destination == FloatOutput::Float32) return bound + std::ldexp(std::fabs(value), -23) + 1e-30;
  tuning::SplitReference reference{float(value), float(residual), float(gate), float(up)};
  const LinearEpilogue e = epilogue == LinearEpilogue::UpWithGate ? LinearEpilogue::GateUp : epilogue;
  // silu's fast-math evaluation adds a few fp32 ulps of the product.
  const double silu = e == LinearEpilogue::GateUp ? 64 * 0x1p-24 * std::fabs(value) : 0;
  return tuning::splitTolerance(e, reference, float(bound)) + silu;
}
double silu(double g) { return g / (1 + std::exp(-g)); }

// ---------------------------------------------------------------- variants

std::string tileName(LinearTile tile) {
  switch (tile) {
    case LinearTile::N128: return "n128";
    case LinearTile::N256: return "n256";
    case LinearTile::Paired128: return "paired128";
    case LinearTile::Paired256: return "paired256";
    case LinearTile::Split32: return "split32";
    case LinearTile::Split64: return "split64";
    case LinearTile::Split128: return "split128";
    case LinearTile::Simdgroup: return "simdgroup";
    case LinearTile::GgufStaged: return "staged";
    case LinearTile::GgufRegister: return "register";
  }
  return "?";
}
std::string configLabel(const LinearConfig &c, LinearPhase phase) {
  std::string label = tileName(c.tile) + ".sg" + std::to_string(uint32_t(c.simdgroups));
  if (phase == LinearPhase::Decode && c.tile != LinearTile::GgufStaged && c.tile != LinearTile::GgufRegister &&
      c.tile != LinearTile::Split128)
    label += ".g" + std::to_string(c.groups);
  if (c.splits > 1 || c.tile == LinearTile::Split128 || c.tile == LinearTile::GgufStaged ||
      c.tile == LinearTile::GgufRegister)
    label += ".s" + std::to_string(c.splits);
  return label;
}

// One timed configuration: the dispatches of one projection (a prefill gate/up pair runs two plans), their
// storage rows, and the plan whose configuration labels it.
struct Variant {
  std::string label, pipeline;
  std::vector<LinearPlan> plans;      // the plans a repetition runs, in order
  uint32_t logicalRows = 0;           // rows the fp64 check holds (the plan's storage may be larger)
  bool policy = false, candidate = false, sums = false, baseline = false;
  uint32_t threads = 0, gridX = 0, gridY = 0;
  std::vector<double> times;
  uint32_t reps = 1;
};

// The persistent group counts --groups names for a sequential tile of `tiles` column tiles.
std::vector<uint32_t> groupCounts(const Options &o, uint32_t tiles, uint32_t cores, uint32_t policyGroups) {
  std::vector<uint32_t> out;
  for (const std::string &g : o.groups) {
    uint32_t count = 0;
    if (g == "full") count = tiles;
    else if (g == "policy") count = policyGroups;
    else if (g == "half") count = (tiles + 1) / 2;
    else count = uint32_t(std::lround(std::stod(g) * cores));
    if (count && count <= tiles && std::find(out.begin(), out.end(), count) == out.end()) out.push_back(count);
  }
  return out;
}

bool keepTile(const Options &o, LinearTile tile) {
  return o.tiles.empty() || std::find(o.tiles.begin(), o.tiles.end(), tileName(tile)) != o.tiles.end();
}

// The configuration a decode label names (configLabel), over `n` columns: Split128 takes the full column grid.
std::optional<LinearConfig> decodeConfig(const std::string &label, uint32_t n) {
  const std::vector<std::string> parts = splitList(label, '.');
  if (parts.size() < 2 || parts[1].rfind("sg", 0) != 0) return std::nullopt;
  for (const LinearTile tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128, LinearTile::Paired256,
                                LinearTile::Split32, LinearTile::Split64, LinearTile::Split128}) {
    if (parts[0] != tileName(tile)) continue;
    LinearConfig config{tile, 0, LinearSimdgroups(std::stoul(parts[1].substr(2))), 1};
    for (size_t i = 2; i < parts.size(); ++i) {
      if (parts[i].size() > 1 && parts[i][0] == 'g') config.groups = uint32_t(std::stoul(parts[i].substr(1)));
      else if (parts[i].size() > 1 && parts[i][0] == 's') config.splits = uint32_t(std::stoul(parts[i].substr(1)));
      else return std::nullopt;
    }
    if (tile == LinearTile::Split128) config.groups = n / 128;
    return config;
  }
  return std::nullopt;
}

// The plans of another build's --list output: its lines marked "*" (shape, rows, label, pipeline, *).
std::map<std::pair<std::string, uint32_t>, std::string> readBaseline(const std::string &path) {
  std::ifstream in(path);
  if (!in) fail("cannot open " + path);
  std::map<std::pair<std::string, uint32_t>, std::string> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string shape, label, pipeline, mark;
    uint32_t rows = 0;
    if (fields >> shape >> rows >> label >> pipeline >> mark && mark == "*") out[{shape, rows}] = label;
  }
  return out;
}

// The affine decode variants of a shape at `rows` rows (its logical rows; @32 variants store 32).
void affineDecodeVariants(const Options &o, const Linear &linear, uint32_t cores, const Shape &s, uint32_t rows,
                          std::vector<Variant> &out) {
  const LinearWorkload w{{s.n, s.k}, rows, LinearPhase::Decode, s.epilogue};
  const LinearPlan policy = linear.plan(w);
  std::vector<LinearConfig> candidates;
  for (const LinearPlan &p : linear.candidates(w)) candidates.push_back(p.configuration());
  const auto add = [&](const LinearWorkload &workload, const LinearConfig &config, const std::string &suffix) {
    LinearPlan plan = Linear::plan(workload, config, s.destination);
    for (const Variant &v : out)
      if (v.logicalRows == rows && v.plans.front().workload() == workload &&
          v.plans.front().configuration() == config)
        return;
    Variant v;
    v.label = configLabel(config, LinearPhase::Decode) + suffix;
    v.logicalRows = rows;
    v.policy = workload.rows == rows && config == policy.configuration();
    v.candidate = workload.rows == rows &&
                  std::find(candidates.begin(), candidates.end(), config) != candidates.end();
    v.pipeline = std::string(plan.pipeline()) +
                 (plan.secondPipeline().empty() ? "" : "+" + std::string(plan.secondPipeline()));
    v.threads = plan.threadsPerThreadgroup();
    v.gridX = config.groups;
    v.gridY = config.splits;
    v.plans.push_back(std::move(plan));
    out.push_back(std::move(v));
  };
  const auto tryAdd = [&](const LinearWorkload &workload, const LinearConfig &config, const std::string &suffix) {
    try {
      add(workload, config, suffix);
    } catch (const std::invalid_argument &) {
      // No kernel instance for this configuration.
    }
  };
  add(w, policy.configuration(), "");
  // The sequential tiles and Split128 at this workload's rows, and the 32-row plans at 24 rows over four lanes of
  // storage.
  std::vector<std::pair<LinearWorkload, std::string>> workloads{{w, ""}};
  // Timing runs measure them as the 32-row variants: the same dispatches over the same storage.
  if (rows == 24 && o.check) workloads.push_back({{{s.n, s.k}, 32, LinearPhase::Decode, s.epilogue}, "@32"});
  for (const auto &[workload, suffix] : workloads) {
    for (const LinearTile tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128, LinearTile::Paired256}) {
      if (!keepTile(o, tile)) continue;
      const uint32_t columns = tile == LinearTile::N256 || tile == LinearTile::Paired256 ? 256 : 128;
      const uint32_t tiles = s.n / columns;
      for (const LinearSimdgroups sg : {LinearSimdgroups::Four, LinearSimdgroups::Eight}) {
        const LinearConfig pc = policy.configuration();
        const uint32_t policyGroups = pc.tile == tile && pc.simdgroups == sg ? pc.groups : 0;
        for (const uint32_t groups : groupCounts(o, tiles, cores, policyGroups))
          tryAdd(workload, {tile, groups, sg, 1}, suffix);
      }
    }
    if (keepTile(o, LinearTile::Split128))
      for (uint32_t splits = 2; splits <= LinearConfig::kMaximumSplits; splits *= 2)
        tryAdd(workload, {LinearTile::Split128, s.n / 128, LinearSimdgroups::Eight, splits}, suffix);
  }
  if (o.legacy && rows == kLaneRows) {
    tryAdd(w, {LinearTile::Split32, s.n / 32, LinearSimdgroups::Four, 1}, "");
    tryAdd(w, {LinearTile::Split64, s.n / 64, LinearSimdgroups::Eight, 1}, "");
  }
  // Another build's plan of this shape and rows, timed beside this build's.
  if (const auto it = o.baseline.find({s.label, rows}); it != o.baseline.end())
    if (const auto config = decodeConfig(it->second, s.n)) {
      tryAdd(w, *config, "");
      for (Variant &v : out)
        if (v.logicalRows == rows && v.plans.front().workload() == w && v.plans.front().configuration() == *config)
          v.baseline = true;
    }
}

// The prefill variants of a shape at `rows` chunk rows: the device's prefill plan and the other prefill tiles
// (N128/N256 at four or eight simdgroups; a gate/up pair runs its gate pass and up-with-gate pass on the same
// tile), and where the model pays one, the input-sums dispatch.
void prefillVariants(const Linear &linear, const Shape &s, uint32_t rows, std::vector<Variant> &out) {
  const bool gateUp = s.epilogue == LinearEpilogue::GateUp;
  const LinearEpilogue epilogue = gateUp ? LinearEpilogue::UpWithGate : s.epilogue;
  const LinearWorkload w{{s.n, s.k}, rows, LinearPhase::Prefill, epilogue};
  const LinearWorkload gate{{s.n, s.k}, rows, LinearPhase::Prefill, LinearEpilogue::None};
  const LinearConfig policy = linear.plan(w).configuration();
  std::vector<LinearConfig> configs{policy};
  for (const LinearTile tile : {LinearTile::N128, LinearTile::N256})
    for (const LinearSimdgroups sg : {LinearSimdgroups::Four, LinearSimdgroups::Eight}) {
      const LinearConfig config{tile, 0, sg, 1};
      if (std::find(configs.begin(), configs.end(), config) == configs.end()) configs.push_back(config);
    }
  for (const LinearConfig &config : configs) {
    Variant v;
    try {
      // The gate pass of a pair runs the plain kernel of the up pass's tile; the fused up-with-gate kernel has an
      // N128 instance only at four simdgroups.
      if (gateUp) v.plans.push_back(Linear::plan(gate, config));
      v.plans.push_back(Linear::plan(w, config));
    } catch (const std::invalid_argument &) {
      continue;
    }
    const LinearPlan &main = v.plans.back();
    v.logicalRows = rows;
    v.policy = config == policy;
    v.label = "prefill." + configLabel(main.configuration(), LinearPhase::Prefill) + (gateUp ? ".gate+up" : "");
    for (const LinearPlan &p : v.plans) v.pipeline += (v.pipeline.empty() ? "" : "+") + std::string(p.pipeline());
    v.threads = main.threadsPerThreadgroup();
    v.gridX = main.storageRows() / 32;
    v.gridY = s.n / main.tileColumns();
    out.push_back(std::move(v));
  }
  // The mixer output and draft context projections read sums of their own input (Linear::addPrefillSums).
  if (s.kind == "mixer_out" || s.kind == "d_context") {
    Variant sums;
    sums.logicalRows = rows;
    sums.sums = true;
    sums.label = "prefill.sums32";
    sums.pipeline = "prefill_linear_q4_sums32";
    sums.plans.push_back(linear.plan(gate));
    sums.threads = 256;
    sums.gridX = (rows + 31) / 32;
    sums.gridY = 1;
    out.push_back(std::move(sums));
  }
}

// ---------------------------------------------------------------- one shape: affine

struct Context {
  Options options;
  MetalBackend *backend = nullptr;
  const Linear *linear = nullptr;
  uint32_t cores = 0, family = 0;
  std::string device;
  std::ofstream csv;
  uint64_t checked = 0, timed = 0;
};

uint32_t storageOf(const Variant &v) {
  uint32_t rows = 0;
  for (const LinearPlan &p : v.plans) rows = std::max(rows, p.storageRows());
  return rows;
}

Sizes variantSizes(const Shape &s, const Variant &v) {
  Sizes z;
  const uint64_t rows = storageOf(v), n = s.n, k = s.k;
  z.input = rows * k * 2;
  z.output = rows * n * elementBytes(s.destination);
  if (s.epilogue == LinearEpilogue::Residual) z.residual = rows * n * 2;
  for (const LinearPlan &p : v.plans) {
    z.gateScratch = std::max(z.gateScratch, p.gateScratchBytes());
    z.sums = std::max(z.sums, p.sumsBytes());
    z.downSums = std::max(z.downSums, p.downSumsBytes());
    z.scratch.include(p.scratchSize());
  }
  // A prefill gate/up pair's gate pass writes the gate scratch the up pass reads.
  if (v.plans.size() == 2) z.gateScratch = std::max(z.gateScratch, rows * n * 2);
  if (v.sums) z.sums = std::max(z.sums, rows * (k / 64) * 4);
  return z;
}

void writeCsvHeader(Context &c) {
  c.csv << "device,family,cores,suite,run,order,shape,model,kind,count,emulated,n,k,epilogue,format,rows,"
           "storage_rows,label,tile,simdgroups,groups,splits,threads,grid_x,grid_y,tgs_per_core,pipeline,policy,"
           "candidate,samples,reps,median_ms,min_ms,max_ms,weight_mb,gbps,baseline\n";
}

double medianOf(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0 : v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

void report(Context &c, const Shape &s, const Variant &v, uint64_t weightBytes) {
  const double ms = medianOf(v.times) * 1e3;
  const double lo = *std::min_element(v.times.begin(), v.times.end()) * 1e3;
  const double hi = *std::max_element(v.times.begin(), v.times.end()) * 1e3;
  const LinearConfig config = v.plans.back().configuration();
  const double tgs = double(v.gridX) * v.gridY / c.cores;
  std::printf("%-28s %4u %-34s %9.4f ms %7.1f GB/s  [%.4f..%.4f]%s\n", s.label.c_str(), v.logicalRows,
              v.label.c_str(), ms, weightBytes / (ms * 1e-3) / 1e9, lo, hi, v.policy ? "  *" : "");
  if (!c.csv.is_open()) return;
  c.csv << c.device << ',' << c.family << ',' << c.cores << ',' << c.options.suite << ',' << c.options.run << ','
        << (c.options.reverse ? "reverse" : "forward") << ',' << s.label << ',' << s.model << ',' << s.kind << ','
        << s.count << ',' << s.emulated << ',' << s.n << ',' << s.k << ',' << epilogueName(s.epilogue) << ','
        << s.format << ',' << v.logicalRows << ',' << storageOf(v) << ',' << v.label << ','
        << (v.sums ? "sums32" : tileName(config.tile)) << ',' << uint32_t(config.simdgroups) << ',' << config.groups
        << ',' << config.splits << ',' << v.threads << ',' << v.gridX << ',' << v.gridY << ',' << tgs << ','
        << v.pipeline << ',' << int(v.policy) << ',' << int(v.candidate) << ',' << v.times.size() << ',' << v.reps
        << ',' << ms << ',' << lo << ',' << hi << ',' << weightBytes / 1e6 << ','
        << weightBytes / (ms * 1e-3) / 1e9 << ',' << int(v.baseline) << '\n';
  c.csv.flush();
}

// Times the variants of one shape: `add(graph, variant, copy, buffers)` encodes one repetition on weight copy
// `copy`. Every variant is first checked (`check`), then `fill` writes the inputs, the ring is streamed once, and
// the samples run.
using Encode = std::function<void(CommandGraph &, const Variant &, uint32_t, const Buffers &)>;
using CheckOne = std::function<void(Variant &)>;
using Fill = std::function<void(const Buffers &)>;
void timeShape(Context &c, const Shape &s, std::vector<Variant> &variants, uint32_t copies, uint64_t weightBytes,
               const Encode &add, const CheckOne &check, const Fill &fill) {
  if (c.options.check || c.options.selfCheck)
    for (Variant &v : variants) check(v);
  if (c.options.check) return;
  Sizes sizes;
  for (const Variant &v : variants) sizes.include(variantSizes(s, v));
  Buffers shared(*c.backend, sizes, 0);
  fill(shared);
  uint32_t cursor = 0;
  const auto sample = [&](Variant &v, uint32_t reps) {
    CommandGraph graph;
    for (uint32_t r = 0; r < reps; ++r) {
      add(graph, v, cursor, shared);
      cursor = (cursor + 1) % copies;
    }
    return c.backend->submitCommand(graph.dispatches()).gpuSeconds / reps;
  };
  // Every copy once, so no sample runs on pages the GPU has not touched yet.
  for (uint32_t done = 0; done < copies;) {
    const uint32_t reps = std::min<uint32_t>(copies - done, 64);
    (void)sample(variants.front(), reps);
    done += reps;
  }
  for (Variant &v : variants) {
    (void)sample(v, 2);
    const double t = sample(v, 4);
    v.reps = uint32_t(std::clamp(c.options.targetMs * 1e-3 / std::max(t, 1e-7), 2.0, 256.0));
  }
  if (c.options.reverse) std::reverse(variants.begin(), variants.end());
  for (uint32_t i = 0; i < c.options.samples; ++i)
    for (size_t j = 0; j < variants.size(); ++j) {
      Variant &v = variants[(i + j) % variants.size()];
      v.times.push_back(sample(v, v.reps));
    }
  if (c.options.reverse) std::reverse(variants.begin(), variants.end());
  if (!shared.intact()) fail(s.label + ": a guard band was overwritten during timing");
  if (!shared.countersZero()) fail(s.label + ": split counters are not back at zero after timing");
  for (const Variant &v : variants) {
    report(c, s, v, v.sums ? uint64_t(storageOf(v)) * s.k * 2 : weightBytes);
    ++c.timed;
  }
}

// Runs a variant twice on the same exact-size buffers (NaN partials before each run, NaN padding rows): the guard
// bands must stay intact, the counters return to zero and the two outputs of the active rows agree bit for bit.
// Returns the buffers of the second run.
std::unique_ptr<Buffers> runTwice(Context &c, const Shape &s, const Variant &v, const std::vector<uint16_t> &x,
                                  const std::vector<uint16_t> &residual,
                                  const std::function<void(CommandGraph &, const Buffers &)> &encode) {
  const Sizes sizes = variantSizes(s, v);
  const uint32_t storage = storageOf(v), rows = v.logicalRows;
  auto b = std::make_unique<Buffers>(*c.backend, sizes, 0xff);
  auto *input = static_cast<uint16_t *>(b->input.view.contents());
  for (uint64_t i = 0; i < uint64_t(storage) * s.k; ++i) input[i] = i < uint64_t(rows) * s.k ? x[i] : kNaN;
  if (b->residual.bytes) std::memcpy(b->residual.view.contents(), residual.data(), b->residual.bytes);
  const Guarded &result = v.sums ? b->sums : b->output;
  // Input sums are tiled by 32 rows (and deterministic for NaN padding rows too): compare all of them.
  const uint64_t active = v.sums ? b->sums.bytes : uint64_t(rows) * s.n * elementBytes(s.destination);
  std::vector<uint8_t> first;
  for (int pass = 0; pass < 2; ++pass) {
    std::fill_n(static_cast<uint32_t *>(b->partials.view.contents()), b->partials.bytes / 4, kNaN32);
    CommandGraph graph;
    encode(graph, *b);
    (void)c.backend->submitCommand(graph.dispatches());
    if (!b->intact()) fail(s.label + " " + v.label + ": a write past a buffer");
    if (!b->countersZero()) fail(s.label + " " + v.label + ": split counters are not back at zero");
    const auto *bytes = static_cast<const uint8_t *>(result.view.contents());
    if (!pass) first.assign(bytes, bytes + active);
    else if (std::memcmp(first.data(), bytes, active)) fail(s.label + " " + v.label + ": two runs differ");
  }
  ++c.checked;
  return b;
}

// Encodes one repetition of an affine variant on (p, gate): the input-sums dispatch, a prefill gate/up pair's two
// passes, or the plan's dispatches. With `sums` a prefill plan's input sums are written first.
void encodeAffine(const Linear &linear, CommandGraph &graph, const Variant &v, const Projection &p,
                  const Projection *gate, const Buffers &b, bool sums) {
  const LinearPlan &first = v.plans.front();
  const bool prefill = first.workload().phase == LinearPhase::Prefill;
  if (v.sums || (sums && prefill)) linear.addPrefillSums(graph, b.input.view, b.sums.view, p, v.logicalRows);
  if (v.sums) return;
  if (prefill && v.plans.size() == 2) {
    (void)linear.add(graph, {.input = b.input.view, .output = b.gateScratch.view, .sums = b.sums.view,
                             .scratch = b.scratch()}, *gate, v.plans[0]);
    (void)linear.add(graph, {.input = b.input.view, .output = b.output.view, .sums = b.sums.view,
                             .gateScratch = b.gateScratch.view, .downSums = b.downSums.used(), .scratch = b.scratch()},
                     p, v.plans[1]);
    return;
  }
  const bool gateUp = first.workload().epilogue == LinearEpilogue::GateUp;
  (void)linear.add(graph, {.input = b.input.view, .output = b.output.view, .sums = b.sums.used(),
                           .residual = b.residual.used(), .gateScratch = b.gateScratch.used(), .scratch = b.scratch()},
                   p, first, gateUp ? gate : nullptr);
}

// Holds a checked affine variant's active rows to their fp64 references: outputs (with the residual, or times silu
// of the gate), the gate scratch of a two-pass gate/up plan, the down sums of an up-with-gate pass, the input sums.
double affineValues(const Shape &s, const Variant &v, const Buffers &b, const std::vector<uint16_t> &x,
                    const std::vector<uint16_t> &residual, const Reference &ref, const Reference *gateRef) {
  const uint32_t rows = v.logicalRows, n = s.n, k = s.k;
  uint64_t outside = 0;
  double worst = 0;
  std::string first;
  const auto miss = [&](const std::string &what) {
    if (!outside++) first = what;
  };
  // Prefill sums are [32-row tile][64-input group][row of the tile] (kernels/prefill/linear_q4.metal).
  const auto sumIndex = [](uint32_t r, uint32_t g, uint32_t groups) {
    return uint64_t(r / 32) * 32 * groups + uint64_t(g) * 32 + r % 32;
  };
  if (v.sums) {
    const auto *sums = static_cast<const float *>(b.sums.view.contents());
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t g = 0; g < k / 64; ++g) {
        double want = 0, magnitude = 0;
        for (uint32_t j = 0; j < 64; ++j) {
          want += bf16ToFloat(x[uint64_t(r) * k + g * 64 + j]);
          magnitude += std::fabs(bf16ToFloat(x[uint64_t(r) * k + g * 64 + j]));
        }
        if (!(std::fabs(sums[sumIndex(r, g, k / 64)] - want) <= 64 * 0x1p-24 * magnitude + 1e-30))
          miss("sum row " + std::to_string(r) + " group " + std::to_string(g));
      }
  } else {
    const bool gateUp = s.epilogue == LinearEpilogue::GateUp;
    const auto *outBf16 = static_cast<const uint16_t *>(b.output.view.contents());
    const auto *outF32 = static_cast<const float *>(b.output.view.contents());
    const uint16_t *gateBf16 = b.gateScratch.bytes ? static_cast<const uint16_t *>(b.gateScratch.view.contents())
                                                   : nullptr;
    for (uint32_t r = 0; r < rows; ++r)
      for (uint32_t col = 0; col < n; ++col) {
        const uint64_t i = uint64_t(r) * n + col;
        const double got = s.destination == FloatOutput::Float32 ? double(outF32[i]) : double(bf16ToFloat(outBf16[i]));
        double want = ref.value[i], bound = ref.bound[r], gate = 0, up = 0, res = 0;
        if (s.epilogue == LinearEpilogue::Residual) {
          res = bf16ToFloat(residual[i]);
          want += res;
        }
        if (gateUp) {
          up = ref.value[i];
          // Two-pass plans multiply by silu of the bf16 gate they stored, fused ones by the fp32 gate.
          gate = gateBf16 ? double(bf16ToFloat(gateBf16[i])) : gateRef->value[i];
          want = silu(gate) * up;
          bound = std::max(bound, gateRef->bound[r]);
          if (gateBf16 && !(std::fabs(gate - gateRef->value[i]) <=
                            ulpBf16(float(gateRef->value[i])) + gateRef->bound[r]))
            miss("gate row " + std::to_string(r) + " column " + std::to_string(col));
        }
        const double tol = tolerance(s.epilogue, s.destination, want, bound, res, gate, up);
        worst = std::max(worst, std::fabs(got - want) / tol);
        if (!(std::isfinite(got) && std::fabs(got - want) <= tol))
          miss("row " + std::to_string(r) + " column " + std::to_string(col) + " got " + std::to_string(got) +
               " fp64 " + std::to_string(want) + " bound " + std::to_string(tol));
      }
    // An up-with-gate pass writes the down projection's input sums of its bf16 outputs.
    if (b.downSums.bytes) {
      const auto *d = static_cast<const float *>(b.downSums.view.contents());
      for (uint32_t r = 0; r < rows; ++r)
        for (uint32_t g = 0; g < n / 64; ++g) {
          double want = 0, magnitude = 0;
          for (uint32_t j = 0; j < 64; ++j) {
            const double value = bf16ToFloat(outBf16[uint64_t(r) * n + g * 64 + j]);
            want += value;
            magnitude += std::fabs(value);
          }
          if (!(std::fabs(d[sumIndex(r, g, n / 64)] - want) <= 64 * 0x1p-24 * magnitude + 1e-30))
            miss("down sum row " + std::to_string(r) + " group " + std::to_string(g));
        }
    }
  }
  if (outside)
    fail(s.label + " rows=" + std::to_string(rows) + " " + v.label + ": " + std::to_string(outside) +
         " values outside the fp64 bound; first: " + first);
  return worst;
}

// --list: the variants of a shape and the plans they run, without GPU work.
bool listed(Context &c, const Shape &s, const std::vector<Variant> &variants) {
  if (!c.options.list) return false;
  for (const Variant &v : variants) {
    std::printf("%-28s %4u %-30s %s %s\n", s.label.c_str(), v.logicalRows, v.label.c_str(), v.pipeline.c_str(),
                v.policy ? "*" : "");
    ++c.checked;
  }
  return true;
}

void runAffine(Context &c, const Shape &s) {
  const Options &o = c.options;
  const bool gateUp = s.epilogue == LinearEpilogue::GateUp;
  const bool prefill = o.suite == "prefill";
  std::vector<Variant> variants;
  if (prefill)
    for (const uint32_t rows : o.prefillRows) prefillVariants(*c.linear, s, rows, variants);
  for (const uint32_t rows : o.rows) affineDecodeVariants(o, *c.linear, c.cores, s, rows, variants);
  if (variants.empty() || listed(c, s, variants)) return;
  const AffineImage up = affineImage(s.n, s.k, 17 + s.n + s.k);
  std::unique_ptr<AffineImage> gate;
  if (gateUp) gate = std::make_unique<AffineImage>(affineImage(s.n, s.k, 1017 + s.n + s.k));
  const uint64_t perCopy = (up.weights.size() + up.scales.size() + up.biases.size()) * (gateUp ? 2 : 1);
  const uint32_t copies = o.check ? 1 : ringCopies(perCopy, o.ringBytes);
  const Ring upRing = makeRing(*c.backend, {&up.weights, &up.scales, &up.biases}, copies);
  Ring gateRing;
  if (gateUp) gateRing = makeRing(*c.backend, {&gate->weights, &gate->scales, &gate->biases}, copies);
  std::vector<Projection> projections, gates;
  for (uint32_t i = 0; i < copies; ++i) {
    projections.emplace_back(s.n, s.k,
                             AffineWeights{upRing.copy(*c.backend, 0, i, up.weights.size()),
                                           upRing.copy(*c.backend, 1, i, up.scales.size()),
                                           upRing.copy(*c.backend, 2, i, up.biases.size())});
    projections.back().destination = s.destination;
    if (gateUp)
      gates.emplace_back(s.n, s.k,
                         AffineWeights{gateRing.copy(*c.backend, 0, i, gate->weights.size()),
                                       gateRing.copy(*c.backend, 1, i, gate->scales.size()),
                                       gateRing.copy(*c.backend, 2, i, gate->biases.size())});
  }
  uint32_t maxRows = 0;
  for (const Variant &v : variants) maxRows = std::max({maxRows, storageOf(v), v.logicalRows});
  std::vector<uint16_t> x(uint64_t(maxRows) * s.k), residual(uint64_t(maxRows) * s.n);
  {
    std::mt19937 random(7 + s.k);
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (auto &e : x) e = floatToBf16(value(random));
    for (uint64_t i = 0; i < residual.size(); ++i) residual[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  }
  std::unique_ptr<Reference> ref, gateRef;
  uint32_t checkRows = 0;
  for (const Variant &v : variants) checkRows = std::max(checkRows, v.logicalRows);
  const auto check = [&](Variant &v) {
    const auto encode = [&](CommandGraph &graph, const Buffers &b) {
      encodeAffine(*c.linear, graph, v, projections[0], gateUp ? &gates[0] : nullptr, b, true);
    };
    const std::unique_ptr<Buffers> b = runTwice(c, s, v, x, residual, encode);
    if (!o.check) return;
    if (!ref) {
      ref = std::make_unique<Reference>(affineReference(up, x, checkRows));
      if (gateUp) gateRef = std::make_unique<Reference>(affineReference(*gate, x, checkRows));
    }
    const double worst = affineValues(s, v, *b, x, residual, *ref, gateRef.get());
    std::printf("PASS %-28s %4u %-26s worst %.3f of the bound (%s)\n", s.label.c_str(), v.logicalRows,
                v.label.c_str(), worst, v.pipeline.c_str());
  };
  const auto encode = [&](CommandGraph &graph, const Variant &v, uint32_t copy, const Buffers &b) {
    encodeAffine(*c.linear, graph, v, projections[copy], gateUp ? &gates[copy] : nullptr, b, false);
  };
  const auto fill = [&](const Buffers &b) {
    std::memcpy(b.input.view.contents(), x.data(), std::min<uint64_t>(b.input.bytes, x.size() * 2));
    if (b.residual.bytes)
      std::memcpy(b.residual.view.contents(), residual.data(), std::min<uint64_t>(b.residual.bytes, residual.size() * 2));
  };
  timeShape(c, s, variants, copies, perCopy, encode, check, fill);
}

// ---------------------------------------------------------------- one shape: gguf

void runGguf(Context &c, const Shape &s) {
  using namespace gguf_reference;
  const Options &o = c.options;
  MetalBackend &backend = *c.backend;
  const Fmt format = fmtNamed(s.format);
  if (format == FMT_COUNT) fail("unknown GGUF format " + s.format);
  const bool gateUp = s.epilogue == LinearEpilogue::GateUp;
  std::mt19937 rng(9 + s.n);
  std::vector<float> upValues, gateValues;
  const Packed upPacked = repack(format, makeNative(format, s.n, s.k, rng), s.n, s.k, o.check ? &upValues : nullptr);
  Packed gatePacked;
  if (gateUp) gatePacked = repack(format, makeNative(format, s.n, s.k, rng), s.n, s.k, o.check ? &gateValues : nullptr);
  const bool hasPlane1 = kQuantFormats[format].plane1_bytes != 0;
  const uint64_t perImage = upPacked.w0.size() + (hasPlane1 ? upPacked.w1.size() : 0) + upPacked.meta.size();
  const uint64_t perCopy = perImage * (gateUp ? 2 : 1);
  const uint32_t copies = o.check ? 1 : ringCopies(perCopy, o.ringBytes);
  const auto ringOf = [&](const Packed &p) { return makeRing(backend, {&p.w0, &p.w1, &p.meta}, copies); };
  const Ring upRing = ringOf(upPacked);
  Ring gateRing;
  if (gateUp) gateRing = ringOf(gatePacked);
  const auto projectionOf = [&](const Ring &ring, const Packed &p, uint32_t i) {
    BlockWeights weights;
    weights.segments.push_back(QuantizedSegment::planes(
        format, s.n, s.k, ring.copy(backend, 0, i, p.w0.size()),
        hasPlane1 ? ring.copy(backend, 1, i, p.w1.size()) : MetalBuffer{}, ring.copy(backend, 2, i, p.meta.size())));
    Projection projection(s.n, s.k, std::move(weights));
    projection.destination = s.destination;
    return projection;
  };
  std::vector<Projection> projections, gates;
  for (uint32_t i = 0; i < copies; ++i) {
    projections.push_back(projectionOf(upRing, upPacked, i));
    if (gateUp) gates.push_back(projectionOf(gateRing, gatePacked, i));
  }
  std::vector<Variant> variants;
  for (const uint32_t rows : o.rows) {
    const LinearWorkload w{{s.n, s.k}, rows, LinearPhase::Decode, s.epilogue, WeightLayout::Block32};
    const LinearPlan policy = c.linear->plan(w, projections[0], gateUp ? &gates[0] : nullptr);
    for (const LinearTile tile : {LinearTile::GgufStaged, LinearTile::GgufRegister}) {
      if (!keepTile(o, tile) && policy.configuration().tile != tile) continue;
      for (uint32_t splits = 1; splits <= LinearConfig::kMaximumSplits; splits *= 2) {
        const bool registerTile = tile == LinearTile::GgufRegister;
        const LinearConfig config{tile, s.n / 64, registerTile ? LinearSimdgroups::Four : LinearSimdgroups::Two,
                                  splits};
        if (!keepTile(o, tile) && config != policy.configuration()) continue;
        try {
          LinearPlan plan = Linear::plan(w, config, s.destination);
          Variant v;
          v.label = configLabel(config, LinearPhase::Decode);
          v.logicalRows = rows;
          v.policy = config == policy.configuration();
          v.threads = registerTile ? 128 : 64;
          v.gridX = s.n / 64;
          v.gridY = splits;
          v.pipeline = s.format + (registerTile ? "_register" : "_staged");
          v.plans.push_back(std::move(plan));
          variants.push_back(std::move(v));
        } catch (const std::invalid_argument &) {
          // No such configuration (K splits of whole units).
        }
      }
    }
  }
  if (variants.empty() || listed(c, s, variants)) return;
  std::vector<uint16_t> x(uint64_t(kMaximumRows) * s.k), residual(uint64_t(kMaximumRows) * s.n);
  {
    std::uniform_real_distribution<float> value(-1.0f, 1.0f);
    for (auto &e : x) e = floatToBf16(value(rng));
    for (uint64_t i = 0; i < residual.size(); ++i) residual[i] = floatToBf16(float(int(i % 31) - 15) / 8);
  }
  // fp64 dots of the rows with the GGML weight values (GgufFormatReference.hpp), at the first check.
  std::vector<Dot> dots, gateDots;
  const auto referenceDots = [&](const std::vector<float> &values, std::vector<Dot> &out) {
    out.assign(uint64_t(kMaximumRows) * s.n, {});
    std::vector<float> xf(x.size());
    for (uint64_t i = 0; i < x.size(); ++i) xf[i] = bf16ToFloat(x[i]);
    const float *xp = xf.data(), *wp = values.data();
    Dot *outp = out.data();
    const uint32_t n = s.n, k = s.k;
    dispatch_apply(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t col) {
      for (uint32_t r = 0; r < kMaximumRows; ++r) outp[uint64_t(r) * n + col] = dot(xp + uint64_t(r) * k, wp + col * uint64_t(k), k);
    });
  };
  const auto encodeOn = [&](CommandGraph &graph, const Variant &v, uint32_t copy, const Buffers &b) {
    (void)c.linear->add(graph, {.input = b.input.view, .output = b.output.view, .residual = b.residual.used(),
                                .gateScratch = b.gateScratch.used(), .scratch = b.scratch()},
                        projections[copy], v.plans.front(), gateUp ? &gates[copy] : nullptr);
  };
  const auto check = [&](Variant &v) {
    const std::unique_ptr<Buffers> b =
        runTwice(c, s, v, x, residual, [&](CommandGraph &graph, const Buffers &b) { encodeOn(graph, v, 0, b); });
    if (!o.check) return;
    if (dots.empty()) {
      referenceDots(upValues, dots);
      if (gateUp) referenceDots(gateValues, gateDots);
    }
    const bool staged = v.plans.front().configuration().tile == LinearTile::GgufStaged;
    const auto *outBf16 = static_cast<const uint16_t *>(b->output.view.contents());
    const auto *outF32 = static_cast<const float *>(b->output.view.contents());
    const auto *gateOut = gateUp ? static_cast<const uint16_t *>(b->gateScratch.view.contents()) : nullptr;
    uint64_t outside = 0;
    double worst = 0;
    std::string first;
    for (uint64_t i = 0; i < uint64_t(v.logicalRows) * s.n; ++i) {
      const double got = s.destination == FloatOutput::Float32 ? double(outF32[i]) : double(bf16ToFloat(outBf16[i]));
      double want = dots[i].value, bound = projectionBound(dots[i], staged), gate = 0, res = 0;
      const double up = dots[i].value;
      if (s.epilogue == LinearEpilogue::Residual) {
        res = bf16ToFloat(residual[i]);
        want += res;
      }
      if (gateUp) {
        // The up pass multiplies by silu of the bf16 gate the gate pass stored.
        gate = bf16ToFloat(gateOut[i]);
        const double gateBound = projectionBound(gateDots[i], staged);
        if (!(std::fabs(gate - gateDots[i].value) <= ulpBf16(float(gateDots[i].value)) + gateBound) && !outside++)
          first = "gate element " + std::to_string(i);
        want = silu(gate) * up;
        bound = std::max(bound, gateBound);
      }
      const double tol = tolerance(s.epilogue, s.destination, want, bound, res, gate, up);
      worst = std::max(worst, std::fabs(got - want) / tol);
      if (!(std::isfinite(got) && std::fabs(got - want) <= tol) && !outside++)
        first = "element " + std::to_string(i) + " got " + std::to_string(got) + " fp64 " + std::to_string(want) +
                " bound " + std::to_string(tol);
    }
    if (outside)
      fail(s.label + " rows=" + std::to_string(v.logicalRows) + " " + v.label + ": " + std::to_string(outside) +
           " values outside the fp64 bound; first: " + first);
    std::printf("PASS %-28s %4u %-26s worst %.3f of the bound\n", s.label.c_str(), v.logicalRows, v.label.c_str(),
                worst);
  };
  const auto fill = [&](const Buffers &b) {
    std::memcpy(b.input.view.contents(), x.data(), std::min<uint64_t>(b.input.bytes, x.size() * 2));
    if (b.residual.bytes)
      std::memcpy(b.residual.view.contents(), residual.data(), std::min<uint64_t>(b.residual.bytes, residual.size() * 2));
  };
  timeShape(c, s, variants, copies, perCopy, encodeOn, check, fill);
}

// ---------------------------------------------------------------- bandwidth

// Stream-read bandwidth of a 1 GiB buffer: 256-thread threadgroups, `per core` of them, each thread reading
// uint4s at a grid stride. A runtime-compiled kernel, independent of the production library.
void runBandwidth(Context &c) {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  NSString *source = @"#include <metal_stdlib>\nusing namespace metal;\n"
                      "kernel void stream_read(device const uint4 *src [[buffer(0)]], device uint *out [[buffer(1)]],"
                      " constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]],"
                      " uint threads [[threads_per_grid]]) {\n"
                      "  uint4 acc = 0;\n"
                      "  for (uint i = id; i < count; i += threads) acc ^= src[i];\n"
                      "  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) out[0] = acc.x;\n"
                      "}\n";
  id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
  if (!library) fail("stream kernel: " + std::string(error.localizedDescription.UTF8String));
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"stream_read"] error:&error];
  if (!pipeline) fail("stream pipeline: " + std::string(error.localizedDescription.UTF8String));
  const uint64_t bytes = 1ull << 30;
  id<MTLBuffer> src = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
  id<MTLBuffer> out = [device newBufferWithLength:16 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> queue = [device newCommandQueue];
  const uint32_t count = uint32_t(bytes / 16);
  const auto run = [&](uint32_t groups, uint32_t reps) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&count length:4 atIndex:2];
    for (uint32_t r = 0; r < reps; ++r)
      [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return (command.GPUEndTime - command.GPUStartTime) / reps;
  };
  (void)run(c.cores * 8, 2);
  for (const uint32_t perCore : {1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 24u, 32u}) {
    std::vector<double> times;
    for (uint32_t i = 0; i < c.options.samples; ++i) times.push_back(run(perCore * c.cores, 4));
    const double t = medianOf(times);
    std::printf("stream-read %2u threadgroups/core: %7.1f GB/s\n", perCore, bytes / t / 1e9);
    if (c.csv.is_open())
      c.csv << c.device << ',' << c.family << ',' << c.cores << ",bandwidth," << c.options.run << ",forward,stream,,,0,0,"
            << bytes << ",0,none,,0,0,stream.tg" << perCore << ",stream,8,0,1,256," << perCore * c.cores << ",1,"
            << perCore << ",stream_read,0,0," << times.size() << ",4," << t * 1e3 << ','
            << *std::min_element(times.begin(), times.end()) * 1e3 << ','
            << *std::max_element(times.begin(), times.end()) * 1e3 << ',' << bytes / 1e6 << ',' << bytes / t / 1e9
            << ",0\n";
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc < 2) {
      std::cerr << "usage: policy-bench <metallib> [--suite affine|prefill|gguf|bandwidth] [options]\n";
      return 2;
    }
    Context c;
    Options &o = c.options;
    try {
      for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
          if (i + 1 >= argc) fail("missing value for " + a);
          return argv[++i];
        };
        if (a == "--suite") o.suite = next();
        else if (a == "--shapes") o.shapeSets = splitList(next());
        else if (a == "--filter") o.filters = splitList(next());
        else if (a == "--emulate") o.emulate = numbers(next());
        else if (a == "--rows") o.rows = numbers(next());
        else if (a == "--prefill-rows") o.prefillRows = numbers(next());
        else if (a == "--groups") o.groups = splitList(next());
        else if (a == "--formats") o.formats = splitList(next());
        else if (a == "--tiles") o.tiles = splitList(next());
        else if (a == "--samples") o.samples = uint32_t(std::stoul(next()));
        else if (a == "--target-ms") o.targetMs = std::stod(next());
        else if (a == "--ring-mib") o.ringBytes = uint64_t(std::stoull(next())) << 20;
        else if (a == "--order") o.reverse = next() == "reverse";
        else if (a == "--run") o.run = uint32_t(std::stoul(next()));
        else if (a == "--csv") o.csv = next();
        else if (a == "--legacy") o.legacy = true;
        else if (a == "--check") o.check = true;
        else if (a == "--no-self-check") o.selfCheck = false;
        else if (a == "--list") o.list = true;
        else if (a == "--baseline") o.baseline = readBaseline(next());
        else fail("unknown option " + a);
      }
      if (o.suite != "affine" && o.suite != "prefill" && o.suite != "gguf" && o.suite != "bandwidth")
        fail("unknown suite " + o.suite);
      for (const uint32_t rows : o.rows)
        if (!rows || rows % kLaneRows || rows > kMaximumRows) fail("decode rows are 8, 16, 24 or 32");
      if (o.samples < 1) fail("--samples must be positive");
      std::cout << std::unitbuf;
      MetalBackend backend(argv[1]);
      const DeviceCapabilities &caps = backend.capabilities();
      const Linear linear(caps);
      c.backend = &backend;
      c.linear = &linear;
      c.cores = caps.gpuCoreCount;
      c.family = caps.appleGpuFamily;
      c.device = caps.deviceName;
      std::replace(c.device.begin(), c.device.end(), ',', ' ');
      const char *validation = std::getenv("MTL_SHADER_VALIDATION");
      std::printf("# policy-bench device=\"%s\" family=%u cores=%u macos=%s suite=%s%s shader_validation=%s\n",
                  caps.deviceName.c_str(), caps.appleGpuFamily, caps.gpuCoreCount, caps.macosVersion().c_str(),
                  o.suite.c_str(), o.check ? " check" : "", validation ? validation : "0");
      if (o.check && (!validation || std::string(validation) != "1"))
        std::printf("# warning: --check without MTL_SHADER_VALIDATION=1 does not bound-check the kernels\n");
      if (!o.csv.empty()) {
        struct stat st;
        const bool fresh = stat(o.csv.c_str(), &st) != 0 || st.st_size == 0;
        c.csv.open(o.csv, std::ios::app);
        if (!c.csv) fail("cannot open " + o.csv);
        if (fresh) writeCsvHeader(c);
      }
      if (o.suite == "bandwidth") {
        runBandwidth(c);
        return 0;
      }
      const std::vector<Shape> shapes = selectShapes(o, c.cores);
      for (const Shape &s : shapes) {
        if (o.suite == "gguf") runGguf(c, s);
        else runAffine(c, s);
      }
      if (o.list) std::printf("# %llu configurations\n", (unsigned long long)c.checked);
      else if (o.check) std::printf("# checked %llu configurations: PASS\n", (unsigned long long)c.checked);
      else std::printf("# timed %llu configurations (self-checked %llu)\n", (unsigned long long)c.timed,
                       (unsigned long long)c.checked);
    } catch (const std::exception &e) {
      std::fflush(stdout);
      std::cerr << "policy-bench: " << e.what() << '\n';
      return 1;
    }
  }
  return 0;
}
