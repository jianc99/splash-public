// Every production plan of the kernel policy, one line each, so that two
// builds can be diffed: the affine and GGUF projections of the 27B and 35B
// targets and their drafts (with K % 1024 != 0 controls) at every decode
// width and prefill chunk length, with their tuning candidates and arena
// bounds, the GGUF float tile, and the 35B MoE layer, on GPU families 9 to
// 11 at the core counts of the measured and emulated machines and an unknown
// count. In each build:
//   build/engine-tests/plan-census > plans.txt
// then diff the two files; a policy change lists exactly the plans it moves.
#include "ops/ExecutionPlans.hpp"
#include "metal/abi/QuantFormat.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;

struct Shape final {
  const char *name;
  LinearMatrix matrix;
  LinearEpilogue epilogue;
  // The vocabulary heads write fp32 logits and run in decode only.
  FloatOutput destination = FloatOutput::BFloat16;
};

// The decode projections of the Qwen3.8-27B and Qwen3.6-35B-A3B targets and
// their DFlash drafts (d_), the 35B shared expert, and shapes with
// K % 1024 != 0.
constexpr std::array kShapes{
    Shape{"27b.gdn_in", {16640, 5120}, LinearEpilogue::None},
    Shape{"27b.attn_in", {14336, 5120}, LinearEpilogue::None},
    Shape{"27b.mixer_out", {5120, 6144}, LinearEpilogue::Residual},
    Shape{"27b.gate_up", {17408, 5120}, LinearEpilogue::GateUp},
    Shape{"27b.down", {5120, 17408}, LinearEpilogue::Residual},
    Shape{"27b.lm_head", {248320, 5120}, LinearEpilogue::None, FloatOutput::Float32},
    Shape{"27b.d_context", {5120, 25600}, LinearEpilogue::None},
    Shape{"27b.d_qkv", {6144, 5120}, LinearEpilogue::None},
    Shape{"27b.d_dynamic", {1280, 5120}, LinearEpilogue::None},
    Shape{"27b.d_out", {5120, 4096}, LinearEpilogue::None},
    Shape{"27b.d_down", {5120, 17408}, LinearEpilogue::None},
    Shape{"27b.d_selector", {256, 5120}, LinearEpilogue::None},
    Shape{"35b.gdn_in", {12544, 2048}, LinearEpilogue::None},
    Shape{"35b.attn_in", {9216, 2048}, LinearEpilogue::None},
    Shape{"35b.mixer_out", {2048, 4096}, LinearEpilogue::Residual},
    Shape{"35b.lm_head", {248320, 2048}, LinearEpilogue::None, FloatOutput::Float32},
    Shape{"35b.shared_gate_up", {512, 2048}, LinearEpilogue::GateUp},
    Shape{"35b.shared_down", {2048, 512}, LinearEpilogue::Residual},
    Shape{"35b.d_context", {2048, 16384}, LinearEpilogue::None},
    Shape{"35b.d_qkv", {6144, 2048}, LinearEpilogue::None},
    Shape{"35b.d_dynamic", {512, 2048}, LinearEpilogue::None},
    Shape{"35b.d_out", {2048, 4096}, LinearEpilogue::None},
    Shape{"35b.d_gate_up", {6144, 2048}, LinearEpilogue::GateUp},
    Shape{"35b.d_down", {2048, 6144}, LinearEpilogue::None},
    Shape{"35b.d_down_residual", {2048, 6144}, LinearEpilogue::Residual},
    Shape{"35b.d_selector", {256, 2048}, LinearEpilogue::None},
    Shape{"k4352", {5120, 4352}, LinearEpilogue::None},
    Shape{"k4352_residual", {5120, 4352}, LinearEpilogue::Residual},
    Shape{"k4352_gate_up", {6144, 4352}, LinearEpilogue::GateUp},
    Shape{"k768", {2048, 768}, LinearEpilogue::None}};

// GGUF segment formats: Q4_K keeps Apple9's register tile, IQ2_XS stages
// wherever the staged tile holds the lanes' rows, Q2_K stages from two lanes,
// and a projection mixing a staged format with Q4_K keeps the register tile.
struct Formats final {
  const char *name;
  std::vector<uint32_t> segments;
};
const std::array<Formats, 4> kFormats{{{"q4k", {GGUF_FMT_Q4K}},
                                       {"iq2xs", {GGUF_FMT_IQ2XS}},
                                       {"q2k", {GGUF_FMT_Q2K}},
                                       {"iq3xxs+q4k", {GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K}}}};

constexpr std::array<uint32_t, 3> kFamilies{9, 10, 11};
// Zero is the unknown count.
constexpr std::array<uint32_t, 15> kCores{0, 10, 12, 14, 16, 18, 20, 24, 30, 32, 38, 40, 60, 76, 80};
constexpr std::array<uint32_t, 21> kPrefillRows{1,  4,  8,  12, 16, 20,  24,  28,   31,  32,  33,
                                                36, 40, 48, 64, 96, 128, 256, 512, 1024, 2048};
constexpr std::array<uint32_t, 4> kDecodeLanes{1, 2, 3, 4};

const char *tileName(LinearTile tile) {
  switch (tile) {
  case LinearTile::N128: return "N128";
  case LinearTile::N256: return "N256";
  case LinearTile::Paired128: return "Paired128";
  case LinearTile::Split32: return "Split32";
  case LinearTile::Split64: return "Split64";
  case LinearTile::Split128: return "Split128";
  case LinearTile::Paired256: return "Paired256";
  case LinearTile::Simdgroup: return "Simdgroup";
  case LinearTile::GgufStaged: return "GgufStaged";
  case LinearTile::GgufRegister: return "GgufRegister";
  }
  return "unknown";
}
const char *epilogueName(LinearEpilogue epilogue) {
  switch (epilogue) {
  case LinearEpilogue::None: return "none";
  case LinearEpilogue::Residual: return "residual";
  case LinearEpilogue::GateUp: return "gate_up";
  case LinearEpilogue::UpWithGate: return "up_with_gate";
  }
  return "unknown";
}
const char *inputName(LinearInput input) {
  switch (input) {
  case LinearInput::Plain: return "plain";
  case LinearInput::Table64: return "table64";
  case LinearInput::Table16: return "table16";
  }
  return "unknown";
}

std::string text(const LinearScratchSize &s) {
  return std::to_string(s.input) + '/' + std::to_string(s.sums) + '/' + std::to_string(s.partials) + '/' +
         std::to_string(s.counters) + '/' + std::to_string(s.rotated);
}
std::string text(const LinearConfig &c) {
  return std::string(tileName(c.tile)) + " g" + std::to_string(c.groups) + " sg" +
         std::to_string(static_cast<uint32_t>(c.simdgroups)) + " s" + std::to_string(c.splits);
}
std::string text(const LinearPlan &p) {
  std::string line = text(p.configuration()) + " | " + std::string(p.pipeline());
  if (!p.secondPipeline().empty()) line += '+' + std::string(p.secondPipeline());
  line += " | rows " + std::to_string(p.storageRows()) + " cols " + std::to_string(p.tileColumns()) + " threads " +
          std::to_string(p.threadsPerThreadgroup()) + " partials " + std::to_string(p.partialSums()) + ' ' +
          inputName(p.input()) + (p.destination() == FloatOutput::Float32 ? " fp32" : "") + " | scratch " +
          text(p.scratchSize()) + " sums " + std::to_string(p.sumsBytes()) + " gate " +
          std::to_string(p.gateScratchBytes()) + " down " + std::to_string(p.downSumsBytes());
  return line;
}
std::string text(const MoeWorkspace &w) {
  std::string line;
  for (const auto field : kMoeWorkspaceFields) line += (line.empty() ? "" : "/") + std::to_string(w.*field);
  return line;
}
std::string text(const MoeConfig &c) {
  return "tile M" + std::to_string(static_cast<uint32_t>(c.expertTile)) + " route " +
         std::to_string(c.routeWideRows) + " m8sg " + std::to_string(static_cast<uint32_t>(c.m8Simdgroups)) +
         (c.ggufTile == MoeGgufTile::Register ? " register" : " staged") +
         (c.ggufRouterTile == FloatTile::NeuralAccelerator ? " router-na" : " router-simdgroup");
}
std::string text(const MoePlan &p) {
  return text(p.configuration()) + " | rows " + std::to_string(p.tileRows()) + " split " +
         std::to_string(p.splitExperts()) + " tiles " + std::to_string(p.maximumTiles()) + " | workspace " +
         text(p.workspace());
}

// A block projection of equal segments tiling its columns, one per format;
// planning reads only their geometry and formats.
Projection blockProjection(LinearMatrix matrix, std::span<const uint32_t> formats) {
  BlockWeights weights;
  const uint32_t width = matrix.outputSize / uint32_t(formats.size());
  for (const uint32_t format : formats) {
    QuantizedSegment s = QuantizedSegment::planes(format, width, matrix.inputSize, {}, {}, {});
    s.columnOffset = uint32_t(weights.segments.size()) * width;
    weights.segments.push_back(s);
  }
  return Projection(matrix.outputSize, matrix.inputSize, std::move(weights));
}

void printLinear(const std::string &device, const ExecutionPlans &plans, const Shape &shape, Projection p,
                 const std::string &layout) {
  const Linear &linear = plans.linear();
  p.destination = shape.destination;
  const std::string prefix = device + ' ' + shape.name + ' ' + std::to_string(shape.matrix.outputSize) + 'x' +
                             std::to_string(shape.matrix.inputSize) + ' ' + layout + ' ';
  // A gate/up plan's gate is a projection like its up projection.
  const Projection *gate = shape.epilogue == LinearEpilogue::GateUp ? &p : nullptr;
  for (const uint32_t lanes : kDecodeLanes) {
    const LinearWorkload w{shape.matrix, lanes * 8, LinearPhase::Decode, shape.epilogue, p.layout()};
    std::string candidates;
    for (const LinearPlan &candidate : linear.candidates(w))
      candidates += (candidates.empty() ? "" : ", ") + text(candidate.configuration());
    std::printf("%sdecode r%u %s: %s | candidates %s | arena %s\n", prefix.c_str(), w.rows,
                epilogueName(w.epilogue), text(linear.plan(w, p, gate)).c_str(), candidates.c_str(),
                text(linear.decodeScratchSize(w)).c_str());
  }
  if (shape.destination == FloatOutput::Float32) return;
  // Prefill runs gate/up as the gate projection and the up projection with
  // the SiLU gate.
  const std::vector<LinearEpilogue> prefillEpilogues =
      shape.epilogue == LinearEpilogue::GateUp
          ? std::vector<LinearEpilogue>{LinearEpilogue::None, LinearEpilogue::UpWithGate}
          : std::vector<LinearEpilogue>{shape.epilogue};
  for (const LinearEpilogue epilogue : prefillEpilogues)
    for (const uint32_t rows : kPrefillRows) {
      const LinearWorkload w{shape.matrix, rows, LinearPhase::Prefill, epilogue, p.layout()};
      std::string candidates;
      for (const LinearPlan &candidate : linear.candidates(w))
        candidates += (candidates.empty() ? "" : ", ") + text(candidate.configuration());
      std::printf("%sprefill r%u %s: %s | candidates %s\n", prefix.c_str(), rows, epilogueName(epilogue),
                  text(linear.plan(w, p)).c_str(), candidates.c_str());
    }
  const ProjectionShape bounds = p.shape();
  std::printf("%sarena: prefill %s", prefix.c_str(), text(linear.prefillScratchSize(bounds)).c_str());
  if (shape.epilogue == LinearEpilogue::GateUp)
    std::printf(" gate/up %llu", static_cast<unsigned long long>(plans.gateUpWorkspace(bounds)));
  std::printf("\n");
}

void printDevice(uint32_t family, uint32_t cores) {
  DeviceCapabilities capabilities;
  capabilities.appleGpuFamily = family;
  capabilities.gpuCoreCount = cores;
  const ExecutionPlans plans(capabilities);
  const std::string device = "f" + std::to_string(family) + " c" + std::to_string(cores);
  for (const Shape &shape : kShapes) {
    printLinear(device, plans, shape, Projection(shape.matrix.outputSize, shape.matrix.inputSize, AffineWeights{}),
                "affine");
    for (const Formats &formats : kFormats) {
      // A gate/up pair and an epilogue take single tensors.
      if (formats.segments.size() > 1 && shape.epilogue != LinearEpilogue::None) continue;
      printLinear(device, plans, shape, blockProjection(shape.matrix, formats.segments),
                  std::string("gguf:") + formats.name);
    }
  }
  // GGUF float segments: the 35B router (256 columns) and alpha/beta (64),
  // the 27B alpha/beta (96).
  for (const uint32_t n : {64U, 96U, 256U}) {
    std::string tiles;
    for (const uint32_t rows : kPrefillRows)
      tiles += (tiles.empty() ? "" : " ") +
               std::string(plans.linear().ggufFloatTile(rows, n) == FloatTile::NeuralAccelerator ? "na" : "simd");
    std::printf("%s float %u columns prefill rows: %s\n", device.c_str(), n, tiles.c_str());
  }
  // The 35B MoE layer in its affine and GGUF weights.
  const std::array<std::pair<const char *, MoeShape>, 4> moeShapes{{
      {"35b.moe affine", {2048, 256, 8, 512}},
      {"35b.moe gguf:q4k", {2048, 256, 8, 512, WeightLayout::Block32, GGUF_FMT_Q4K}},
      {"35b.moe gguf:iq2xs", {2048, 256, 8, 512, WeightLayout::Block32, GGUF_FMT_IQ2XS}},
      {"35b.moe gguf:q2k", {2048, 256, 8, 512, WeightLayout::Block32, GGUF_FMT_Q2K}}}};
  for (const auto &[name, shape] : moeShapes) {
    const auto candidates = [&](const MoeWorkload &w) {
      std::string line;
      for (const MoePlan &candidate : plans.moeCandidates(w))
        line += (line.empty() ? "" : ", ") + text(candidate.configuration());
      return line;
    };
    for (const uint32_t lanes : kDecodeLanes)
      std::printf("%s %s decode r%u: %s | candidates %s\n", device.c_str(), name, lanes * 8,
                  text(plans.moeDecode(shape, lanes)).c_str(),
                  candidates({shape, lanes * 8, MoePhase::Decode}).c_str());
    for (const uint32_t rows : kPrefillRows)
      std::printf("%s %s prefill r%u: %s | candidates %s\n", device.c_str(), name, rows,
                  text(plans.moePrefill(shape, rows)).c_str(), candidates({shape, rows, MoePhase::Prefill}).c_str());
    std::printf("%s %s arena: decode per lane %s prefill %s\n", device.c_str(), name,
                text(plans.moeDecodeWorkspacePerLane(shape)).c_str(),
                text(plans.moePrefillWorkspace(shape, SPLASH_PREFILL_TOKEN_BUDGET)).c_str());
  }
}

} // namespace

int main() {
  try {
    for (const uint32_t family : kFamilies)
      for (const uint32_t cores : kCores) printDevice(family, cores);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "plan-census: %s\n", error.what());
    return 1;
  }
}
