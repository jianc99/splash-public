#include "AffineQ4Fixture.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/QuantFormat.h"
#include "tuning/LinearNumerics.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using namespace splash;
using namespace splash::ops;
using test::mix;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Function> void rejects(Function function) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error("invalid Linear plan or buffer was accepted");
}


Linear gpu(uint32_t family, uint32_t cores) {
  DeviceCapabilities device;
  device.appleGpuFamily = family;
  device.gpuCoreCount = cores;
  return Linear(device);
}

// Throws naming the rule a plan broke and the plan.
[[noreturn]] void broke(const char *rule, uint32_t family, uint32_t cores, LinearWorkload w) {
  throw std::runtime_error(std::string(rule) + ": family " + std::to_string(family) + ", " +
                           std::to_string(cores) + " cores, N " + std::to_string(w.matrix.outputSize) +
                           ", K " + std::to_string(w.matrix.inputSize) + ", " + std::to_string(w.rows) +
                           (w.phase == LinearPhase::Decode ? " decode" : " prefill") + " rows, epilogue " +
                           std::to_string(unsigned(w.epilogue)));
}

// The kernels a plan names follow from its configuration: pipeline names,
// geometry and scratch are restated here independently of the operator.

// One simdgroup kernel per epilogue, in both phases.
std::string simdgroupPipeline(LinearEpilogue epilogue) {
  return epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
      epilogue == LinearEpilogue::UpWithGate ? "decode_linear_q4_sg_up_silu" :
      epilogue == LinearEpilogue::Residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
}

// The rows suffix of a batched decode kernel.
std::string rowsSuffix(uint32_t lanes) { return lanes > 1 ? "_m" + std::to_string(lanes * 8) : ""; }

std::string expectedPipeline(LinearConfig expected, uint32_t lanes, LinearEpilogue epilogue) {
  if (expected.tile == LinearTile::Simdgroup) return simdgroupPipeline(epilogue);
  // Gate/up runs the plain split kernel as its gate pass.
  if (expected.tile == LinearTile::Split128)
    return std::string("decode_linear_q4_n128_split") +
        (epilogue == LinearEpilogue::Residual ? "_residual" : "") + rowsSuffix(lanes);
  if (expected.tile == LinearTile::Paired256) return "decode_linear_q4_n256_paired_sg4";
  if (epilogue == LinearEpilogue::GateUp)
    return lanes == 1 ? "decode_linear_q4_n256_gate_up" : lanes == 2 ? "decode_linear_q4_n256_gate_up_m16"
        : lanes == 3 ? "decode_linear_q4_n256_m24" : "decode_linear_q4_n256_m32";
  std::string name = expected.tile == LinearTile::N256 ? "decode_linear_q4_n256" : "decode_linear_q4_n128";
  if (epilogue == LinearEpilogue::Residual) name += "_residual";
  if (expected.tile == LinearTile::Paired128) return name + "_paired";
  if (lanes > 1) name += "_m" + std::to_string(lanes * 8);
  if (expected.simdgroups == LinearSimdgroups::Four) name += "_sg4";
  return name;
}

// Split128 at every lane count and the N256 gate/up tile at three and four
// lanes run gate/up as the gate projection and then the up projection with the
// SiLU product.
std::string expectedSecondPipeline(LinearConfig expected, uint32_t lanes, LinearEpilogue epilogue) {
  if (epilogue != LinearEpilogue::GateUp || expected.tile == LinearTile::Simdgroup) return {};
  if (expected.tile == LinearTile::Split128) return "decode_linear_q4_n128_split_up_silu" + rowsSuffix(lanes);
  if (lanes < 3) return {};
  return "decode_linear_q4_n256_up_silu_m" + std::to_string(lanes * 8);
}

std::string expectedPrefillPipeline(LinearConfig expected, LinearEpilogue epilogue) {
  if (expected.tile == LinearTile::Simdgroup) return simdgroupPipeline(epilogue);
  std::string name = expected.tile == LinearTile::N256 ? "prefill_linear_q4_n256" : "prefill_linear_q4_n128";
  if (epilogue == LinearEpilogue::UpWithGate) name += "_up_silu_sums";
  if (epilogue == LinearEpilogue::Residual) name += "_residual";
  if (expected.simdgroups == LinearSimdgroups::Four) name += "_sg4";
  return name;
}

// The simdgroup tile reads the Table64 activation table its producer writes
// (tableBytes and tableSumsBytes) for whole eight-row lanes and, split over
// K, reduces two fp32 fragment streams per partition, row and column with one
// completion counter per lane and column tile; one partition binds
// one-element placeholders. Split128 reduces one fp32 partial per partition,
// row and column with one counter per column tile, which covers every lane.
// Every other affine tile reads the plain rows and binds no scratch.
constexpr uint64_t kFragmentStreams = 2;
LinearScratchSize expectedScratch(LinearConfig expected, LinearWorkload w, uint32_t tileColumns) {
  const auto [n, k] = w.matrix;
  if (expected.tile == LinearTile::Split128)
    return {0, 0, uint64_t{expected.splits} * w.rows * n * sizeof(float), uint64_t{n / tileColumns} * sizeof(uint32_t)};
  if (expected.tile != LinearTile::Simdgroup) return {};
  const uint64_t lanes = (w.rows + 7) / 8, rows = lanes * 8;
  const bool split = expected.splits > 1;
  return {tableBytes(k, rows), tableSumsBytes(LinearInput::Table64, k, rows),
          split ? expected.splits * kFragmentStreams * rows * n * sizeof(float) : sizeof(float),
          split ? lanes * (n / tileColumns) * sizeof(uint32_t) : sizeof(uint32_t)};
}

bool sameScratch(LinearScratchSize a, LinearScratchSize b) {
  return a.input == b.input && a.sums == b.sums && a.partials == b.partials && a.counters == b.counters;
}

// A plan the policy makes is one its kernels run: the scope, pipelines, tile
// geometry, input layout and scratch its configuration names, storage for its
// rows, fp32 partial sums from its K splits alone, and an arena bound
// (`arena`: the plan's decode or prefill scratch bound) that covers it.
void requireRunnable(const LinearPlan &plan, LinearScratchSize arena, uint32_t family, uint32_t cores) {
  const LinearWorkload w = plan.workload();
  const LinearConfig c = plan.configuration();
  const auto rule = [&](bool holds, const char *name) { if (!holds) broke(name, family, cores, w); };
  const LinearScratchSize scratch = plan.scratchSize();
  rule(plan.storageRows() >= w.rows && plan.threadsPerThreadgroup() == static_cast<uint32_t>(c.simdgroups) * 32 &&
           plan.partialSums() == c.splits,
       "plan storage, scope or partial sums differ from its configuration");
  rule(arena.input >= scratch.input && arena.sums >= scratch.sums && arena.partials >= scratch.partials &&
           arena.counters >= scratch.counters,
       "arena bound below a plan");
  if (w.weightLayout == WeightLayout::Block32) {
    rule(plan.tileColumns() == 64 &&
             plan.input() == (c.tile == LinearTile::GgufRegister ? LinearInput::Table16 : LinearInput::Plain),
         "GGUF plan geometry or input layout differs from its tile");
    return;
  }
  const bool simdgroup = c.tile == LinearTile::Simdgroup;
  const uint32_t columns = simdgroup ? (w.epilogue == LinearEpilogue::GateUp ? 32U : 64U)
      : c.tile == LinearTile::N256 || c.tile == LinearTile::Paired256 ? 256U : 128U;
  const uint32_t lanes = w.rows / 8;
  const bool decode = w.phase == LinearPhase::Decode;
  rule(plan.tileColumns() == columns && plan.input() == (simdgroup ? LinearInput::Table64 : LinearInput::Plain) &&
           sameScratch(scratch, expectedScratch(c, w, columns)),
       "affine plan geometry, input layout or scratch differs from its tile");
  rule(plan.pipeline() == (decode ? expectedPipeline(c, lanes, w.epilogue) : expectedPrefillPipeline(c, w.epilogue)) &&
           plan.secondPipeline() == (decode ? expectedSecondPipeline(c, lanes, w.epilogue) : ""),
       "affine plan pipelines differ from its configuration");
  // The simdgroup tile reads its table's sums instead of the MPP tile's Q4 sums.
  rule(decode || (plan.sumsBytes() == 0) == simdgroup, "affine prefill sums differ from its tile");
}

struct ProductionShape final { LinearMatrix matrix; LinearEpilogue epilogue; };
// Every decode projection of the Qwen3.8-27B and Qwen3.6-35B-A3B targets and
// their DFlash drafts (N x K, epilogue), plus K % 1024 != 0 controls.
constexpr std::array kProductionShapes{
    ProductionShape{{16640, 5120}, LinearEpilogue::None},
    ProductionShape{{14336, 5120}, LinearEpilogue::None},
    ProductionShape{{5120, 6144}, LinearEpilogue::Residual},
    ProductionShape{{17408, 5120}, LinearEpilogue::GateUp},
    ProductionShape{{5120, 17408}, LinearEpilogue::Residual},
    ProductionShape{{248320, 5120}, LinearEpilogue::None},
    ProductionShape{{1280, 5120}, LinearEpilogue::None},
    ProductionShape{{6144, 5120}, LinearEpilogue::None},
    ProductionShape{{5120, 4096}, LinearEpilogue::None},
    ProductionShape{{5120, 17408}, LinearEpilogue::None},
    ProductionShape{{5120, 25600}, LinearEpilogue::None},
    ProductionShape{{256, 5120}, LinearEpilogue::None},
    ProductionShape{{12544, 2048}, LinearEpilogue::None},
    ProductionShape{{9216, 2048}, LinearEpilogue::None},
    ProductionShape{{2048, 4096}, LinearEpilogue::Residual},
    ProductionShape{{248320, 2048}, LinearEpilogue::None},
    ProductionShape{{512, 2048}, LinearEpilogue::None},
    ProductionShape{{6144, 2048}, LinearEpilogue::None},
    ProductionShape{{2048, 4096}, LinearEpilogue::None},
    ProductionShape{{6144, 2048}, LinearEpilogue::GateUp},
    ProductionShape{{2048, 6144}, LinearEpilogue::None},
    ProductionShape{{2048, 6144}, LinearEpilogue::Residual},
    ProductionShape{{2048, 16384}, LinearEpilogue::None},
    ProductionShape{{256, 2048}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::None},
    ProductionShape{{5120, 4352}, LinearEpilogue::Residual},
    ProductionShape{{6144, 4352}, LinearEpilogue::GateUp},
    ProductionShape{{2048, 768}, LinearEpilogue::None}};

// Anchors from the measured machines: 16- and 20-core Apple10 GPUs (M5 Pro)
// and a 40-core Apple9 GPU (M3 Max). Changing a rule must change these
// knowingly.
void anchorPlans() {
  const auto configured = [](uint32_t family, uint32_t cores, LinearWorkload workload) {
    return gpu(family, cores).plan(workload).configuration();
  };
  const LinearWorkload gateUp{{17408, 5120}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  require(configured(10, 16, gateUp) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 20, gateUp) == LinearConfig{LinearTile::N256, 48} &&
              configured(9, 40, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 2} &&
              // Unknown counts use the same intermediate estimate on both families.
              configured(10, 0, gateUp) == configured(10, 32, gateUp) &&
              configured(9, 0, gateUp) == configured(9, 32, gateUp),
          "fused gate/up grid does not follow the balanced two-tile rule");
  // Apple9 matrix K splits cover all decode widths; broad plain projections
  // retain their old multi-lane grids.
  require(configured(9, 16, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 1} &&
              configured(9, 20, gateUp) == LinearConfig{LinearTile::Simdgroup, 544, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 2} &&
              configured(9, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 1} &&
              configured(9, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 65},
          "Apple9 decode grids changed without a measurement");
  // Apple10 splits K where the Split128 grid fits four threadgroups per core
  // and keeps the sequential tiles elsewhere; Apple9 simdgroup and wide
  // paired N256 anchors are unchanged.
  const LinearWorkload mixer27{{5120, 6144}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload mixer35{{2048, 4096}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload draftGateUp{{6144, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  const auto split = [](uint32_t n, uint32_t splits) {
    return LinearConfig{LinearTile::Split128, n / 128, LinearSimdgroups::Eight, splits};
  };
  require(configured(10, 20, mixer27) == split(5120, 2) &&
              configured(10, 16, mixer27) == LinearConfig{LinearTile::Paired128, 40} &&
              configured(10, 40, mixer27) == split(5120, 4) &&
              configured(10, 20, mixer35) == split(2048, 4) &&
              configured(10, 16, mixer35) == split(2048, 4) &&
              configured(10, 10, mixer35) == split(2048, 2) &&
              configured(10, 40, mixer35) == split(2048, 8) &&
              configured(10, 10, {{5120, 17408}, 8}) == LinearConfig{LinearTile::N128, 40} &&
              configured(10, 20, {{1280, 5120}, 8}) == split(1280, 8) &&
              configured(10, 16, {{1280, 5120}, 8}) == split(1280, 4) &&
              configured(10, 16, {{256, 5120}, 8}) == split(256, 8) &&
              configured(10, 20, {{6144, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 48} &&
              configured(10, 40, {{6144, 5120}, 8}) == split(6144, 2) &&
              configured(10, 20, draftGateUp) == LinearConfig{LinearTile::N256, 24} &&
              configured(10, 16, draftGateUp) == LinearConfig{LinearTile::N256, 24} &&
              configured(10, 40, draftGateUp) == split(6144, 2),
          "Apple10 K split anchors changed");
  require(configured(9, 40, mixer27) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(9, 40, {{256, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 4, LinearSimdgroups::Four, 4} &&
              configured(9, 18, mixer27) == LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 4},
          "Apple9 simdgroup policy anchors changed");
  const LinearConfig paired256Apple10_20{LinearTile::Paired256, 80, LinearSimdgroups::Four};
  require(configured(10, 20, {{248320, 5120}, 8}) == paired256Apple10_20 &&
              configured(10, 20, {{248320, 2048}, 8}) == paired256Apple10_20 &&
              configured(10, 16, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 64, LinearSimdgroups::Four} &&
              configured(10, 10, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 40, LinearSimdgroups::Four} &&
              configured(9, 40, {{248320, 5120}, 8}) ==
                  LinearConfig{LinearTile::Simdgroup, 3880, LinearSimdgroups::Four, 1} &&
              configured(9, 80, {{248320, 2048}, 8}) ==
                  LinearConfig{LinearTile::Simdgroup, 3880, LinearSimdgroups::Four, 1} &&
              configured(10, 20, {{25600, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 80, LinearSimdgroups::Four} &&
              configured(10, 20, {{25344, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 80},
          "one-lane paired N256 anchors changed");
  // K % 1024 != 0 is legal for matrix tiles, and Split128 partitions differ by
  // at most one 256-input block (17 into 8 and 9, 3 into 1 and 2).
  require(configured(10, 20, {{5120, 4352}, 8}) == split(5120, 2) &&
              configured(9, 40, {{5120, 4352}, 8, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 4} &&
              configured(9, 40, {{6144, 4352}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::Simdgroup, 192, LinearSimdgroups::Four, 4} &&
              configured(10, 20, {{2048, 768}, 8}) == split(2048, 2) &&
              configured(10, 20, {{2048, 4096}, 16}) == split(2048, 4) &&
              configured(9, 40, {{5120, 6144}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
                  LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8} &&
              configured(10, 20, {{6144, 2048}, 32, LinearPhase::Decode, LinearEpilogue::GateUp}) ==
                  LinearConfig{LinearTile::N256, 24},
          "one-lane fallbacks or multi-lane rules changed");
  // Balanced two-tile groups above one wave: 130 paired tiles keep three
  // groups per core on 20 cores and one full wave of longer chains on 16; 98
  // tiles land on 60 and 50 groups; the M16 grid holds to five per core.
  require(configured(10, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 70} &&
              configured(10, 16, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 64} &&
              configured(10, 20, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 60} &&
              configured(10, 16, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 50} &&
              configured(10, 20, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 75} &&
              configured(10, 16, {{16640, 5120}, 16}) == LinearConfig{LinearTile::N128, 64} &&
              configured(10, 20, {{12544, 2048}, 16}) == LinearConfig{LinearTile::N128, 98} &&
              configured(10, 16, {{16640, 5120}, 24}) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 20, {{16640, 5120}, 24}) ==
                  LinearConfig{LinearTile::N128, 130, LinearSimdgroups::Four} &&
              configured(10, 20, {{16640, 5120}, 32}) == LinearConfig{LinearTile::N256, 45} &&
              configured(10, 20, {{248320, 5120}, 16}) == LinearConfig{LinearTile::N128, 1940} &&
              configured(9, 20, {{16640, 5120}, 8}) == LinearConfig{LinearTile::Simdgroup, 260, LinearSimdgroups::Four, 2} &&
              configured(10, 0, {{16640, 5120}, 8}) == configured(10, 32, {{16640, 5120}, 8}),
          "persistent decode groups changed for the measured shapes");
  require(configured(10, 16, {{14336, 5120}, 24}) == LinearConfig{LinearTile::N256, 40} &&
          configured(10, 16, {{14336, 5120}, 32}) == LinearConfig{LinearTile::N256, 40} &&
          configured(10, 40, {{14336, 5120}, 32}) == split(14336, 2) &&
          configured(9, 40, {{14336, 5120}, 24}) ==
              LinearConfig{LinearTile::Simdgroup, 224, LinearSimdgroups::Four, 4} &&
          configured(9, 40, {{248320, 5120}, 24}) ==
              LinearConfig{LinearTile::N128, 1940, LinearSimdgroups::Four} &&
          configured(9, 40, {{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) ==
              LinearConfig{LinearTile::Simdgroup, 80, LinearSimdgroups::Four, 8},
          "decode tile rules changed for the measured shapes");
  // The tensor decode law (kAffineTensorTiles, kAffineSplitTiers), measured
  // DRAM-cold on the 20-core M5 Pro and the 12-core M6 with other core counts
  // emulated, and on Splish's 40-core M5 Max (device-policy.md, "Tensor decode
  // law"). One lane: unpaired N128 from 3.2 to 4 N128 tiles per core (M6 27B
  // mixer output, 3.33: 0.82 of the paired time; 20 cores, 35B attention
  // input, 3.6: 0.85; 40 cores, 27B GDN input, 3.25: 0.93), paired just past
  // three (35B GDN input on 32 cores, 3.06: paired 2.4% faster emulated) and
  // paired N256 from five N256 tiles per core (27B attention input on 10
  // cores: 0.86-0.89).
  const LinearWorkload mixer27Residual{{5120, 6144}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  require(configured(10, 12, mixer27Residual) == LinearConfig{LinearTile::N128, 40} &&
              configured(11, 12, mixer27Residual) == LinearConfig{LinearTile::N128, 40} &&
              configured(10, 20, {{9216, 2048}, 8}) == LinearConfig{LinearTile::N128, 72} &&
              configured(10, 40, {{16640, 5120}, 8}) == LinearConfig{LinearTile::N128, 130} &&
              configured(10, 32, {{12544, 2048}, 8}) == LinearConfig{LinearTile::Paired128, 98} &&
              configured(10, 12, {{14336, 5120}, 8}) == LinearConfig{LinearTile::Paired128, 48} &&
              configured(10, 10, {{14336, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 40, LinearSimdgroups::Four} &&
              configured(10, 12, {{16640, 5120}, 8}) ==
                  LinearConfig{LinearTile::Paired256, 48, LinearSimdgroups::Four},
          "one-lane tensor tile law anchors changed");
  // Three lanes: N256 from 3.5 N256 tiles per core (M6 27B attention input,
  // 4.67: 0.82 of the four-simdgroup N128 time); four lanes from 1.6 (20
  // cores, 35B attention input, 1.8: 0.80; M6, the 27B draft output, 1.67:
  // 0.84). Steps of two MPP fragments split partitions of at least 512 inputs
  // (35B draft 512 x 2048: eight splits at one lane, four at four lanes,
  // 0.93 of the eight-split time) and long partitions up to three
  // threadgroups per core (27B draft qkv on 20 cores: 0.89 of its sequential
  // time at four lanes).
  require(configured(10, 12, {{14336, 5120}, 24}) == LinearConfig{LinearTile::N256, 32} &&
              configured(10, 20, {{9216, 2048}, 24}) ==
                  LinearConfig{LinearTile::N128, 72, LinearSimdgroups::Four} &&
              configured(10, 20, {{9216, 2048}, 32}) == LinearConfig{LinearTile::N256, 36} &&
              configured(10, 12, {{5120, 4096}, 32}) == LinearConfig{LinearTile::N256, 20} &&
              configured(10, 20, {{512, 2048}, 8}) == split(512, 8) &&
              configured(10, 20, {{512, 2048}, 16}) == split(512, 8) &&
              configured(10, 20, {{512, 2048}, 24}) == split(512, 4) &&
              configured(10, 20, {{512, 2048}, 32}) == split(512, 4) &&
              configured(10, 20, {{6144, 5120}, 16}) == LinearConfig{LinearTile::N128, 48} &&
              configured(10, 20, {{6144, 5120}, 32}) == split(6144, 2),
          "multi-lane tensor tile or split law anchors changed");
  require(configured(10, 16, {{5120, 17408}, 8}) == LinearConfig{LinearTile::Paired128, 40} &&
          configured(10, 16, {{5120, 17408}, 16}) == LinearConfig{LinearTile::N128, 40} &&
          configured(10, 16, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(10, 16, {{17408, 5120}, 2048, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four} &&
          configured(9, 40, {{6144, 5120}, 2048, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N256, 0} &&
          configured(9, 40, {{6144, 5120}, 33, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0} &&
          configured(9, 40, {{17408, 5120}, 33, LinearPhase::Prefill,
                              LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::N256, 0},
          "one-lane pipelining or prefill tile rule changed for the measured shapes");
  // Apple9 prefill chunks of up to 32 rows take the decode tile's grid and
  // split; Apple10 keeps its prefill tile.
  require(configured(9, 40, {{6144, 5120}, 32, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::Simdgroup, 96, LinearSimdgroups::Four, 4} &&
          configured(9, 40, {{17408, 5120}, 17, LinearPhase::Prefill, LinearEpilogue::UpWithGate}) ==
              LinearConfig{LinearTile::Simdgroup, 272, LinearSimdgroups::Four, 4} &&
          configured(9, 40, {{5120, 17408}, 1, LinearPhase::Prefill, LinearEpilogue::Residual}) ==
              configured(9, 40, {{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual}) &&
          configured(10, 16, {{6144, 5120}, 32, LinearPhase::Prefill}) ==
              LinearConfig{LinearTile::N128, 0, LinearSimdgroups::Four},
          "short prefill chunks changed tiles for the measured shapes");
}

// `widestCandidates` accumulates the largest candidate set seen, so main() can
// check that the bound below is reached and not merely respected.
void planContracts(uint32_t family, uint32_t cores, size_t &widestCandidates) {
  Linear linear = gpu(family, cores);
  // Include a wide Apple9 projection with four distinct persistent grids,
  // both paired N256 grids and all four K splits to reach the candidate bound.
  for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                    LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                    LinearMatrix{23040, 2048}, LinearMatrix{131072, 4096}}) {
    for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
        const LinearWorkload workload{matrix, lanes * 8, LinearPhase::Decode, epilogue};
        const auto candidates = linear.candidates(workload);
        require(linear.plan(workload).configuration().tile != LinearTile::Split32 &&
                    linear.plan(workload).configuration().tile != LinearTile::Split64,
                "offline split-K candidate became a serving default");
        require(!candidates.empty() && candidates.size() <= Linear::kMaximumCandidates,
                "Linear candidates exceed the bounded set");
        widestCandidates = std::max(widestCandidates, candidates.size());
        require(candidates.front().configuration() == linear.plan(workload).configuration(),
                "Linear baseline is not first candidate");
        uint32_t fourScopeCandidates = 0, splitCandidates = 0, split128Candidates = 0, simdgroupCandidates = 0;
        for (size_t index = 0; index < candidates.size(); ++index) {
          const auto &plan = candidates[index];
          const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
          const auto tile = plan.configuration().tile;
          const bool split = tile == LinearTile::Split32 || tile == LinearTile::Split64;
          const bool split128 = tile == LinearTile::Split128;
          if (split) {
            ++splitCandidates;
            require(lanes == 1 && matrix.inputSize % 1024 == 0 && plan.partialSums() == 4 &&
                        plan.configuration().groups == matrix.outputSize / plan.tileColumns() &&
                        (epilogue != LinearEpilogue::GateUp || tile == LinearTile::Split32) &&
                        plan.secondPipeline().empty(),
                    "split-K candidate escaped its one-lane full-grid contract");
          } else if (split128) {
            ++split128Candidates;
            const uint32_t splits = plan.configuration().splits;
            require(splits >= 2 && matrix.inputSize / 256 >= splits && plan.partialSums() == splits &&
                        plan.configuration().groups == matrix.outputSize / 128 &&
                        plan.secondPipeline().empty() == (epilogue != LinearEpilogue::GateUp),
                    "Split128 candidate escaped its full-grid contract");
          } else if (plan.usesSimdgroup()) {
            ++simdgroupCandidates;
            require(family == 9 && four &&
                        plan.partialSums() == plan.configuration().splits &&
                        plan.configuration().groups == matrix.outputSize / plan.tileColumns(),
                    "simdgroup candidate escaped its full-grid contract");
          } else {
            require(plan.partialSums() == 1, "sequential candidate reports partial sums");
          }
          if (four) {
            ++fourScopeCandidates;
            const bool oneLane = lanes == 1 &&
                (tile == LinearTile::Split32 ||
                 (tile == LinearTile::Paired256 && epilogue == LinearEpilogue::None));
            require(((lanes == 3 && tile == LinearTile::N128 && epilogue != LinearEpilogue::GateUp) ||
                     oneLane || plan.usesSimdgroup()) && plan.secondPipeline().empty(),
                    "four-SIMDgroup candidate escaped its precompiled workload set");
            if (lanes == 3 && !plan.usesSimdgroup())
              require(plan.pipeline() == (epilogue == LinearEpilogue::Residual
                          ? "decode_linear_q4_n128_residual_m24_sg4" : "decode_linear_q4_n128_m24_sg4"),
                      "four-SIMDgroup plan chose the wrong pipeline");
          }
          require(plan.threadsPerThreadgroup() == (four ? 128 : 256),
                  "Linear plan scope/thread count disagree");
          require(plan.storageRows() == lanes * 8 && !plan.sumsBytes() && !plan.downSumsBytes(),
                  "decode storage/sums contract changed");
          require(plan.gateScratchBytes() ==
                      (epilogue == LinearEpilogue::GateUp && (split128 || (lanes >= 3 && !plan.usesSimdgroup()))
                           ? uint64_t{lanes} * 8 * matrix.outputSize * 2 : 0),
                  "decode gate scratch disagrees with decomposition");
          for (size_t prior = 0; prior < index; ++prior)
            require(candidates[prior].configuration() != plan.configuration(),
                    "duplicate Linear candidates");
        }
        // One lane always lists the paired N256 tile for plain projections and
        // the split tiles when K allows them: Split32 for every decode
        // epilogue, Split64 for the single-stream ones.
        require((fourScopeCandidates != 0) ==
                    (family == 9 || (lanes == 3 && epilogue != LinearEpilogue::GateUp) ||
                     (lanes == 1 && (family == 9 || epilogue == LinearEpilogue::None || matrix.inputSize % 1024 == 0))),
                "Linear candidate set omitted or added four-SIMDgroup plans");
        uint32_t legalSplits = 0;
        if (family == 9)
          for (uint32_t split : {1U, 2U, 4U, 8U})
            legalSplits += matrix.inputSize % (64 * split) == 0;
        require(simdgroupCandidates == legalSplits,
                "Linear candidates omit a legal Apple9 K split");
        require(splitCandidates == (lanes == 1 && matrix.inputSize % 1024 == 0
                                        ? (epilogue == LinearEpilogue::GateUp ? 1U : 2U) : 0U),
                "Linear candidate set omitted or added split-K plans");
        // Apple10 lists Split128 at every lane count and each K split its
        // blocks allow.
        uint32_t legalSplit128 = 0;
        if (family >= 10)
          for (uint32_t split : {2U, 4U, 8U}) legalSplit128 += matrix.inputSize / 256 >= split;
        require(split128Candidates == legalSplit128, "Linear candidate set omitted or added Split128 plans");
      }
    }
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                               LinearEpilogue::UpWithGate}) {
      // Every prefill epilogue has the eight-simdgroup N256 tile and the
      // four-simdgroup N128 tile; the plain and residual ones also N128/8.
      const auto candidates = linear.candidates({matrix, 2048, LinearPhase::Prefill, epilogue});
      const bool up = epilogue == LinearEpilogue::UpWithGate;
      require(candidates.size() == (up ? 2U : 3U) &&
                  candidates.front().configuration() ==
                      linear.plan({matrix, 2048, LinearPhase::Prefill, epilogue}).configuration(),
              "prefill candidate set changed");
      // Chunks of up to 32 rows list the same tiles, but on Apple9 only its
      // simdgroup tile at every legal K split.
      for (const uint32_t rows : {1U, 17U, 32U}) {
        const LinearWorkload chunk{matrix, rows, LinearPhase::Prefill, epilogue};
        const auto chunkCandidates = linear.candidates(chunk);
        uint32_t simdgroups = 0, legalSplits = 0;
        for (const auto &plan : chunkCandidates) simdgroups += plan.usesSimdgroup();
        for (const uint32_t split : {1U, 2U, 4U, 8U}) legalSplits += matrix.inputSize % (64 * split) == 0;
        require(chunkCandidates.front().configuration() == linear.plan(chunk).configuration() &&
                    (family == 9 ? simdgroups == chunkCandidates.size() && simdgroups == legalSplits
                                 : !simdgroups && chunkCandidates.size() == (up ? 2U : 3U)),
                "short prefill candidate set changed");
      }
      for (const auto &plan : candidates) {
        const bool four = plan.configuration().simdgroups == LinearSimdgroups::Four;
        require(plan.configuration().groups == 0 && plan.threadsPerThreadgroup() == (four ? 128 : 256) &&
                    (!four || plan.configuration().tile == LinearTile::N128) &&
                    (!up || four || plan.configuration().tile == LinearTile::N256) &&
                    (plan.pipeline().ends_with("_sg4") == four),
                "prefill candidate scope, tile or pipeline name disagree");
      }
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        const auto plan = linear.plan({matrix, rows, LinearPhase::Prefill, epilogue});
        // Apple9's simdgroup tile pads to whole lanes and reads no Q4 sums.
        const bool simdgroup = family == 9 && rows <= 32;
        const uint64_t storageRows = simdgroup ? (rows + 7) / 8 * 8 : (rows + 31) / 32 * 32;
        require(plan.usesSimdgroup() == simdgroup && plan.storageRows() == storageRows &&
                    plan.sumsBytes() == (simdgroup ? 0 : storageRows * (matrix.inputSize / 64) * 4),
                "prefill padding or sums bound incorrect");
        require(plan.gateScratchBytes() == (up ? storageRows * matrix.outputSize * 2 : 0) &&
                    plan.downSumsBytes() == (up && !simdgroup ? storageRows * (matrix.outputSize / 64) * 4 : 0),
                "prefill gate/output sums bound incorrect");
      }
    }
  }
  const LinearWorkload valid{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::None};
  for (const LinearWorkload invalid : {
           LinearWorkload{{0, 256}, 8}, LinearWorkload{{128, 256}, 8},
           LinearWorkload{{384, 256}, 8}, LinearWorkload{{512, 64}, 8},
           LinearWorkload{{512, 320}, 8}, LinearWorkload{{512, 0}, 8},
           LinearWorkload{{512, 256}, 0}, LinearWorkload{{512, 256}, 7},
           LinearWorkload{{512, 256}, 40},
           LinearWorkload{{512, 256}, 8, static_cast<LinearPhase>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, static_cast<LinearEpilogue>(255)},
           LinearWorkload{{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::UpWithGate},
           LinearWorkload{{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::GateUp},
           LinearWorkload{{512, 256}, 2049, LinearPhase::Prefill}})
    rejects([&] { (void)linear.plan(invalid); });
  for (const LinearConfig invalid : {
           LinearConfig{LinearTile::N128, 0}, LinearConfig{LinearTile::N128, 5},
           LinearConfig{static_cast<LinearTile>(255), 1}})
    rejects([&] { (void)Linear::plan(valid, invalid); });
  rejects([&] { (void)Linear::plan({{512, 256}, 16}, {LinearTile::Paired128, 1}); });
  rejects([&] { (void)Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Linear::plan({{512, 256}, 32, LinearPhase::Prefill}, {LinearTile::Paired128, 0}); });
  rejects([&] { (void)Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::Residual}, {LinearTile::N256, 1}); });
  rejects([&] { (void)Linear::plan({{512, 256}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}, {LinearTile::N128, 1}); });
  rejects([&] { (void)Linear::plan({{512, 256}, 8, LinearPhase::Prefill, LinearEpilogue::UpWithGate}, {LinearTile::N128, 0}); });
  // Split tiles: one lane, K % 1024 == 0, the full grid and the kernel's own
  // simdgroup count; Split64 has no gate/up form. Split128: the full grid,
  // eight simdgroups and two to eight K partitions of at least one 256-input
  // block, at every lane count and epilogue. Paired256: one lane, plain
  // epilogue, four simdgroups. None exists in prefill.
  const LinearWorkload splitWorkload{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::None};
  const LinearWorkload splitResidual{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearWorkload splitGateUp{{512, 1024}, 8, LinearPhase::Decode, LinearEpilogue::GateUp};
  const LinearConfig matrixTile{LinearTile::Simdgroup, 8, LinearSimdgroups::Four, 4};
  for (const uint32_t splits : {0U, 3U, 16U}) {
    auto invalid = matrixTile;
    invalid.splits = splits;
    rejects([&] { (void)Linear::plan(splitWorkload, invalid); });
  }
  rejects([&] { (void)Linear::plan({{512, 768}, 8},
      {LinearTile::Simdgroup, 8, LinearSimdgroups::Four, 8}); });
  for (uint32_t rows : {8U, 16U, 24U, 32U}) {
    const LinearWorkload workload{{512, 1024}, rows};
    const LinearPlan plan = Linear::plan(workload, matrixTile);
    require(sameScratch(plan.scratchSize(), expectedScratch(matrixTile, workload, plan.tileColumns())),
            "matrix row tiles must own disjoint input, sums, both fragment streams' partials and counters");
  }
  // Prefill chunks of up to 32 rows run the same tile over whole lanes, one
  // kernel per epilogue, reading its table instead of Q4 sums; longer chunks
  // and the other prefill tiles' zero grid are refused.
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate}) {
    const LinearWorkload chunk{{512, 1024}, 17, LinearPhase::Prefill, epilogue};
    const LinearPlan plan = Linear::plan(chunk, matrixTile);
    require(plan.storageRows() == 24 && plan.pipeline() == simdgroupPipeline(epilogue) &&
                plan.input() == LinearInput::Table64 && !plan.sumsBytes() && !plan.downSumsBytes() &&
                sameScratch(plan.scratchSize(), expectedScratch(matrixTile, chunk, plan.tileColumns())),
            "simdgroup prefill plan geometry, pipeline or sums are wrong");
    rejects([&] { (void)Linear::plan({{512, 1024}, 33, LinearPhase::Prefill, epilogue}, matrixTile); });
    rejects([&] { (void)Linear::plan(chunk, {LinearTile::Simdgroup, 0, LinearSimdgroups::Four, 4}); });
  }
  rejects([&] { (void)Linear::plan(splitWorkload,
      {LinearTile::Simdgroup, 4, LinearSimdgroups::Four, 4}); });
  rejects([&] { (void)Linear::plan(splitWorkload,
      {LinearTile::Simdgroup, 8, LinearSimdgroups::Eight, 4}); });
  const LinearConfig split32{LinearTile::Split32, 16, LinearSimdgroups::Four};
  const LinearConfig split64{LinearTile::Split64, 8, LinearSimdgroups::Eight};
  const LinearConfig split128{LinearTile::Split128, 4, LinearSimdgroups::Eight, 4};
  const LinearConfig paired256{LinearTile::Paired256, 2, LinearSimdgroups::Four};
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const std::string rows = rowsSuffix(lanes);
    const LinearWorkload plain{{512, 1024}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None};
    const auto plan = Linear::plan(plain, split128);
    require(plan.pipeline() == "decode_linear_q4_n128_split" + rows && plan.threadsPerThreadgroup() == 256 &&
                plan.tileColumns() == 128 && plan.partialSums() == 4 && plan.secondPipeline().empty() &&
                plan.storageRows() == lanes * 8 && plan.input() == LinearInput::Plain &&
                sameScratch(plan.scratchSize(), expectedScratch(split128, plain, 128)),
            "Split128 plan geometry, pipeline or scratch is wrong");
    require(Linear::plan({plain.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual}, split128)
                    .pipeline() == "decode_linear_q4_n128_split_residual" + rows,
            "Split128 residual pipeline is wrong");
    const auto gateUpPlan = Linear::plan({plain.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::GateUp},
                                         split128);
    require(gateUpPlan.pipeline() == "decode_linear_q4_n128_split" + rows &&
                gateUpPlan.secondPipeline() == "decode_linear_q4_n128_split_up_silu" + rows &&
                gateUpPlan.gateScratchBytes() == uint64_t{lanes} * 8 * 512 * 2,
            "Split128 gate/up runs a gate pass into the gate scratch and an up pass");
  }
  {
    const auto plan = Linear::plan(splitWorkload, split64);
    require(plan.pipeline() == "decode_linear_q4_n64_split4" && plan.threadsPerThreadgroup() == 256 &&
                plan.tileColumns() == 64 && plan.partialSums() == 4 && plan.secondPipeline().empty(),
            "Split64 plan geometry or pipeline is wrong");
    const auto residual = Linear::plan(splitResidual, split32);
    require(residual.pipeline() == "decode_linear_q4_n32_split4_residual" &&
                residual.threadsPerThreadgroup() == 128 && residual.tileColumns() == 32 &&
                residual.partialSums() == 4,
            "Split32 residual plan geometry or pipeline is wrong");
    require(Linear::plan(splitResidual, split64).pipeline() == "decode_linear_q4_n64_split4_residual" &&
                Linear::plan(splitWorkload, split32).pipeline() == "decode_linear_q4_n32_split4",
            "split plan pipeline names are wrong");
    const auto gateUpPlan = Linear::plan(splitGateUp, split32);
    require(gateUpPlan.pipeline() == "decode_linear_q4_n32_split4_gate_up" &&
                gateUpPlan.threadsPerThreadgroup() == 128 && gateUpPlan.secondPipeline().empty() &&
                gateUpPlan.gateScratchBytes() == 0,
            "Split32 gate/up plan geometry or pipeline is wrong");
    const auto wide = Linear::plan(splitWorkload, paired256);
    require(wide.pipeline() == "decode_linear_q4_n256_paired_sg4" && wide.threadsPerThreadgroup() == 128 &&
                wide.tileColumns() == 256 && wide.partialSums() == 1,
            "Paired256 plan geometry or pipeline is wrong");
  }
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split32, 16, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split64, 8, LinearSimdgroups::Four}); });
  // Split128 takes eight simdgroups, the full grid and 2, 4 or 8 partitions
  // of at least one block: 1024 inputs hold four.
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split128, 4, LinearSimdgroups::Four, 4}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split128, 2, LinearSimdgroups::Eight, 4}); });
  for (const uint32_t splits : {0U, 1U, 3U, 8U, 16U})
    rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split128, 4, LinearSimdgroups::Eight, splits}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Paired256, 2, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split32, 8, LinearSimdgroups::Four}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split64, 4, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Linear::plan(splitWorkload, {LinearTile::Split64, 16, LinearSimdgroups::Eight}); });
  rejects([&] { (void)Linear::plan(splitGateUp, split64); });
  rejects([&] { (void)Linear::plan(splitResidual, paired256); });
  rejects([&] { (void)Linear::plan(splitGateUp, paired256); });
  for (const LinearConfig config : {split32, split64})
    rejects([&] { (void)Linear::plan({{512, 768}, 8}, config); });
  for (uint32_t rows : {16U, 24U, 32U})
    for (const LinearConfig config : {split32, split64, paired256})
      rejects([&] { (void)Linear::plan({{512, 1024}, rows}, config); });
  for (const auto tile : {LinearTile::Split32, LinearTile::Split64, LinearTile::Paired256})
    rejects([&] { (void)Linear::plan({{512, 1024}, 32, LinearPhase::Prefill},
        {tile, 0, tile == LinearTile::Split64 ? LinearSimdgroups::Eight : LinearSimdgroups::Four}); });
  rejects([&] { (void)Linear::plan({{512, 1024}, 32, LinearPhase::Prefill},
                                   {LinearTile::Split128, 0, LinearSimdgroups::Eight, 4}); });
  const LinearWorkload fourWorkload{{512, 256}, 24, LinearPhase::Decode, LinearEpilogue::None};
  const LinearConfig fourConfig{LinearTile::N128, 1, LinearSimdgroups::Four};
  for (uint32_t scope : {0U, 1U, 2U, 3U, 16U, 255U})
    rejects([&] { (void)Linear::plan(fourWorkload,
        {LinearTile::N128, 1, static_cast<LinearSimdgroups>(scope)}); });
  for (uint32_t rows : {8U, 16U, 32U})
    rejects([&] { (void)Linear::plan({{512, 256}, rows}, fourConfig); });
  for (const auto tile : {LinearTile::N256, LinearTile::Paired128})
    rejects([&] { (void)Linear::plan(fourWorkload,
        {tile, 1, LinearSimdgroups::Four}); });
  for (const auto epilogue : {LinearEpilogue::GateUp, LinearEpilogue::UpWithGate})
    rejects([&] { (void)Linear::plan({{512, 256}, 24, LinearPhase::Decode, epilogue},
                                      fourConfig); });
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                             LinearEpilogue::UpWithGate}) {
    const LinearWorkload prefill{{512, 256}, 24, LinearPhase::Prefill, epilogue};
    require(Linear::plan(prefill, {LinearTile::N128, 0, LinearSimdgroups::Four})
                    .threadsPerThreadgroup() == 128,
            "four-SIMDgroup prefill plan was rejected");
    rejects([&] { (void)Linear::plan(prefill, {LinearTile::N256, 0, LinearSimdgroups::Four}); });
    rejects([&] { (void)Linear::plan(prefill, {LinearTile::N128, 1, LinearSimdgroups::Four}); });
  }

  const LinearWorkload other{{768, 768}, 16, LinearPhase::Decode, LinearEpilogue::None};
  const auto original = linear.plan(other).configuration();
  const LinearConfig selected{LinearTile::N256, 1};
  const std::array choices{LinearChoice{other, selected}, LinearChoice{valid, selected}};
  linear.setChoices(choices);
  require(linear.plan(valid).configuration() == selected &&
              linear.plan(other).configuration() == selected,
          "unsorted profile choices did not take effect");
  const std::array duplicate{choices[0], choices[0]};
  rejects([&] { linear.setChoices(duplicate); });
  const std::array invalid{LinearChoice{valid, {LinearTile::N128, 0}}};
  rejects([&] { linear.setChoices(invalid); });
  require(linear.plan(other).configuration() == selected,
          "invalid profile update changed installed choices");
  linear.setChoices({});
  require(linear.plan(other).configuration() == original,
          "clearing profile did not restore the shipped baseline");
  const auto baselineFourWorkload = linear.plan(fourWorkload);
  const std::array fourChoices{LinearChoice{fourWorkload, fourConfig}};
  linear.setChoices(fourChoices);
  require(linear.plan(fourWorkload).configuration() == fourConfig &&
              linear.plan(fourWorkload).threadsPerThreadgroup() == 128,
          "installed four-SIMDgroup choice did not reach the selected plan");
  const std::array invalidScope{LinearChoice{valid, fourConfig}};
  rejects([&] { linear.setChoices(invalidScope); });
  require(linear.plan(fourWorkload).configuration() == fourConfig,
          "invalid scope update changed installed choices");
  linear.setChoices({});
  require(linear.plan(fourWorkload).configuration() == baselineFourWorkload.configuration() &&
              linear.plan(fourWorkload).threadsPerThreadgroup() ==
                  baselineFourWorkload.threadsPerThreadgroup(),
          "clearing choices did not restore the shipped execution scope");
}

// A block projection of equal segments tiling its columns, one per format;
// planning reads only their geometry and formats.
Projection blockProjection(uint32_t n, uint32_t k, std::span<const uint32_t> formats) {
  BlockWeights weights;
  const uint32_t width = n / uint32_t(formats.size());
  for (const uint32_t format : formats) {
    QuantizedSegment s = QuantizedSegment::planes(format, width, k, {}, {}, {});
    s.columnOffset = uint32_t(weights.segments.size()) * width;
    weights.segments.push_back(s);
  }
  return Projection(n, k, std::move(weights));
}
// ... of `segments` segments in `format`, Q4_K unless it says otherwise.
Projection blockProjection(uint32_t n, uint32_t k, uint32_t segments, uint32_t format = GGUF_FMT_Q4K) {
  return blockProjection(n, k, std::vector<uint32_t>(segments, format));
}

// fp32 destinations (the logits): the plan of a projection with an fp32
// destination keeps the configuration, tile kernel, input table and scratch of
// its bf16 plan on every family and core count, and only plain decode
// workloads take one. Returns the kernels of the plain decode candidates,
// whose fp32 instances floatInstances looks up.
std::set<std::string_view> floatOutputPlans() {
  const auto fp32 = [](Projection p) {
    p.destination = FloatOutput::Float32;
    return p;
  };
  std::set<std::string_view> kernels;
  for (const uint32_t family : {9U, 10U, 11U})
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      const Linear linear = gpu(family, cores);
      for (const uint32_t n : {256U, 5120U, 16640U, 248320U})
        for (const uint32_t k : {2048U, 5120U})
          for (const Projection &p : {Projection(n, k, AffineWeights{}), blockProjection(n, k, 1)})
            for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
              const LinearWorkload w{{n, k}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None, p.layout()};
              const LinearPlan bf16 = linear.plan(w, p), head = linear.plan(w, fp32(p));
              if (head.configuration() != bf16.configuration() || head.destination() != FloatOutput::Float32 ||
                  head.pipeline() != bf16.pipeline() || head.input() != bf16.input() ||
                  head.storageRows() != bf16.storageRows() || !sameScratch(head.scratchSize(), bf16.scratchSize()))
                broke("an fp32 plan differs from its bf16 plan", family, cores, w);
              for (const LinearPlan &candidate : linear.candidates(w))
                if (!candidate.pipeline().empty()) kernels.insert(candidate.pipeline());
            }
    }
  std::vector<LinearWorkload> others{{{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                     {{5120, 2048}, 8, LinearPhase::Decode, LinearEpilogue::GateUp}};
  for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
    for (const uint32_t rows : {8U, 2048U}) others.push_back({{5120, 2048}, rows, LinearPhase::Prefill, epilogue});
  const Linear linear = gpu(10, 20);
  for (const Projection &p : {Projection(5120, 2048, AffineWeights{}), blockProjection(5120, 2048, 1)})
    for (const LinearWorkload &w : others) {
      (void)linear.plan(w, p);
      rejects([&] { (void)linear.plan(w, fp32(p)); });
    }
  return kernels;
}

// Every property of a plan that encoding and arena sizing read.
bool samePlan(const LinearPlan &a, const LinearPlan &b) {
  return a.workload() == b.workload() && a.configuration() == b.configuration() &&
         a.destination() == b.destination() && a.pipeline() == b.pipeline() &&
         a.secondPipeline() == b.secondPipeline() && a.storageRows() == b.storageRows() && a.input() == b.input() &&
         sameScratch(a.scratchSize(), b.scratchSize()) && a.sumsBytes() == b.sumsBytes() &&
         a.gateScratchBytes() == b.gateScratchBytes() && a.downSumsBytes() == b.downSumsBytes();
}
bool sameMoePlan(const MoePlan &a, const MoePlan &b) {
  return a.shape() == b.shape() && a.rows() == b.rows() && a.configuration() == b.configuration() &&
         a.splitExperts() == b.splitExperts() && a.maximumTiles() == b.maximumTiles() &&
         a.workspace() == b.workspace();
}

// The GPU families after Apple10 (Apple11 is the M6) run its policy: every
// Linear plan and candidate, GGUF plan and float tile, MoE plan and arena
// bound equals Apple10's at every core count, the unknown one included.
void newerFamilyPlans() {
  constexpr MoeShape affineMoe{2048, 256, 8, 512};
  constexpr MoeShape q4kMoe{2048, 256, 8, 512, WeightLayout::Block32, GGUF_FMT_Q4K};
  constexpr MoeShape iq2Moe{2048, 256, 8, 512, WeightLayout::Block32, GGUF_FMT_IQ2XS};
  for (uint32_t family = 11; family <= DeviceCapabilities::kNewestAppleGpuFamily; ++family)
    for (uint32_t cores = 0; cores <= 128; ++cores) {
      DeviceCapabilities device;
      device.appleGpuFamily = 10;
      device.gpuCoreCount = cores;
      const ExecutionPlans apple10(device);
      device.appleGpuFamily = family;
      const ExecutionPlans newer(device);
      const Linear &a = apple10.linear(), &b = newer.linear();
      const auto check = [&](bool same, const char *what, LinearWorkload w) {
        if (!same) broke(what, family, cores, w);
      };
      for (const ProductionShape &shape : kProductionShapes) {
        const auto [n, k] = shape.matrix;
        // Apple9 stages IQ2_XS where it keeps Q4_K on its register tile.
        const Projection affine(n, k, AffineWeights{}), q4k = blockProjection(n, k, 1),
            iq2 = blockProjection(n, k, 1, GGUF_FMT_IQ2XS), fused = blockProjection(n, k, 3);
        for (const Projection *p : {&affine, &q4k, &iq2, &fused}) {
          for (uint32_t lanes = 1; lanes <= 4; ++lanes)
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::GateUp}) {
              if (p == &fused && e != LinearEpilogue::None) continue;
              const LinearWorkload w{shape.matrix, lanes * 8, LinearPhase::Decode, e, p->layout()};
              const Projection *gate = e == LinearEpilogue::GateUp ? p : nullptr;
              check(samePlan(a.plan(w, *p, gate), b.plan(w, *p, gate)) &&
                        std::ranges::equal(a.candidates(w), b.candidates(w), samePlan) &&
                        sameScratch(a.decodeScratchSize(w), b.decodeScratchSize(w)),
                    "a newer family's decode plan differs from Apple10's", w);
            }
          for (const uint32_t rows : {1U, 8U, 17U, 32U, 33U, 128U, 2048U})
            for (const auto e : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate}) {
              if (p == &fused && e != LinearEpilogue::None) continue;
              const LinearWorkload w{shape.matrix, rows, LinearPhase::Prefill, e, p->layout()};
              check(samePlan(a.plan(w, *p), b.plan(w, *p)) &&
                        std::ranges::equal(a.candidates(w), b.candidates(w), samePlan),
                    "a newer family's prefill plan differs from Apple10's", w);
            }
          const ProjectionShape projection = p->shape();
          check(sameScratch(a.prefillScratchSize(projection), b.prefillScratchSize(projection)) &&
                    apple10.gateUpWorkspace(projection) == newer.gateUpWorkspace(projection),
                "a newer family's prefill or gate/up arena bound differs from Apple10's",
                {shape.matrix, 8, LinearPhase::Decode, shape.epilogue, p->layout()});
        }
      }
      for (const uint32_t n : {64U, 256U})
        for (uint32_t rows = 1; rows <= 2048; ++rows)
          check(a.ggufFloatTile(rows, n) == b.ggufFloatTile(rows, n), "a newer family's float tile differs from Apple10's",
                {{n, 256}, rows, LinearPhase::Prefill});
      for (const MoeShape shape : {affineMoe, MoeShape{768, 7, 3, 256}, q4kMoe, iq2Moe}) {
        const auto sameCandidates = [&](const MoeWorkload &w) {
          return std::ranges::equal(apple10.moeCandidates(w), newer.moeCandidates(w), sameMoePlan);
        };
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          require(sameMoePlan(apple10.moeDecode(shape, lanes), newer.moeDecode(shape, lanes)) &&
                      sameCandidates({shape, lanes * 8, MoePhase::Decode}),
                  "a newer family's MoE decode plan differs from Apple10's");
        for (const uint32_t rows : {1U, 8U, 17U, 32U, 33U, 256U, 257U, 320U, 321U, 1040U, 2048U})
          require(sameMoePlan(apple10.moePrefill(shape, rows), newer.moePrefill(shape, rows)) &&
                      sameCandidates({shape, rows, MoePhase::Prefill}),
                  "a newer family's MoE prefill plan differs from Apple10's");
        require(apple10.moeDecodeWorkspacePerLane(shape) == newer.moeDecodeWorkspacePerLane(shape) &&
                    apple10.moePrefillWorkspace(shape, 2048) == newer.moePrefillWorkspace(shape, 2048),
                "a newer family's MoE arena bound differs from Apple10's");
      }
    }
}

// A GGUF decode projection whose split count is pinned on one core count.
struct SplitAnchor final {
  const char *projection;
  uint32_t cores, n, k, rows;
  LinearEpilogue epilogue;
  uint32_t segments, splits;
};

LinearPlan anchorPlan(uint32_t family, const SplitAnchor &anchor) {
  return gpu(family, anchor.cores).plan({{anchor.n, anchor.k}, anchor.rows, LinearPhase::Decode, anchor.epilogue},
                                        blockProjection(anchor.n, anchor.k, anchor.segments));
}

void requireAnchorSplits(const LinearPlan &plan, const SplitAnchor &anchor, const char *policy) {
  const uint32_t splits = plan.configuration().splits;
  if (splits != anchor.splits)
    throw std::runtime_error(std::string(policy) + ": " + anchor.projection + " on " + std::to_string(anchor.cores) +
                             " cores plans " + std::to_string(splits) + " K splits, not " +
                             std::to_string(anchor.splits));
}

constexpr auto kNone = LinearEpilogue::None, kResidual = LinearEpilogue::Residual, kGateUp = LinearEpilogue::GateUp;

// Six threadgroups per core, at least 512 inputs per partition, for every
// projection kind.
constexpr std::array kStagedSplitAnchors{
    SplitAnchor{"27B down", 16, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 20, 5120, 17408, 8, kResidual, 1, 2},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 1},
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"35B output projection", 16, 2048, 4096, 8, kResidual, 1, 4},
    SplitAnchor{"35B shared-expert gate/up", 16, 512, 2048, 8, kGateUp, 1, 4},
    SplitAnchor{"27B gate/up", 16, 17408, 5120, 8, kGateUp, 1, 1},
    SplitAnchor{"two-segment fused projection", 16, 4096, 5120, 8, kNone, 2, 2},
    SplitAnchor{"27B three-segment GDN input", 40, 16640, 5120, 8, kNone, 3, 1},
    SplitAnchor{"27B vocabulary head", 16, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 256 tensor", 16, 1024, 256, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 3072 tensor", 16, 1024, 3072, 8, kNone, 1, 4}};

// One wave (four threadgroups per core) down to one 256-input unit per
// partition, eight waves while partitions keep 1024 inputs, at most eight.
constexpr std::array kRegisterSplitAnchors{
    SplitAnchor{"27B down", 40, 5120, 17408, 8, kResidual, 1, 8},
    SplitAnchor{"27B out_proj, four lanes", 40, 5120, 6144, 32, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input, two lanes", 40, 16640, 5120, 16, kNone, 3, 4},
    SplitAnchor{"27B three-segment attention input", 40, 14336, 5120, 8, kNone, 3, 4},
    SplitAnchor{"27B gate/up, three lanes", 40, 17408, 5120, 24, kGateUp, 1, 4},
    SplitAnchor{"27B vocabulary head", 40, 248320, 5120, 8, kNone, 1, 1},
    SplitAnchor{"27B down", 10, 5120, 17408, 8, kResidual, 1, 4},
    SplitAnchor{"27B three-segment GDN input", 10, 16640, 5120, 8, kNone, 3, 2},
    SplitAnchor{"35B two-segment GDN input", 40, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B two-segment GDN input", 80, 12288, 2048, 8, kNone, 2, 2},
    SplitAnchor{"35B shared-expert gate/up", 40, 512, 2048, 8, kGateUp, 1, 8},
    SplitAnchor{"35B shared-expert down", 40, 2048, 512, 8, kResidual, 1, 2},
    SplitAnchor{"35B output projection", 40, 2048, 4096, 8, kResidual, 1, 8},
    SplitAnchor{"35B vocabulary head", 40, 248320, 2048, 8, kNone, 1, 1},
    SplitAnchor{"narrow 1024 x 5120 tensor", 40, 1024, 5120, 8, kNone, 1, 8},
    SplitAnchor{"narrow 1024 x 1024 tensor", 40, 1024, 1024, 8, kNone, 1, 4}};

// fp32 K-split partials over the plan's rows and columns and one completion
// counter per 64-column tile (LinearPlan::scratchSize).
uint64_t splitPartialsBytes(const LinearPlan &plan) {
  return uint64_t{plan.partialSums()} * plan.storageRows() * plan.workload().matrix.outputSize * sizeof(float);
}
uint64_t splitCountersBytes(const LinearPlan &plan) {
  return uint64_t{plan.configuration().groups} * sizeof(uint32_t);
}
// The bf16 gate projection a GGUF gate/up plan writes before its up pass.
uint64_t gateBytes(const LinearPlan &plan) {
  return uint64_t{plan.storageRows()} * plan.workload().matrix.outputSize * sizeof(uint16_t);
}

// GGUF projections plan with their segments: the staged split policy for
// every projection kind, exact split scratch over the tile's rows, and
// prefill tiles of 8, 16, 32 or 128 rows.
void ggufPlans() {
  const Linear linear = gpu(10, 16);
  // A block projection without segments would reach the dispatch paths with
  // nothing to index or encode.
  rejects([] { (void)Projection(5120, 17408, BlockWeights{}); });
  // Its segments tile the leading columns in order: a gap, an overlap, another
  // input width or a segment past the end is not a projection; padding past
  // the last segment is.
  const auto segment = [](uint32_t n, uint32_t k, uint32_t offset) {
    QuantizedSegment s = QuantizedSegment::planes(GGUF_FMT_Q4K, n, k, {}, {}, {});
    s.columnOffset = offset;
    return s;
  };
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 320)}}); });
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 128)}}); });
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 512, 0)}}); });
  rejects([&] { (void)Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(768, 256, 256)}}); });
  require(Projection(768, 256, BlockWeights{{segment(256, 256, 0), segment(256, 256, 256)}}).blocks().segments.size() == 2,
          "a projection with padding past its segments was rejected");
  const LinearWorkload down{{5120, 17408}, 8, LinearPhase::Decode, LinearEpilogue::Residual};
  const LinearPlan single = linear.plan(down, blockProjection(5120, 17408, 1));
  require(single.workload().weightLayout == WeightLayout::Block32 &&
              single.configuration() == LinearConfig{LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 2} &&
              single.input() == LinearInput::Plain && single.partialSums() == 2 &&
              single.scratchSize().partials == splitPartialsBytes(single) &&
              single.scratchSize().counters == splitCountersBytes(single) && single.scratchSize().input == 0,
          "GGUF single-tensor decode plan");
  for (const SplitAnchor &anchor : kStagedSplitAnchors)
    requireAnchorSplits(anchorPlan(10, anchor), anchor, "GGUF staged split policy");
  const LinearPlan gateUpPlan = linear.plan({{512, 2048}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                            blockProjection(512, 2048, 1));
  require(gateUpPlan.partialSums() == 4 && gateUpPlan.gateScratchBytes() == gateBytes(gateUpPlan) &&
              gateUpPlan.scratchSize().partials == splitPartialsBytes(gateUpPlan),
          "GGUF staged gate/up runs a gate pass into the gate scratch");
  // Decode tiles hold 8, 16 or 32 rows: three lanes run the 32-row tile.
  const LinearPlan three = linear.plan({{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
                                       blockProjection(5120, 17408, 1));
  require(three.storageRows() == 32 && three.configuration() == single.configuration() &&
              three.scratchSize().partials == splitPartialsBytes(three),
          "GGUF staged three-lane plans run the 32-row tile");
  for (const auto [rows, storage] : {std::pair{1U, 8U}, {8U, 8U}, {9U, 16U}, {17U, 32U}, {25U, 32U}, {32U, 32U},
                                     {33U, 128U}, {100U, 128U}, {129U, 256U}, {2048U, 2048U}}) {
    const LinearPlan prefill = linear.plan({{5120, 17408}, rows, LinearPhase::Prefill, LinearEpilogue::UpWithGate},
                                           blockProjection(5120, 17408, 1));
    // Chunks of up to 32 rows take the decode tile and its split rule (two
    // partitions of the 80-tile grid on 16 cores); 128-row tiles take none.
    const uint32_t splits = rows <= 32 ? 2 : 1;
    require(prefill.storageRows() == storage && prefill.sumsBytes() == 0 && prefill.downSumsBytes() == 0 &&
                prefill.gateScratchBytes() == gateBytes(prefill) &&
                prefill.configuration().splits == splits && prefill.partialSums() == splits &&
                prefill.scratchSize().partials == (splits > 1 ? splitPartialsBytes(prefill) : 0) &&
                prefill.threadsPerThreadgroup() == (rows <= 32 ? 64U : 128U),
            "GGUF prefill tile rows and splits");
  }
  // The decode tiles hold at most a decode batch.
  LinearWorkload longPrefill{{5120, 17408}, 33, LinearPhase::Prefill, LinearEpilogue::None, WeightLayout::Block32};
  rejects([&] { (void)Linear::plan(longPrefill, {LinearTile::GgufStaged, 0, LinearSimdgroups::Two}); });
  // Affine and GGUF plans do not mix.
  rejects([&] { (void)Linear::plan(down, {LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 8}); });
  LinearWorkload gguf = down;
  gguf.weightLayout = WeightLayout::Block32;
  rejects([&] { (void)Linear::plan(gguf, {LinearTile::N128, 40}); });
  rejects([&] { (void)Linear::plan(gguf, {LinearTile::GgufStaged, 40, LinearSimdgroups::Two, 8}); });
  rejects([&] { (void)Linear::plan(gguf, {LinearTile::GgufStaged, 80, LinearSimdgroups::Two, 3}); });
  // The arena bound is the single-tensor plan, which fused and gate/up plans share.
  require(linear.decodeScratchSize(gguf).partials == single.scratchSize().partials,
          "GGUF decode scratch bound");
  // Float projections take the neural accelerator tile from three of its
  // 64 x 32 tiles per two cores: on 16 cores the 35B router (N 256) from 129
  // rows, alpha/beta (N 64) from 705; never on Apple9 or below 16 rows.
  const Linear oneCore = gpu(10, 1);
  require(linear.ggufFloatTile(32, 256) == FloatTile::Simdgroup && linear.ggufFloatTile(128, 256) == FloatTile::Simdgroup &&
              linear.ggufFloatTile(129, 256) == FloatTile::NeuralAccelerator &&
              linear.ggufFloatTile(2048, 256) == FloatTile::NeuralAccelerator &&
              linear.ggufFloatTile(704, 64) == FloatTile::Simdgroup &&
              linear.ggufFloatTile(705, 64) == FloatTile::NeuralAccelerator &&
              oneCore.ggufFloatTile(15, 256) == FloatTile::Simdgroup &&
              oneCore.ggufFloatTile(16, 256) == FloatTile::NeuralAccelerator,
          "GGUF float tile rule");

  // Apple9 decodes every GGUF width with the exact register tile, all lanes
  // in one threadgroup, and K splits from the core count; prefill stages.
  for (const SplitAnchor &anchor : kRegisterSplitAnchors) {
    const LinearPlan plan = anchorPlan(9, anchor);
    require(plan.configuration().tile == LinearTile::GgufRegister &&
                plan.configuration().groups == anchor.n / 64 &&
                plan.configuration().simdgroups == LinearSimdgroups::Four &&
                plan.input() == LinearInput::Table16,
            "Apple9 GGUF register plan");
    requireAnchorSplits(plan, anchor, "Apple9 GGUF register split policy");
  }
  const Linear m3 = gpu(9, 40);
  for (const uint32_t rows : {8U, 32U}) {
    const LinearPlan plan = m3.plan({{5120, 17408}, rows, LinearPhase::Decode, LinearEpilogue::Residual},
                                    blockProjection(5120, 17408, 1));
    const LinearScratchSize size = plan.scratchSize();
    require(plan.partialSums() == 8 && size.input == tableBytes(17408, rows) &&
                size.sums == tableSumsBytes(LinearInput::Table16, 17408, rows) &&
                size.partials == splitPartialsBytes(plan) && size.counters == splitCountersBytes(plan) &&
                plan.gateScratchBytes() == 0,
            "Apple9 GGUF register scratch");
  }
  const LinearPlan head = m3.plan({{248320, 5120}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                  blockProjection(248320, 5120, 1));
  require(head.scratchSize().partials == sizeof(float) && head.scratchSize().counters == sizeof(uint32_t),
          "Apple9 GGUF register scratch without splits binds placeholders");
  const LinearPlan registerGateUp = m3.plan({{17408, 5120}, 16, LinearPhase::Decode, LinearEpilogue::GateUp},
                                            blockProjection(17408, 5120, 1));
  require(registerGateUp.gateScratchBytes() == gateBytes(registerGateUp),
          "Apple9 GGUF gate/up runs a gate pass into the gate scratch");
  require(m3.plan({{5120, 17408}, 100, LinearPhase::Prefill, LinearEpilogue::Residual},
                  blockProjection(5120, 17408, 1)).configuration().tile == LinearTile::GgufStaged,
          "Apple9 GGUF prefill stages");
  require(m3.ggufFloatTile(2048, 256) == FloatTile::Simdgroup, "Apple9 float projections take the simdgroup tile");
  LinearWorkload registerDown = down;
  registerDown.weightLayout = WeightLayout::Block32;
  require(m3.decodeScratchSize(registerDown).partials ==
              m3.plan(down, blockProjection(5120, 17408, 1)).scratchSize().partials,
          "Apple9 GGUF decode scratch bound");
  // Apple9 stages the IQ2, IQ3_XXS and IQ1 formats wherever the staged tile
  // holds the lanes' rows unpadded, and Q2_K from two lanes. A projection
  // stages only when every quantized segment's format does; its plan binds
  // the register tile's rows, and the decode scratch bound covers both tiles.
  for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
    const auto plan = [&](std::initializer_list<uint32_t> formats) {
      const uint32_t n = uint32_t(formats.size()) * 1024;
      return m3.plan({{n, 5120}, lanes * 8, LinearPhase::Decode, LinearEpilogue::None},
                     blockProjection(n, 5120, formats));
    };
    const LinearTile iq = lanes == 3 ? LinearTile::GgufRegister : LinearTile::GgufStaged;
    const LinearTile q2k = lanes == 2 || lanes == 4 ? LinearTile::GgufStaged : LinearTile::GgufRegister;
    bool staged = true;
    for (const uint32_t format :
         {GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2XXS, GGUF_FMT_IQ2XS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1S, GGUF_FMT_IQ1M})
      staged = staged && plan({format}).configuration().tile == iq;
    require(staged && plan({GGUF_FMT_IQ3XXS, GGUF_FMT_IQ2S, GGUF_FMT_IQ1M}).configuration().tile == iq,
            "Apple9 stages the IQ2, IQ3_XXS and IQ1 formats at unpadded lanes");
    require(plan({GGUF_FMT_Q2K}).configuration().tile == q2k, "Apple9 stages Q2_K from two unpadded lanes");
    for (const uint32_t format : {GGUF_FMT_Q4K, GGUF_FMT_Q6K, GGUF_FMT_IQ4XS, GGUF_FMT_IQ3S, GGUF_FMT_Q80,
                                  GGUF_FMT_Q40, GGUF_FMT_MXFP4})
      require(plan({format}).configuration().tile == LinearTile::GgufRegister, "Apple9 keeps other formats' registers");
    require(plan({GGUF_FMT_IQ3XXS, GGUF_FMT_Q4K}).configuration().tile == LinearTile::GgufRegister &&
                plan({GGUF_FMT_IQ3XXS}).storageRows() == plan({GGUF_FMT_Q4K}).storageRows(),
            "Apple9 keeps a mixed projection's registers, and a staged plan binds the register rows");
    // A gate/up plan runs its gate on the same tile: it stages only when the gate's formats do too.
    const LinearWorkload gateUp{{1024, 5120}, lanes * 8, LinearPhase::Decode, LinearEpilogue::GateUp};
    const Projection up = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XXS),
                     iqGate = blockProjection(1024, 5120, 1, GGUF_FMT_IQ2XS), q4kGate = blockProjection(1024, 5120, 1);
    require(m3.plan(gateUp, up, &iqGate).configuration().tile == iq &&
                m3.plan(gateUp, up, &q4kGate).configuration().tile == LinearTile::GgufRegister,
            "Apple9 staged a gate/up plan whose gate keeps the register tile");
    const LinearPlan stagedDown = m3.plan({down.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                          blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XS));
    const LinearPlan registerPlan = m3.plan({down.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual},
                                            blockProjection(5120, 17408, 1));
    const LinearScratchSize bound = m3.decodeScratchSize(
        {down.matrix, lanes * 8, LinearPhase::Decode, LinearEpilogue::Residual, WeightLayout::Block32});
    for (const LinearScratchSize size : {stagedDown.scratchSize(), registerPlan.scratchSize()})
      require(bound.input >= size.input && bound.sums >= size.sums && bound.partials >= size.partials &&
                  bound.counters >= size.counters,
              "Apple9 GGUF decode scratch bound covers both tiles");
  }
  require(linear.plan({{5120, 17408}, 24, LinearPhase::Decode, LinearEpilogue::Residual},
                      blockProjection(5120, 17408, 1, GGUF_FMT_IQ2XXS)).configuration().tile == LinearTile::GgufStaged,
          "Apple10 stages every format");
  // Apple9's staged tile, in decode and in prefill chunks, splits K by the
  // register tile's tiers: on 40 cores 17408 x 5120 in four, 5120 x 17408 in
  // eight.
  for (const auto [n, k, splits] : {std::tuple{17408U, 5120U, 4U}, {5120U, 17408U, 8U}}) {
    const LinearPlan decode = m3.plan({{n, k}, 8, LinearPhase::Decode, LinearEpilogue::None},
                                      blockProjection(n, k, 1, GGUF_FMT_IQ2XXS));
    const LinearPlan chunk = m3.plan({{n, k}, 8, LinearPhase::Prefill, LinearEpilogue::None}, blockProjection(n, k, 1));
    require(decode.configuration() == LinearConfig{LinearTile::GgufStaged, n / 64, LinearSimdgroups::Two, splits} &&
                chunk.configuration().splits == splits,
            "Apple9 staged split tiers");
  }
  for (const LinearConfig config : {LinearConfig{LinearTile::GgufRegister, 80, LinearSimdgroups::Four, 3},
                                    LinearConfig{LinearTile::GgufRegister, 80, LinearSimdgroups::Four, 16},
                                    LinearConfig{LinearTile::GgufRegister, 40, LinearSimdgroups::Four, 8},
                                    LinearConfig{LinearTile::GgufRegister, 80, LinearSimdgroups::Two, 8}})
    rejects([&] { (void)Linear::plan(registerDown, config); });
  // Split boundaries fall on 256-input units, and prefill has no register tile.
  rejects([&] {
    (void)Linear::plan({{5120, 512}, 8, LinearPhase::Decode, LinearEpilogue::None, WeightLayout::Block32},
                         {LinearTile::GgufRegister, 80, LinearSimdgroups::Four, 4});
  });
  rejects([&] {
    (void)Linear::plan({{5120, 17408}, 128, LinearPhase::Prefill, LinearEpilogue::None, WeightLayout::Block32},
                         {LinearTile::GgufRegister, 0, LinearSimdgroups::Four, 1});
  });
}

// The policy's laws, over GPU families 9 and 10 (newerFamilyPlans holds 11 to
// 10's plans), every core count up to 128 and the unknown one (0), and a grid
// of widths, inputs, rows and epilogues, affine and GGUF (one to three Q4_K
// segments). They state what each rule may read; anchorPlans and ggufPlans pin
// the measured decisions.
// - Every plan is one its kernels run, within its arena bound (requireRunnable).
// - Primitive: a tensor GPU plans no register tile; a register GPU plans no
//   Apple10 tile and leaves its register tile only for plain decode
//   projections at three and four lanes (the wide-plain rule).
// - Floors: a split count is a power of two up to eight whose partitions keep
//   whole steps of its kernel family and the inputs of its smallest tier that
//   holds for the plan's rows.
// - Rows: decode plans bind their lanes' rows, but for the GGUF staged tile at
//   three lanes (32 rows). The rules declared rows-dependent are the only ones
//   that read them: the row quanta, the tensor tile law (kAffineTensorTiles:
//   per lane count) and the split tiers' row ranges (Split128: one MPP
//   fragment or two). So a GGUF decode plan is the same at every batch width,
//   an affine K-split plan the same at every width whose rows the same tiers
//   of its split law hold for and whose width runs one, and Apple10 splits K
//   at all or none of those widths. A prefill chunk runs a decode tile exactly
//   when it has at most 32 rows and is GGUF or on Apple9.
// - Scale: doubling the width and the core count keeps the tile, scope and
//   split count, and whether a decode plan takes the full column grid, but for
//   Apple9 prefill across its one absolute core count (32).
// - Monotony: more cores never lower a split count and a wider grid never
//   raises it or narrows the tile; once a float projection's chunk takes the
//   neural accelerator, longer chunks and fewer cores keep it.
void policyLaws() {
  const auto policy = [](uint32_t family, uint32_t cores) {
    DeviceCapabilities device;
    device.appleGpuFamily = family;
    device.gpuCoreCount = cores;
    return DevicePolicy(device);
  };
  require(policy(9, 40).primitive == Primitive::Register && policy(10, 20).primitive == Primitive::Tensor &&
              policy(11, 12).primitive == Primitive::Tensor && policy(0, 0).primitive == Primitive::Tensor &&
              policy(9, 40).cores == 40 && policy(10, 0).cores == kAssumedGpuCores,
          "the family decides the primitive and a missing core count takes the assumed one");
  // Neighbouring widths differ by at most 2x, so a threshold that ignored
  // the core count would split some width from its double.
  constexpr std::array<uint32_t, 17> widths{256,  512,   768,   1024,  1280,  2048,  2560,   4096,  5120,
                                            6144, 9216,  12544, 17408, 32768, 65536, 131072, 248320};
  // 4352 and 6400 inputs leave Apple9's eight even partitions short of whole
  // quant groups.
  constexpr std::array<uint32_t, 12> inputs{256, 512, 768, 1024, 2048, 4096, 4352, 5120, 6144, 6400, 17408, 25600};
  constexpr std::array<uint32_t, 8> prefillRows{1, 8, 17, 32, 33, 64, 127, 2048};
  for (const uint32_t n : widths)
    for (const uint32_t k : inputs)
      // No segments: affine weights.
      for (uint32_t segments = 0; segments <= 3; ++segments) {
        const bool gguf = segments > 0;
        const Projection weights = gguf ? blockProjection(n, k, segments) : Projection(n, k, AffineWeights{});
        const Projection wider = gguf ? blockProjection(2 * n, k, segments) : Projection(2 * n, k, AffineWeights{});
        // A fused projection takes no epilogue; prefill runs gate/up as a gate
        // and an up-with-gate projection.
        const auto epilogues = [&](LinearPhase phase) {
          if (segments > 1) return std::vector<LinearEpilogue>{LinearEpilogue::None};
          return std::vector<LinearEpilogue>{LinearEpilogue::None, LinearEpilogue::Residual,
                                             phase == LinearPhase::Decode ? LinearEpilogue::GateUp
                                                                          : LinearEpilogue::UpWithGate};
        };
        for (const uint32_t family : {9U, 10U})
          for (uint32_t cores = 0; cores <= 128; ++cores) {
            const Linear linear = gpu(family, cores), more = gpu(family, cores + 1), twice = gpu(family, 2 * cores);
            const Primitive primitive = policy(family, cores).primitive;
            const auto plan = [&](const Linear &l, const Projection &p, uint32_t rows, LinearPhase phase,
                                  LinearEpilogue e) {
              return l.plan({{p.outputSize, k}, rows, phase, e}, p, e == LinearEpilogue::GateUp ? &p : nullptr);
            };
            const auto rule = [&](bool holds, const char *name, const LinearPlan &p) {
              if (!holds) broke(name, family, cores, p.workload());
            };
            // The laws every plan keeps, whatever its phase.
            const auto common = [&](const LinearPlan &p, LinearScratchSize arena) {
              requireRunnable(p, arena, family, cores);
              const LinearConfig c = p.configuration();
              const LinearTile t = c.tile;
              rule(primitive == Primitive::Register
                       ? t != LinearTile::Split128 && t != LinearTile::Paired128 && t != LinearTile::Paired256
                       : t != LinearTile::Simdgroup && t != LinearTile::GgufRegister,
                   "a plan's tile is not its primitive's", p);
              const SplitLaw &law = p.kernelFamily().split;
              uint32_t floor = 0;
              for (const SplitTier &tier : law.tiers(primitive))
                if (tier.holds(p.storageRows()) && (!floor || tier.inputs < floor)) floor = tier.inputs;
              rule(c.validSplits() &&
                       (c.splits == 1 ||
                        (floor && k >= c.splits * floor &&
                         (law.evenPartitions ? k % (c.splits * law.partitionInputs) == 0
                                             : k / law.partitionInputs >= c.splits))),
                   "a split count breaks its kernel family's floors", p);
              if (!cores) return;
              const LinearWorkload w = p.workload();
              const LinearPlan scaled = plan(twice, wider, w.rows, w.phase, w.epilogue);
              const LinearConfig s = scaled.configuration();
              const bool prefillClass = primitive == Primitive::Register && !gguf && w.phase == LinearPhase::Prefill &&
                  w.rows > kMaximumDecodeTileRows && cores <= 32 && 2 * cores > 32;
              const bool fullGrid = c.groups == n / p.tileColumns(),
                         scaledFullGrid = s.groups == 2 * n / scaled.tileColumns();
              rule(prefillClass || (s.tile == t && s.simdgroups == c.simdgroups && s.splits == c.splits &&
                                    (w.phase == LinearPhase::Prefill || fullGrid == scaledFullGrid)),
                   "a plan depends on more than its grid per core", p);
              const LinearPlan moreCores = plan(more, weights, w.rows, w.phase, w.epilogue),
                               widerGrid = plan(linear, wider, w.rows, w.phase, w.epilogue);
              rule(moreCores.configuration().splits >= c.splits && widerGrid.configuration().splits <= c.splits &&
                       widerGrid.tileColumns() >= p.tileColumns(),
                   "a split count or tile width is not monotone in cores and width", p);
            };
            for (const LinearEpilogue e : epilogues(LinearPhase::Decode)) {
              std::vector<LinearPlan> lanePlans;
              for (uint32_t lanes = 1; lanes <= 4; ++lanes) {
                const LinearPlan p = plan(linear, weights, lanes * 8, LinearPhase::Decode, e);
                common(p, linear.decodeScratchSize(p.workload()));
                const LinearTile t = p.configuration().tile;
                rule(primitive == Primitive::Tensor || gguf || t == LinearTile::Simdgroup ||
                         (lanes >= 3 && e == LinearEpilogue::None),
                     "Apple9 left its register tile beyond the wide plain projections", p);
                rule(p.storageRows() == (t == LinearTile::GgufStaged && lanes == 3 ? 32U : lanes * 8),
                     "a decode plan's storage rows differ from its lanes' rows", p);
                lanePlans.push_back(p);
              }
              const auto splitsK = [](const LinearPlan &p) {
                const LinearTile t = p.configuration().tile;
                return t == LinearTile::Simdgroup || t == LinearTile::Split128;
              };
              // The split law deciding whether a width splits, and the tiers
              // of it that hold for each width's rows.
              const SplitLaw &splitLaw = gguf ? lanePlans.front().kernelFamily().split
                                         : primitive == Primitive::Register ? kAffineRegisterTile.split
                                                                            : kAffineTensorSplit.split;
              const auto heldTiers = [&](const LinearPlan &p) {
                std::vector<bool> held;
                for (const SplitTier &tier : splitLaw.tiers(primitive)) held.push_back(tier.holds(p.storageRows()));
                return held;
              };
              for (const LinearPlan &p : lanePlans)
                for (const LinearPlan &q : lanePlans)
                  if (heldTiers(p) == heldTiers(q))
                    rule(gguf ? p.configuration() == q.configuration()
                              : (!splitsK(p) || !splitsK(q) || q.configuration() == p.configuration()) &&
                                    (primitive == Primitive::Register || splitsK(p) == splitsK(q)),
                         "a K-split plan depends on the batch width beyond its tiers' row ranges", p);
            }
            const LinearScratchSize prefillArena = linear.prefillScratchSize(weights.shape());
            for (const LinearEpilogue e : epilogues(LinearPhase::Prefill))
              for (const uint32_t rows : prefillRows) {
                const LinearPlan p = plan(linear, weights, rows, LinearPhase::Prefill, e);
                common(p, prefillArena);
                const LinearConfig c = p.configuration();
                const bool decodeTile = c.tile == LinearTile::Simdgroup ||
                    (c.tile == LinearTile::GgufStaged && c.simdgroups == LinearSimdgroups::Two);
                rule(decodeTile == (rows <= kMaximumDecodeTileRows && (gguf || primitive == Primitive::Register)),
                     "a prefill chunk ran the wrong tile for its rows", p);
              }
          }
      }
  // The float tile follows the grid per core: once a chunk takes the neural
  // accelerator, longer chunks and fewer cores keep it; never below 16 rows or
  // on Apple9.
  for (uint32_t cores = 1; cores <= 128; ++cores)
    for (const uint32_t n : {16U, 64U, 256U, 1024U}) {
      const Linear linear = gpu(10, cores), more = gpu(10, cores + 1);
      bool accelerator = false;
      for (uint32_t rows = 1; rows <= 2048; ++rows) {
        const bool now = linear.ggufFloatTile(rows, n) == FloatTile::NeuralAccelerator;
        require((!accelerator || now) && (!now || rows >= 16) &&
                    (more.ggufFloatTile(rows, n) == FloatTile::Simdgroup || now),
                "GGUF float tile is not monotone in rows and cores");
        accelerator = now;
      }
      require(gpu(9, cores).ggufFloatTile(2048, n) == FloatTile::Simdgroup, "Apple9 GGUF float tile");
    }
}

// Exercise continuous core counts, not just measured SKU anchors. These
// contracts check legal grids, bounded candidates and override workspace;
// they do not claim performance on simulated hardware.
void scalingContracts() {
  for (uint32_t family : {9U, 10U, 11U}) {
    for (uint32_t index = 0; index <= 129; ++index) {
      const uint32_t reported = index == 129 ? 4096 : index;
      const uint32_t cores = reported ? reported : 32U;
      Linear linear = gpu(family, reported);
      for (uint32_t n : {256U, 5120U, 131072U}) {
        for (uint32_t k : {256U, 768U, 1024U, 4096U, 5120U, 17408U}) {
          for (uint32_t rows : {8U, 16U, 24U, 32U}) {
            for (auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::GateUp}) {
              const LinearWorkload w{{n, k}, rows, LinearPhase::Decode, epilogue};
              const auto baseline = linear.plan(w);
              const auto candidates = linear.candidates(w);
              require(candidates.front().configuration() == baseline.configuration() &&
                          candidates.size() <= Linear::kMaximumCandidates,
                      "core scaling lost or displaced the baseline");
              for (size_t i = 0; i < candidates.size(); ++i) {
                const auto &plan = candidates[i];
                require(plan.configuration().groups > 0 &&
                            plan.configuration().groups <= n / plan.tileColumns(),
                        "core scaling produced an invalid grid");
                for (size_t j = 0; j < i; ++j)
                  require(plan.configuration() != candidates[j].configuration(),
                          "core scaling produced duplicate candidates");
                const std::array choice{LinearChoice{w, plan.configuration()}};
                linear.setChoices(choice);
                const auto scratch = linear.decodeScratchSize(w);
                for (const auto required : {baseline.scratchSize(), plan.scratchSize()})
                  require(scratch.input >= required.input && scratch.sums >= required.sums &&
                              scratch.partials >= required.partials && scratch.counters >= required.counters,
                          "installed candidate exceeds admitted Q4 workspace");
              }
              linear.setChoices({});
              // N128 is available for every non-gated decode workload. Its
              // candidate waves must follow the device, including tiny GPUs.
              if (epilogue != LinearEpilogue::GateUp) {
                for (uint32_t wave : {2U, 3U, 4U}) {
                  const LinearConfig wanted{LinearTile::N128, std::min(n / 128, cores * wave)};
                  require(std::any_of(candidates.begin(), candidates.end(), [&](const auto &p) {
                    return p.configuration() == wanted;
                  }), "candidate waves do not scale with GPU core count");
                }
              }
            }
          }
        }
      }
    }
  }
}

metal::MetalBuffer allocate(metal::MetalBackend &backend, uint64_t bytes) {
  if (!bytes) return {};
  auto buffer = backend.allocateBuffer(bytes);
  std::memset(buffer.contents(), 0, bytes);
  return buffer;
}

// A block projection dispatches only as the plan's matrix, and each segment
// fills whole 64-column tiles, with every buffer the plan needs: the matching
// projection encodes its dispatch, so only the projection can reject.
void ggufProjectionMatrix(metal::MetalBackend &backend) {
  const Linear linear = gpu(10, 16);
  const auto segment = [](uint32_t n, uint32_t offset) {
    QuantizedSegment s = QuantizedSegment::planes(GGUF_FMT_Q4K, n, 17408, {}, {}, {});
    s.columnOffset = offset;
    return s;
  };
  const Projection matching(5120, 17408, BlockWeights{{segment(5120, 0)}});
  const LinearPlan plan = linear.plan({{5120, 17408}, 8}, matching);
  const uint64_t rows = plan.storageRows();
  const LinearScratchSize scratch = plan.scratchSize();
  const LinearBuffers buffers{
      .input = allocate(backend, rows * 17408 * 2),
      .output = allocate(backend, rows * 5120 * 2),
      .sums = allocate(backend, plan.sumsBytes()),
      .gateScratch = allocate(backend, plan.gateScratchBytes()),
      .downSums = allocate(backend, plan.downSumsBytes()),
      .scratch = {allocate(backend, scratch.input), allocate(backend, scratch.sums),
                  allocate(backend, scratch.partials), allocate(backend, scratch.counters)}};
  {
    metal::CommandGraph graph;
    (void)linear.add(graph, buffers, matching, plan);
    require(graph.dispatches().size() == 1, "the matching block projection did not encode its dispatch");
  }
  // The padding past a projection's segments is part of it, not room for a
  // narrower plan; a segment of 32 columns leaves a partial tile.
  for (const Projection &projection : {Projection(5376, 17408, BlockWeights{{segment(5120, 0)}}),
                                       Projection(5120, 17408, BlockWeights{{segment(5088, 0), segment(32, 5088)}})}) {
    metal::CommandGraph graph;
    rejects([&] { (void)linear.add(graph, buffers, projection, plan); });
    require(graph.empty(), "a block projection that is not the plan's tiles encoded a dispatch");
  }
  // A projection with an fp32 destination writes fp32 rows through the
  // kernel's fp32 instance; the fused kernels have none.
  Projection head = matching, fused(5120, 17408, BlockWeights{{segment(2560, 0), segment(2560, 2560)}});
  head.destination = fused.destination = FloatOutput::Float32;
  const LinearPlan fp32 = linear.plan({{5120, 17408}, 8}, head);
  LinearBuffers fp32Buffers = buffers;
  fp32Buffers.output = allocate(backend, rows * 5120 * sizeof(float));
  metal::CommandGraph graph;
  rejects([&] { (void)linear.add(graph, buffers, head, fp32); });
  rejects([&] { (void)linear.add(graph, fp32Buffers, fused, linear.plan({{5120, 17408}, 8}, fused)); });
  require(graph.empty(), "an invalid fp32 block projection encoded a dispatch");
  (void)linear.add(graph, fp32Buffers, head, fp32);
  require(graph.dispatches().size() == 1 && graph.dispatches()[0].pipelineName == "gguf_decode_q4k_m8_a_f32",
          "the fp32 block projection did not encode its fp32 kernel");
}

std::array<uint64_t, 3> projectionFingerprint(const Projection &projection) {
  std::array<uint64_t, 3> result{};
  const std::array buffers{projection.affine().weights, projection.affine().scales, projection.affine().biases};
  for (size_t slot = 0; slot < buffers.size(); ++slot) {
    const auto *bytes = static_cast<const uint8_t *>(buffers[slot].contents());
    uint64_t hash = 14695981039346656037ULL;
    for (uint64_t i = 0; i < buffers[slot].sizeBytes(); ++i)
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    result[slot] = hash;
  }
  return result;
}

// Scalar reference uses the actual StorageN=256 bytes and FP32 per-group
// affine accumulation, including the BF16 boundary before each epilogue.
float affineReference(const Projection &p, const uint16_t *input,
                        uint32_t row, uint32_t column) {
  const auto *weights = static_cast<const uint8_t *>(p.affine().weights.contents());
  const auto *scales = static_cast<const uint16_t *>(p.affine().scales.contents());
  const auto *biases = static_cast<const uint16_t *>(p.affine().biases.contents());
  const uint32_t groups = p.inputSize / 64;
  float result = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    const uint64_t parameter = (uint64_t{column / 256} * groups + group) * 256 + column % 256;
    float partial = 0, sum = 0;
    for (uint32_t k = 0; k < 64; ++k) {
      const uint8_t byte = weights[parameter * 32 + k / 2];
      const uint32_t quantized = (byte >> ((k & 1) * 4)) & 15;
      const float x = tuning::bf16ToFloat(input[uint64_t{row} * p.inputSize + group * 64 + k]);
      sum += x;
      partial += x * quantized;
    }
    result += partial * tuning::bf16ToFloat(scales[parameter]) + sum * tuning::bf16ToFloat(biases[parameter]);
  }
  return tuning::bf16ToFloat(tuning::floatToBf16(result));
}

void checkReference(const Projection &p, const Projection &gate,
                      LinearWorkload workload, const LinearBuffers &buffers,
                      const uint16_t *savedResidual = nullptr, bool split = false, float slack = 0) {
  const auto *input = static_cast<const uint16_t *>(buffers.input.contents());
  const auto *output = static_cast<const uint16_t *>(buffers.output.contents());
  const auto *residual = savedResidual ? savedResidual
      : static_cast<const uint16_t *>(buffers.residual.contents());
  const auto *gateValues = static_cast<const uint16_t *>(buffers.gateScratch.contents());
  for (const uint32_t row : {0U, workload.rows / 2, workload.rows - 1}) {
    for (const uint32_t column : {0U, 127U, 128U, 255U,
                                  p.outputSize / 2, p.outputSize - 1}) {
      const float projection = affineReference(p, input, row, column);
      float expected = projection;
      float residualValue = 0, gateValue = 0;
      const uint64_t index = uint64_t{row} * p.outputSize + column;
      if (workload.epilogue == LinearEpilogue::Residual) {
        residualValue = tuning::bf16ToFloat(residual[index]);
        expected += residualValue;
      }
      if (workload.epilogue == LinearEpilogue::GateUp ||
          workload.epilogue == LinearEpilogue::UpWithGate) {
        gateValue = workload.epilogue == LinearEpilogue::GateUp
            ? affineReference(gate, input, row, column) : tuning::bf16ToFloat(gateValues[index]);
        expected *= gateValue / (1 + std::exp(-gateValue));
      }
      expected = tuning::bf16ToFloat(tuning::floatToBf16(expected));
      // The oracle accumulates in the sequential kernel's order. A split tile
      // reassociates that sum, so it is held to the derived bf16 bound
      // (tuning/LinearNumerics.hpp) on top of the oracle's own margin.
      float tolerance = 0.004f;
      if (split)
        tolerance += tuning::splitTolerance(workload.epilogue,
            {expected, residualValue, gateValue, projection}, slack);
      const float actual = tuning::bf16ToFloat(output[index]);
      if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::cerr << "reference row=" << row << " col=" << column
                  << " actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("Linear failed independent packed-Q4 oracle");
      }
    }
  }
  // The MPP tile's fused up projection also writes its output's Q4 sums.
  if (buffers.downSums) {
    const auto *sums = static_cast<const float *>(buffers.downSums.contents());
    const uint32_t quantGroups = p.outputSize / 64;
    for (uint32_t row = 0; row < workload.rows; ++row) {
      for (uint32_t group = 0; group < quantGroups; ++group) {
        double expected = 0;
        for (uint32_t k = 0; k < 64; ++k)
          expected += tuning::bf16ToFloat(output[uint64_t{row} * p.outputSize + group * 64 + k]);
        const uint64_t index = uint64_t{row / 32} * 32 * quantGroups + group * 32 + row % 32;
        require(std::abs(sums[index] - expected) <= 1e-6 * std::max(1.0, std::abs(expected)),
                "fused prefill output sums have wrong layout/value");
      }
    }
  }
}

void bufferContracts(metal::MetalBackend &backend, Linear &linear,
                        LinearBuffers buffers, const Projection &p,
                        const Projection &gate, const LinearPlan &plan) {
  const bool gateUp = plan.workload().epilogue == LinearEpilogue::GateUp;
  const auto add = [&](metal::CommandGraph &graph, LinearBuffers b,
                        const Projection &projection, const Projection *g) {
    linear.add(graph, b, projection, plan, g);
  };
  const std::array members{&LinearBuffers::input, &LinearBuffers::output,
                           &LinearBuffers::sums, &LinearBuffers::residual,
                           &LinearBuffers::gateScratch, &LinearBuffers::downSums};
  for (auto member : members) {
    if (!(buffers.*member)) continue;
    auto shortBuffers = buffers;
    shortBuffers.*member = backend.view(buffers.*member, 0, (buffers.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, shortBuffers, p, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid Linear buffers partially encoded a graph");
  }
  // Every scratch field the plan uses (the simdgroup tile's table, sums,
  // partials and counters; the Split128 partials and counters).
  const LinearScratchSize scratch = plan.scratchSize();
  for (const auto [member, bytes] : {std::pair{&LinearScratch::input, scratch.input},
                                     std::pair{&LinearScratch::sums, scratch.sums},
                                     std::pair{&LinearScratch::partials, scratch.partials},
                                     std::pair{&LinearScratch::counters, scratch.counters}}) {
    if (!bytes) continue;
    auto shortBuffers = buffers;
    shortBuffers.scratch.*member = backend.view(buffers.scratch.*member, 0,
                                                (buffers.scratch.*member).sizeBytes() - 1);
    metal::CommandGraph graph;
    rejects([&] { add(graph, shortBuffers, p, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid split workspace partially encoded a graph");
  }
  for (auto member : {&AffineWeights::weights, &AffineWeights::scales, &AffineWeights::biases}) {
    AffineWeights planes = p.affine();
    planes.*member = backend.view(p.affine().*member, 0, (p.affine().*member).sizeBytes() - 1);
    const Projection shortProjection(p.outputSize, p.inputSize, planes);
    metal::CommandGraph graph;
    rejects([&] { add(graph, buffers, shortProjection, gateUp ? &gate : nullptr); });
    require(graph.empty(), "invalid projection partially encoded a graph");
  }
  auto mismatch = p;
  mismatch.inputSize += 256;
  metal::CommandGraph graph;
  rejects([&] { add(graph, buffers, mismatch, gateUp ? &gate : nullptr); });
  rejects([&] { add(graph, buffers, p, gateUp ? nullptr : &gate); });
  require(graph.empty(), "invalid Linear gate/projection partially encoded graph");
  if (plan.sumsBytes()) {
    const auto workload = plan.workload();
    rejects([&] {
      linear.addPrefillSums(graph,
          backend.view(buffers.input, 0, buffers.input.sizeBytes() - 1), buffers.sums,
          p, workload.rows);
    });
    rejects([&] {
      linear.addPrefillSums(graph, buffers.input,
          backend.view(buffers.sums, 0, buffers.sums.sizeBytes() - 1),
          p, workload.rows);
    });
    for (const LinearMatrix matrix : {LinearMatrix{0, workload.matrix.inputSize},
           LinearMatrix{128, workload.matrix.inputSize},
           LinearMatrix{workload.matrix.outputSize, 0},
           LinearMatrix{workload.matrix.outputSize, 63}})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums,
                                         Projection(matrix.outputSize, matrix.inputSize, p.affine()),
                                         workload.rows); });
    for (uint32_t rows : {0U, SPLASH_PREFILL_TOKEN_BUDGET + 1U})
      rejects([&] { linear.addPrefillSums(graph, buffers.input, buffers.sums, p, rows); });
    require(graph.empty(), "invalid prefill sums input partially encoded graph");
  }
}

void numericalCase(metal::MetalBackend &backend, Linear &linear,
                      const Projection &p, const Projection &gate,
                      LinearWorkload workload, bool inPlaceResidual = false) {
  require(!inPlaceResidual || workload.epilogue == LinearEpilogue::Residual,
          "in-place residual fixture requires residual epilogue");
  // Candidates never mix tiles of different storage rows (Linear::candidates).
  const auto candidates = linear.candidates(workload);
  const uint32_t storageRows = candidates[0].storageRows();
  // A plan's own scratch, exact so that bufferContracts can shorten it.
  const auto scratchFor = [&](const LinearPlan &plan) {
    const auto scratch = plan.scratchSize();
    return LinearScratch{allocate(backend, scratch.input), allocate(backend, scratch.sums),
                         allocate(backend, scratch.partials), allocate(backend, scratch.counters)};
  };
  auto input = allocate(backend, uint64_t{storageRows} * p.inputSize * 2);
  auto *inputValues = static_cast<uint16_t *>(input.contents());
  for (uint64_t i = 0; i < uint64_t{workload.rows} * p.inputSize; ++i)
    inputValues[i] = tuning::floatToBf16(float(int(mix(uint32_t(i) + 1949) % 257) - 128) / 257.0f);
  // Sequential candidates share their output bytes; split-K candidates are
  // held to the derived bound against them once every candidate has run.
  std::vector<uint16_t> baseline;
  struct SplitOutput final {
    std::vector<uint16_t> output, residual;
    std::string_view pipeline;
    bool simdgroup = false;
  };
  std::vector<SplitOutput> splitOutputs;
  for (const auto &plan : candidates) {
    const uint64_t outputBytes = uint64_t{storageRows} * p.outputSize * 2;
    const uint64_t guardBytes = uint64_t{8} * p.outputSize * 2;
    auto outputBacking = allocate(backend, outputBytes + guardBytes);
    std::memset(static_cast<uint8_t *>(outputBacking.contents()) + outputBytes, 0x5a, guardBytes);
    const uint64_t gateBytes = plan.gateScratchBytes();
    auto gateBacking = allocate(backend, gateBytes ? gateBytes + guardBytes : 0);
    if (gateBytes)
      std::memset(static_cast<uint8_t *>(gateBacking.contents()) + gateBytes, 0x5a, guardBytes);
    LinearBuffers b{input, backend.view(outputBacking, 0, outputBytes),
                     allocate(backend, plan.sumsBytes()), {},
                     gateBytes ? backend.view(gateBacking, 0, gateBytes) : metal::MetalBuffer{},
                     allocate(backend, plan.downSumsBytes()), scratchFor(plan)};
    if (workload.epilogue == LinearEpilogue::Residual) {
      b.residual = inPlaceResidual ? b.output : allocate(backend, b.output.sizeBytes());
      auto *residual = static_cast<uint16_t *>(b.residual.contents());
      for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i)
        residual[i] = tuning::floatToBf16(float(int(mix(uint32_t(i) + 7919) % 257) - 128) / 257.0f);
    }
    bufferContracts(backend, linear, b, p, gate, plan);
    metal::CommandGraph graph;
    if (plan.sumsBytes())
      linear.addPrefillSums(graph, input, b.sums, p, workload.rows);
    if (workload.epilogue == LinearEpilogue::UpWithGate) {
      const auto gatePlan = linear.plan({workload.matrix, workload.rows, LinearPhase::Prefill,
                                         LinearEpilogue::None});
      linear.add(graph, {input, b.gateScratch, b.sums, {}, {}, {}, scratchFor(gatePlan)}, gate, gatePlan);
    }
    const size_t prepasses = graph.dispatches().size();
    LinearDispatchStats stats;
    linear.add(graph, b, p, plan, workload.epilogue == LinearEpilogue::GateUp ? &gate : nullptr, &stats);
    const auto &last = graph.dispatches().back();
    require(last.pipelineName == (plan.secondPipeline().empty() ? plan.pipeline() : plan.secondPipeline()),
            "production dispatch differs from Linear plan");
    for (const auto &dispatch : graph.dispatches().subspan(prepasses))
      require(dispatch.threadsPerThreadgroup.x == plan.threadsPerThreadgroup() &&
                  dispatch.threadsPerThreadgroup.y == 1 && dispatch.threadsPerThreadgroup.z == 1,
              "production dispatch threads differ from Linear plan scope");
    if (workload.phase == LinearPhase::Decode) {
      const uint32_t lanes = workload.rows / 8;
      const uint32_t dispatches = plan.usesSimdgroup() ? 2 : plan.secondPipeline().empty() ? 1 : 2;
      require(last.threadgroups.x == plan.configuration().groups &&
                  graph.dispatches().size() == dispatches,
              "Linear decode plan/graph geometry mismatch");
      // These counters describe projection fusion, excluding input preparation.
      const uint32_t projections = plan.secondPipeline().empty() ? 1 : 2;
      require(stats.fusedSourceOperations == (lanes == 1 ? 0 : lanes * projections) &&
                  stats.m16Dispatches == (lanes == 2 ? projections : 0) &&
                  stats.m24Dispatches == (lanes == 3 ? projections : 0) &&
                  stats.m32Dispatches == (lanes == 4 ? projections : 0),
              "Linear dispatch statistics changed");
    } else if (plan.usesSimdgroup()) {
      // The table's preparation, then the split grid over every lane.
      require(last.threadgroups.x == plan.configuration().groups &&
                  last.threadgroups.y == plan.configuration().splits && last.threadgroups.z == storageRows / 8 &&
                  graph.dispatches().size() == prepasses + 2,
              "Linear simdgroup prefill plan/graph geometry mismatch");
    } else {
      require(last.threadgroups.x == storageRows / 32 &&
                  last.threadgroups.y == p.outputSize / plan.tileColumns() &&
                  graph.dispatches().size() == prepasses + 1,
              "Linear prefill plan/graph geometry mismatch");
    }
    const auto snapshot = [](const metal::MetalBuffer &buffer) {
      if (!buffer) return std::vector<uint8_t>{};
      const auto *begin = static_cast<const uint8_t *>(buffer.contents());
      return std::vector<uint8_t>{begin, begin + buffer.sizeBytes()};
    };
    const auto immutableInput = snapshot(b.input);
    std::vector<uint16_t> immutableResidual;
    if (b.residual) {
      const auto *values = static_cast<const uint16_t *>(b.residual.contents());
      immutableResidual.assign(values, values + b.residual.sizeBytes() / sizeof(uint16_t));
    }
    (void)backend.submitCommand(graph.dispatches());
    if (workload.phase == LinearPhase::Decode && workload.rows == 24) {
      const auto firstOutput = snapshot(b.output);
      if (inPlaceResidual)
        std::memcpy(b.residual.contents(), immutableResidual.data(),
                    immutableResidual.size() * sizeof(uint16_t));
      (void)backend.submitCommand(graph.dispatches());
      require(std::memcmp(firstOutput.data(), b.output.contents(), firstOutput.size()) == 0,
              "repeated M24 dispatch changed its output bytes");
    }
    const auto checkGuard = [&](const metal::MetalBuffer &backing, uint64_t payload,
                                 const char *role) {
      const auto *guard = static_cast<const uint8_t *>(backing.contents()) + payload;
      for (uint64_t i = 0; i < guardBytes; ++i) {
        if (guard[i] != 0x5a) {
          std::cerr << role << " guard overwritten matrix=" << p.outputSize << 'x' << p.inputSize
                    << " rows=" << workload.rows << " epilogue=" << uint32_t(workload.epilogue)
                    << " pipeline=" << plan.pipeline() << " first_extra_byte=" << i << '\n';
          throw std::runtime_error("Linear wrote beyond its exact workspace/output view");
        }
      }
    };
    checkGuard(outputBacking, outputBytes, "output");
    if (gateBytes) checkGuard(gateBacking, gateBytes, "gate scratch");
    require(std::memcmp(immutableInput.data(), b.input.contents(), immutableInput.size()) == 0,
            "Linear modified its immutable input");
    if (!inPlaceResidual && !immutableResidual.empty())
      require(std::memcmp(immutableResidual.data(), b.residual.contents(),
                          immutableResidual.size() * sizeof(uint16_t)) == 0,
              "Linear modified its immutable residual");
    try {
      checkReference(p, gate, workload, b,
                     inPlaceResidual ? immutableResidual.data() : nullptr,
                     plan.partialSums() > 1 || plan.usesSimdgroup(),
                     plan.usesSimdgroup() ? std::max(tuning::simdgroupSlack(workload, b.input, p),
                         tuning::simdgroupSlack(workload, b.input, gate)) : 0);
    } catch (const std::exception &) {
      std::cerr << "matrix=" << p.outputSize << 'x' << p.inputSize
                << " rows=" << workload.rows << " phase=" << uint32_t(workload.phase)
                << " epilogue=" << uint32_t(workload.epilogue)
                << " tile=" << uint32_t(plan.configuration().tile)
                << " groups=" << plan.configuration().groups
                << " simdgroups=" << uint32_t(plan.configuration().simdgroups)
                << " in_place_residual=" << inPlaceResidual
                << " pipeline=" << plan.pipeline() << '\n';
      throw;
    }
    const auto *output = static_cast<const uint16_t *>(b.output.contents());
    const uint64_t elements = uint64_t{storageRows} * p.outputSize;
    if (workload.phase == LinearPhase::Decode && workload.epilogue == LinearEpilogue::None) {
      // The fp32 instance (the logits) holds the values the bf16 one
      // rounds, bit for bit, and writes nothing past its rows.
      const LinearPlan fp32 = Linear::plan(workload, plan.configuration(), FloatOutput::Float32);
      const uint64_t fp32Bytes = elements * sizeof(float);
      auto fp32Backing = allocate(backend, fp32Bytes + guardBytes);
      std::memset(static_cast<uint8_t *>(fp32Backing.contents()) + fp32Bytes, 0x5a, guardBytes);
      LinearBuffers fp32Buffers = b;
      fp32Buffers.output = backend.view(fp32Backing, 0, fp32Bytes);
      metal::CommandGraph fp32Graph;
      linear.add(fp32Graph, fp32Buffers, p, fp32);
      require(fp32Graph.dispatches().back().pipelineName == kernelInstance(plan.pipeline(), FloatOutput::Float32),
              "an fp32 plan did not dispatch its kernel's fp32 instance");
      (void)backend.submitCommand(fp32Graph.dispatches());
      checkGuard(fp32Backing, fp32Bytes, "fp32 output");
      const auto *values = static_cast<const float *>(fp32Buffers.output.contents());
      for (uint64_t i = 0; i < elements; ++i)
        if (tuning::floatToBf16(values[i]) != output[i]) {
          std::cerr << "fp32 element=" << i << " value=" << values[i] << " bf16=" << tuning::bf16ToFloat(output[i])
                    << " pipeline=" << fp32.pipeline() << '\n';
          throw std::runtime_error("an fp32 output does not round to its bf16 plan's output");
        }
    }
    if (plan.partialSums() > 1 || plan.usesSimdgroup()) {
      splitOutputs.push_back({{output, output + elements}, immutableResidual, plan.pipeline(), plan.usesSimdgroup()});
    } else {
      if (baseline.empty()) baseline.assign(output, output + elements);
      require(std::memcmp(baseline.data(), output, elements * 2) == 0,
              "Linear candidate differs from baseline output bytes");
    }
    for (uint64_t i = uint64_t{workload.rows} * p.outputSize; i < elements; ++i)
      require(output[i] == 0, "padded prefill rows were not zero");
  }
  // Apple9's short prefill chunks list only simdgroup plans, which the oracle
  // above bounds (q4-prefill-projection holds them to the MPP prefill tile).
  if (splitOutputs.empty() || (baseline.empty() && workload.phase == LinearPhase::Prefill)) return;
  require(!baseline.empty(), "split-K candidates have no sequential reference");
  // The gate/up bound needs the exact gate and up projections: the sequential
  // N128 plain plan on both weight sets.
  std::vector<uint16_t> gateReference, upReference;
  if (workload.epilogue == LinearEpilogue::GateUp) {
    const auto plain = Linear::plan({workload.matrix, workload.rows, LinearPhase::Decode,
                                       LinearEpilogue::None},
                                      {LinearTile::N128, p.outputSize / 128});
    auto gateOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    auto upOutput = allocate(backend, uint64_t{storageRows} * p.outputSize * 2);
    metal::CommandGraph graph;
    linear.add(graph, {input, gateOutput, {}, {}, {}, {}}, gate, plain);
    linear.add(graph, {input, upOutput, {}, {}, {}, {}}, p, plain);
    (void)backend.submitCommand(graph.dispatches());
    const auto *g = static_cast<const uint16_t *>(gateOutput.contents());
    const auto *u = static_cast<const uint16_t *>(upOutput.contents());
    gateReference.assign(g, g + baseline.size());
    upReference.assign(u, u + baseline.size());
  }
  float maxAbs = 0;
  for (const uint16_t value : baseline) maxAbs = std::max(maxAbs, std::fabs(tuning::bf16ToFloat(value)));
  const float slack = tuning::reassociationSlack(p.inputSize, maxAbs);
  const float operandSlack = std::max(tuning::simdgroupSlack(workload, input, p),
                                      tuning::simdgroupSlack(workload, input, gate));
  for (const auto &split : splitOutputs) {
    const float toleranceSlack = slack + (split.simdgroup ? operandSlack : 0);
    for (uint64_t i = 0; i < uint64_t{workload.rows} * p.outputSize; ++i) {
      tuning::SplitReference reference{tuning::bf16ToFloat(baseline[i])};
      if (workload.epilogue == LinearEpilogue::Residual) reference.residual = tuning::bf16ToFloat(split.residual[i]);
      if (workload.epilogue == LinearEpilogue::GateUp) {
        reference.gate = tuning::bf16ToFloat(gateReference[i]);
        reference.up = tuning::bf16ToFloat(upReference[i]);
      }
      if (!tuning::withinSplitTolerance(tuning::bf16ToFloat(split.output[i]), workload.epilogue, reference,
                                        toleranceSlack)) {
        std::cerr << "split element=" << i << " actual=" << tuning::bf16ToFloat(split.output[i])
                  << " reference=" << reference.value << " residual=" << reference.residual
                  << " gate=" << reference.gate << " up=" << reference.up
                  << " bound=" << tuning::splitTolerance(workload.epilogue, reference, toleranceSlack)
                  << " matrix=" << p.outputSize << 'x' << p.inputSize
                  << " epilogue=" << uint32_t(workload.epilogue)
                  << " pipeline=" << split.pipeline << '\n';
        throw std::runtime_error("split-K candidate exceeds its bf16 bound against the sequential tiles");
      }
    }
  }
}

// Every plain decode kernel of floatOutputPlans has the fp32 instance its
// fp32 plans run.
void floatInstances(const char *metallib, const std::set<std::string_view> &kernels) {
  id<MTLLibrary> library = [MTLCreateSystemDefaultDevice() newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]] error:nil];
  require(library != nil, "could not load the Linear library");
  for (const std::string_view kernel : kernels)
    require([library newFunctionWithName:[NSString stringWithUTF8String:
                kernelInstance(kernel, FloatOutput::Float32).c_str()]] != nil,
            "a plain decode kernel has no fp32 instance");
}

// Every Linear pipeline a candidate plan of any family and core count
// launches, with its threads per threadgroup: one pipeline never takes two
// execution scopes. Device-free.
std::map<std::string, uint32_t> pipelineScopes() {
  std::map<std::string, uint32_t> names{{"prefill_linear_q4_sums32", 256}};
  const auto collect = [&](const Linear &linear, LinearWorkload workload) {
    const bool plain = workload.phase == LinearPhase::Decode && workload.epilogue == LinearEpilogue::None;
    for (const auto &candidate : linear.candidates(workload)) {
      std::vector<LinearPlan> plans{candidate};
      if (plain) plans.push_back(Linear::plan(workload, candidate.configuration(), FloatOutput::Float32));
      for (const auto &plan : plans)
        for (auto name : {plan.pipeline(), plan.secondPipeline()}) {
          if (name.empty()) continue;
          const auto [found, inserted] = names.emplace(kernelInstance(name, plan.destination()),
                                                       plan.threadsPerThreadgroup());
          require(inserted || found->second == plan.threadsPerThreadgroup(),
                  "one Linear pipeline was assigned incompatible execution scopes");
        }
    }
  };
  for (const uint32_t family : {9U, 10U, 11U})
    for (const uint32_t cores : {0U, 10U, 16U, 20U, 40U, 80U}) {
      const Linear linear = gpu(family, cores);
      for (uint32_t lanes = 1; lanes <= 4; ++lanes)
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::GateUp})
          collect(linear, {{16640, 5120}, lanes * 8, LinearPhase::Decode, epilogue});
      for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                 LinearEpilogue::UpWithGate})
        for (const uint32_t rows : {17U, 33U})
          collect(linear, {{16640, 5120}, rows, LinearPhase::Prefill, epilogue});
    }
  return names;
}

// This device's resources for every pipeline of pipelineScopes: its static
// threadgroup memory fits the device, its thread limit covers the plans'
// threads, and SIMD groups are 32 wide. Shader validation instruments the
// pipelines and inflates these numbers, so this runs without it.
void pipelineCapabilities(const char *metallib, const DeviceCapabilities &capabilities,
                          const std::map<std::string, uint32_t> &names) {
  require(!std::getenv("MTL_SHADER_VALIDATION"),
          "the pipeline resource check inspects production pipelines: run it without MTL_SHADER_VALIDATION");
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib]] error:&error];
  require(library != nil, "could not load Linear library for resource inspection");
  uint64_t largestStaticMemory = 0;
  uint64_t smallestThreadLimit = std::numeric_limits<uint64_t>::max();
  for (const auto &[name, threads] : names) {
    id<MTLFunction> function = [library newFunctionWithName:
        [NSString stringWithUTF8String:name.c_str()]];
    require(function != nil, "missing precompiled Linear candidate");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    require(pipeline != nil, "Linear candidate pipeline could not be created");
    largestStaticMemory = std::max(largestStaticMemory, uint64_t(pipeline.staticThreadgroupMemoryLength));
    smallestThreadLimit = std::min(smallestThreadLimit, uint64_t(pipeline.maxTotalThreadsPerThreadgroup));
    require(pipeline.staticThreadgroupMemoryLength <= capabilities.maxThreadgroupMemoryBytes &&
                pipeline.maxTotalThreadsPerThreadgroup >= threads && pipeline.threadExecutionWidth == 32,
            "Linear candidate exceeds current device resources");
  }
  std::cout << "Linear pipelines=" << names.size() << " maximum_static_tg_bytes="
            << largestStaticMemory << " minimum_thread_limit=" << smallestThreadLimit
            << " apple_family=" << capabilities.appleGpuFamily << " PASS\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool capabilities = argc == 3 && std::string_view(argv[1]) == "--capabilities";
    require(argc == 2 || capabilities,
            "usage: linear-plan <production.metallib|--cpu|--capabilities production.metallib>");
    if (capabilities) {
      metal::MetalBackend backend(argv[2]);
      pipelineCapabilities(argv[2], backend.capabilities(), pipelineScopes());
      return 0;
    }
    anchorPlans();
    newerFamilyPlans();
    policyLaws();
    ggufPlans();
    const std::set<std::string_view> plainKernels = floatOutputPlans();
    scalingContracts();
    // Apple9 at the assumed core count reaches the expanded split set;
    // Apple10 exercises the MPP-only candidate bound.
    size_t widestCandidates = 0;
    planContracts(9, 0, widestCandidates);
    planContracts(10, 16, widestCandidates);
    require(widestCandidates == Linear::kMaximumCandidates,
            "no covered workload reaches the Linear candidate bound");
    // The resources behind these scopes are --capabilities' check.
    static_cast<void>(pipelineScopes());
    if (std::string_view(argv[1]) == "--cpu") {
      std::cout << "Linear CPU plans: PASS\n";
      return 0;
    }
    metal::MetalBackend backend(argv[1]);
    floatInstances(argv[1], plainKernels);
    ggufProjectionMatrix(backend);
    Linear linear(backend.capabilities());
    // Apple9's short prefill chunks run the simdgroup kernels, which every
    // family's GPU runs too.
    Linear apple9 = gpu(9, 40);
    for (const LinearMatrix matrix : {LinearMatrix{512, 256}, LinearMatrix{768, 768},
                                      LinearMatrix{16640, 5120}, LinearMatrix{12544, 2048},
                                      LinearMatrix{5120, 17408},
                                      LinearMatrix{256, 64}, LinearMatrix{512, 320}}) {
      const auto p = test::deterministicQ4Projection(backend, matrix, 31);
      const auto gate = test::deterministicQ4Projection(backend, matrix, 157);
      const auto immutableProjection = projectionFingerprint(p);
      const auto immutableGateProjection = projectionFingerprint(gate);
      if (matrix.inputSize % 256 == 0)
        for (uint32_t lanes = 1; lanes <= 4; ++lanes)
          for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                     LinearEpilogue::GateUp}) {
            numericalCase(backend, linear, p, gate,
                          {matrix, lanes * 8, LinearPhase::Decode, epilogue});
            if (epilogue == LinearEpilogue::Residual)
              numericalCase(backend, linear, p, gate,
                            {matrix, lanes * 8, LinearPhase::Decode, epilogue}, true);
          }
      for (const uint32_t rows : {1U, 33U, 2048U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::UpWithGate})
          numericalCase(backend, linear, p, gate,
                        {matrix, rows, LinearPhase::Prefill, epilogue});
      for (const uint32_t rows : {7U, 17U, 32U})
        for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual,
                                   LinearEpilogue::UpWithGate})
          numericalCase(backend, apple9, p, gate, {matrix, rows, LinearPhase::Prefill, epilogue});
      require(projectionFingerprint(p) == immutableProjection &&
                  projectionFingerprint(gate) == immutableGateProjection,
              "Linear changed immutable Q4 weights or quantization metadata");
      std::cout << "Linear N=" << matrix.outputSize << " K=" << matrix.inputSize
                << " all epilogues/candidates/row cases PASS\n";
    }
    std::cout << "Linear plans and candidates: PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "Linear plans: FAIL: " << error.what() << '\n';
    return 1;
  }
}
