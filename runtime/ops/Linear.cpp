#include "Linear.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/Linear.h"
#include "ops/ChoiceTable.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::ops {
namespace {

constexpr uint32_t kQuantGroup = 64;
static_assert(SPLASH_TARGET_VERIFY_ROWS == 8,
              "simdgroup Q4 tiles require eight verify rows per lane");
// The kernels sum their input per block of four quant groups (256 inputs).
// Split128 partitions K in whole blocks; the one-lane split tiles hold four
// partitions of whole blocks.
constexpr uint32_t kInputSumBlock = 4 * kQuantGroup;
constexpr uint32_t kSplitPartitions = 4;
constexpr uint32_t kSplitInputBlock = kSplitPartitions * kInputSumBlock;
static_assert(kAffineRegisterTile.split.partitionInputs == kQuantGroup &&
                  kAffineTensorSplit.split.partitionInputs == kInputSumBlock,
              "the affine split laws partition K in the kernels' steps");
static_assert(sizeof(LinearMatrix) == 8);

bool oneLaneSplit(LinearTile tile) noexcept {
  return tile == LinearTile::Split32 || tile == LinearTile::Split64;
}
bool oneLaneTile(LinearTile tile) noexcept {
  return tile == LinearTile::Paired128 || tile == LinearTile::Paired256 || oneLaneSplit(tile);
}
// Simdgroups fixed by the kernel instance: the one-lane split tiles run four
// partitions of one (N32) or two (N64) simdgroups, the paired N256 tile runs
// four and Split128 the N128 tile's eight.
std::optional<LinearSimdgroups> fixedSimdgroups(LinearTile tile) noexcept {
  switch (tile) {
  case LinearTile::Split32:
  case LinearTile::Simdgroup:
  case LinearTile::Paired256: return LinearSimdgroups::Four;
  case LinearTile::Split64:
  case LinearTile::Split128: return LinearSimdgroups::Eight;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::GgufStaged:
  case LinearTile::GgufRegister: return std::nullopt;
  }
  return std::nullopt;
}

void validate(LinearWorkload w) {
  if (w.weightLayout != WeightLayout::Affine64 && w.weightLayout != WeightLayout::Block32)
    throw std::invalid_argument("invalid linear weight layout");
  if (!w.matrix.outputSize || w.matrix.outputSize % 256 ||
      !w.matrix.inputSize || w.matrix.inputSize % kQuantGroup)
    throw std::invalid_argument("invalid linear matrix");
  if (w.phase == LinearPhase::Prefill) {
    if (!w.rows || w.rows > SPLASH_PREFILL_TOKEN_BUDGET ||
        w.epilogue == LinearEpilogue::GateUp)
      throw std::invalid_argument("invalid linear prefill workload");
  } else if (w.phase == LinearPhase::Decode) {
    if (w.matrix.inputSize % 256 || !w.rows || w.rows % SPLASH_TARGET_VERIFY_ROWS ||
        w.rows > SPLASH_TARGET_VERIFY_ROWS * SPLASH_MAXIMUM_BATCH_WIDTH ||
        w.epilogue == LinearEpilogue::UpWithGate)
      throw std::invalid_argument("invalid linear decode workload");
  } else {
    throw std::invalid_argument("invalid linear phase");
  }
  if (w.epilogue != LinearEpilogue::None && w.epilogue != LinearEpilogue::Residual &&
      w.epilogue != LinearEpilogue::GateUp && w.epilogue != LinearEpilogue::UpWithGate)
    throw std::invalid_argument("invalid linear epilogue");
}

// A buffer a plan does not use needs no bytes and may be absent.
void requireBytes(const metal::MetalBuffer &buffer, uint64_t bytes, const char *what) {
  if (bytes && (!buffer || buffer.sizeBytes() < bytes))
    throw std::invalid_argument(std::string("projection ") + what + " buffer holds " +
                                std::to_string(buffer.sizeBytes()) + " bytes, needs " + std::to_string(bytes));
}

LinearWorkload decode(LinearMatrix matrix, uint32_t lanes, LinearEpilogue epilogue) {
  if (!lanes || lanes > SPLASH_MAXIMUM_BATCH_WIDTH)
    throw std::invalid_argument("invalid linear decode batch width");
  return {matrix, lanes * SPLASH_TARGET_VERIFY_ROWS, LinearPhase::Decode, epilogue};
}

// The four-simdgroup kernels: every prefill N128 tile, the decode M24 and M32
// N128 plain and residual projections, all matrix row tiles of up to a decode
// batch, and the one-lane Split32 (plain, residual and gate/up) and Paired256
// (plain) tiles.
bool supportsFourSimdgroups(LinearWorkload w, LinearTile tile) noexcept {
  if (tile == LinearTile::Simdgroup) return w.rows <= kMaximumDecodeTileRows;
  if (tile == LinearTile::Split32)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS;
  // Only the affine paired N256 kernel is instantiated: this tile is used
  // for wide plain projections; residual and gate/up retain their own tiles.
  if (tile == LinearTile::Paired256)
    return w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS &&
        w.epilogue == LinearEpilogue::None;
  if (tile != LinearTile::N128) return false;
  return w.phase == LinearPhase::Prefill ||
      ((w.rows == 24 || w.rows == 32) &&
       (w.epilogue == LinearEpilogue::None || w.epilogue == LinearEpilogue::Residual));
}

} // namespace

// Table16 holds its sums per eight-row tile (metal/abi/Gguf.h), a lane's rows.
uint64_t tableSumsBytes(LinearInput layout, uint32_t width, uint64_t rows) noexcept {
  return layout == LinearInput::Table16
             ? rows / SPLASH_TARGET_VERIFY_ROWS * table16_sums_per_tile(width) * sizeof(float)
       : layout == LinearInput::Table64 ? uint64_t{width} * rows / 16 : 0;
}

void requireAffineProjection(const Projection &p, LinearMatrix matrix) {
  if (p.layout() != WeightLayout::Affine64 || p.outputSize != matrix.outputSize ||
      p.inputSize != matrix.inputSize)
    throw std::invalid_argument("affine projection does not match plan");
  requireBytes(p.affine().weights, uint64_t{matrix.outputSize} * matrix.inputSize / 2, "weight");
  const uint64_t bytes = uint64_t{matrix.outputSize} * (matrix.inputSize / kQuantGroup) * 2;
  requireBytes(p.affine().scales, bytes, "scale");
  requireBytes(p.affine().biases, bytes, "bias");
}

const KernelFamily &LinearPlan::kernelFamily() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup: return kAffineRegisterTile;
  case LinearTile::Split128: return kAffineTensorSplit;
  case LinearTile::GgufRegister: return kGgufRegisterTile;
  case LinearTile::GgufStaged:
    return config_.simdgroups == LinearSimdgroups::Two ? kGgufStagedTile : kGgufStagedPrefill;
  case LinearTile::N128:
  case LinearTile::N256:
  case LinearTile::Paired128:
  case LinearTile::Split32:
  case LinearTile::Split64:
  case LinearTile::Paired256: break;
  }
  return workload_.phase == LinearPhase::Prefill ? kAffineTensorPrefill : kAffineTensorDecode;
}
uint32_t LinearPlan::storageRows() const noexcept { return kernelFamily().rows.round(workload_.rows); }
uint32_t LinearPlan::tileColumns() const noexcept {
  switch (config_.tile) {
  case LinearTile::Simdgroup: return workload_.epilogue == LinearEpilogue::GateUp ? 32 : 64;
  case LinearTile::Split32: return 32;
  case LinearTile::Split64:
  case LinearTile::GgufStaged:
  case LinearTile::GgufRegister: return 64;
  case LinearTile::N256:
  case LinearTile::Paired256: return 256;
  case LinearTile::N128:
  case LinearTile::Paired128:
  case LinearTile::Split128: return 128;
  }
  return 0;
}
uint32_t LinearPlan::threadsPerThreadgroup() const noexcept {
  return static_cast<uint32_t>(config_.simdgroups) * 32;
}
uint32_t LinearPlan::partialSums() const noexcept {
  return oneLaneSplit(config_.tile) ? kSplitPartitions : config_.splits;
}
bool LinearPlan::usesSimdgroup() const noexcept { return config_.tile == LinearTile::Simdgroup; }
LinearInput LinearPlan::input() const noexcept {
  if (config_.tile == LinearTile::GgufRegister) return LinearInput::Table16;
  return usesSimdgroup() ? LinearInput::Table64 : LinearInput::Plain;
}
LinearScratchSize LinearPlan::scratchSize() const noexcept {
  if (workload_.weightLayout == WeightLayout::Block32) return blockScratchSize();
  const auto [n, k] = workload_.matrix;
  // Split128: [split][row][column] fp32 partials over every row of the step
  // and one counter per column tile.
  if (config_.tile == LinearTile::Split128)
    return {0, 0, uint64_t{config_.splits} * workload_.rows * n * sizeof(float),
            uint64_t{n / tileColumns()} * sizeof(uint32_t)};
  if (!usesSimdgroup()) return {};
  const uint64_t rows = storageRows();
  const uint64_t lanes = rows / SPLASH_TARGET_VERIFY_ROWS;
  // Each row tile owns two fp32 fragment streams per K partition and one
  // completion counter per column tile. Single-partition kernels use neither.
  return {tableBytes(k, rows), tableSumsBytes(LinearInput::Table64, k, rows),
          config_.splits > 1 ? config_.splits * 2 * rows * n * sizeof(float) : sizeof(float),
          config_.splits > 1 ? lanes * (n / tileColumns()) * sizeof(uint32_t) : sizeof(uint32_t)};
}

uint64_t LinearPlan::sumsBytes() const noexcept {
  return workload_.phase == LinearPhase::Prefill && workload_.weightLayout == WeightLayout::Affine64 &&
      !usesSimdgroup() ? uint64_t{storageRows()} * (workload_.matrix.inputSize / kQuantGroup) * 4 : 0;
}
uint64_t LinearPlan::gateScratchBytes() const noexcept {
  // GGUF tiles run gate/up as a gate pass and an up-with-gate pass.
  const bool needed = workload_.epilogue == LinearEpilogue::UpWithGate ||
      (workload_.epilogue == LinearEpilogue::GateUp &&
       (!secondPipeline_.empty() || workload_.weightLayout == WeightLayout::Block32));
  return needed ? uint64_t{storageRows()} * workload_.matrix.outputSize * 2 : 0;
}
uint64_t LinearPlan::downSumsBytes() const noexcept {
  return workload_.epilogue == LinearEpilogue::UpWithGate && workload_.weightLayout == WeightLayout::Affine64 &&
      !usesSimdgroup() ? uint64_t{storageRows()} * (workload_.matrix.outputSize / kQuantGroup) * 4 : 0;
}

LinearPlan::LinearPlan(LinearWorkload w, LinearConfig config, FloatOutput destination)
    : workload_(w), config_(config), destination_(destination) {
  validate(w);
  if (destination == FloatOutput::Float32 &&
      (w.phase != LinearPhase::Decode || w.epilogue != LinearEpilogue::None))
    throw std::invalid_argument("an fp32 destination takes a plain decode projection");
  const bool ggufTile = config.tile == LinearTile::GgufStaged || config.tile == LinearTile::GgufRegister;
  if (ggufTile != (w.weightLayout == WeightLayout::Block32))
    throw std::invalid_argument("block projections run the GGUF tiles, affine ones the Q4 tiles");
  if (ggufTile) {
    // Kernel names follow the segment formats (LinearGguf.cpp).
    requireBlockConfiguration();
    return;
  }
  const bool splitsK = config.tile == LinearTile::Simdgroup || config.tile == LinearTile::Split128;
  if (!splitsK && config.splits != 1)
    throw std::invalid_argument("K splits require the simdgroup or Split128 Q4 tile");
  if (config.tile != LinearTile::N128 && config.tile != LinearTile::N256 && !splitsK &&
      !oneLaneTile(config.tile))
    throw std::invalid_argument("invalid Q4 linear tile");
  if ((config.simdgroups != LinearSimdgroups::Four &&
       config.simdgroups != LinearSimdgroups::Eight) ||
      (config.simdgroups == LinearSimdgroups::Four &&
       !supportsFourSimdgroups(w, config.tile)))
    throw std::invalid_argument("invalid Q4 cooperative execution scope");
  if (const auto fixed = fixedSimdgroups(config.tile); fixed && config.simdgroups != *fixed)
    throw std::invalid_argument("Q4 tile requires its kernel's simdgroup count");
  if (w.matrix.outputSize % tileColumns())
    throw std::invalid_argument("Q4 matrix is not divisible by tile columns");
  const bool residual = w.epilogue == LinearEpilogue::Residual;
  const bool four = config.simdgroups == LinearSimdgroups::Four;
  // One simdgroup kernel per epilogue serves decode steps and prefill chunks
  // of up to a decode batch (supportsFourSimdgroups) alike.
  if (usesSimdgroup()) {
    const uint32_t groups = w.matrix.inputSize / kQuantGroup;
    if (config.groups != w.matrix.outputSize / tileColumns() || !config.validSplits() || groups % config.splits)
      throw std::invalid_argument("simdgroup Q4 requires full column grid and whole power-of-two K partitions");
    pipeline_ = w.epilogue == LinearEpilogue::GateUp ? "decode_linear_q4_sg_gate_up" :
        w.epilogue == LinearEpilogue::UpWithGate ? "decode_linear_q4_sg_up_silu" :
        residual ? "decode_linear_q4_sg_residual" : "decode_linear_q4_sg";
    return;
  }
  if (w.phase == LinearPhase::Prefill) {
    if (config.groups || oneLaneTile(config.tile) || config.tile == LinearTile::Split128)
      throw std::invalid_argument("invalid Q4 prefill configuration");
    if (four) {
      pipeline_ = w.epilogue == LinearEpilogue::UpWithGate
          ? "prefill_linear_q4_n128_up_silu_sums_sg4"
          : residual ? "prefill_linear_q4_n128_residual_sg4" : "prefill_linear_q4_n128_sg4";
    } else if (w.epilogue == LinearEpilogue::UpWithGate) {
      if (config.tile != LinearTile::N256)
        throw std::invalid_argument(
            "Q4 fused prefill up requires N256 or four simdgroups");
      pipeline_ = "prefill_linear_q4_n256_up_silu_sums";
    } else if (residual) {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128_residual" : "prefill_linear_q4_n256_residual";
    } else {
      pipeline_ = config.tile == LinearTile::N128
          ? "prefill_linear_q4_n128" : "prefill_linear_q4_n256";
    }
    return;
  }
  if (!config.groups || config.groups > w.matrix.outputSize / tileColumns())
    throw std::invalid_argument("invalid Q4 decode group count");
  const uint32_t lane = w.rows / SPLASH_TARGET_VERIFY_ROWS - 1;
  if (oneLaneTile(config.tile) && (lane != 0 || w.matrix.outputSize % 256))
    throw std::invalid_argument("paired or split Q4 tile requires one lane and paired columns");
  if (oneLaneSplit(config.tile)) {
    // Each partition takes a quarter of K in whole 256-input blocks, and the
    // split kernels are dispatched one threadgroup per tile.
    if (w.matrix.inputSize % kSplitInputBlock ||
        config.groups != w.matrix.outputSize / tileColumns())
      throw std::invalid_argument("split Q4 tile requires K % 1024 == 0 and the full grid");
    const bool n32 = config.tile == LinearTile::Split32;
    if (w.epilogue == LinearEpilogue::GateUp) {
      // Only the N32 two-stream split kernel is instantiated.
      if (!n32) throw std::invalid_argument("Q4 split gate/up requires Split32");
      pipeline_ = "decode_linear_q4_n32_split4_gate_up";
    } else if (residual) {
      pipeline_ = n32 ? "decode_linear_q4_n32_split4_residual"
                      : "decode_linear_q4_n64_split4_residual";
    } else {
      pipeline_ = n32 ? "decode_linear_q4_n32_split4" : "decode_linear_q4_n64_split4";
    }
    return;
  }
  if (config.tile == LinearTile::Split128) {
    // Every partition holds at least one of the kernel's 256-input blocks.
    if (config.groups != w.matrix.outputSize / tileColumns() || !config.validSplits() ||
        config.splits < 2 || w.matrix.inputSize / kInputSumBlock < config.splits)
      throw std::invalid_argument("Split128 requires the full column grid and 2-8 K partitions of 256-input blocks");
    constexpr std::array plainNames{"decode_linear_q4_n128_split", "decode_linear_q4_n128_split_m16",
        "decode_linear_q4_n128_split_m24", "decode_linear_q4_n128_split_m32"};
    constexpr std::array residualNames{"decode_linear_q4_n128_split_residual",
        "decode_linear_q4_n128_split_residual_m16", "decode_linear_q4_n128_split_residual_m24",
        "decode_linear_q4_n128_split_residual_m32"};
    constexpr std::array upSiluNames{"decode_linear_q4_n128_split_up_silu",
        "decode_linear_q4_n128_split_up_silu_m16", "decode_linear_q4_n128_split_up_silu_m24",
        "decode_linear_q4_n128_split_up_silu_m32"};
    // Gate/up at every lane count: a plain gate pass into the gate scratch,
    // then the up pass whose epilogue applies the SiLU gate.
    pipeline_ = residual ? residualNames[lane] : plainNames[lane];
    if (w.epilogue == LinearEpilogue::GateUp) secondPipeline_ = upSiluNames[lane];
    return;
  }
  if (config.tile == LinearTile::Paired256) {
    pipeline_ = "decode_linear_q4_n256_paired_sg4";
    return;
  }
  if (four) {
    if (lane == 2)
      pipeline_ = residual ? "decode_linear_q4_n128_residual_m24_sg4" : "decode_linear_q4_n128_m24_sg4";
    else
      pipeline_ = residual ? "decode_linear_q4_n128_residual_m32_sg4" : "decode_linear_q4_n128_m32_sg4";
    return;
  }
  if (w.epilogue == LinearEpilogue::GateUp) {
    if (config.tile != LinearTile::N256)
      throw std::invalid_argument("Q4 gate/up requires N256");
    constexpr std::array names{"decode_linear_q4_n256_gate_up", "decode_linear_q4_n256_gate_up_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
    if (lane >= 2)
      secondPipeline_ = lane == 2 ? "decode_linear_q4_n256_up_silu_m24"
                                  : "decode_linear_q4_n256_up_silu_m32";
  } else if (residual) {
    if (config.tile == LinearTile::N256)
      throw std::invalid_argument("Q4 decode residual requires N128");
    constexpr std::array names{"decode_linear_q4_n128_residual", "decode_linear_q4_n128_residual_m16",
        "decode_linear_q4_n128_residual_m24", "decode_linear_q4_n128_residual_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
        ? "decode_linear_q4_n128_residual_paired" : names[lane];
  } else if (config.tile == LinearTile::N256) {
    constexpr std::array names{"decode_linear_q4_n256", "decode_linear_q4_n256_m16",
        "decode_linear_q4_n256_m24", "decode_linear_q4_n256_m32"};
    pipeline_ = names[lane];
  } else {
    constexpr std::array names{"decode_linear_q4_n128", "decode_linear_q4_n128_m16",
        "decode_linear_q4_n128_m24", "decode_linear_q4_n128_m32"};
    pipeline_ = config.tile == LinearTile::Paired128
                    ? "decode_linear_q4_n128_paired" : names[lane];
  }
}

namespace {

// Decode groups stream output tiles. Under round-robin group placement, the
// most loaded core sets dispatch latency. Use the full grid for small workloads,
// balanced two-tile groups at intermediate sizes, and one wave for longer
// chains; sufficiently large grids balance themselves.
struct DecodeGroupPolicy final {
  // The one-tile grid wins up to this many groups per core.
  uint32_t fullGridGroupsPerCore;
  // Groups per core of one wave.
  uint32_t waveGroupsPerCore;
  // From this many tiles per core the many-wave grid wins again.
  uint32_t manyWaveTilesPerCore;
};
// N128 runs one wave of its effective concurrency (kTensorConcurrency), and
// its full grid up to one such wave per core (up to five at 16 rows). The
// others were measured on 16/20-core Apple10 GPUs: N256 and gate/up keep
// three per core although their k is 2 (two per core, the 2026-09-27
// hardware characterization's k, loses up to 16% of a three- or four-lane
// step), four simdgroups keep eight (k = 5 loses up to 1.4%). Gate/up's
// many-wave threshold follows N256; the four-simdgroup threshold scales from
// N128. Those two extrapolations remain unmeasured.
constexpr DecodeGroupPolicy kN128Groups{kTensorConcurrency.n128, kTensorConcurrency.n128, 12},
    kN128M16Groups{5, kTensorConcurrency.n128, 12}, kN256Groups{3, 3, 8}, kGateUpGroups{3, 3, 8},
    kFourSimdgroupGroups{8, 8, 24};

// Tiles on the most loaded core when `groups` threadgroups are placed
// round-robin on `cores` and group g streams tiles g, g + groups, ...
uint32_t maxCoreTiles(uint32_t tiles, uint32_t groups, uint32_t cores) noexcept {
  uint32_t worst = 0;
  for (uint32_t core = 0; core < cores; ++core) {
    uint32_t load = 0;
    for (uint32_t group = core; group < groups; group += cores)
      load += (tiles - group + groups - 1) / groups;
    worst = std::max(worst, load);
  }
  return worst;
}

uint32_t decodeGroups(uint32_t tiles, uint32_t cores,
                      DecodeGroupPolicy policy) noexcept {
  const uint32_t wave = policy.waveGroupsPerCore * cores;
  if (tiles <= policy.fullGridGroupsPerCore * cores ||
      tiles >= policy.manyWaveTilesPerCore * cores)
    return tiles;
  const uint32_t twoTile = (tiles + 1) / 2;
  // Here wave < twoTile <= tiles, so the wave is a valid count (LinearPlan
  // rejects more groups than tiles) whatever the per-core constants are.
  if (twoTile > wave) return wave;
  // The smallest balanced two-tile count keeping three quarters of the
  // full-grid limit resident. A multiple of the core count is always
  // balanced, so the search ends within `cores` steps and below `tiles`.
  const uint32_t balanced = (tiles + cores - 1) / cores;
  uint32_t groups =
      std::max(twoTile, policy.fullGridGroupsPerCore * cores * 3 / 4);
  while (maxCoreTiles(tiles, groups, cores) != balanced) ++groups;
  return groups;
}
// Apple9 leaves its register tile for plain projections of three and four
// lanes whose N256 grid keeps two tiles per core: its register tile runs one
// lane per threadgroup and so streams the weights once per lane.
constexpr uint32_t kApple9WidePlainTilesPerCore = 2;
// Apple9 N256 prefill needs eight threadgroups per core to amortize its larger
// tile. Paired-A/B tuning (tune-kernels) and the per-shape microprofile
// (benchmark-prefill) on a 32-core Apple9 GPU (M4 Max) measured the
// four-simdgroup N128 tile ahead of N256 on every prefill shape and probed
// row count: +6..10% GPU wherever the margin cleared the tuning threshold,
// never behind. Apple9 GPUs at or below that measured core count therefore
// share the Apple10 prefill rule. Larger Apple9 GPUs (40-core class) keep the
// wide-tile rule below; it was sized for them and remains unremeasured there.
constexpr uint32_t kApple9MeasuredPrefillCores = 32;
constexpr double kApple9WidePrefillGroupsPerCore = 8.0;

// The paired N256 tile runs one wave of its effective concurrency, four
// threadgroups per core (kTensorConcurrency). The one-lane split tiles remain offline
// candidates: they beat Split128 or the sequential tile on a few 35B one-lane
// shapes (by up to 7% at 20 cores), 0.55% of a one-lane step, too little for a
// second split rule.
constexpr uint32_t kPaired256WaveGroupsPerCore = kTensorConcurrency.paired256;

// Apple10's sequential tiles, for steps Split128 does not split: the tile law
// (kAffineTensorTiles) over decodeGroups' persistent grids. Gate/up runs N256,
// its fused kernel at one and two lanes; the N128 steps of three lanes run
// four simdgroups.
LinearConfig tensorSequentialConfig(LinearWorkload w, uint32_t cores) {
  // validate() requires outputSize % 256 == 0, so every tile width divides it.
  const uint32_t tiles128 = w.matrix.outputSize / 128, tiles256 = w.matrix.outputSize / 256;
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  if (w.epilogue == LinearEpilogue::GateUp) return {LinearTile::N256, decodeGroups(tiles256, cores, kGateUpGroups)};
  const std::optional<TilesPerCore> &wide = kAffineTensorTiles.wide[lanes - 1];
  const bool widePlain = w.epilogue == LinearEpilogue::None && wide && wide->reachedBy(tiles256, cores);
  if (lanes == 1) {
    if (widePlain)
      return {LinearTile::Paired256, std::min(tiles256, kPaired256WaveGroupsPerCore * cores), LinearSimdgroups::Four};
    const bool unpaired = kAffineTensorTiles.unpaired.reachedBy(tiles128, cores) &&
                          tiles128 <= kTensorConcurrency.n128 * cores;
    return {unpaired ? LinearTile::N128 : LinearTile::Paired128, decodeGroups(tiles128, cores, kN128Groups)};
  }
  if (widePlain) return {LinearTile::N256, decodeGroups(tiles256, cores, kN256Groups)};
  if (lanes == 3 || (lanes == 4 && kAffineTensorTiles.fourLaneFourSimdgroups.reachedBy(tiles128, cores)))
    return {LinearTile::N128, decodeGroups(tiles128, cores, kFourSimdgroupGroups), LinearSimdgroups::Four};
  return {LinearTile::N128, decodeGroups(tiles128, cores, lanes == 2 ? kN128M16Groups : kN128Groups)};
}

// Apple9's simdgroup tile over the full column grid, 32 columns per gate/up
// threadgroup and 64 otherwise, split by its family's law. It ignores the
// rows, so a lane's outputs are the same at every batch width and in prefill
// chunks.
LinearConfig simdgroupConfig(LinearWorkload w, const DevicePolicy &device) {
  const uint32_t grid = w.matrix.outputSize / (w.epilogue == LinearEpilogue::GateUp ? 32 : 64);
  return {LinearTile::Simdgroup, grid, LinearSimdgroups::Four,
          splitK(kAffineRegisterTile, device, grid, w.matrix.inputSize, kAffineRegisterTile.rows.round(w.rows))};
}

} // namespace

uint32_t Linear::decodeStorageRows(uint32_t rows, ProjectionShape shape) const {
  return plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Decode, LinearEpilogue::None, shape.layout})
      .storageRows();
}

void Linear::account(LinearDispatchStats &stats, uint32_t lanes, uint32_t dispatches) noexcept {
  if (lanes == 1) return;
  stats.fusedSourceOperations += uint64_t{lanes} * dispatches;
  if (lanes == 2) stats.m16Dispatches += dispatches;
  else if (lanes == 3) stats.m24Dispatches += dispatches;
  else stats.m32Dispatches += dispatches;
}

// The primitive selects variants; core count and workload tile counts
// determine parallelism.
LinearConfig Linear::baseline(LinearWorkload w, std::span<const Projection *const> projections) const {
  validate(w);
  if (w.weightLayout == WeightLayout::Block32) return ggufBaseline(w, projections);
  const uint32_t tiles128 = w.matrix.outputSize / 128;
  const uint32_t tiles256 = w.matrix.outputSize / 256;
  const bool registerTiles = device_.primitive == Primitive::Register;
  if (w.phase == LinearPhase::Prefill) {
    // Apple9 runs chunks of up to a decode batch on its decode tile with the
    // decode split rule, as ggufBaseline does. The MPP tile's grid of such a
    // chunk is one row of 128-column tiles, each streaming all of K: on a
    // 40-core M3 Max it took 1.25 ms for the 27B FFN down projection (17408
    // inputs, 5120 outputs) at 17-32 rows, about ten times its bandwidth floor.
    if (registerTiles && w.rows <= kMaximumDecodeTileRows) return simdgroupConfig(w, device_);
    if (!registerTiles || device_.cores <= kApple9MeasuredPrefillCores)
      return {LinearTile::N128, 0, LinearSimdgroups::Four};
    const uint32_t rowTiles = (w.rows + kAffinePrefillTileRows - 1) / kAffinePrefillTileRows;
    const bool wide = double(rowTiles) * tiles256 >=
        kApple9WidePrefillGroupsPerCore * device_.cores;
    return {w.epilogue == LinearEpilogue::UpWithGate || wide ? LinearTile::N256
                                                              : LinearTile::N128, 0};
  }
  if (registerTiles) {
    // Apple9's wide plain projections run MPP tiles over one-tile grids (the
    // round-robin groups were measured on Apple10): four-simdgroup N128 at
    // three lanes, N256 at four.
    const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
    if (lanes < 3 || w.epilogue != LinearEpilogue::None || tiles256 < kApple9WidePlainTilesPerCore * device_.cores)
      return simdgroupConfig(w, device_);
    return lanes == 3 ? LinearConfig{LinearTile::N128, tiles128, LinearSimdgroups::Four}
                      : LinearConfig{LinearTile::N256, tiles256};
  }
  if (const uint32_t splits = splitK(kAffineTensorSplit, device_, tiles128, w.matrix.inputSize, w.rows); splits > 1)
    return {LinearTile::Split128, tiles128, LinearSimdgroups::Eight, splits};
  return tensorSequentialConfig(w, device_.cores);
}

LinearPlan Linear::plan(LinearWorkload workload) const {
  return LinearPlan(workload, chosenConfiguration(choices_, workload, baseline(workload)));
}
LinearPlan Linear::plan(LinearWorkload workload, LinearConfig config, FloatOutput destination) {
  return LinearPlan(workload, config, destination);
}
LinearPlan Linear::plan(LinearWorkload w, const Projection &p, const Projection *gate) const {
  w.weightLayout = p.layout();
  const std::array<const Projection *, 2> projections{&p, gate};
  return LinearPlan(w, chosenConfiguration(choices_, w, baseline(w, projections)), p.destination);
}
void Linear::setChoices(std::span<const LinearChoice> choices) {
  std::vector<LinearChoice> pending(choices.begin(), choices.end());
  for (const auto &choice : pending) {
    if (choice.workload.weightLayout == WeightLayout::Block32)
      throw std::invalid_argument("block projection plans are not tuned");
    (void)plan(choice.workload, choice.configuration);
  }
  sortUniqueChoices(pending);
  choices_ = std::move(pending);
}

std::vector<LinearPlan> Linear::candidates(LinearWorkload w) const {
  std::vector<LinearPlan> result;
  result.reserve(kMaximumCandidates);
  result.push_back(LinearPlan(w, baseline(w)));
  if (w.weightLayout == WeightLayout::Block32) return result;
  const auto append = [&](LinearConfig config) {
    for (const auto &existing : result)
      if (existing.configuration() == config) return;
    result.push_back(LinearPlan(w, config));
  };
  // Prefill chunks the simdgroup tile runs list only its K splits, so a tuning
  // fixture never mixes its lanes and table with the MPP tiles' padded rows
  // and Q4 sums.
  const bool simdgroupPrefill = w.phase == LinearPhase::Prefill && result.front().usesSimdgroup();
  for (const auto tile : {LinearTile::N128, LinearTile::N256, LinearTile::Paired128}) {
    const uint32_t columns = tile == LinearTile::N256 ? 256 : 128;
    if (simdgroupPrefill || w.matrix.outputSize % columns ||
        (tile == LinearTile::Paired128 && (w.phase != LinearPhase::Decode ||
         w.rows != SPLASH_TARGET_VERIFY_ROWS || w.matrix.outputSize % 256)) ||
        (w.epilogue == LinearEpilogue::GateUp && tile != LinearTile::N256) ||
        (w.phase == LinearPhase::Decode && w.epilogue == LinearEpilogue::Residual && tile == LinearTile::N256))
      continue;
    if (w.phase == LinearPhase::Prefill) {
      // The fused up projection has no eight-simdgroup N128 kernel.
      if (tile == LinearTile::N256 || w.epilogue != LinearEpilogue::UpWithGate)
        append({tile, 0});
      if (supportsFourSimdgroups(w, tile))
        append({tile, 0, LinearSimdgroups::Four});
    } else {
      const uint32_t tiles = w.matrix.outputSize / columns;
      // Sample two, three and four groups per core plus the full grid.
      // Always retain the measured baseline above, including its balanced
      // group count. Fixed counts tied to one GPU miss these waves elsewhere.
      for (const uint32_t groups : {2 * device_.cores, 3 * device_.cores, 4 * device_.cores, tiles}) {
        append({tile, std::min(groups, tiles)});
        if (supportsFourSimdgroups(w, tile))
          append({tile, std::min(groups, tiles), LinearSimdgroups::Four});
      }
    }
  }
  if ((w.phase == LinearPhase::Decode && device_.primitive == Primitive::Register) || simdgroupPrefill) {
    const uint32_t n = w.matrix.outputSize;
    const uint32_t columns = w.epilogue == LinearEpilogue::GateUp ? 32 : 64;
    for (uint32_t splits = 1; splits <= LinearConfig::kMaximumSplits; splits *= 2)
      if ((w.matrix.inputSize / kQuantGroup) % splits == 0)
        append({LinearTile::Simdgroup, n / columns, LinearSimdgroups::Four, splits});
  }
  // Apple10 lists Split128 at every K split its 256-input blocks allow, at
  // every lane count.
  if (w.phase == LinearPhase::Decode && device_.primitive == Primitive::Tensor)
    for (uint32_t splits = 2;
         splits <= LinearConfig::kMaximumSplits && splits <= w.matrix.inputSize / kInputSumBlock; splits *= 2)
      append({LinearTile::Split128, w.matrix.outputSize / 128, LinearSimdgroups::Eight, splits});
  // One-lane tiles: the split forms at their full grid and the paired N256
  // tile at one wave and at its full grid.
  if (w.phase == LinearPhase::Decode && w.rows == SPLASH_TARGET_VERIFY_ROWS) {
    const uint32_t n = w.matrix.outputSize;
    if (w.matrix.inputSize % kSplitInputBlock == 0) {
      append({LinearTile::Split32, n / 32, LinearSimdgroups::Four});
      if (w.epilogue != LinearEpilogue::GateUp)
        append({LinearTile::Split64, n / 64, LinearSimdgroups::Eight});
    }
    if (w.epilogue == LinearEpilogue::None)
      for (const uint32_t groups : {kPaired256WaveGroupsPerCore * device_.cores, n / 256})
        append({LinearTile::Paired256, std::min(groups, n / 256), LinearSimdgroups::Four});
  }
  return result;
}

LinearPlan Linear::decodePlan(const Projection &p, uint32_t lanes, LinearEpilogue epilogue,
                              const Projection *gate) const {
  return plan(decode({p.outputSize, p.inputSize}, lanes, epilogue), p, gate);
}
LinearPlan Linear::prefillPlan(const Projection &p, uint32_t rows, LinearEpilogue epilogue) const {
  return plan({{p.outputSize, p.inputSize}, rows, LinearPhase::Prefill, epilogue}, p);
}

LinearScratchSize Linear::decodeScratchSize(LinearWorkload w) const {
  if (w.weightLayout == WeightLayout::Block32) return ggufDecodeScratchSize(w);
  return LinearPlan(w, baseline(w)).scratchSize().include(plan(w).scratchSize());
}
LinearScratchSize Linear::prefillScratchSize(ProjectionShape shape) const {
  LinearScratchSize bound;
  for (uint32_t rows = 1; rows <= kMaximumDecodeTileRows; ++rows)
    for (const auto epilogue : {LinearEpilogue::None, LinearEpilogue::Residual, LinearEpilogue::UpWithGate})
      bound.include(
          plan({{shape.outputSize, shape.inputSize}, rows, LinearPhase::Prefill, epilogue, shape.layout}).scratchSize());
  return bound;
}


PreparedInput Linear::add(metal::CommandGraph &graph, LinearBuffers b,
    const Projection &p, const LinearPlan &selected, const Projection *gate,
    LinearDispatchStats *stats) const {
  const LinearWorkload w = selected.workload();
  const auto [n, k] = w.matrix;
  if (p.layout() != w.weightLayout || (gate && gate->layout() != w.weightLayout))
    throw std::invalid_argument("projection layout does not match execution plan");
  if ((w.epilogue == LinearEpilogue::GateUp) != (gate != nullptr))
    throw std::invalid_argument("a gate/up plan takes a gate projection and no other plan does");
  const uint64_t rows = selected.storageRows();
  requireBytes(b.input, rows * k * 2, "input");
  requireBytes(b.output, rows * n * elementBytes(selected.destination()), "output");
  if (w.epilogue == LinearEpilogue::Residual) requireBytes(b.residual, rows * n * 2, "residual");
  requireBytes(b.sums, selected.sumsBytes(), "sums");
  requireBytes(b.gateScratch, selected.gateScratchBytes(), "gate scratch");
  requireBytes(b.downSums, selected.downSumsBytes(), "down sums");
  const LinearScratchSize scratch = selected.scratchSize();
  requireBytes(b.scratch.input, scratch.input, "scratch table");
  requireBytes(b.scratch.sums, scratch.sums, "scratch sums");
  requireBytes(b.scratch.partials, scratch.partials, "partials");
  requireBytes(b.scratch.counters, scratch.counters, "counters");
  if (p.layout() == WeightLayout::Block32) {
    if (p.rotation) requireBytes(b.scratch.rotated, rotatedBytes(k, rows), "rotated input");
    addGguf(graph, b, p, selected, gate, stats);
    // A rotated projection's plan prepares its table, if any, from the
    // rotated rows, which no other plan reads.
    if (p.rotation) return {};
    // Only quantized segments run the plan's tile: float segments alone
    // leave the scratch table as it was.
    const std::vector<QuantizedSegment> &segments = p.blocks().segments;
    const bool tiled =
        std::any_of(segments.begin(), segments.end(), [](const QuantizedSegment &s) { return !s.isFloat(); });
    return tiled && selected.input() != LinearInput::Plain ? PreparedInput{b.input, selected.input()} : b.prepared;
  }
  requireAffineProjection(p, w.matrix);
  if (gate) requireAffineProjection(*gate, w.matrix);
  const AffineWeights &weights = p.affine();
  if (selected.usesSimdgroup()) {
    const uint32_t lanes = selected.storageRows() / SPLASH_TARGET_VERIFY_ROWS;
    if (b.prepared.layout != LinearInput::Table64 || !b.prepared.source.sameView(b.input))
      graph.add("decode_linear_q4_prepare", {b.input, b.scratch.input, b.scratch.sums},
                k, {k / 32, lanes, 1}, {128, 1, 1});
    const AffineWeights &first = gate ? gate->affine() : weights;
    std::vector<metal::MetalBuffer> bindings{b.scratch.input, first.weights, first.scales, first.biases,
                                             b.output, b.scratch.sums, b.scratch.partials, b.scratch.counters};
    if (gate) bindings.insert(bindings.end(), {weights.weights, weights.scales, weights.biases});
    else if (w.epilogue == LinearEpilogue::Residual) bindings.push_back(b.residual);
    else if (w.epilogue == LinearEpilogue::UpWithGate) bindings.push_back(b.gateScratch);
    graph.add(kernelInstance(selected.pipeline(), selected.destination()), std::move(bindings),
        Q4Params{n, k, selected.configuration().splits},
        {selected.configuration().groups, selected.configuration().splits, lanes}, {128, 1, 1});
    if (stats && w.phase == LinearPhase::Decode) account(*stats, lanes, 1);
    return {b.input, LinearInput::Table64};
  }
  const auto dispatch = [&](std::string_view name,
      std::initializer_list<metal::MetalBuffer> bindings) {
    if (w.phase == LinearPhase::Prefill)
      graph.add(std::string(name), bindings,
          Q4PrefillParams{w.matrix.outputSize, w.matrix.inputSize},
          {selected.storageRows() / kAffinePrefillTileRows, n / selected.tileColumns(), 1},
          {selected.threadsPerThreadgroup(), 1, 1});
    else {
      // Split128 binds its partials and counters after the sequential
      // kernel's buffers; every other decode tile runs one K split.
      const LinearConfig config = selected.configuration();
      std::vector<metal::MetalBuffer> buffers(bindings);
      if (config.tile == LinearTile::Split128)
        buffers.insert(buffers.end(), {b.scratch.partials, b.scratch.counters});
      graph.add(kernelInstance(name, selected.destination()), std::move(buffers), Q4Params{n, k, config.groups},
                {config.groups, config.splits, 1}, {selected.threadsPerThreadgroup(), 1, 1});
    }
  };
  const bool prefill = w.phase == LinearPhase::Prefill;
  if (w.epilogue == LinearEpilogue::GateUp) {
    const AffineWeights &g = gate->affine();
    if (selected.secondPipeline().empty())
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases,
                                     b.output, weights.weights, weights.scales, weights.biases});
    else {
      dispatch(selected.pipeline(), {b.input, g.weights, g.scales, g.biases, b.gateScratch});
      dispatch(selected.secondPipeline(),
               {b.input, weights.weights, weights.scales, weights.biases, b.gateScratch, b.output});
    }
  } else if (w.epilogue == LinearEpilogue::UpWithGate)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                   b.gateScratch, b.output, b.sums, b.downSums});
  else if (w.epilogue == LinearEpilogue::Residual) {
    if (prefill)
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output, b.sums});
    else
      dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases,
                                     b.residual, b.output});
  } else if (prefill)
    dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output, b.sums});
  else dispatch(selected.pipeline(), {b.input, weights.weights, weights.scales, weights.biases, b.output});
  if (stats && !prefill)
    account(*stats, w.rows / SPLASH_TARGET_VERIFY_ROWS, selected.secondPipeline().empty() ? 1 : 2);
  return b.prepared;
}

void Linear::addPrefillSums(metal::CommandGraph &graph, metal::MetalBuffer input, metal::MetalBuffer sums,
                            const Projection &consumer, uint32_t rows) const {
  validate({{consumer.outputSize, consumer.inputSize}, rows, LinearPhase::Prefill});
  const uint32_t storageRows = kAffineTensorPrefill.rows.round(rows);
  requireBytes(input, uint64_t{storageRows} * consumer.inputSize * 2, "input");
  requireBytes(sums, uint64_t{storageRows} * (consumer.inputSize / kQuantGroup) * 4, "sums");
  graph.add("prefill_linear_q4_sums32", {input, sums}, Q4PrefillParams{consumer.outputSize, consumer.inputSize},
            {storageRows / kAffinePrefillTileRows, 1, 1});
}
PreparedInput Linear::addPrefill(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                 metal::MetalBuffer output, metal::MetalBuffer sums, uint32_t rows,
                                 LinearScratch scratch, PreparedInput prepared) const {
  return add(graph, {.input = input, .output = output, .sums = sums, .scratch = scratch, .prepared = prepared}, p,
             prefillPlan(p, rows, LinearEpilogue::None));
}
PreparedInput Linear::addPrefillResidual(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                         metal::MetalBuffer residual, metal::MetalBuffer output,
                                         metal::MetalBuffer sums, uint32_t rows, LinearScratch scratch,
                                         PreparedInput prepared) const {
  return add(graph,
             {.input = input, .output = output, .sums = sums, .residual = residual, .scratch = scratch,
              .prepared = prepared},
             p, prefillPlan(p, rows, LinearEpilogue::Residual));
}
PreparedInput Linear::addPrefillUpWithGate(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &up,
                                           metal::MetalBuffer gateScratch, metal::MetalBuffer output,
                                           metal::MetalBuffer sums, metal::MetalBuffer downSums, uint32_t rows,
                                           LinearScratch scratch, PreparedInput prepared) const {
  return add(graph,
             {.input = input, .output = output, .sums = sums, .gateScratch = gateScratch, .downSums = downSums,
              .scratch = scratch, .prepared = prepared},
             up, prefillPlan(up, rows, LinearEpilogue::UpWithGate));
}
PreparedInput Linear::addDecode(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                metal::MetalBuffer output, LinearScratch scratch) const {
  return add(graph, {.input = input, .output = output, .scratch = scratch}, p, decodePlan(p, 1));
}
PreparedInput Linear::addDecodeBatch(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                     metal::MetalBuffer output, uint32_t lanes, LinearDispatchStats &stats,
                                     LinearScratch scratch, PreparedInput prepared) const {
  return add(graph, {.input = input, .output = output, .scratch = scratch, .prepared = prepared}, p,
             decodePlan(p, lanes), nullptr, &stats);
}
PreparedInput Linear::addResidualBatch(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &p,
                                       metal::MetalBuffer residual, metal::MetalBuffer output, uint32_t lanes,
                                       LinearDispatchStats &stats, LinearScratch scratch,
                                       PreparedInput prepared) const {
  return add(graph,
             {.input = input, .output = output, .residual = residual, .scratch = scratch, .prepared = prepared},
             p, decodePlan(p, lanes, LinearEpilogue::Residual), nullptr, &stats);
}
PreparedInput Linear::addGateUpBatch(metal::CommandGraph &graph, metal::MetalBuffer input, const Projection &gate,
                                     const Projection &up, metal::MetalBuffer gateScratch,
                                     metal::MetalBuffer output, uint32_t lanes, LinearDispatchStats &stats,
                                     LinearScratch scratch, PreparedInput prepared) const {
  return add(graph,
             {.input = input, .output = output, .gateScratch = gateScratch, .scratch = scratch,
              .prepared = prepared},
             up, decodePlan(up, lanes, LinearEpilogue::GateUp, &gate), &gate, &stats);
}

} // namespace splash::ops
