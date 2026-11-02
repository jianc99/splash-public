#include "ops/KernelSelection.hpp"

#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"
#include "metal/abi/QuantFormat.h"

#include <algorithm>
#include <array>
#include <variant>

namespace splash::ops {
namespace {

static_assert(kApple9Projection.stagedPrefillRows <= kMaximumStagedPrefillRows &&
                  kApple9SmallProjection.stagedPrefillRows <= kMaximumStagedPrefillRows &&
                  kApple10Projection.stagedPrefillRows <= kMaximumStagedPrefillRows,
              "the staged tile holds the prefill chunks it runs");

// The most bits per weight a format stores, in sixteenths (BF16 and F16).
constexpr uint32_t kMostWeightBits16 = [] {
  uint32_t most = 0;
  for (const QuantFormat &f : kQuantFormats) most = std::max(most, f.block_bytes * 128 / f.block_elements);
  return most;
}();

// The split count doubles while some tier asks (TierLaw).
uint32_t tierSplits(uint32_t n, uint32_t k, uint32_t cores, std::span<const SplitTier> tiers) {
  const uint64_t grid = n / GGUF_TILE_COLUMNS;
  uint32_t splits = 1;
  const auto asks = [&](const SplitTier &t) {
    return splits < t.splits && grid * splits < uint64_t{t.threadgroupsPerCore} * cores &&
           k / (2 * splits) >= t.inputs;
  };
  while ((k / 32) % (2 * splits) == 0 && std::any_of(tiers.begin(), tiers.end(), asks)) splits *= 2;
  return splits;
}

// The time of a dispatch of `groups` threadgroups of a staged tile, `steps`
// K steps each, on `cores` cores, in bits a core streams at its share of DRAM,
// times the cores: every wave but the last holds `resident` threadgroups on
// each core and the last the rest, each threadgroup streaming the weights of
// a step of 64 columns, of `bits16` sixteenths of a bit each, per step; the
// last wave takes at least `saturating` threadgroups' time when it is the
// grid's only one and `following`'s otherwise. A split grid's threadgroups
// each also store their fp32 partials of `rows` rows, which the last
// partition reads back.
uint64_t stagedTime(uint64_t groups, uint32_t steps, uint32_t splits, uint32_t rows, uint32_t bits16,
                    uint32_t cores, StagedOccupancy tile) {
  const uint64_t perCore = (groups + cores - 1) / cores, waves = (perCore + tile.resident - 1) / tile.resident;
  const uint64_t full = (waves - 1) * tile.resident * cores,
                 floor = uint64_t{waves > 1 ? tile.following : tile.saturating} * cores;
  const uint64_t stepBits = uint64_t{GGUF_TILE_COLUMNS} * 32 * bits16 / 16,
                 partialBits = splits > 1 ? 2 * uint64_t{rows} * GGUF_TILE_COLUMNS * 32 : 0;
  return steps * stepBits * (full + std::max(groups - full, floor)) + groups * partialBits;
}

// The wave law's split of `rows` rows (a gate/up pair's two tensors when it
// may run in one pass), and passes, of least time; ties keep fewer
// partitions, then one pass. The law tells the pass modes apart only where a
// last wave falls below its floor: on the 32-core M5 Max the 27B gate/up pair
// puts 17 one-pass threadgroups on each core, the 17th of which runs a second
// wave alone (333 us against 245 in two passes of 8.5 per core, one lane).
// Where both modes' last waves clear their floors it predicts the same time
// for both, and the tie takes one pass: in 133 of the 152 tied cases of the M5
// Max's and the M5 Pro's sweeps (one to four lanes) one pass takes up to 22%
// less time, a 20-core M5 Pro's 27B pair (27.2 per core) 8% (395 us against
// 430), a 40-core M5 Max's (13.6) 8-9%; at 20 and 36 one-pass threadgroups per
// core it takes up to 20% more, most at one lane. A 16-core M5 Pro puts 34
// (16, 16 and 2), where the law takes two passes at every lane count: emulated
// by width on the M5 Max they take 9-12% less time than one unsplit pass at
// one and two lanes but 7% more at three and four, where the 32-row tile's
// lower residency slows them. Below ProjectionCalibration::onePassFromCores a
// pair runs two passes whatever the law.
SplitChoice waveSplits(const WaveLaw &law, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores,
                       Passes passes) {
  const uint32_t tileRows = stagedTileRows(rows);
  SplitChoice best;
  uint64_t least = UINT64_MAX;
  for (uint32_t splits = 1; splits <= law.maximumSplits && (k / 32) % splits == 0 &&
                            (splits == 1 || k / splits >= law.partitionInputs);
       splits *= 2) {
    const uint32_t steps = k / 32 / splits;
    const uint64_t tiles = uint64_t{n / GGUF_TILE_COLUMNS} * splits;
    if (passes != Passes::Two) {
      const uint64_t time = stagedTime(2 * tiles, steps, splits, tileRows, bits16, cores, law.gateUpPair);
      if (time < least) least = time, best = {splits, true};
    }
    if (passes != Passes::One) {
      // A pair's two passes take twice one tensor's time.
      const uint64_t time =
          (passes == Passes::Either ? 2 : 1) * stagedTime(tiles, steps, splits, tileRows, bits16, cores, law.tile);
      if (time < least) least = time, best = {splits, false};
    }
  }
  return best;
}

// Whether the register tile runs a decode step of `lanes` lanes over
// `columnTiles` column tiles of a unit of `formatClass` (RegisterBoundary).
bool registers(const RegisterBoundary &boundary, FormatClass formatClass, uint32_t lanes, uint64_t columnTiles,
               uint32_t cores) {
  return 4 * columnTiles * lanes < uint64_t{boundary.belowQuarters[size_t(formatClass)][lanes - 1]} * cores;
}

// The staged tile's configuration of `rows` rows.
LinearConfig staged(const ProjectionCalibration &p, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16,
                    uint32_t cores, Passes passes) {
  const SplitChoice choice = splitsFor(p.stagedSplits, n, k, rows, bits16, cores, passes);
  return {.tile = LinearTile::GgufStaged, .splits = choice.splits, .oneGateUpPass = choice.onePass};
}

bool gridLookup(uint32_t format) noexcept {
  switch (format) {
  case GGUF_FMT_IQ3XXS: case GGUF_FMT_IQ2XXS: case GGUF_FMT_IQ2XS: case GGUF_FMT_IQ2S:
  case GGUF_FMT_IQ1S: case GGUF_FMT_IQ1M: return true;
  default: return false;
  }
}

// The FormatClass of quantized segments in `formats`.
class Classifier final {
public:
  void add(uint32_t format) noexcept {
    grid_ &= gridLookup(format) || format == GGUF_FMT_Q2K;
    q2k_ |= format == GGUF_FMT_Q2K;
    fourBits_ &= quant_four_bits(format);
    any_ = true;
  }
  [[nodiscard]] FormatClass formatClass() const noexcept {
    if (any_ && grid_) return q2k_ ? FormatClass::GridLookupOrQ2K : FormatClass::GridLookup;
    return fourBits_ ? FormatClass::FourBitsOrMore : FormatClass::UnderFourBits;
  }

private:
  bool grid_ = true, q2k_ = false, fourBits_ = true, any_ = false;
};

} // namespace

UnitFormats unitFormats(std::span<const Projection *const> projections) noexcept {
  UnitFormats unit;
  Classifier classes;
  uint32_t bits = 0, count = 0;
  for (const Projection *p : projections) {
    if (!p) continue;
    bits += p->shape().weightBits16, ++count;
    for (const QuantizedSegment &s : p->blocks().segments) {
      if (s.isFloat()) continue;
      unit.registerTakes &= quant_register_tensor(s.formatId, s.inputSize);
      classes.add(s.formatId);
    }
  }
  unit.formatClass = classes.formatClass();
  unit.weightBits16 = count ? bits / count : 0;
  if (projections.size() == 2 && projections[0] && projections[1]) {
    const std::vector<QuantizedSegment> &up = projections[0]->blocks().segments,
                                        &gate = projections[1]->blocks().segments;
    unit.oneFormat = up.size() == 1 && gate.size() == 1 && !up.front().isFloat() &&
                     up.front().formatId == gate.front().formatId;
  }
  return unit;
}

UnitFormats unitFormats(const ProjectionShape &shape) noexcept {
  return {.formatClass = shape.fourBits ? FormatClass::FourBitsOrMore : FormatClass::UnderFourBits,
          .registerTakes = shape.registerFormats,
          .weightBits16 = shape.weightBits16};
}

SplitChoice splitsFor(const SplitLaw &law, uint32_t n, uint32_t k, uint32_t rows, uint32_t bits16, uint32_t cores,
                      Passes passes) noexcept {
  if (const TierLaw *tiers = std::get_if<TierLaw>(&law)) return {tierSplits(n, k, cores, tiers->tiers), false};
  return waveSplits(std::get<WaveLaw>(law), n, k, rows, bits16 ? bits16 : kMostWeightBits16, cores, passes);
}

// Prefill chunks of up to stagedPrefillRows rows run the staged tile at their
// rows' height with stagedSplits, longer ones the prefill tile unsplit. A
// decode step runs the register tile where it takes every quantized segment
// and the class's RegisterBoundary holds, a one-pass-eligible gate/up pair's
// column tiles counting both tensors', splitting by registerSplits; else the
// staged tile, splitting by stagedSplits (a wave law also picks one or two
// passes for an eligible pair), its walks spread from spreadFromCores, and one
// unsplit lane under sixteenRowBelowBits16 on the 16-row tile, a pair's only
// where both its tensors' own plans take it.
LinearConfig selectProjection(const KernelCalibration &calibration, LinearWorkload w, const UnitFormats &unit,
                              std::span<const UnitFormats> pairTensors) {
  const auto [n, k] = w.matrix;
  const ProjectionCalibration &p = calibration.projection;
  const uint32_t cores = calibration.cores;
  if (w.phase == LinearPhase::Prefill)
    return w.rows <= p.stagedPrefillRows ? staged(p, n, k, w.rows, unit.weightBits16, cores, Passes::Two)
                                         : LinearConfig{.tile = LinearTile::GgufPrefill};
  const uint32_t lanes = w.rows / SPLASH_TARGET_VERIFY_ROWS;
  const bool gateUp = w.epilogue == LinearEpilogue::GateUp,
             onePass = gateUp && unit.oneFormat && cores >= p.onePassFromCores;
  if (unit.registerTakes &&
      registers(p.decodeRegister, unit.formatClass, lanes, (onePass ? 2 : 1) * uint64_t{n / GGUF_TILE_COLUMNS},
                cores))
    return {.tile = LinearTile::GgufRegister,
            .splits = splitsFor(p.registerSplits, n, k, w.rows, unit.weightBits16, cores).splits};
  LinearConfig config = staged(p, n, k, w.rows, unit.weightBits16, cores, onePass ? Passes::Either : Passes::Two);
  config.spread = cores >= p.spreadFromCores;
  config.sixteenRows =
      lanes == 1 && config.splits == 1 &&
      (gateUp ? !pairTensors.empty() && std::all_of(pairTensors.begin(), pairTensors.end(),
                                                    [&](const UnitFormats &tensor) {
                                                      return selectProjection(calibration, {w.matrix, w.rows}, tensor)
                                                          .sixteenRows;
                                                    })
              : unit.weightBits16 && unit.weightBits16 < p.sixteenRowBelowBits16);
  return config;
}

// The plan of the shape's flags; where the register tile may run (it runs a
// unit of FourBitsOrMore), a unit of other formats or one it does not take
// stages, of the most bits per weight, whose partials weigh least against the
// weights: the most K partitions of each pass mode, since the law's split
// never falls with the bits per weight and never rises with the rows (a
// split's partials grow with its rows alone, the same in either pass mode);
// and a gate/up pair of one format in one pass where it may.
std::vector<LinearConfig> decodeConfigurations(const KernelCalibration &calibration, LinearWorkload w,
                                               bool registerTakes) {
  const ProjectionCalibration &p = calibration.projection;
  const uint32_t cores = calibration.cores, n = w.matrix.outputSize, k = w.matrix.inputSize;
  std::vector<LinearConfig> configurations{
      selectProjection(calibration, w, {.registerTakes = registerTakes})};
  if (registers(p.decodeRegister, FormatClass::FourBitsOrMore, w.rows / SPLASH_TARGET_VERIFY_ROWS,
                n / GGUF_TILE_COLUMNS, cores))
    configurations.push_back(staged(p, n, k, w.rows, 0, cores, Passes::Two));
  if (w.epilogue == LinearEpilogue::GateUp && cores >= p.onePassFromCores)
    configurations.push_back(staged(p, n, k, w.rows, 0, cores, Passes::One));
  return configurations;
}

FloatTile selectFloatTile(const KernelCalibration &calibration, uint32_t rows, uint32_t outputSize,
                          uint32_t formatId) noexcept {
  const FloatTileCalibration &f = calibration.floatTile;
  const uint64_t tiles = uint64_t{(rows + 63) / 64} * ((outputSize + 31) / 32);
  // The grid in half tiles per core.
  const uint32_t halves = formatId == GGUF_FMT_BF16                 ? f.bf16Halves
                          : formatId == QuantizedSegment::kFloat32 ? f.f32Halves
                                                                   : f.quantizedRowsHalves;
  return f.neuralAccelerator && rows >= f.minimumRows && 2 * tiles >= uint64_t{halves} * calibration.cores
             ? FloatTile::NeuralAccelerator
             : FloatTile::Simdgroup;
}

MoeFormats moeFormats(std::span<const BlockMoeWeights *const> blocks) noexcept {
  // Routed elements per format, and routers per format, F32 counted one past
  // the GGUF formats.
  std::array<uint64_t, GGUF_FMT_COUNT> weights{};
  std::array<uint32_t, GGUF_FMT_COUNT + 1> routers{};
  MoeFormats formats;
  for (const BlockMoeWeights *block : blocks) {
    for (const BlockExpertProjection *projection : {&block->gate, &block->up, &block->down}) {
      for (const QuantizedSegment *s : {&projection->routed, &projection->shared})
        formats.experts.registerTakes &= s->isFloat() || quant_register_tensor(s->formatId, s->inputSize);
      if (!projection->routed.isFloat())
        weights[projection->routed.formatId] += uint64_t{projection->routed.outputSize} * projection->routed.inputSize;
    }
    ++routers[std::min<uint32_t>(block->router.formatId, GGUF_FMT_COUNT)];
  }
  Classifier classes;
  if (const auto most = std::max_element(weights.begin(), weights.end()); *most)
    classes.add(uint32_t(most - weights.begin()));
  formats.experts.formatClass = classes.formatClass();
  const auto router = std::max_element(routers.begin(), routers.end());
  formats.router = *router && router < routers.end() - 1 ? uint32_t(router - routers.begin())
                                                         : QuantizedSegment::kFloat32;
  return formats;
}

// The register tile where it takes the experts and the class's entry holds;
// else the staged tile, on 8-row tiles while a prefill chunk's routes average
// at most one row per expert (rows * topK <= experts), where the two tiles
// take the same time, and 32-row tiles beyond, which stream an expert's
// weights once for up to 32 of its rows (its tiles run 8-, 16- or 32-row
// matmuls by their live rows, moe_live_rows). The router's float tile follows
// selectFloatTile by its rows' format.
MoeConfig selectMoe(const KernelCalibration &calibration, const MoeShape &shape, const MoeFormats &formats,
                    uint32_t rows, MoePhase phase) noexcept {
  MoeConfig config;
  config.ggufTile = formats.experts.registerTakes && calibration.moe.registerTile[size_t(formats.experts.formatClass)]
                        ? MoeGgufTile::Register
                        : MoeGgufTile::Staged;
  config.expertTile = phase == MoePhase::Prefill && config.ggufTile == MoeGgufTile::Staged &&
                              uint64_t{rows} * shape.expertsPerToken > shape.experts
                          ? MoeExpertTile::M32
                          : MoeExpertTile::M8;
  config.ggufRouterTile = selectFloatTile(calibration, rows, shape.experts, formats.router);
  return config;
}

} // namespace splash::ops
