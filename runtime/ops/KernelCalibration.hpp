#pragma once

// Every measured constant of kernel selection, by GPU class and, where a
// measurement says so, by core range, each with what measured it. The rules
// that read them are in KernelSelection.cpp; KernelCalibration::of is the
// only place that picks a row by the device's class.

#include "metal/DeviceCapabilities.hpp"
#include "metal/abi/ExecutionGeometry.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

namespace splash::ops {

// A "from N cores" threshold no device reaches, and a register boundary
// every grid is below.
inline constexpr uint32_t kNever = UINT32_MAX;
inline constexpr uint32_t kAlways = UINT32_MAX;

// How selection classes a weight unit's quantized segments together (a
// projection, a gate/up pair, a MoE block's experts; formatClass):
// - GridLookup: every segment in a format whose register-tile operands come
//   from grid lookups beside its matrix operations: IQ3_XXS, the IQ2 formats
//   and IQ1;
// - GridLookupOrQ2K: every segment GridLookup or Q2_K, at least one Q2_K;
// - FourBitsOrMore: otherwise, every segment of at least four bits per weight
//   (quant_four_bits), or no quantized segment;
// - UnderFourBits: otherwise.
enum class FormatClass : uint8_t { GridLookup, GridLookupOrQ2K, UnderFourBits, FourBitsOrMore };
inline constexpr size_t kFormatClasses = 4;

// ---- K-split laws ---------------------------------------------------------

// K splits by tiers. A tier asks for more partitions, up to its `splits`,
// while the grid holds fewer than `threadgroupsPerCore` threadgroups per core
// and each partition would still keep `inputs` inputs; the split count
// doubles, in whole 32-input groups per partition, while some tier asks. The
// tiers ignore the batch width: bounds that depended on it did not pay on
// Apple9. The register tile and Apple9's staged tile run their matrix
// operations on the FP32 pipe (Apple9) or as exact bf16 register operations,
// without a residency step in their time (a 10-core M4 over 2 to 40 register
// threadgroups per core): the Apple10 staged tiles' wave law does not
// describe them.
struct SplitTier final {
  uint32_t threadgroupsPerCore;
  uint32_t inputs;
  uint32_t splits = 8;
};
struct TierLaw final {
  std::span<const SplitTier> tiers;
};

// The Apple10 staged decode tiles' wave law. A tile kernel's threadgroups each
// walk K / 32 / splits steps over a 64-column tile's weights (a one-pass
// gate/up pair's 32 columns of both tensors). A core runs `resident` of them at
// once, so a grid runs in waves of that many per core; it streams its share of
// DRAM from `saturating` of them on, so a grid's first wave of fewer takes as
// long as one of that many, waiting on its chains' latency; a later wave starts
// as the one before drains, which hides part of that latency, and takes as long
// as at least `following`.
struct StagedOccupancy final {
  uint32_t resident, saturating, following;
};
struct WaveLaw final {
  // gguf_decode_<format>_m<rows>_<e> and gguf_decode_fused_m<rows>: two
  // simdgroups, 64 columns.
  StagedOccupancy tile;
  // gguf_decode_<format>_m<rows>_gate_up: one simdgroup, 32 columns of each
  // tensor.
  StagedOccupancy gateUpPair;
  // Partitions keep this many inputs, maximumSplits at most.
  uint32_t partitionInputs;
  uint32_t maximumSplits;
};
using SplitLaw = std::variant<TierLaw, WaveLaw>;

// Register tile against the staged tile for a decode step of `lanes` lanes,
// by the unit's FormatClass: register where 4 * columnTiles * lanes <
// belowQuarters * cores (in quarter threadgroups per core); kAlways where it
// takes the formats whatever the grid, 0 never.
struct RegisterBoundary final {
  std::array<std::array<uint32_t, SPLASH_MAXIMUM_BATCH_WIDTH>, kFormatClasses> belowQuarters;
};

struct ProjectionCalibration final {
  SplitLaw registerSplits;
  SplitLaw stagedSplits;
  RegisterBoundary decodeRegister;
  // From this many cores the staged decode walks spread (LinearConfig::spread).
  uint32_t spreadFromCores;
  // From this many cores a gate/up pair of one format may run in one pass.
  uint32_t onePassFromCores;
  // One unsplit decode lane of fewer bits per weight than this, in
  // sixteenths, runs the staged 16-row tile; 0 never.
  uint32_t sixteenRowBelowBits16;
  // Prefill chunks of up to this many rows run the staged tile
  // (<= kMaximumStagedPrefillRows); longer ones the prefill tile.
  uint32_t stagedPrefillRows;
};

struct FloatTileCalibration final {
  // The device has a neural accelerator per core.
  bool neuralAccelerator;
  // The accelerator tile's fewest rows, and the grid it needs in half tiles
  // per core for BF16 values, F32 values and a quantized format's rows.
  uint32_t minimumRows;
  uint32_t bf16Halves, f32Halves, quantizedRowsHalves;
};

struct MoeCalibration final {
  // The expert tile by the block's FormatClass, in both phases: register
  // where true.
  std::array<bool, kFormatClasses> registerTile;
};

// Every measured constant of kernel selection for one device.
struct KernelCalibration final {
  // DeviceCapabilities::plannedGpuCores.
  uint32_t cores;
  ProjectionCalibration projection;
  FloatTileCalibration floatTile;
  MoeCalibration moe;

  [[nodiscard]] static KernelCalibration of(const DeviceCapabilities &device) noexcept;
};

// ---- Projections ----------------------------------------------------------

// Register tile (128 threads). A core holds four of its threadgroups at
// once, so the tile's time steps with every four threadgroups per core.
// Below one wave a core must fill it, down to 256 inputs per partition, whole
// coefficient units in every format (quant_register_unit); below eight waves
// more threadgroups shrink the last wave's tail while partitions of 1024
// inputs amortize the partial sums (both measured on a 40-core M3 Max). Both
// stop at eight partitions. A grid that eight leave below three threadgroups
// per core takes sixteen while partitions keep 512 inputs, a tier measured on
// Apple10 alone (Apple9 takes it unmeasured): Qwen3.8-Flash-Next's
// hyper-connection down projection, Q8_0 320 x 10240 on five column tiles,
// then takes 11-17% less time at one to four lanes on the 20-core M5 Pro and
// 6-20% less on the 32-core M5 Max, and 32 partitions more than 16 on both.
// On the M5 Max, sixteen take 4% less to 10% more 512 columns wide (two
// threadgroups per core at eight), up to 18% more 832 wide (3.25), 4-15% more
// on the same 320 x 10240 in BF16, twice Q8_0's bytes (Flash-Next loads such
// tensors as Q8_0), and 22-38% more on Q4_K 512 x 5120, whose partitions
// would keep 320 inputs. From five threadgroups per core a 10-core M4 takes
// 3-7% less time at one lane on one or two partitions than on the tiers' four
// (MLX 4-bit, 5120 inputs), where at two to four lanes the tiers' are the
// fastest: telling one lane apart needs an Apple9 GPU of more cores.
// Status: tiers 1-2 measured (M3 Max); tier 3 measured on Apple10, unmeasured
// on Apple9 (U1).
inline constexpr SplitTier kRegisterTiers[] = {{4, 256}, {32, 1024}, {3, 512, 16}};

// Apple9's staged tile takes the register tile's tiers of eight partitions
// (the first two): on a 40-core M3 Max over the 27B and 35B dense shapes at
// one to four lanes they come within 0-9% of each shape's fastest split (20%
// at one lane on 2048 x 512, a 0.012 ms projection), where a fitted tier of
// six threadgroups per core is 13-27% slower on 17408 x 5120 and
// 12288 x 5120 and 50% on 2048 x 512. Status: measured (M3 Max).
inline constexpr std::span<const SplitTier> kApple9StagedTiers = std::span(kRegisterTiers).first(2);

// The wave law's constants, measured on the 32-core M5 Max, the 20-core M5 Pro
// and the 12-core M6 (gguf-projection-benchmark over 2 to 40 threadgroups per
// core, MLX 4-bit, Q4_K to Q8_0, IQ3_XXS and IQ2_XS, 2048 and 5120 inputs, one
// to four lanes): time per byte steps up past 10 of the two-simdgroup tile's
// threadgroups per core at one and two lanes on the M5 Max and the M6 (M5 Max,
// one lane: 140 us at 10, 190 at 11 for 1.1 times the bytes), past 8 or 9 on
// the M5 Pro, and past 16 of the pair's at every lane count (229 us at 16, 334
// at 17). The 32-row tile of three and four lanes steps past 9 on the M5 Max, 8
// on the M5 Pro and 11 on the M6, which the law leaves out: it takes 10 at
// every row count. A lone chain of 160 steps takes as long as 5.4 and 9
// threadgroups' bytes at DRAM's rate, and one threadgroup per core past a full
// wave adds 2.7-3.5 of the tile's threadgroups' time (M5 Max and M6, one and
// two lanes) and 6-8.5 of the pair's (M5 Max and M5 Pro). `saturating` and
// `following` are not those measurements but a grid search's fit, with the
// residencies, to the fastest split of every measured case (the 16-row tile
// where one lane takes it): 6 and 10 for the measured 5.4 and 9, 2 and 3 for
// the measured 2.7-3.5 and 6-8.5; they stand in for what the law leaves out,
// such as the 32-row residency and the pair's per-byte advantage. Floors at the
// measured trailing waves (3 and 7) take as much time in total on the same
// cases but leave 18 of them more than 3% slower than the earlier rules,
// against 5 (Q8_0 and Q6_K split at 12 threadgroups per core, three- and
// four-lane pairs at 10 per pass). On those cases, which fitted it
// (in-sample), the law's choices take 1.1%, 1.6% and 0.05% less time than the
// earlier rules' in total on the M5 Max, the M5 Pro and the M6, 29, 30 and 3
// cases 3-33% less, and 2, 2 and 1 cases 4-12% more (a 34-us pair of four
// threadgroups per core over 2048 inputs, two lanes). Status: measured over
// 2048 and 5120 inputs and pairs in MLX 4-bit; the down projections' 17408
// inputs, pairs in other formats and low-bit pairs take the law unmeasured
// (U6). Staged partitions keep 512 inputs, eight at most.
inline constexpr WaveLaw kApple10StagedLaw{{10, 6, 2}, {16, 10, 3}, 512, 8};
static_assert(kApple10StagedLaw.tile.resident >= kApple10StagedLaw.tile.saturating &&
                  kApple10StagedLaw.gateUpPair.resident >= kApple10StagedLaw.gateUpPair.saturating,
              "a full wave streams at DRAM's rate");

// Apple9 decodes on the register tile but for a unit the staged tile holds
// unpadded, every quantized segment in a GridLookup format, or in Q2_K from
// two lanes. On a 40-core M3 Max at 17408x5120 and 5120x17408, best split of
// each tile, the staged tile takes 6-21% less time on those formats at one
// lane, 2-11% at two and 2-9% at four, and on Q2_K 24-33% less at two and
// four lanes but 5-8% more at one. Three lanes pad its 24 rows to 32, where it
// takes 16-47% more for every format; the other formats keep the register
// tile, 6-44% faster at one lane. Status: measured on the 40-core M3 Max;
// unmeasured on a 10-core Apple9 (U3), which takes it.
inline constexpr RegisterBoundary kApple9DecodeRegister{{{
    {0, 0, kAlways, 0},             // GridLookup
    {kAlways, 0, kAlways, 0},       // GridLookupOrQ2K
    {kAlways, kAlways, kAlways, kAlways},  // UnderFourBits
    {kAlways, kAlways, kAlways, kAlways},  // FourBitsOrMore
}}};

// Apple10 runs decode steps on the register tile on narrow grids: where the
// grid, a threadgroup per 64 columns (a gate/up pair's that may run in one
// pass staged) and lane, holds fewer than one and a half threadgroups per core,
// half that in formats of fewer than four bits per weight. There the staged
// tile's 64 threads per column tile leave its cores' loads too few and the
// register tile's 128 run more of them; its time grows with each lane, where
// the staged tile's matmuls take up to 16 rows at the cost of 8, and with its
// decode, where fewer bits leave the staged tile less to stream. Register
// against staged time on one lane's 8-row tile, best split of each tile:
// - 32-core M5 Max: 13-43% less below the bound (BF16 and Q8_0 320 x 10240,
//   Q4_K and Q5_K 512 x 2560, Q8_0 512 x 2048, Q6_K 1024 x 5120, Q4_K and
//   Q6_K 2048 x 4096); at 2560 x 6144 (1.25 per core) 6-9% less in Q4_K, Q5_K,
//   Q6_K, IQ4_XS, IQ4_NL and MXFP4, BF16 2%, Q8_0 the same, and 2-20% more in
//   Q3_K, IQ3_S, IQ3_XXS, Q2_K, Q2_0, IQ2_XS, IQ2_XXS and IQ1_M, which take
//   7-41% less below theirs (IQ2_XXS and Q3_K 1024 x 5120, IQ3_XXS, Q3_K and
//   Q2_K 512 x 2560, Q3_K 320 x 10240) and as much at 1.0 (Q3_K 2048 x 4096);
// - 20-core M5 Pro: 18-33% less at 320 and 512 columns, 4-9% more past the
//   bound (Q4_K and Q6_K 2560 x 6144, Q4_K 2048 x 4096); in fewer bits 23-33%
//   less below theirs (Q3_K, Q2_K, Q2_0 512 x 2560, Q3_K 320 x 10240), 3% and
//   14% more past (Q3_K and IQ2_XXS 1024 x 5120).
// A one-pass gate/up pair of 1024 x 5120 takes 3% (Q4_K) and 23% (IQ2_XXS)
// less time staged than in the register tile's two passes on the M5 Pro, 11%
// more (Q4_K, 1.0 per core) and 10% less (IQ2_XXS) on the M5 Max; one of
// 512 x 2560 (Q4_K) takes 17% and 25% less in the two. Status: measured.
inline constexpr RegisterBoundary kApple10DecodeRegister{{{
    {3, 3, 3, 3},  // GridLookup
    {3, 3, 3, 3},  // GridLookupOrQ2K
    {3, 3, 3, 3},  // UnderFourBits
    {6, 6, 6, 6},  // FourBitsOrMore
}}};

// Decode tiles walking K in lockstep all fetch the same slice of the input
// at each step, just written by the operator before and in no cache yet,
// and the fetch sits on every tile's critical path; starting each column
// tile at its own step lets most tiles find their slice brought in by
// others. On Apple10 and later the 27B decodes 4-10% faster end to end on a
// 40-core M5 Max and 4-8% on a 20-core M5 Pro (8-bit weights, whose tiles
// stream twice the bytes, 1% slower at one and two lanes and 1-2% faster
// at three and four on the M5 Max); a 12-core M6 and Apple9 gain nothing,
// lockstep as fast or faster. 16 cores lies between the measured 12 and 20.
// Status: measured at 12, 20 and 40 cores; the threshold between 12 and 20
// unmeasured (U7: no owner Mac has 13-19 cores).
inline constexpr uint32_t kApple10SpreadFromCores = 16;

// There a gate/up pair of one format may also run in one pass, each
// simdgroup staging both tensors' columns, where the wave law takes it: its
// threadgroups of one simdgroup reach DRAM's rate from fewer columns per core
// than two passes' and hold more per wave, and lose a second wave where the
// pair puts a few more than a wave on each core. On a 40-core M5 Max (13.6
// per core) the 27B pair alone took 8-9% less time in one pass in MLX 4-bit
// and Q4_K at one and two lanes, the 27B decode cycle 2.4-3.2% less in MLX
// 4-bit and 0.6-2.0% in UD-Q4_K_M; 8-bit, whose pair streams near DRAM's rate
// either way, 0.1-0.6% more. A 12-core M6 holds 45 of the pair's threadgroups
// per core, where one pass took 2-11% more: fewer cores run two passes.
// Status: as kApple10SpreadFromCores (U7).
inline constexpr uint32_t kApple10OnePassFromCores = 16;

// One lane of a projection whose quantized segments store fewer than four
// and a half bits per weight together runs unsplit on Apple10's 16-row
// tile, the second lane of its storage padding: the lane's rows are the
// bits of the 8-row tile's (gguf-projection checks it), in less time. On
// the 27B UD-IQ3_XXS's shapes, DRAM-cold and unsplit, a 12-core M6 takes
// 2-10% less (its fused inputs 8-10%, an IQ3_S down projection 1% more)
// and a 32-core M5 Max 5-7% less (a one-pass gate/up pair 10%), and the
// 27B decodes one lane 3.3-3.6% faster on the M6; Flash-Next's fused
// inputs take 7-9% less on a 20-core M5 Pro, its one-lane cycle about 1%.
// At 4.5 bits the M6 takes up to 1.6% more (Q4_K, the vocabulary head's).
// Split, the padding lane's partials double the split's traffic. At the
// same split the 16-row tile takes 2-8% less time on the M6 at 4 to 20
// threadgroups per core (IQ3_XXS and IQ2_XS, 5120 inputs, two and four
// partitions), but on the narrow grids where the law splits K most it
// takes 4-36% more on a 32-core M5 Max (IQ3_XXS, IQ2_XS and Q2_K in four
// and eight partitions on the 35B's 2048 x 4096, 2048 x 6144 and
// 2048 x 16384, Flash-Next's 2560 x 6144 and the 122B's 3072 x 8192, 1 to
// 1.5 threadgroups per core unsplit; IQ4_XS on the last), against 7% less
// to 2% more on the 27B's 5120 x 6144 and 5120 x 17408 in four. An unsplit
// gate/up pair runs it where both its tensors' own plans do. Status: measured.
inline constexpr uint32_t kApple10SixteenRowBelowBits16 = 72;

// Prefill: 128-row tiles. A chunk of up to 32 rows runs the staged tile
// of its rows (8, 16 or 32, two simdgroups) with the decode split law:
// the same half stage and matmul rows, so its outputs equal the prefill
// tile's up to the K split's fp32 reassociation (gguf-projection checks it),
// and each simdgroup streams its own 32 columns instead of four 8-row
// simdgroups sharing a stage. Unsplit, on a 17408 x 5120 Q4_K projection
// that is 1.8-2.9x faster on a 16-core M5 Pro (its neural accelerator pads
// 8 rows to 16) and 1.1-2.8x on a 40-core M3 Max; the 27B down and output
// projections split in two gain 24-37% more on the 16-core M5 Pro.
//
// Past 32 rows the prefill tile takes about as long for 33 rows as for 64:
// 0.58 ms on that Q4_K projection on the 16-core M5 Pro, against 0.25 ms
// for the 32-row staged tile. On Apple10 a chunk of up to 64 rows runs the
// staged tile's 64-row height, which reads the weights once: 0.48 ms there,
// and 6-43% less than the prefill tile over the 27B's Q4_K, MLX 4-bit and
// Q8_0 projections at the split law's choice (gguf-projection-benchmark
// prefill=33,48,64). Status: Apple9 33-64 rows unmeasured (U2), so it keeps
// the prefill tile.
inline constexpr uint32_t kApple9StagedPrefillRows = SPLASH_MAXIMUM_BATCH_WIDTH * SPLASH_TARGET_VERIFY_ROWS;
inline constexpr uint32_t kApple10StagedPrefillRows = 64;

inline constexpr ProjectionCalibration kApple9Projection{
    .registerSplits = TierLaw{kRegisterTiers},
    .stagedSplits = TierLaw{kApple9StagedTiers},
    .decodeRegister = kApple9DecodeRegister,
    .spreadFromCores = kNever,
    .onePassFromCores = kNever,
    .sixteenRowBelowBits16 = 0,
    .stagedPrefillRows = kApple9StagedPrefillRows,
};
inline constexpr ProjectionCalibration kApple10Projection{
    .registerSplits = TierLaw{kRegisterTiers},
    .stagedSplits = kApple10StagedLaw,
    .decodeRegister = kApple10DecodeRegister,
    .spreadFromCores = kApple10SpreadFromCores,
    .onePassFromCores = kApple10OnePassFromCores,
    .sixteenRowBelowBits16 = kApple10SixteenRowBelowBits16,
    .stagedPrefillRows = kApple10StagedPrefillRows,
};

// ---- Float projections ----------------------------------------------------

// The neural accelerator tile needs one (Apple9's matrix operations share the
// FP32 pipe, where three bf16 matmuls cost three fp32 ones) and enough of its
// 64-row by 32-column tiles per core. A tile runs K / 32 dependent steps, each
// staging the step's weights in threadgroup memory as bf16 parts and running
// a matmul per part (kernels/shared/gguf_float.metal), while the fp32 kernel
// spreads fewer rows over 8-column tiles with 8 K partitions each and
// finishes first below that grid. The grid it needs grows with that staging:
// 1.5 tiles per core for BF16 values, one part each, 2 for F32 values, three
// parts, and 3 for a quantized format's rows, decoded into three parts.
// Where the two tiles take the same time (gguf-projection-benchmark
// float-<format>, rows in steps of 64) on a 32-core M5 Max and a 12-core M6:
// F32 at 2.25-2.5 and 2.0-2.67 tiles per core for 256 columns of 2048 or 3072
// inputs, but 1.75-2.0 on the M6 and 2.0 and more on the M5 Max for the 64
// and 96 of GDN alpha/beta; quantized rows at 3.5 (8-bit affine) and
// 2.67-4.0 (8-bit affine, Q8_0, 4-bit affine, Q4_K); on a 20-core M5 Pro F32
// at about 2.2 for the 35B's router. Over those points the rule's tile takes
// at most 13% (F32) and 6% (quantized) longer than the faster one, where 1.5
// for every weight took up to 42% and 61% longer. BF16 crosses at 0.67-1.33
// (the M5 Pro: about 1.5 for Flash-Next's router), below its rule, which
// keeps a decode step's 16 to 32 rows on the simdgroup tile on 12 cores (a
// 512-column router: 1.6-1.8x faster there). Status: measured.
inline constexpr FloatTileCalibration kApple9FloatTile{false, 16, 3, 4, 6};
inline constexpr FloatTileCalibration kApple10FloatTile{true, 16, 3, 4, 6};

// ---- MoE ------------------------------------------------------------------

// Apple9 stages experts mostly in a GridLookup format: one 35B-shaped layer on
// a 40-core M3 Max decodes UD-Q2_K_XL's IQ2_XS and IQ3_XXS experts, and the
// IQ2, IQ3_XXS and IQ1 formats alone, 4-21% faster staged at B1-B4, where
// Q4_K/Q5_K, Q2_K and IQ4_XS experts take 3-34% longer. In decode, as Apple9's
// dense GGUF projections do: its matrix operations share the FP32 pipe, where
// the register tile beats staging. In prefill as last measured, on the 35B's
// real routes on the 40-core M3 Max (ms per layer): against the staged 32-row
// tiles, faster at 512 rows (3.52 vs 3.72) and slower at 2048 (12.6-13.1 vs
// 11.0-11.7). The M3 Max's numbers predate the staged tile's one gate/up pass
// and its column tile for an expert's lone tile of one or two rows: at their
// code the M4 too ran Q4_K/Q5_K, Q2_K and IQ4_XS experts 2-15% faster on the
// register tile, so larger GPUs, unmeasured since, keep it. Status:
// unmeasured since on Apple9 GPUs of more than 10 cores (U4); no such GPU is
// reachable, so today's choice stays in both phases.
inline constexpr MoeCalibration kApple9Moe{{false, true, true, true}};

// Apple9 GPUs of 10 cores or fewer (the base M3 and M4) stage experts in every
// format, in both phases: on a 10-core M4 such a layer in each of the 25
// formats the register tile takes, and one of Flash-Next's shape in 4-bit
// formats, take 12-45% less time staged at B1-B4 with rows routed at random,
// up to 17% less with each lane's rows routed among few experts (IQ4_NL at B1:
// 1% more) and 7-28% less on a 2048-row chunk (at most 4% more on 128- to
// 512-row chunks). Status: measured at 10 cores and on the 40-core M3 Max; the
// boundary between them unmeasured (U5).
inline constexpr uint32_t kApple9SmallGpuCores = 10;
inline constexpr MoeCalibration kApple9SmallMoe{{false, false, false, false}};

// Apple10 stages experts in every format.
inline constexpr MoeCalibration kApple10Moe{{false, false, false, false}};

inline KernelCalibration KernelCalibration::of(const DeviceCapabilities &device) noexcept {
  const uint32_t cores = device.plannedGpuCores();
  if (device.gpuClass() == GpuClass::Apple10) return {cores, kApple10Projection, kApple10FloatTile, kApple10Moe};
  return {cores, kApple9Projection, kApple9FloatTile, cores > kApple9SmallGpuCores ? kApple9Moe : kApple9SmallMoe};
}

// ---- Attention ------------------------------------------------------------

// Each prefill query tile's history splits: one per this many pages a tile's
// rows read (PagedAttention.cpp, prefillSplits). A chunk of a few hundred rows
// over a long history otherwise leaves much of the GPU idle: on an M5 Max the
// attention of a 256-row chunk after 32K-128K tokens ran 23-26% faster at
// these splits (Qwen3.6-35B-A3B's and Qwen3.8-27B's), of a 512-row chunk 23%
// and 10%, within 7% of the fastest split count. Status: measured on the
// 32-core M5 Max and the 20-core M5 Pro; unmeasured on Apple9 and on 12 cores.
inline constexpr uint32_t kPrefillPagesPerSplit = 32;

} // namespace splash::ops
