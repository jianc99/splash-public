#pragma once

#include "metal/DeviceCapabilities.hpp"
#include "metal/abi/ExecutionGeometry.h"
#include "metal/abi/Gguf.h"

#include <cstdint>
#include <span>

namespace splash::ops {

// What the projection kernels' policy reads of the device, and the laws their
// kernel families share (dev/benchmarks/device-policy.md). The GPU family
// decides the primitive only. Each family below holds the constants of its
// row quanta and split law, with the machine and date they were measured on.

// Apple9 has no per-core matrix unit: its register tiles run bf16 simdgroup
// matrix operations on the FP32 pipe. Apple10 and later run MPP tensor tiles
// on each core's neural accelerator. The runtime refuses families below 9 at
// startup; an unknown family (0, in tests) takes the tensor tiles.
enum class Primitive : uint8_t { Register, Tensor };

// Missing IORegistry core metadata: one estimate for every kernel, within the
// 16-40-core range of the measured machines. A fallback, not a calibrated
// optimum; a reported count always wins.
inline constexpr uint32_t kAssumedGpuCores = 32;

// The device as the kernel policies see it, derived once from its capabilities.
struct DevicePolicy final {
  Primitive primitive;
  uint32_t cores;
  explicit DevicePolicy(const DeviceCapabilities &device) noexcept;
};

// The rows a kernel family's tile runs for `rows` rows: whole multiples of
// `quantum`, or with `doubling`, the smallest of quantum, 2 quantum,
// 4 quantum, ... that holds them.
struct RowQuanta final {
  uint32_t quantum;
  bool doubling = false;
  [[nodiscard]] constexpr uint32_t round(uint32_t rows) const noexcept {
    if (!doubling) return (rows + quantum - 1) / quantum * quantum;
    uint32_t height = quantum;
    while (height < rows) height *= 2;
    return height;
  }
};

// One tier of a split law: it asks for twice the K partitions while the grid
// holds fewer than `groupsPerCore` threadgroups per core (at most that many
// for an inclusive law) and twice the partitions would each keep `inputs`
// inputs.
struct SplitTier final {
  uint32_t groupsPerCore;
  uint32_t inputs;
};

// A kernel family's split law: its tiers on each primitive (none: it never
// splits there), the comparator, and its partitions of whole
// `partitionInputs`-input steps, equal in size or differing by at most one.
struct SplitLaw final {
  std::span<const SplitTier> registerTiers, tensorTiers;
  bool inclusive = false;
  uint32_t partitionInputs = 0;
  bool evenPartitions = false;
  [[nodiscard]] constexpr std::span<const SplitTier> tiers(Primitive primitive) const noexcept {
    return primitive == Primitive::Register ? registerTiers : tensorTiers;
  }
};

struct KernelFamily final {
  RowQuanta rows;
  SplitLaw split;
};

// The projection kernel families of Linear.cpp (affine Q4) and LinearGguf.cpp.

// Apple9's affine register tile (decode_linear_q4_sg*), one lane per
// threadgroup: decode steps, and prefill chunks padded to whole lanes. Splits
// aim for sixteen independent column/K threadgroups per core, in equal
// partitions of at least twelve 64-input quant groups to amortize the
// reduction: 40-core M3 Max, 2026-09-20 (2c788c0; apple9-simdgroup.md: 27B
// decode cycle 89 -> 60 ms, 35B 27 -> 25 ms).
inline constexpr SplitTier kAffineRegisterTiers[] = {{16, 768}};
inline constexpr KernelFamily kAffineRegisterTile{{SPLASH_TARGET_VERIFY_ROWS},
                                                  {kAffineRegisterTiers, {}, false, 64, true}};

// Apple10's sequential affine MPP decode tiles (decode_linear_q4_n128, _n256
// and their paired and one-lane split forms): 8, 16, 24 or 32 rows, exactly.
inline constexpr KernelFamily kAffineTensorDecode{{SPLASH_TARGET_VERIFY_ROWS}, {}};

// Split128 (decode_linear_q4_n128_split*): the N128 tile (256 threads), every
// lane in each threadgroup, over K partitions of whole 256-input blocks. It
// splits while the grid holds at most two threadgroups per core, so the split
// grid fits four per core (1024 threads, twice the 512-thread occupancy
// knee), with at least one block per partition. Measured DRAM-cold on a
// 20-core M5 Pro, 2026-09-27 (b6752e1), over the 27B and 35B MLX 4-bit decode
// projections and their drafts at one to four lanes, and 16 and 40 cores
// emulated by width: grids that only reach the knee leave time (the 20 tiles
// of 2560 x 4096 take 0.48-0.60 of the sequential time with four splits,
// 0.60-0.71 with two), and a grid past four per core loses to its second wave
// (6144 x 5120 in two splits is 9% slower at one lane). The law is within 5%
// of each shape's fastest split in 145 of 172 shapes and lanes and never
// slower than the sequential tiles; partitions longer than one block gained
// nothing measurable.
inline constexpr SplitTier kAffineSplitTiers[] = {{2, 256}};
inline constexpr KernelFamily kAffineTensorSplit{{SPLASH_TARGET_VERIFY_ROWS},
                                                 {{}, kAffineSplitTiers, true, 256, false}};

// The affine MPP prefill tiles (prefill_linear_q4_*).
inline constexpr uint32_t kAffinePrefillTileRows = 32;
inline constexpr KernelFamily kAffineTensorPrefill{{kAffinePrefillTileRows}, {}};

// Apple9's GGUF register tile (gguf_decode_sg_*, 128 threads), every lane in
// each threadgroup, over K partitions of whole 256-input coefficient units.
// Four of its threadgroups are resident on a core at once: on a 40-core M3
// Max its time steps every four per core (Q4_K, K = 8192, one lane, ms: 3 per
// core 0.156, 4 0.157, 5 0.220, 7 0.281, 8 0.286; the same steps at two to
// four lanes and for Q8_0). Below one wave a core must fill it, down to one
// unit per partition; below eight waves more threadgroups shrink the last
// wave's tail while partitions of 1024 inputs amortize the partial sums (flat
// from eight to 32 waves). Over every 27B and 35B projection kind at one to
// four lanes and 10-80 cores emulated by width, 2026-09-24 (417a4fc), the
// decode step's projections run 0.95% slower than the fastest split of each
// shape on average and 2.3% at worst (sixteen threadgroups per core with two
// units per partition: 2.8%, 7.8%).
inline constexpr SplitTier kGgufRegisterTiers[] = {{4, 256}, {32, 1024}};
inline constexpr KernelFamily kGgufRegisterTile{{SPLASH_TARGET_VERIFY_ROWS},
                                                {kGgufRegisterTiers, {}, false, 256, false}};

// The GGUF staged decode tile (gguf_decode_*_m<rows>, 64 threads), which also
// runs prefill chunks of up to 32 rows: MPP computes 16-row fragments, so it
// holds 8, 16 or 32 rows, and a three-lane step runs the 32-row tile over four
// lanes of storage. It splits K in equal partitions of whole 32-input groups.
// On Apple10 one fitted tier, six threadgroups per core with 512 inputs per
// partition. Six is not a residency (12-17 of these threadgroups run at once
// per core on the M5 Pro): past it a core's memory and neural accelerator are
// busy and more partitions only add reduction. Over the 27B and 35B dense
// shapes, all formats, one to four lanes, on the 16- and 20-core M5 Pro and
// 10-, 30- and 40-core GPUs emulated by width, 2026-09-24 (417a4fc): 3.6% over
// the fastest split of each shape in total and 36% at worst on a 15-us shape
// (the register tiers in threads per core: 6.6%; the previous 32 per core with
// 1024 inputs and unsplit fused and gate/up kernels: 6.4%). Apple9 cores take
// as many of its threadgroups as of the register tile's, and its tiers: on a
// 40-core M3 Max, 2026-09-25 (1bf892b), over the 27B and 35B dense shapes at
// one to four lanes they come within 0-9% of each shape's fastest split (20%
// at one lane on 2048 x 512, a 0.012 ms projection), where the Apple10 tier is
// 13-27% slower on 17408 x 5120 and 12288 x 5120 and 50% on 2048 x 512.
inline constexpr SplitTier kGgufStagedTiers[] = {{6, 512}};
inline constexpr KernelFamily kGgufStagedTile{{SPLASH_TARGET_VERIFY_ROWS, true},
                                              {kGgufRegisterTiers, kGgufStagedTiers, false, GGUF_STAGED_STEP, true}};

// The GGUF staged prefill tile (gguf_prefill_*).
inline constexpr KernelFamily kGgufStagedPrefill{{GGUF_PREFILL_ROWS}, {}};

// The K partitions of a family's tile over a grid of `grid` threadgroups per
// partition. The count doubles, up to eight, while a tier of the device's
// primitive asks and the doubled partitions keep the family's steps. It reads
// the grid per core and K, never the rows, so a request's sums are the same
// alone and batched.
[[nodiscard]] uint32_t splitK(const KernelFamily &family, const DevicePolicy &device, uint32_t grid,
                              uint32_t inputs) noexcept;

} // namespace splash::ops
