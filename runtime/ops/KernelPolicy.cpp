#include "ops/KernelPolicy.hpp"

#include "ops/Linear.hpp"

#include <algorithm>

namespace splash::ops {

DevicePolicy::DevicePolicy(const DeviceCapabilities &device) noexcept
    : primitive(device.appleGpuFamily == 9 ? Primitive::Register : Primitive::Tensor),
      cores(device.gpuCoreCount ? device.gpuCoreCount : kAssumedGpuCores) {}

uint32_t splitK(const KernelFamily &family, const DevicePolicy &device, uint32_t grid, uint32_t inputs,
                uint32_t rows) noexcept {
  const SplitLaw &law = family.split;
  const std::span<const SplitTier> tiers = law.tiers(device.primitive);
  uint32_t splits = 1;
  const auto partitioned = [&](uint32_t partitions) {
    return law.evenPartitions ? inputs % (uint64_t{partitions} * law.partitionInputs) == 0
                              : inputs / law.partitionInputs >= partitions;
  };
  const auto asks = [&](const SplitTier &tier) {
    const uint64_t threadgroups = uint64_t{grid} * splits, target = uint64_t{tier.groupsPerCore} * device.cores;
    return tier.holds(rows) && (law.inclusive ? threadgroups <= target : threadgroups < target) &&
           inputs >= uint64_t{2 * splits} * tier.inputs;
  };
  while (splits < LinearConfig::kMaximumSplits && std::any_of(tiers.begin(), tiers.end(), asks) &&
         partitioned(2 * splits))
    splits *= 2;
  return splits;
}

} // namespace splash::ops
