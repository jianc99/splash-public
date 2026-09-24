#pragma once

#include "model/PreparedWeights.hpp"
#include "ops/Vision.hpp"

namespace splash::model {

enum class VisionSource : uint8_t { Packed, Safetensors, Gguf, None };

struct PreparedVision {
  std::filesystem::path path;
  std::vector<ops::VisionPrecision> precision;
  uint64_t bytes = 0;
};

[[nodiscard]] uint64_t
preparedVisionBytes(const std::filesystem::path &directory, VisionSource source,
                    const ops::VisionLayout &layout);
[[nodiscard]] PreparedVision
prepareVisionWeights(const std::filesystem::path &directory,
                     VisionSource source, const ops::VisionLayout &layout,
                     const PreparationCheck &check = {},
                     const PreparationCheck &prepareCheck = {});

} // namespace splash::model
