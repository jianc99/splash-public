#pragma once

#include "model/PreparedWeights.hpp"
#include "ops/Vision.hpp"

#include <filesystem>
#include <memory>

namespace splash::model {

enum class VisionSource : uint8_t { Packed, Safetensors, Gguf, None };

// Source adapter for the vision tower: the vision_tower.* tensors of an MLX
// checkpoint or a GGUF mmproj. Both prepare the packed vision/model.bin layout
// (MDFV0001, every tensor BF16). A BF16 tensor is copied; an F32 or F16 tensor
// is converted only when every value is exactly a BF16, and preparation fails
// otherwise. Construction validates the source's metadata and plans the
// prepared file; tensor values are read only when preparing.
class VisionPreparation final {
public:
  VisionPreparation(const std::filesystem::path &directory, VisionSource source,
                    const ops::VisionLayout &layout, PreparationCheck check = {});
  ~VisionPreparation();
  VisionPreparation(const VisionPreparation &) = delete;
  VisionPreparation &operator=(const VisionPreparation &) = delete;

  [[nodiscard]] const ops::VisionLayout &layout() const noexcept;
  // The prepared file's cache identity and size, for disk budgeting.
  [[nodiscard]] const PreparedWeight &weight() const noexcept;
  // The prepared file, reused or written now. admitConversion admits the
  // conversion workspace on a cache miss; check (the constructor's) runs on
  // every load.
  [[nodiscard]] std::filesystem::path
  prepare(const PreparationCheck &admitConversion = {}) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Bytes of the packed layout, which every source prepares.
[[nodiscard]] uint64_t preparedVisionBytes(const ops::VisionLayout &layout);

} // namespace splash::model
