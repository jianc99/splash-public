#include "model/QwenVision.hpp"

#include <string>
#include <utility>

namespace splash::model {
namespace {

void validateLayout(const ops::VisionLayout &layout) {
  if (!layout.depth || !layout.hiddenSize || !layout.patchDimension ||
      !layout.intermediateSize || !layout.paddedIntermediateSize ||
      !layout.mergedHiddenSize || !layout.outputHiddenSize || !layout.heads ||
      !layout.headDimension || !layout.positionGridSide || !layout.patchSize ||
      !layout.spatialMerge ||
      layout.heads * layout.headDimension != layout.hiddenSize ||
      layout.paddedIntermediateSize < layout.intermediateSize ||
      layout.mergedHiddenSize !=
          layout.hiddenSize * layout.spatialMerge * layout.spatialMerge ||
      layout.patchDimension != 3 * 2 * layout.patchSize * layout.patchSize) {
    throw WeightStoreError("Qwen vision layout is inconsistent");
  }
}

class VisionReader {
public:
  VisionReader(WeightFile &file,
               const std::vector<ops::VisionPrecision> &precision)
      : file_(file), precision_(precision) {}
  std::pair<metal::MetalBuffer, ops::VisionPrecision>
  tensor(uint64_t elements, std::string_view label) {
    if (cursor_ == precision_.size())
      throw WeightStoreError("missing vision storage type");
    const auto type = precision_[cursor_++];
    const uint64_t unit = type == ops::VisionPrecision::Float32 ? 4 : 2;
    return {file_.section(
                checkedWeightMultiply(elements, unit, "vision tensor bytes"),
                label),
            type};
  }
  ops::VisionAffine affine(uint32_t rows, uint32_t columns,
                           std::string_view label) {
    auto [weight, weightType] = tensor(uint64_t(rows) * columns, label);
    auto [bias, biasType] = tensor(rows, label);
    return {std::move(weight), std::move(bias), weightType, biasType};
  }
  ops::VisionNorm norm(uint32_t width, std::string_view label) {
    auto [weight, weightType] = tensor(width, label);
    auto [bias, biasType] = tensor(width, label);
    if (weightType != biasType)
      throw WeightStoreError("vision norm storage types differ");
    return {std::move(weight), std::move(bias), weightType};
  }

private:
  WeightFile &file_;
  const std::vector<ops::VisionPrecision> &precision_;
  size_t cursor_ = 0;
};

} // namespace

QwenVisionWeights loadQwenVisionWeights(metal::MetalBackend &backend,
                                        const std::filesystem::path &directory,
                                        ops::VisionLayout layout,
                                        VisionSource source,
                                        PreparationCheck prepareCheck) {
  validateLayout(layout);
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  QwenVisionWeights result;
  result.tensors.layout = layout;
  result.tensors.blocks.reserve(layout.depth);

  const auto prepared = prepareVisionWeights(
      directory, source, layout, [&backend] { backend.checkOperation(); },
      prepareCheck);
  WeightFile file(backend, prepared.path, "vision/model.bin", kVisionMagic,
                  layout.depth, source == VisionSource::Gguf ? 1 : 0,
                  source == VisionSource::Packed
                      ? ""
                      : prepared.path.parent_path().filename().string());
  VisionReader reader(file, prepared.precision);
  result.tensors.patchEmbedding =
      reader.affine(layout.hiddenSize, layout.patchDimension, "patch-embed");
  auto [position, positionType] =
      reader.tensor(uint64_t(layout.positionGridSide) *
                        layout.positionGridSide * layout.hiddenSize,
                    "position-table");
  result.tensors.positionTable = std::move(position);
  result.tensors.positionPrecision = positionType;

  for (uint32_t blockIndex = 0; blockIndex < layout.depth; ++blockIndex) {
    ops::VisionBlock block;
    block.norm1 = reader.norm(layout.hiddenSize, "norm1");
    block.qkv = reader.affine(3 * layout.hiddenSize, layout.hiddenSize, "qkv");
    block.projection =
        reader.affine(layout.hiddenSize, layout.hiddenSize, "proj");
    block.norm2 = reader.norm(layout.hiddenSize, "norm2");
    block.upProjection =
        reader.affine(layout.paddedIntermediateSize, layout.hiddenSize, "fc1");
    block.downProjection =
        reader.affine(layout.hiddenSize, layout.paddedIntermediateSize, "fc2");
    result.tensors.blocks.push_back(std::move(block));
  }
  result.tensors.mergerNorm = reader.norm(layout.hiddenSize, "merger-norm");
  result.tensors.mergerUpProjection = reader.affine(
      layout.mergedHiddenSize, layout.mergedHiddenSize, "merger-fc1");
  result.tensors.mergerDownProjection = reader.affine(
      layout.outputHiddenSize, layout.mergedHiddenSize, "merger-fc2");
  file.finish();
  result.files.push_back(file.record());

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

} // namespace splash::model
