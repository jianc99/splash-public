#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"

#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace splash::model {

void requireCompatibleModelPackage(const ModelPackage &package) {
  if (!package.descriptor.valid() ||
      package.descriptor.draft != package.draft.layout ||
      !std::visit(
          [&](const auto &target) {
            return package.descriptor.target == TargetLayout{target.layout} &&
                   target.layout.vocabularySize ==
                       package.draft.layout.vocabularySize;
          },
          package.target)) {
    throw std::invalid_argument(
        "target and draft model interfaces are incompatible");
  }
}

namespace {

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_8Layout &layout,
                         const QwenTargetFiles<Qwen3_8Layout> &files) {
  return loadQwen3_8Weights(backend, layout, files);
}

TargetWeights readTarget(metal::MetalBackend &backend, const Qwen3_6MoeLayout &layout,
                         const QwenTargetFiles<Qwen3_6MoeLayout> &files) {
  return loadQwen3_6MoeWeights(backend, layout, files);
}

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const std::filesystem::path &root,
                         ModelDescriptor descriptor, PreparationCheck admitConversion = {}) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  const PreparationCheck check = [&backend] { backend.checkOperation(); };
  // Every prepared file of the model, the vision tower's and the target's, is
  // planned before the first is written, so one disk check budgets them all.
  std::vector<PreparedWeight> prepared;
  std::optional<VisionLoader> vision;
  if (result.descriptor.visionSource == VisionSource::Safetensors ||
      result.descriptor.visionSource == VisionSource::Gguf) {
    vision.emplace(root / "vision", result.descriptor.visionSource,
                   result.descriptor.vision, check, admitConversion);
    prepared.push_back(vision->weight());
  }
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        using Layout = std::remove_cvref_t<decltype(layout)>;
        const std::filesystem::path directory = root / "target";
        const auto read = [&](const QwenTargetFiles<Layout> &files,
                              std::span<const PreparedWeight> target) {
          prepared.insert(prepared.end(), target.begin(), target.end());
          if (!prepared.empty()) PreparedWeights().requireSpace(prepared, check);
          return readTarget(backend, layout, files);
        };
        switch (result.descriptor.targetSource) {
        case TargetSource::Packed:
          return read(PackedTargetFiles<Layout>{backend, directory, layout}, {});
        case TargetSource::Affine: {
          AffineTargetLoader loader(backend, directory, layout, admitConversion);
          return read(loader, loader.weights());
        }
        case TargetSource::Gguf: {
          GgufTargetLoader loader(backend, findTargetGguf(directory), ggufTargetGeometry(layout),
                                  admitConversion);
          return read(loader, loader.weights());
        }
        }
        throw std::invalid_argument("unknown target source");
      },
      result.descriptor.target);
  result.draft = loadDFlashDraftWeights(
      backend, root / "draft", result.descriptor.draft);
  if (vision)
    result.vision = loadQwenVisionWeights(backend, *vision);
  else if (result.descriptor.visionSource == VisionSource::Packed)
    result.vision = loadQwenVisionWeights(backend, root / "vision", result.descriptor.vision);

  std::vector<WeightFileRecord> records(result.targetFiles().begin(),
                                        result.targetFiles().end());
  records.insert(records.end(), result.draft.files.begin(),
                 result.draft.files.end());
  records.insert(records.end(), result.vision.files.begin(),
                 result.vision.files.end());
  result.manifestFingerprintSha256 = weightManifestFingerprint(records);
  requireCompatibleModelPackage(result);
  return result;
}

} // namespace

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root) {
  return loadPackage(backend, root, inspectModelPackage(root));
}

ModelPackage loadModelPackage(metal::MetalBackend &backend,
                              const std::filesystem::path &root,
                              const ModelDescriptor &descriptor, PreparationCheck admitConversion) {
  return loadPackage(backend, root, descriptor, std::move(admitConversion));
}

uint64_t preparedModelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    WeightSource source(findTargetGguf(root / "target"));
    const GgufFile file(source);
    bytes = std::visit([&](const auto &layout) {
      return gguf::ImagePlanner(file, ggufTargetGeometry(layout)).totalBytes();
    }, descriptor.target);
  } else if (descriptor.targetSource == TargetSource::Affine) {
    bytes = std::visit([](const auto &layout) { return preparedAffineBytes(layout); }, descriptor.target);
  }
  if (descriptor.visionSource == VisionSource::Safetensors || descriptor.visionSource == VisionSource::Gguf)
    bytes += preparedVisionBytes(descriptor.vision);
  for (std::string_view directory : {"target", "draft", "vision"}) {
    if (directory == "vision" && descriptor.visionSource != VisionSource::Packed) continue;
    if (directory == "target" && descriptor.targetSource != TargetSource::Packed) continue;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(root / directory)) {
      if (!entry.is_regular_file()) continue;
      const uint64_t size = entry.file_size();
      if (size > std::numeric_limits<uint64_t>::max() - bytes) throw std::overflow_error("model weight size overflows");
      bytes += size;
    }
  }
  if (!bytes) throw std::invalid_argument("model package contains no regular files");
  return bytes;
}

} // namespace splash::model
