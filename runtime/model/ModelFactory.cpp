#include "ModelFactory.hpp"
#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"
#include <limits>

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

ModelPackage loadPackage(metal::MetalBackend &backend,
                         const std::filesystem::path &root,
                         ModelDescriptor descriptor, PreparationCheck prepareCheck = {}) {
  ModelPackage result;
  result.descriptor = std::move(descriptor);
  if (!result.descriptor.valid())
    throw std::invalid_argument("model descriptor is invalid");
  result.target = std::visit(
      [&](const auto &layout) -> TargetWeights {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(layout)>,
                                     Qwen3_8Layout>)
          return loadQwen3_8Weights(backend, root / "target", layout,
                                    result.descriptor.targetSource, prepareCheck);
        else
          return loadQwen3_6MoeWeights(backend, root / "target", layout,
                                       result.descriptor.targetSource, prepareCheck);
      },
      result.descriptor.target);
  result.draft = loadDFlashDraftWeights(
      backend, root / "draft", result.descriptor.draft);
  result.vision = loadQwenVisionWeights(
      backend, root / "vision", result.descriptor.vision);

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
                              const ModelDescriptor &descriptor, PreparationCheck prepareCheck) {
  return loadPackage(backend, root, descriptor, std::move(prepareCheck));
}

uint64_t preparedModelWeightBytes(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  uint64_t bytes = 0;
  if (descriptor.targetSource == TargetSource::Gguf) {
    const GgufFile source(findTargetGguf(root / "target"));
    bytes = std::visit([&](const auto &layout) {
      return gguf::ImagePlanner(source, ggufTargetGeometry(layout)).totalBytes();
    }, descriptor.target);
  } else if (descriptor.targetSource == TargetSource::Affine) {
    bytes = std::visit([](const auto &layout) { return preparedAffineBytes(layout); }, descriptor.target);
  }
  for (std::string_view directory : {"target", "draft", "vision"}) {
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
