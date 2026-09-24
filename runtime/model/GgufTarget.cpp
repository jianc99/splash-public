#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"

namespace splash::model {
std::filesystem::path findTargetGguf(const std::filesystem::path &directory) {
  std::filesystem::path found;
  std::error_code error;
  for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
    if (entry.path().extension() != ".gguf") continue;
    if (!found.empty()) throw GgufError("target directory holds more than one GGUF: " + directory.string());
    found = entry.path();
  }
  if (error) throw GgufError("cannot list target directory: " + directory.string());
  if (found.empty()) throw GgufError("target directory holds no GGUF: " + directory.string());
  return found;
}

GgufTargetLoader::GgufTargetLoader(metal::MetalBackend &backend, std::filesystem::path path,
                                   gguf::TargetGeometry geometry, PreparationCheck check,
                                   std::span<const PreparedWeight> alsoPrepared)
    : backend_(&backend), check_(std::move(check)), source_(path, [&backend] { backend.checkOperation(); }),
      file_(std::move(path)), planner_(file_, geometry) {
  source_.checkUnchanged();
  std::vector<PreparedWeight> weights(alsoPrepared.begin(), alsoPrepared.end());
  const auto include = [&](const gguf::Image &image) {
    weights.push_back({ggufImageKey(source_.digest(), image), image.bytes});
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    backend.checkOperation();
    include(planner_.layer(layer));
  }
  include(planner_.head());
  include(planner_.embedding());
  cache_.requireSpace(weights, [&backend] { backend.checkOperation(); });
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  backend_->checkOperation();
  const gguf::Image image = planner_.layer(index);
  return build(image, index, planner_.geometry().isFullAttentionLayer(index) ? 1u : 0u);
}

WeightFile GgufTargetLoader::head() {
  backend_->checkOperation();
  return build(planner_.head(), planner_.geometry().layers, 2u);
}

WeightFile GgufTargetLoader::embedding() {
  backend_->checkOperation();
  return build(planner_.embedding(), planner_.geometry().vocabularySize,
               planner_.geometry().hiddenSize);
}

WeightFile GgufTargetLoader::build(const gguf::Image &image, uint32_t expectedLayer,
                                   uint32_t expectedType) {
  source_.checkUnchanged();
  const auto key = ggufImageKey(source_.digest(), image);
  const auto path = cache_.prepare({key, image.bytes, image.name, file_.path().string()},
      [&](int destination) {
        prepareGgufImage(*backend_, source_.descriptor(), destination, image, check_);
        source_.checkUnchanged();
      },
      [&] { backend_->checkOperation(); }, check_);
  source_.checkUnchanged();
  return WeightFile(*backend_, path, "target/" + image.name, kGgufImageMagic,
                    expectedLayer, expectedType, key);
}

} // namespace splash::model
