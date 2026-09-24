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
                                   gguf::TargetGeometry geometry, PreparationCheck admitConversion,
                                   std::span<const PreparedWeight> alsoPrepared)
    : backend_(&backend), admitConversion_(std::move(admitConversion)),
      source_(path, [&backend] { backend.checkOperation(); }) {
  const GgufFile file(source_);
  source_.checkUnchanged();
  // Validates the whole source before its tensor data is hashed.
  const gguf::ImagePlanner planner(file, geometry);
  std::vector<PreparedWeight> weights(alsoPrepared.begin(), alsoPrepared.end());
  const auto plan = [&](gguf::Image image) {
    backend.checkOperation();
    auto weight = ggufImageWeight(source_, image);
    weights.push_back(weight);
    images_.push_back({std::move(image), std::move(weight)});
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) plan(planner.layer(layer));
  plan(planner.head());
  plan(planner.embedding());
  cache_.requireSpace(weights, [&backend] { backend.checkOperation(); });
}

WeightFile GgufTargetLoader::layer(uint32_t index) {
  if (index + 2 >= images_.size()) throw GgufError("target layer is out of range");
  return build(images_[index]);
}

WeightFile GgufTargetLoader::head() { return build(images_[images_.size() - 2]); }

WeightFile GgufTargetLoader::embedding() { return build(images_.back()); }

WeightFile GgufTargetLoader::build(const Planned &planned) {
  const gguf::Image &image = planned.image;
  const auto path = cache_.prepare(planned.weight,
      [&](int destination, const PreparationCheck &admit) {
        writeGgufImage(*backend_, source_, destination, image, admit);
      },
      {[this] { backend_->checkOperation(); }, admitConversion_, [this] { source_.checkUnchanged(); }});
  return WeightFile(*backend_, path, planned.weight.component, kGgufImageMagic, image.layer, image.type,
                    planned.weight.key);
}

} // namespace splash::model
