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
  const GgufFile file(std::move(path));
  source_.checkUnchanged();
  dataOffset_ = file.dataOffset();
  const gguf::ImagePlanner planner(file, geometry);
  std::vector<PreparedWeight> weights(alsoPrepared.begin(), alsoPrepared.end());
  const auto plan = [&](gguf::Image image) {
    backend.checkOperation();
    auto key = ggufImageKey(source_.digest(), image);
    weights.push_back({key, image.bytes});
    images_.push_back({std::move(image), std::move(key)});
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
  backend_->checkOperation();
  source_.checkUnchanged();
  const auto path = cache_.prepare({planned.key, image.bytes, image.name, source_.path().string()},
      [&](int destination) {
        prepareGgufImage(*backend_, source_.descriptor(), dataOffset_, destination, image,
                         [&] { backend_->checkOperation(); if (admitConversion_) admitConversion_(); });
        source_.checkUnchanged();
      },
      [&] { backend_->checkOperation(); }, admitConversion_);
  source_.checkUnchanged();
  return WeightFile(*backend_, path, "target/" + image.name, kGgufImageMagic, image.layer, image.type,
                    planned.key);
}

} // namespace splash::model
