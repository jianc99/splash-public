#pragma once

// Source adapter for a Qwen GGUF. Preparation writes immutable cached files;
// serving uses the same read-only WeightFile mappings as packaged weights.

#include <filesystem>
#include <memory>

#include "metal/MetalBackend.hpp"
#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedWeights.hpp"
#include "model/WeightStore.hpp"

namespace splash::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  GgufTargetLoader(metal::MetalBackend &backend, std::filesystem::path path,
                   gguf::TargetGeometry geometry, PreparationCheck check = {});
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  [[nodiscard]] const GgufFile &file() const noexcept { return file_; }
  [[nodiscard]] const gguf::ImagePlanner &planner() const noexcept { return planner_; }

private:
  [[nodiscard]] WeightFile build(const gguf::Image &image, uint32_t expectedLayer,
                                 uint32_t expectedType);

  metal::MetalBackend *backend_;
  PreparationCheck check_;
  WeightSource source_;
  GgufFile file_;
  gguf::ImagePlanner planner_;
  PreparedWeights cache_;
};

// The GGUF geometry of a Qwen layout: a dense FFN, or a sparse MoE when the
// layout has experts.
template <class Layout>
[[nodiscard]] gguf::TargetGeometry ggufTargetGeometry(const Layout &layout) {
  gguf::TargetGeometry geometry;
  geometry.layers = layout.layers;
  geometry.hiddenSize = layout.hiddenSize;
  geometry.vocabularySize = layout.vocabularySize;
  geometry.gdnKeyHeads = layout.gdnKeyHeads;
  geometry.gdnValueHeads = layout.gdnValueHeads;
  geometry.gdnHeadDimension = layout.gdnHeadDimension;
  geometry.convolutionDimension = layout.convolutionDimension;
  geometry.attentionWidth = layout.attentionWidth;
  geometry.attentionKvHeads = layout.attentionKvHeads;
  geometry.attentionHeadDimension = layout.attentionHeadDimension;
  geometry.fullAttentionPeriod = layout.fullAttentionPeriod;
  if constexpr (requires { layout.experts; }) {
    geometry.intermediateSize = 0;
    geometry.experts = layout.experts;
    geometry.expertsPerToken = layout.expertsPerToken;
    geometry.expertIntermediateSize = layout.expertIntermediateSize;
  } else {
    geometry.intermediateSize = layout.intermediateSize;
  }
  return geometry;
}

// readQwenTargetWeights' files (QwenTarget.hpp): the images of a loader.
struct GgufTargetFiles final {
  GgufTargetLoader &loader;
  [[nodiscard]] WeightFile layer(uint32_t index, bool) const { return loader.layer(index); }
  [[nodiscard]] WeightFile head(uint32_t) const { return loader.head(); }
  [[nodiscard]] WeightFile embedding(uint32_t, uint32_t) const { return loader.embedding(); }
};

} // namespace splash::model
