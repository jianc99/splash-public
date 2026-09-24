#pragma once

// Source adapter for a Qwen GGUF. Preparation writes immutable cached files;
// serving uses the same read-only WeightFile mappings as packaged weights.

#include <filesystem>
#include <span>
#include <vector>

#include "model/GgufFile.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedFiles.hpp"

namespace splash::model {

// The single .gguf in a target directory (shards are not supported).
[[nodiscard]] std::filesystem::path findTargetGguf(const std::filesystem::path &directory);

class GgufTargetLoader final {
public:
  // Plans every image from the GGUF's metadata once. Before the first image
  // is written, the disk check budgets every missing image together with
  // alsoPrepared, the model's other prepared files.
  GgufTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &path,
                   const gguf::TargetGeometry &geometry, PreparationCheck admitConversion = {},
                   std::span<const PreparedWeight> alsoPrepared = {});
  GgufTargetLoader(const GgufTargetLoader &) = delete;
  GgufTargetLoader &operator=(const GgufTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();

private:
  [[nodiscard]] WeightFile open(size_t index);

  metal::MetalBackend &backend_;
  WeightSource source_;
  std::vector<gguf::Image> images_; // layers, head, embedding
  std::vector<PreparedWeight> weights_;
  PreparedFiles files_;
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
    geometry.experts = layout.experts;
    geometry.expertsPerToken = layout.expertsPerToken;
    geometry.expertIntermediateSize = layout.expertIntermediateSize;
  } else {
    geometry.intermediateSize = layout.intermediateSize;
  }
  return geometry;
}

} // namespace splash::model
