#include "model/Qwen3_8.hpp"

#include <utility>

namespace splash::model {
namespace {

void validateLayout(const Qwen3_8Layout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.intermediateSize || !layout.attentionQueryHeads ||
      !layout.attentionKvHeads || !layout.attentionHeadDimension ||
      !layout.rotaryPairs || !(layout.rotaryTheta > 0.0F) ||
      !layout.fullAttentionPeriod) {
    throw WeightStoreError("Qwen3.8 layout contains a zero dimension");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.intermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.intermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
  if (layout.attentionQueryHeads * layout.attentionHeadDimension !=
      layout.attentionWidth) {
    throw WeightStoreError("Qwen3.8 attention layout is inconsistent");
  }
}

} // namespace

Qwen3_8Weights loadQwen3_8Weights(metal::MetalBackend &backend,
                                  const std::filesystem::path &directory,
                                  Qwen3_8Layout layout, TargetSource source, PreparationCheck admitConversion,
                                  std::span<const PreparedWeight> alsoPrepared) {
  validateLayout(layout);
  // The dense FFN reads the same projections from either format.
  const auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  return loadQwenTarget<Qwen3_8Weights>(backend, directory, layout, source, std::move(admitConversion),
                                        alsoPrepared, readFfn, readFfn);
}

} // namespace splash::model
