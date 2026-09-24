#include "model/Qwen3_8.hpp"
#include "model/QwenTargetLoader.hpp"

namespace splash::model {

Qwen3_8Weights loadQwen3_8Weights(metal::MetalBackend &backend, Qwen3_8Layout layout,
                                  const QwenTargetFiles<Qwen3_8Layout> &files) {
  // The dense FFN reads the same projections from either format.
  const auto readFfn = [&](WeightFile &file, Qwen3_8LayerWeights &layer, const auto &format) {
    layer.gateProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-gate");
    layer.upProjection =
        format.projection(file, layout.intermediateSize, layout.hiddenSize, "mlp-up");
    layer.downProjection =
        format.projection(file, layout.hiddenSize, layout.intermediateSize, "mlp-down");
  };
  return loadQwenTarget<Qwen3_8Weights>(backend, layout, files, readFfn);
}

} // namespace splash::model
