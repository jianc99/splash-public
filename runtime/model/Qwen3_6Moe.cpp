#include "Qwen3_6Moe.hpp"

#include "model/GgufTarget.hpp"
#include "model/AffineTarget.hpp"

#include <string_view>

namespace splash::model {
namespace {

constexpr std::string_view kHeadMagic = "MDFM0002";

void requireLayout(const Qwen3_6MoeLayout &layout) {
  if (!layout.maximumContextTokens || !layout.layers || !layout.hiddenSize ||
      !layout.vocabularySize || !layout.packedGdnWidth ||
      !layout.packedFullWidth || !layout.convolutionDimension ||
      !layout.gdnKeyHeads || !layout.gdnValueHeads ||
      !layout.gdnHeadDimension || !layout.attentionWidth ||
      !layout.attentionQueryHeads || !layout.attentionKvHeads ||
      !layout.attentionHeadDimension || !layout.rotaryPairs ||
      !(layout.rotaryTheta > 0.0F) || !layout.fullAttentionPeriod ||
      !layout.experts || !layout.expertsPerToken ||
      !layout.expertIntermediateSize) {
    throw WeightStoreError("Qwen3.6 MoE layout contains a zero dimension");
  }
  if (layout.gdnValueHeads % layout.gdnKeyHeads ||
      layout.convolutionDimension !=
          (2 * layout.gdnKeyHeads + layout.gdnValueHeads) *
              layout.gdnHeadDimension ||
      layout.attentionWidth !=
          layout.attentionQueryHeads * layout.attentionHeadDimension ||
      layout.packedFullWidth !=
          2 * layout.attentionWidth +
              2 * layout.attentionKvHeads * layout.attentionHeadDimension ||
      layout.expertsPerToken > layout.experts ||
      layout.hiddenCaptureLayers.back() >= layout.layers ||
      !layout.kvLayout().valid() || !layout.gdnStateLayout().valid()) {
    throw WeightStoreError("Qwen3.6 MoE layout is inconsistent");
  }
  validateQ4Layout(layout.packedGdnWidth, layout.hiddenSize);
  validateQ4Layout(layout.packedFullWidth, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.attentionWidth);
  validateQ4Layout(layout.expertIntermediateSize, layout.hiddenSize);
  validateQ4Layout(layout.hiddenSize, layout.expertIntermediateSize);
  validateQ4Layout(layout.vocabularySize, layout.hiddenSize);
}

} // namespace

Qwen3_6MoeWeights
loadQwen3_6MoeWeights(metal::MetalBackend &backend,
                      const std::filesystem::path &directory,
                      Qwen3_6MoeLayout layout, TargetSource source, PreparationCheck prepareCheck) {
  requireLayout(layout);
  if (source == TargetSource::Gguf) {
    // Source-specific preparation ends at immutable WeightFile views.
    GgufTargetLoader loader(backend, findTargetGguf(directory),
                            ggufTargetGeometry(layout), std::move(prepareCheck));
    return readQwenTargetWeights<Qwen3_6MoeWeights>(
        backend, layout, GgufTargetFiles{loader},
        [](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
          ops::BlockMoeWeights ffn;
          ffn.router = readQuantizedSegment(file, "router");
          ffn.gate.routed = readQuantizedSegment(file, "experts-gate");
          ffn.up.routed = readQuantizedSegment(file, "experts-up");
          ffn.down.routed = readQuantizedSegment(file, "experts-down");
          ffn.gate.shared = readQuantizedSegment(file, "shared-expert-gate");
          ffn.up.shared = readQuantizedSegment(file, "shared-expert-up");
          ffn.down.shared = readQuantizedSegment(file, "shared-expert-down");
          ffn.sharedExpertGate =
              readQuantizedSegment(file, "shared-expert-scalar-gate");
          layer.ffn = std::move(ffn);
        },
        true);
  }
  auto readFfn = [&](WeightFile &file, Qwen3_6MoeLayerWeights &layer) {
        layer.ffn.affine().router = readQ8Projection(
            file, backend, layout.experts, layout.hiddenSize, "router");
        layer.ffn.affine().expertGate = readExpertProjection(
            file, layout.experts, layout.expertIntermediateSize,
            layout.hiddenSize, "experts-gate");
        layer.ffn.affine().expertUp = readExpertProjection(
            file, layout.experts, layout.expertIntermediateSize,
            layout.hiddenSize, "experts-up");
        layer.ffn.affine().expertDown = readExpertProjection(
            file, layout.experts, layout.hiddenSize,
            layout.expertIntermediateSize, "experts-down");
        layer.ffn.affine().sharedGate = readExpertProjection(
            file, 1, layout.expertIntermediateSize, layout.hiddenSize,
            "shared-expert-gate");
        layer.ffn.affine().sharedUp = readExpertProjection(
            file, 1, layout.expertIntermediateSize, layout.hiddenSize,
            "shared-expert-up");
        layer.ffn.affine().sharedDown = readExpertProjection(
            file, 1, layout.hiddenSize, layout.expertIntermediateSize,
            "shared-expert-down");
        layer.ffn.affine().sharedExpertGate = readQ8Projection(
            file, backend, kQ4StorageN, layout.hiddenSize,
            "shared-expert-scalar-gate");
      };
  if (source == TargetSource::Affine) {
    AffineTargetLoader loader(backend, directory, layout, std::move(prepareCheck));
    return readQwenTargetWeights<Qwen3_6MoeWeights>(backend, layout, loader, readFfn, false);
  }
  return loadQwenTargetWeights<Qwen3_6MoeWeights>(backend, directory, layout, kHeadMagic, readFfn);
}

} // namespace splash::model
