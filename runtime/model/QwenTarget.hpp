#pragma once

#include "Model.hpp"
#include "StateLayout.hpp"
#include "WeightStore.hpp"
#include "model/AffineTarget.hpp"
#include "model/GgufTarget.hpp"
#include "model/PreparedWeights.hpp"
#include "ops/GDN.hpp"
#include "ops/ExecutionPlans.hpp"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/Normalization.hpp"
#include "ops/PagedAttention.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace splash::model {

struct Qwen3_8Weights;
struct Qwen3_6MoeWeights;

enum class QwenFfnKind : uint8_t { Dense, SparseMoe };

// Both supported targets bind the same mixer tensors per hybrid layer; only
// the FFN differs between them.
struct QwenGdnWeights final {
  ops::Projection inputProjection;
  metal::MetalBuffer convolutionWeights;
  metal::MetalBuffer decay;
  metal::MetalBuffer timeBias;
  ops::NormWeights mixerNorm;
  ops::Projection outputProjection;
  // The value-head order of outputProjection's input columns, in which the
  // GDN writes its output.
  ops::GdnHeadOrder outputHeadOrder = ops::GdnHeadOrder::Grouped;
};

struct QwenAttentionWeights final {
  ops::Projection inputProjection;
  ops::NormWeights queryNorm;
  ops::NormWeights keyNorm;
  ops::Projection outputProjection;
};

using QwenMixerWeights = std::variant<QwenGdnWeights, QwenAttentionWeights>;

// Sizes of the mixer sections in a packed layer file.
struct QwenMixerGeometry final {
  uint32_t hiddenSize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedAttentionWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionHeadDimension = 0;
};

// How a target's files store its tensors; loadQwenTarget pairs each source's
// files with their format. Affine files, packed or prepared from MLX, hold
// every projection, a fused one too, as one affine Q4 tensor, and bf16 norms.
struct AffineTargetFormat final {
  static constexpr bool float32Norms = false;
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Grouped;
  metal::MetalBackend &backend;

  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readAffineProjection(file, backend, outputSize, inputSize, label);
  }
  // The tensor `label`; block images keep the projection as `tensors`.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view label,
                                      std::initializer_list<std::string_view>) const {
    return projection(file, outputSize, inputSize, label);
  }
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readAffineEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Prepared GGUF images hold each GGUF tensor as one block-quantized segment,
// a fused projection as its tensors in output column order, and the GGUF's
// F32 norms. The GGUF keeps the GDN output projection's input columns in
// llama.cpp's tiled value-head order, so the GDN writes its output in it.
struct BlockTargetFormat final {
  static constexpr bool float32Norms = true;
  static constexpr ops::GdnHeadOrder gdnOutputOrder = ops::GdnHeadOrder::Tiled;

  [[nodiscard]] ops::Projection projection(WeightFile &file, uint32_t outputSize,
                                           uint32_t inputSize, std::string_view label) const {
    return readGgufProjection(file, outputSize, inputSize, label);
  }
  // The tensors, which may leave padding columns past the last one
  // (LinearGguf.cpp requireSegments); affine files keep one tensor.
  [[nodiscard]] ops::Projection fused(WeightFile &file, uint32_t outputSize, uint32_t inputSize,
                                      std::string_view,
                                      std::initializer_list<std::string_view> tensors) const;
  [[nodiscard]] ops::EmbeddingWeights embedding(WeightFile &file, uint32_t outputSize,
                                                uint32_t inputSize) const {
    return readGgufEmbedding(file, outputSize, inputSize, "embedding");
  }
};

// Reads the mixer sections that follow a layer's input norm, in file order
// (instantiated for both formats).
template <class Format>
[[nodiscard]] QwenMixerWeights readQwenMixer(WeightFile &file, const Format &format,
                                             const QwenMixerGeometry &geometry,
                                             bool fullAttention);

inline constexpr std::string_view kEmbeddingMagic = "MDFE0001";

// Opens the packed files of a target directory: one per hybrid layer, head.bin
// and embedding.bin.
struct PackedTargetFiles final {
  metal::MetalBackend &backend;
  std::filesystem::path directory;
  std::string_view layerMagic;
  std::string_view headMagic;
  std::string_view embeddingMagic;
  [[nodiscard]] WeightFile layer(uint32_t index, bool fullAttention) const {
    const std::string filename = "layer-" + std::to_string(index) + ".bin";
    return WeightFile(backend, directory / filename, "target/" + filename, layerMagic, index,
                      fullAttention ? 1U : 0U);
  }
  [[nodiscard]] WeightFile head(uint32_t layers) const {
    return WeightFile(backend, directory / "head.bin", "target/head.bin", headMagic, layers, 2);
  }
  [[nodiscard]] WeightFile embedding(uint32_t vocabulary, uint32_t hidden) const {
    return WeightFile(backend, directory / "embedding.bin", "target/embedding.bin",
                      embeddingMagic, vocabulary, hidden);
  }
};

// Reads a target through immutable WeightFiles, packaged or prepared locally,
// in their format: per layer the input norm, mixer, post-attention norm and
// the architecture's FFN through readFfn, then the head and the token
// embedding. Weights is the architecture's weight struct.
template <class Weights, class Layout, class Files, class Format, class ReadFfn>
[[nodiscard]] Weights
readQwenTargetWeights(metal::MetalBackend &backend, const Layout &layout, Files &&files,
                      const Format &format, ReadFfn readFfn) {
  const uint64_t allocationBaseline = backend.memoryStats().allocatedBytes;
  Weights result;
  result.layout = layout;
  result.layers.reserve(layout.layers);

  for (uint32_t layerIndex = 0; layerIndex < layout.layers; ++layerIndex) {
    const bool fullAttention = layout.isFullAttentionLayer(layerIndex);
    WeightFile file = files.layer(layerIndex, fullAttention);
    auto &layer = result.layers.emplace_back();
    layer.inputNorm = readNorm(file, layout.hiddenSize, Format::float32Norms, "input-norm");
    layer.mixer = readQwenMixer(file, format, layout.mixerGeometry(), fullAttention);
    layer.postAttentionNorm =
        readNorm(file, layout.hiddenSize, Format::float32Norms, "post-attention-norm");
    readFfn(file, layer, format);
    file.finish();
    result.files.push_back(file.record());
  }

  {
    WeightFile file = files.head(layout.layers);
    result.finalNorm = readNorm(file, layout.hiddenSize, Format::float32Norms, "final-norm");
    result.logitsProjection =
        format.projection(file, layout.vocabularySize, layout.hiddenSize, "logits");
    file.finish();
    result.files.push_back(file.record());
  }
  {
    WeightFile file = files.embedding(layout.vocabularySize, layout.hiddenSize);
    result.tokenEmbedding = format.embedding(file, layout.vocabularySize, layout.hiddenSize);
    file.finish();
    result.files.push_back(file.record());
  }

  result.manifestFingerprintSha256 = weightManifestFingerprint(result.files);
  result.actualAllocatedBytes = metal::allocationDelta(
      allocationBaseline, backend.memoryStats().allocatedBytes);
  return result;
}

// Loads a target from its source: packed files (splash-packed-q4 formats), or
// an MLX or GGUF source prepared into cached files, whose disk check budgets
// alsoPrepared, the model's other prepared files, too. The architecture reads
// its FFN from affine files through readAffineFfn and from GGUF images through
// readBlockFfn, each called with the file, the layer and the format.
template <class Weights, class Layout, class ReadAffineFfn, class ReadBlockFfn>
[[nodiscard]] Weights
loadQwenTarget(metal::MetalBackend &backend, const std::filesystem::path &directory,
               const Layout &layout, TargetSource source, PreparationCheck admission,
               std::span<const PreparedWeight> alsoPrepared, ReadAffineFfn readAffineFfn,
               ReadBlockFfn readBlockFfn) {
  const AffineTargetFormat affine{backend};
  switch (source) {
  case TargetSource::Packed:
    return readQwenTargetWeights<Weights>(
        backend, layout,
        PackedTargetFiles{backend, directory, Layout::layerMagic, Layout::headMagic, kEmbeddingMagic},
        affine, readAffineFfn);
  case TargetSource::Affine: {
    AffineTargetLoader loader(backend, directory, layout, std::move(admission), alsoPrepared);
    return readQwenTargetWeights<Weights>(backend, layout, loader, affine, readAffineFfn);
  }
  case TargetSource::Gguf: {
    GgufTargetLoader loader(backend, findTargetGguf(directory), ggufTargetGeometry(layout),
                            std::move(admission), alsoPrepared);
    return readQwenTargetWeights<Weights>(backend, layout, GgufTargetFiles{loader},
                                          BlockTargetFormat{}, readBlockFfn);
  }
  }
  throw WeightStoreError("unknown target source");
}

// Runtime-visible tensor geometry shared by the supported Qwen hybrid
// targets. It describes semantics only; operators remain responsible for
// choosing device-specific Metal pipelines and compute tiles.
struct QwenTargetGeometry final {
  static constexpr uint32_t maximumCaptureLayers = 8;

  uint32_t maximumContextTokens = 0;
  uint32_t layers = 0;
  uint32_t hiddenSize = 0;
  uint32_t vocabularySize = 0;
  uint32_t packedGdnWidth = 0;
  uint32_t packedAttentionWidth = 0;
  uint32_t convolutionDimension = 0;
  uint32_t gdnKeyHeads = 0;
  uint32_t gdnValueHeads = 0;
  uint32_t gdnHeadDimension = 0;
  uint32_t attentionWidth = 0;
  uint32_t attentionQueryHeads = 0;
  uint32_t attentionKvHeads = 0;
  uint32_t attentionHeadDimension = 0;
  uint32_t rotaryPairs = 0;
  float rotaryTheta = 0.0F;
  uint32_t denseIntermediateSize = 0;
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  QwenFfnKind ffnKind = QwenFfnKind::Dense;
  // The weight layout every sparse MoE block of the target shares.
  ops::WeightLayout moeLayout = ops::WeightLayout::Affine64;
  uint32_t maskToken = 0;
  std::array<uint32_t, 2> stopTokens{};
  std::array<uint32_t, maximumCaptureLayers> captureLayerValues{};
  uint32_t captureLayerCount = 0;
  kv::Layout kvLayout{};
  GdnStateLayout stateLayout{};
  // Distinct operator requirements, collected from the loaded weights.
  std::vector<ops::ProjectionShape> prefillProjections;
  std::vector<ops::ProjectionShape> decodeProjections;
  std::vector<ops::ProjectionShape> gateUpProjections;

  [[nodiscard]] constexpr uint32_t gdnKeyWidth() const noexcept {
    return gdnKeyHeads * gdnHeadDimension;
  }
  [[nodiscard]] constexpr uint32_t capturedHiddenSize() const noexcept {
    return hiddenSize * captureLayerCount;
  }
  [[nodiscard]] constexpr ops::MoeShape moeShape() const noexcept {
    return {hiddenSize, experts, expertsPerToken, expertIntermediateSize, moeLayout};
  }
  [[nodiscard]] constexpr uint32_t ffnScratchWidth() const noexcept {
    return ffnKind == QwenFfnKind::Dense ? denseIntermediateSize
                                         : expertIntermediateSize;
  }
  [[nodiscard]] constexpr std::span<const uint32_t>
  captureLayers() const noexcept {
    return {captureLayerValues.data(), captureLayerCount};
  }
  [[nodiscard]] constexpr ops::GdnShape gdnShape() const noexcept {
    return {gdnKeyHeads, gdnValueHeads, gdnHeadDimension,
            convolutionDimension, packedGdnWidth};
  }
  // The projection lists hold every projection the weights dispatch, which
  // each have sizes.
  [[nodiscard]] bool valid() const noexcept {
    const auto sized = [](const std::vector<ops::ProjectionShape> &shapes) {
      return !shapes.empty() && std::all_of(shapes.begin(), shapes.end(), [](const auto &shape) {
        return shape.outputSize && shape.inputSize;
      });
    };
    return maximumContextTokens && layers && hiddenSize && vocabularySize &&
           packedGdnWidth && packedAttentionWidth && convolutionDimension &&
           gdnKeyHeads && gdnValueHeads && gdnHeadDimension &&
           attentionWidth && attentionQueryHeads && attentionKvHeads &&
           attentionHeadDimension && rotaryPairs && rotaryTheta > 0.0F &&
           captureLayerCount && captureLayerCount <= maximumCaptureLayers &&
           kvLayout.valid() && stateLayout.valid() &&
           stateLayout.layers + kvLayout.attentionLayers == layers &&
           gdnKeyWidth() * 2 + attentionWidth <= packedGdnWidth &&
           attentionWidth == attentionQueryHeads * attentionHeadDimension &&
           kvLayout.kvHeads == attentionKvHeads &&
           kvLayout.headDimension == attentionHeadDimension &&
           sized(prefillProjections) && sized(decodeProjections) &&
           ((ffnKind == QwenFfnKind::Dense && denseIntermediateSize && sized(gateUpProjections)) ||
            (ffnKind == QwenFfnKind::SparseMoe && moeShape().valid()));
  }
};

struct QwenTargetPrefillCapture final {
  uint32_t sourceStart = 0;
  uint32_t destinationStart = 0;
  uint32_t rows = 0;
};

struct QwenTargetPrefillSequence final {
  uint32_t rowBegin = 0;
  uint32_t rows = 0;
  uint32_t attentionStride = 0;
  uint64_t queryOffset = 0;
  uint64_t kvOffset = 0;
  kv::Q8ChunkedPrefillParams q8;
  metal::MetalBuffer pageTable;
  std::span<const metal::MetalBuffer> convolutionIn;
  std::span<const metal::MetalBuffer> convolutionOut;
  std::span<const metal::MetalBuffer> recurrentIn;
  std::span<const metal::MetalBuffer> recurrentOut;
  std::array<QwenTargetPrefillCapture, 2> captures{};
  uint32_t captureCount = 0;
};

struct QwenTargetPrefillBuffers final {
  // Split projections of chunks of up to 32 rows (LinearGguf.cpp).
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer captured;
  metal::MetalBuffer gdnPacked;
  metal::MetalBuffer gdnQueries;
  metal::MetalBuffer gdnKeys;
  metal::MetalBuffer gdnValues;
  metal::MetalBuffer gdnDecay;
  metal::MetalBuffer gdnBeta;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseGateScratch;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer projectionSums;
  metal::MetalBuffer downProjectionSums;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer chunkKeys;
  metal::MetalBuffer chunkValues;
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer groupedSums;
};

struct QwenTargetVerifyBuffers final {
  ops::LinearScratch linearScratch{};
  std::array<metal::MetalBuffer, 2> hidden;
  metal::MetalBuffer normalized;
  metal::MetalBuffer recurrent;
  metal::MetalBuffer gdnHidden;
  metal::MetalBuffer gdnOutput;
  metal::MetalBuffer denseIntermediate;
  metal::MetalBuffer fullPacked;
  metal::MetalBuffer fullQueries;
  metal::MetalBuffer attentionPartials;
  metal::MetalBuffer attentionStatistics;
  metal::MetalBuffer fullAttention;
  metal::MetalBuffer attentionHidden;
  metal::MetalBuffer attentionOutput;
  metal::MetalBuffer ropeCos;
  metal::MetalBuffer ropeSin;
  metal::MetalBuffer arrived;
  metal::MetalBuffer generation;
  metal::MetalBuffer capturedTargetHidden;
  metal::MetalBuffer finalHidden;
  metal::MetalBuffer logits;
  metal::MetalBuffer denseGateScratch;
  std::span<const metal::MetalBuffer> gdnPacked;
  std::span<const metal::MetalBuffer> gdnMixed;
  std::span<const metal::MetalBuffer> gdnDecay;
  std::span<const metal::MetalBuffer> gdnBeta;
  std::span<const metal::MetalBuffer> chunkKeys;
  std::span<const metal::MetalBuffer> chunkValues;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextGdnStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      pageTables;
  metal::MetalBuffer selectedExperts;
  metal::MetalBuffer routingWeights;
  metal::MetalBuffer tileDescriptors;
  metal::MetalBuffer tileCount;
  metal::MetalBuffer groupedRoutes;
  metal::MetalBuffer routeRows;
  metal::MetalBuffer groupedInput;
  metal::MetalBuffer expertIntermediate;
  metal::MetalBuffer expertOutput;
  metal::MetalBuffer groupedSums;
};

struct QwenTargetCommitBuffers final {
  metal::MetalBuffer packed;
  metal::MetalBuffer mixed;
  metal::MetalBuffer decay;
  metal::MetalBuffer beta;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      currentStates;
  std::array<metal::MetalBuffer, ExecutionLimits::maximumBatchWidth>
      nextStates;
  metal::MetalBuffer retainedCounts;
};

[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_8Weights &weights);
[[nodiscard]] QwenTargetGeometry
qwenTargetGeometry(const Qwen3_6MoeWeights &weights);

// Builds the shared Qwen GDN/attention layer graph with the target's dense
// or sparse-MoE FFN. Architecture-specific loaders supply the package tensors.
class QwenTarget final {
public:
  QwenTarget(const Qwen3_8Weights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);
  QwenTarget(const Qwen3_6MoeWeights &weights, metal::MetalBackend &backend,
             const ops::ExecutionPlans &operators,
             kv::Format format = kv::Format::Int8);

  [[nodiscard]] const QwenTargetGeometry &geometry() const noexcept {
    return geometry_;
  }
  [[nodiscard]] const ops::Projection &
  vocabularyProjection() const noexcept;
  // Lanes of storage the tensors of a decode step of `lanes` lanes bind: a
  // linear tile may hold more rows than the step (LinearPlan::storageRows;
  // a three-lane GGUF step on the staged tile runs its 32-row tile over four
  // lanes). Every op still processes the step's lanes; padding rows read
  // stale activations and write results no active row reads.
  [[nodiscard]] uint32_t decodeStorageLanes(uint32_t lanes) const;

  void addPrefill(
      metal::CommandGraph &graph, QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::LayerStorage> kvLayers) const;
  void addVerify(
      metal::CommandGraph &graph, QwenTargetVerifyBuffers buffers,
      std::span<const kv::LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::LinearDispatchStats &stats) const;
  void addHead(metal::CommandGraph &graph, metal::MetalBuffer hidden,
               metal::MetalBuffer finalHidden, metal::MetalBuffer logits,
               uint32_t normalizedRows, ops::LinearScratch scratch = {}) const;
  void addEmbedding(metal::CommandGraph &graph, metal::MetalBuffer tokens,
                    metal::MetalBuffer hidden, uint32_t rows) const;
  void addStateCommit(metal::CommandGraph &graph,
                      QwenTargetCommitBuffers buffers, uint32_t lanes) const;

private:
  using WeightView =
      std::variant<const Qwen3_8Weights *, const Qwen3_6MoeWeights *>;

  template <class Weights>
  void addPrefillImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetPrefillBuffers buffers,
      std::span<const QwenTargetPrefillSequence> sequences, uint32_t rows,
      std::span<const kv::LayerStorage> kvLayers) const;
  template <class Weights>
  void addVerifyImpl(
      const Weights &weights, metal::CommandGraph &graph,
      QwenTargetVerifyBuffers buffers,
      std::span<const kv::LayerStorage> kvLayers,
      std::span<const kv::Q8ChunkedPrefillParams> q8,
      std::span<const kv::Q8VerifyAttentionParams> verify, uint32_t lanes,
      ops::LinearDispatchStats &stats) const;

  WeightView weights_;
  QwenTargetGeometry geometry_;
  metal::MetalBackend &backend_;
  const ops::ExecutionPlans &operators_;
};

} // namespace splash::model
