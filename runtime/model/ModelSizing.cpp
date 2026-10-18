#include "model/ModelSizing.hpp"

#include "model/DraftCheckpoint.hpp"
#include "model/ExpertStore.hpp"
#include "model/GgufTarget.hpp"
#include "model/Qwen4Exp.hpp"
#include "model/SafetensorsImage.hpp"
#include "model/SparseRowStore.hpp"
#include "model/VisionLoader.hpp"

#include <span>
#include <vector>

namespace splash::model {
namespace {

// The rows of a family's streamed table that a step stages and prefetches,
// which its row cache's working memory holds; the Qwen3.5 families have no
// table.
SparseRowStoreOptions tableRows(const Qwen4ExpLayout &layout) { return streamedTableRows(layout); }
template <class Layout> SparseRowStoreOptions tableRows(const Layout &) { return {}; }

// What a target of `layout` planned as `images` takes, while their sources
// are open.
template <class Layout> StreamableWeights targetWeights(const Layout &layout, std::span<const gguf::Image> images) {
  StreamableWeights result;
  for (const gguf::Image &image : images) {
    result.loadedBytes += gguf::loadedBytes(image);
    result.expertBytes += gguf::streamedExpertBytes(image);
    result.tableBytes += gguf::streamedTableBytes(image);
    if (image.streamedRows) {
      const SparseRowStoreOptions rows = tableRows(layout);
      const uint32_t rowBytes = image.streamedRows->rowBytes;
      result.tableFloorBytes = SparseRowStore::workingBytes(rowBytes, rows.prefetchRows, rows.stageRows) +
                               SparseRowStore::cacheBytesFor(1, rowBytes);
    }
  }
  if (result.expertBytes) {
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(images.first(layout.layers));
    result.expertFloorBytes = ExpertStore::minimumBytes(
        blocks, {layout.hiddenSize, layout.experts, layout.expertsPerToken, layout.expertIntermediateSize});
  }
  return result;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const gguf::PlanOptions streamed{.streamExperts = layout.ffnKind == QwenFfnKind::SparseMoe,
                                         .streamTable = true};
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          return targetWeights(layout, gguf::planImages(target.file(), layout.ggufArchitecture, layout, streamed));
        }
        const SafetensorsCheckpoint checkpoint(directory);
        return targetWeights(layout, safetensors::planImages(checkpoint, layout, streamed));
      },
      descriptor.target);
  result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision()) result.loadedBytes += visionImageBytes(descriptor.vision);
  return result;
}

} // namespace splash::model
