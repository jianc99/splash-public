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

uint64_t loadedBytes(std::span<const gguf::Image> images) {
  uint64_t bytes = 0;
  for (const gguf::Image &image : images)
    bytes += gguf::loadedBytes(image);
  return bytes;
}

// What a target of `layout` whose images `plan` plans for what streams
// takes, while their sources are open: what loads while every part
// streams, and what each part adds where it loads instead, the routed
// experts' slot table with them.
template <class Layout, class Plan> StreamableWeights targetWeights(const Layout &layout, const Plan &plan) {
  const bool moe = layout.ffnKind == QwenFfnKind::SparseMoe;
  const std::vector<gguf::Image> streamed = plan({.streamExperts = moe, .streamTable = true});
  StreamableWeights result{.loadedBytes = loadedBytes(streamed)};
  result.tableBytes = loadedBytes(plan({.streamExperts = moe, .streamTable = false})) - result.loadedBytes;
  for (const gguf::Image &image : streamed)
    if (image.streamedRows) {
      const SparseRowStoreOptions rows = tableRows(layout);
      const uint32_t rowBytes = image.streamedRows->rowBytes;
      result.tableFloorBytes = SparseRowStore::workingBytes(rowBytes, rows.prefetchRows, rows.stageRows) +
                               SparseRowStore::cacheBytesFor(1, rowBytes);
    }
  if (moe) {
    result.expertBytes = loadedBytes(plan({.streamExperts = false, .streamTable = true})) - result.loadedBytes +
                         residentExpertSlotsBytes(layout.experts);
    StreamedBlocks blocks;
    blocks.blocks = gguf::streamedExperts(std::span(streamed).first(layout.layers));
    result.expertFloorBytes = ExpertStore::minimumBytes(
        blocks, {layout.hiddenSize, layout.experts, layout.expertsPerToken, layout.expertIntermediateSize});
  }
  return result;
}

} // namespace

StreamableWeights streamableWeights(const std::filesystem::path &root, const ModelDescriptor &descriptor) {
  StreamableWeights result = std::visit(
      [&](const auto &layout) {
        const std::filesystem::path directory = root / "target";
        if (descriptor.targetSource == TargetSource::Gguf) {
          const OpenGguf target(findTargetGgufs(directory));
          return targetWeights(layout, [&](gguf::PlanOptions options) {
            return gguf::planImages(target.file(), layout.ggufArchitecture, layout, options);
          });
        }
        const SafetensorsCheckpoint checkpoint(directory);
        return targetWeights(layout, [&](gguf::PlanOptions options) {
          return safetensors::planImages(checkpoint, layout, options);
        });
      },
      descriptor.target);
  result.loadedBytes += draftImageBytes(root / "draft", descriptor.draft);
  if (descriptor.hasVision()) result.loadedBytes += visionImageBytes(descriptor.vision);
  return result;
}

} // namespace splash::model
