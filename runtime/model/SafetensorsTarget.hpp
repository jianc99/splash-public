#pragma once

// Source adapter for a safetensors target (MLX's, Model Optimizer's or
// compressed-tensors'): its block images (model/SafetensorsImage.hpp) are
// written into memory, and QwenTargetLoader reads them as block-quantized
// weights (BlockTargetFormat).

#include "model/ExpertStore.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/QwenHybridLayout.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/SparseRowStore.hpp"
#include "model/WeightImages.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace splash::model {

class SafetensorsTargetLoader final {
public:
  // Plans every image of the checkpoint in directory once, of the family
  // whose GGUFs are of `architecture` (safetensors::planImages). With a
  // table store (table.cacheBytes), the rows of a table the family adds
  // stream through it (streamTable()) instead of loading (table()). With
  // expert memory (experts.memoryBytes), a MoE target's routed experts
  // stream (streamedExperts()) instead of loading with their layers.
  SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images, const std::filesystem::path &directory,
                          const QwenTargetDimensions &geometry, std::string_view architecture,
                          SparseRowStoreOptions table = {}, ExpertStoreOptions experts = {});
  SafetensorsTargetLoader(const SafetensorsTargetLoader &) = delete;
  SafetensorsTargetLoader &operator=(const SafetensorsTargetLoader &) = delete;

  [[nodiscard]] WeightFile layer(uint32_t index);
  [[nodiscard]] WeightFile head();
  [[nodiscard]] WeightFile embedding();
  // The image of the table the family adds after the embedding (Qwen3.8-
  // Flash-Next's PLE n-gram table), refused when its rows stream.
  [[nodiscard]] WeightFile table();
  // The streamed table, its store reading the table's native rows from the
  // checkpoint's open files (readRepackRows over its shards); none when the
  // table loads.
  [[nodiscard]] std::optional<GgufTargetLoader::StreamedTable> streamTable() const;
  // The checkpoint, for what a family reads beside its images (Flash-Next's
  // PLE hash).
  [[nodiscard]] const SafetensorsCheckpoint &checkpoint() const noexcept { return planned_->checkpoint; }
  // Whether its images hold F32 norms (safetensors::float32Norms).
  [[nodiscard]] bool float32Norms() const noexcept { return float32Norms_; }
  // Whether the routed experts stream, then the store's options and every
  // MoE layer's experts (a block each, those of the leading dense layers'
  // none), which read the checkpoint's open files.
  [[nodiscard]] bool streamsExperts() const noexcept { return experts_.memoryBytes != 0; }
  [[nodiscard]] const ExpertStoreOptions &expertStoreOptions() const noexcept { return experts_; }
  [[nodiscard]] StreamedBlocks streamedExperts() const;

private:
  // The checkpoint and its images, which their writers share.
  struct Planned {
    explicit Planned(const std::filesystem::path &directory) : checkpoint(directory) {}
    SafetensorsCheckpoint checkpoint;
    std::vector<gguf::Image> images; // layers, head, embedding, a family's table
  };
  [[nodiscard]] WeightFile open(size_t index);
  // The plan of the table's image, or null for a family without one.
  [[nodiscard]] const gguf::Image *tablePlan() const noexcept;

  metal::MetalBackend &backend_;
  WeightImages &images_;
  std::shared_ptr<Planned> planned_;
  uint32_t layers_ = 0, firstMoeLayer_ = 0;
  bool float32Norms_ = false;
  SparseRowStoreOptions table_;
  ExpertStoreOptions experts_;
};

} // namespace splash::model
