#include "model/SafetensorsTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/SafetensorsImage.hpp"
#include "metal/abi/QuantFormat.h"

#include <span>

namespace splash::model {
namespace {

// The native rows of a table that streams from a checkpoint, its shards'
// rows in order (the copies of its image), as one file of rows back to back
// that SparseRowStore reads: each read builds whole rows from their codes,
// scales and biases (readRepackRows).
class TableReader final : public RowReader {
public:
  TableReader(gguf::Repack table, std::shared_ptr<const void> planned, const SafetensorsCheckpoint &checkpoint)
      : table_(std::move(table)), planned_(std::move(planned)), checkpoint_(checkpoint),
        rowBytes_(ggufRowBytes(kQuantFormats[table_.format], table_.columns)) {}
  void read(uint64_t offset, std::span<uint8_t> bytes) override {
    if (offset % rowBytes_ || bytes.size() % rowBytes_)
      throw SparseRowError("a streamed table is read by whole rows");
    readRepackRows(table_, offset / rowBytes_, bytes.size() / rowBytes_, bytes.data());
  }
  void checkUnchanged() override { checkpoint_.checkUnchanged(); }
  [[nodiscard]] uint64_t size() const override { return table_.rows * rowBytes_; }

private:
  gguf::Repack table_;
  std::shared_ptr<const void> planned_; // holds the checkpoint
  const SafetensorsCheckpoint &checkpoint_;
  uint64_t rowBytes_;
};

} // namespace

SafetensorsTargetLoader::SafetensorsTargetLoader(metal::MetalBackend &backend, WeightImages &images,
                                                 const std::filesystem::path &directory,
                                                 const QwenTargetDimensions &geometry, std::string_view architecture,
                                                 SparseRowStoreOptions table, ExpertStoreOptions experts)
    : backend_(backend), images_(images), planned_(std::make_shared<Planned>(directory)),
      layers_(geometry.layers), firstMoeLayer_(geometry.leadingDenseLayers), table_(std::move(table)),
      experts_(experts) {
  planned_->images =
      safetensors::planImages(planned_->checkpoint, geometry, architecture,
                              {.streamExperts = experts.memoryBytes != 0, .streamTable = table_.cacheBytes != 0});
  float32Norms_ = safetensors::float32Norms(safetensors::moduleNames(planned_->checkpoint), architecture);
}

StreamedBlocks SafetensorsTargetLoader::streamedExperts() const {
  const std::span layers = std::span(planned_->images).first(layers_);
  return {gguf::streamedExperts(layers.subspan(firstMoeLayer_)), planned_,
          {[planned = planned_] { planned->checkpoint.checkUnchanged(); }}};
}

WeightFile SafetensorsTargetLoader::open(size_t index) {
  return loadImage(backend_, images_, "target/", {planned_, &planned_->images[index]},
                   [planned = planned_] { planned->checkpoint.checkUnchanged(); });
}

WeightFile SafetensorsTargetLoader::layer(uint32_t index) {
  if (index >= layers_) throw WeightStoreError("target layer is out of range");
  return open(index);
}

const gguf::Image *SafetensorsTargetLoader::tablePlan() const noexcept {
  const size_t index = layers_ + 2;
  return index < planned_->images.size() ? &planned_->images[index] : nullptr;
}

WeightFile SafetensorsTargetLoader::table() {
  const gguf::Image *plan = tablePlan();
  if (!plan) throw WeightStoreError("the target has no table");
  if (plan->streamedRows) throw WeightStoreError("the target's table streams");
  return open(layers_ + 2);
}

std::optional<GgufTargetLoader::StreamedTable> SafetensorsTargetLoader::streamTable() const {
  const gguf::Image *plan = tablePlan();
  if (!plan || !plan->streamedRows) return std::nullopt;
  const gguf::StreamedRows &rows = *plan->streamedRows;
  const QuantFormat &format = kQuantFormats[rows.format];
  gguf::Repack table;
  table.format = rows.format;
  table.rows = rows.rows;
  table.columns = rows.rowBytes / format.block_bytes * format.block_elements;
  for (const gguf::Copy &copy : plan->copies) table.sources.push_back(copy.source);
  planned_->checkpoint.checkUnchanged();
  return GgufTargetLoader::StreamedTable{
      std::make_shared<SparseRowStore>(SparseRowTable{rows.tensor, rows.rows, rows.rowBytes, 0},
                                       std::make_unique<TableReader>(std::move(table), planned_, planned_->checkpoint),
                                       table_),
      rows.format, imageRecord(images_, "target/", *plan)};
}

WeightFile SafetensorsTargetLoader::head() { return open(layers_); }

WeightFile SafetensorsTargetLoader::embedding() { return open(layers_ + 1); }

} // namespace splash::model
