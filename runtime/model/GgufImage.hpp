#pragma once

// Plans the prepared MDGG0001 images of a Qwen3.8 (qwen35) or Qwen3.6 MoE
// (qwen35moe) target read straight from a llama.cpp GGUF, from its metadata
// alone: section offsets, the header and descriptor bytes, and the source
// rows each tensor section is written from (model/GgufPreparation.hpp).
// Layout: 16-byte header (magic, layer, type), then 16 KiB-aligned sections;
// each quantized tensor is a 64-byte descriptor, then its plane0, optional
// plane1 and meta planes in the layout of its format (metal/abi/QuantFormat.h);
// each float tensor (F32) a descriptor and its rows as stored. A 3-D expert
// tensor is one quantized tensor of experts * N rows.

#include <cstdint>
#include <string>
#include <vector>

#include "model/GgufFile.hpp"

namespace splash::model::gguf {

struct TargetGeometry {
  uint32_t layers = 64;
  uint32_t hiddenSize = 5120;
  uint32_t vocabularySize = 248320;
  uint32_t intermediateSize = 17408; // dense FFN
  uint32_t gdnKeyHeads = 16;
  uint32_t gdnValueHeads = 48;
  uint32_t gdnHeadDimension = 128;
  uint32_t convolutionDimension = 10240;
  uint32_t attentionWidth = 6144;
  uint32_t attentionKvHeads = 4;
  uint32_t attentionHeadDimension = 256;
  uint32_t fullAttentionPeriod = 4;
  // A sparse MoE FFN (qwen35moe) when experts is set; the shared expert has
  // the routed experts' intermediate width.
  uint32_t experts = 0;
  uint32_t expertsPerToken = 0;
  uint32_t expertIntermediateSize = 0;
  [[nodiscard]] bool isFullAttentionLayer(uint32_t layer) const noexcept {
    return (layer + 1) % fullAttentionPeriod == 0;
  }
  [[nodiscard]] bool sparseMoe() const noexcept { return experts != 0; }
  // The general.architecture of a GGUF of this target.
  [[nodiscard]] const char *architecture() const noexcept {
    return sparseMoe() ? "qwen35moe" : "qwen35";
  }
};

// The order of a tensor's rows in the image. Rows below `from` keep their
// order; from there on, blocks of headRows rows are value heads, which
// llama.cpp stores tiled (group * groupHeads + head) and splash groups by key
// head.
struct RowOrder {
  uint64_t from = UINT64_MAX; // UINT64_MAX: rows as stored
  uint32_t headRows = 0;
  uint32_t groupHeads = 0; // heads per key group
  uint32_t groups = 0;     // value heads per key head
};

// Rows [0, rows) of one source tensor in image order.
struct TensorRows {
  std::string name;
  uint32_t type = 0;   // ggml type
  uint64_t offset = 0; // the tensor's data, from the start of the data section
  uint64_t rows = 0;
  uint64_t rowBytes = 0;
  RowOrder order{};
};

// Header and descriptor bytes.
struct Fill {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
};
// Rows written back to back as stored or, for F32 rows the kernels read as
// bf16, as the bf16 values they equal exactly.
struct Copy {
  uint64_t destination = 0;
  TensorRows source;
  bool bfloat16 = false;
};
// Quantized rows repacked into the planes of their format; the rows of the
// sources in order, then zero rows up to `rows`.
struct Repack {
  uint32_t format = 0; // GGUF_FMT_*
  uint64_t rows = 0;
  uint64_t columns = 0;
  uint64_t plane0 = 0, plane1 = 0, meta = 0; // image offsets; plane1 when the format has one
  std::vector<TensorRows> sources;
};
struct Image {
  std::string name; // layer-N.bin, head.bin, embedding.bin
  uint32_t layer = 0;
  uint32_t type = 0;
  uint64_t bytes = 0;
  std::vector<Fill> fills;
  std::vector<Copy> copies;
  std::vector<Repack> repacks;
};

class ImagePlanner final {
public:
  // Validates architecture, geometry and every tensor's presence, shape and
  // type; throws GgufError listing all offending tensors.
  ImagePlanner(const GgufFile &file, TargetGeometry geometry);
  [[nodiscard]] Image layer(uint32_t index) const;
  [[nodiscard]] Image head() const;
  [[nodiscard]] Image embedding() const;
  [[nodiscard]] const TargetGeometry &geometry() const noexcept { return geometry_; }
  // Sum of all image bytes, for weight admission before preparation.
  [[nodiscard]] uint64_t totalBytes() const;

private:
  const GgufFile &file_;
  TargetGeometry geometry_;
};

} // namespace splash::model::gguf
