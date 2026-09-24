#include "model/GgufImage.hpp"

#include "metal/abi/Gguf.h"
#include "model/StateLayout.hpp"
#include "model/WeightLayout.hpp"

#include <optional>

namespace splash::model::gguf {
namespace {

static_assert(kQuantFormats[GGUF_FMT_Q4K].ggml_type == ggml::kQ4_K &&
                  kQuantFormats[GGUF_FMT_IQ4XS].ggml_type == ggml::kIQ4_XS &&
                  kQuantFormats[GGUF_FMT_IQ4NL].ggml_type == ggml::kIQ4_NL &&
                  kQuantFormats[GGUF_FMT_Q5K].ggml_type == ggml::kQ5_K &&
                  kQuantFormats[GGUF_FMT_Q6K].ggml_type == ggml::kQ6_K &&
                  kQuantFormats[GGUF_FMT_Q3K].ggml_type == ggml::kQ3_K &&
                  kQuantFormats[GGUF_FMT_Q80].ggml_type == ggml::kQ8_0 &&
                  kQuantFormats[GGUF_FMT_IQ3S].ggml_type == ggml::kIQ3_S,
              "format table types are the GGUF type ids");
static_assert(GGUF_TYPE_F32 == ggml::kF32, "float segments carry the GGUF type id");

uint64_t alignUp(uint64_t value) {
  return (value + kWeightFileAlignment - 1) / kWeightFileAlignment * kWeightFileAlignment;
}

void appendLittle32(std::vector<uint8_t> &out, uint32_t value) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}
void appendLittle64(std::vector<uint8_t> &out, uint64_t value) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(value >> (8 * i)));
}

class Builder {
public:
  Builder(const GgufFile &file, const TargetGeometry &geometry, std::string name,
          uint32_t layer, uint32_t type)
      : file_(file), geometry_(geometry) {
    image_.name = std::move(name);
    image_.layer = layer;
    image_.type = type;
    std::vector<uint8_t> header(kGgufImageMagic.begin(), kGgufImageMagic.end());
    appendLittle32(header, layer);
    appendLittle32(header, type);
    image_.fills.push_back({0, std::move(header)});
    cursor_ = 16;
  }

  // A norm as stored: F32, which the norm kernels read unrounded, as
  // llama.cpp does (ops::NormWeights).
  void floatNorm(const std::string &name, uint64_t elements) {
    const GgufTensor &tensor = requireFloat(name, elements);
    copy(rows(tensor, 1, tensor.bytes));
  }

  // Quantized rows [N, K] repacked into planes, rows in `order`.
  void quantized(const GgufTensor &tensor, uint64_t rows, uint64_t columns, RowOrder order = {}) {
    const uint32_t format = gguf_format_of(tensor.type);
    if (format == GGUF_FMT_COUNT)
      throw GgufError("unsupported tensor type " + ggmlTypeName(tensor.type) + " for " + tensor.name);
    if (tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("unexpected shape for " + tensor.name);
    Repack repack = planes(format, rows, columns, tensor.name);
    repack.sources.push_back(this->rows(tensor, rows, rowBytes(format, columns), order));
    image_.repacks.push_back(std::move(repack));
  }

  // beta (value heads rows) | alpha (value heads rows) | zero rows as one
  // 256-row Q8_0 tensor, rows in grouped head order.
  void alphaBeta(const GgufTensor &beta, const GgufTensor &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {&beta, &alpha})
      if (t->type != ggml::kQ8_0 || t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be Q8_0 [" + std::to_string(heads) + ", hidden]: " + t->name);
    if (2 * heads > kQ4StorageN) throw GgufError("alpha/beta rows exceed one 256-row tile");
    Repack repack = planes(GGUF_FMT_Q80, kQ4StorageN, hidden, alpha.name);
    for (const GgufTensor *t : {&beta, &alpha})
      repack.sources.push_back(rows(*t, heads, rowBytes(GGUF_FMT_Q80, hidden), grouped(0, 1)));
    image_.repacks.push_back(std::move(repack));
  }

  // F32 beta (value heads rows) | alpha (value heads rows), rows in grouped
  // head order, as one float tensor.
  void floatAlphaBeta(const GgufTensor &beta, const GgufTensor &alpha) {
    const uint32_t heads = geometry_.gdnValueHeads, hidden = geometry_.hiddenSize;
    for (const GgufTensor *t : {&beta, &alpha})
      if (t->type != ggml::kF32 || t->rows() != heads || t->columns() != hidden)
        throw GgufError("alpha/beta must be [" + std::to_string(heads) + ", hidden]: " + t->name);
    descriptor(ggml::kF32, 2ull * heads, hidden, 0, 0, 0, 0, beta.bytes + alpha.bytes, 0, 0);
    uint64_t destination = section(beta.bytes + alpha.bytes);
    for (const GgufTensor *t : {&beta, &alpha}) {
      image_.copies.push_back({destination, rows(*t, heads, t->bytes / heads, grouped(0, 1))});
      destination += t->bytes;
    }
  }

  // The convolution taps of every channel, q and k channels as stored, then
  // value channels in grouped head order, as the exact bf16 values the GDN
  // kernels read.
  void convolution(const GgufTensor &tensor, uint64_t keyRows) {
    const uint32_t channels = geometry_.convolutionDimension;
    const GgufTensor &conv = requireFloat(tensor.name, uint64_t{channels} * kGdnConvolutionTaps);
    copy(rows(conv, channels, conv.bytes / channels, grouped(keyRows, geometry_.gdnHeadDimension)), true);
  }

  // A per value head F32 vector in grouped head order: as stored, or as the
  // exact bf16 values the kernels read.
  void headVector(const std::string &name, bool bfloat16) {
    const uint32_t heads = geometry_.gdnValueHeads;
    const GgufTensor &tensor = requireFloat(name, heads);
    copy(rows(tensor, heads, tensor.bytes / heads, grouped(0, 1)), bfloat16);
  }

  // Native token rows, gathered by the embedding kernel.
  void embeddingRows(const GgufTensor &tensor) {
    if (tensor.type != ggml::kQ4_K && tensor.type != ggml::kQ6_K && tensor.type != ggml::kQ8_0)
      throw GgufError("unsupported token embedding type " + ggmlTypeName(tensor.type));
    if (tensor.rows() != geometry_.vocabularySize || tensor.columns() != geometry_.hiddenSize)
      throw GgufError("unexpected shape for " + tensor.name);
    copiedRows(tensor);
  }

  // An F32 tensor [rows, columns] as stored (the MoE router and the
  // shared-expert gate, which llama.cpp keeps unquantized).
  void floatTensor(const GgufTensor &tensor, uint64_t rows, uint64_t columns) {
    if (tensor.type != ggml::kF32 || tensor.rows() != rows || tensor.columns() != columns)
      throw GgufError("expected an F32 [" + std::to_string(rows) + ", " + std::to_string(columns) +
                      "] tensor: " + tensor.name);
    copiedRows(tensor);
  }

  Image finish() {
    image_.bytes = alignUp(cursor_);
    return std::move(image_);
  }

private:
  uint64_t section(uint64_t bytes) {
    if (!bytes) throw GgufError("empty image section in " + image_.name);
    const uint64_t start = alignUp(cursor_);
    cursor_ = start + bytes;
    return start;
  }

  const GgufTensor &requireFloat(const std::string &name, uint64_t elements) const {
    const GgufTensor &tensor = file_.require(name);
    if (tensor.elements() != elements) throw GgufError("unexpected shape for " + tensor.name);
    if (tensor.type != ggml::kF32) throw GgufError("expected an F32 tensor: " + tensor.name);
    return tensor;
  }

  RowOrder grouped(uint64_t from, uint32_t headRows) const {
    return {from, headRows, geometry_.gdnKeyHeads, geometry_.gdnValueHeads / geometry_.gdnKeyHeads};
  }

  static uint64_t rowBytes(uint32_t format, uint64_t columns) {
    return columns / kQuantFormats[format].block_elements * kQuantFormats[format].block_bytes;
  }

  static TensorRows rows(const GgufTensor &tensor, uint64_t rows, uint64_t rowBytes, RowOrder order = {}) {
    if (!rows || rows * rowBytes != tensor.bytes) throw GgufError("unexpected size for " + tensor.name);
    return {tensor.name, tensor.type, tensor.offset, rows, rowBytes, order};
  }

  // Rows written as stored, or converted to bf16, into their own section.
  void copy(TensorRows source, bool bfloat16 = false) {
    const uint64_t bytes = source.rows * source.rowBytes / (bfloat16 ? 2 : 1);
    image_.copies.push_back({section(bytes), std::move(source), bfloat16});
  }

  // A tensor's rows as stored, after their descriptor.
  void copiedRows(const GgufTensor &tensor) {
    descriptor(tensor.type, tensor.rows(), tensor.columns(), 0, 0, 0, 0, tensor.bytes, 0, 0);
    copy(rows(tensor, tensor.rows(), tensor.bytes / tensor.rows()));
  }

  // The descriptor and planes of a [rows, columns] quantized tensor.
  Repack planes(uint32_t format, uint64_t rows, uint64_t columns, const std::string &name) {
    if (rows % kQ4StorageN || columns % 256) throw GgufError("tensor is not tile aligned: " + name);
    const QuantFormat &layout = kQuantFormats[format];
    const uint64_t groups = columns / 32;
    const uint64_t plane0 = rows * groups * layout.plane0_bytes;
    const uint64_t plane1 = rows * groups * layout.plane1_bytes;
    const uint64_t meta = rows * (groups / layout.meta_groups) * layout.meta_bytes;
    descriptor(layout.ggml_type, rows, columns, layout.plane0_bytes, layout.plane1_bytes,
               layout.meta_bytes, layout.meta_groups, plane0, plane1, meta);
    Repack repack;
    repack.format = format;
    repack.rows = rows;
    repack.columns = columns;
    repack.plane0 = section(plane0);
    repack.plane1 = plane1 ? section(plane1) : 0;
    repack.meta = section(meta);
    return repack;
  }

  // Word 7 is reserved (0).
  void descriptor(uint32_t type, uint64_t rows, uint64_t columns, uint32_t p0, uint32_t p1,
                  uint32_t metaBytes, uint32_t metaGroups,
                  uint64_t plane0Bytes, uint64_t plane1Bytes, uint64_t metaTotal) {
    std::vector<uint8_t> bytes;
    for (uint32_t word : {type, static_cast<uint32_t>(rows), static_cast<uint32_t>(columns), p0, p1,
                          metaBytes, metaGroups, 0u})
      appendLittle32(bytes, word);
    appendLittle64(bytes, plane0Bytes);
    appendLittle64(bytes, plane1Bytes);
    appendLittle64(bytes, metaTotal);
    bytes.resize(64, 0);
    image_.fills.push_back({section(bytes.size()), std::move(bytes)});
  }

  const GgufFile &file_;
  const TargetGeometry &geometry_;
  Image image_;
  uint64_t cursor_ = 0;
};

std::string prefix(uint32_t layer) { return "blk." + std::to_string(layer) + "."; }

} // namespace

ImagePlanner::ImagePlanner(const GgufFile &file, TargetGeometry geometry)
    : file_(file), geometry_(geometry) {
  const std::string arch = geometry.architecture();
  if (file.architecture() != arch)
    throw GgufError("GGUF architecture is " + file.architecture() + ", but the package's target is " + arch);
  // The target geometry the metadata declares; one error names every mismatch.
  std::string mismatched;
  const auto expect = [&](const char *key, uint64_t value) {
    const std::optional<uint64_t> found = file.unsignedValue(arch + "." + key);
    if (found != value)
      mismatched += (mismatched.empty() ? "" : ", ") + std::string(key) + " " +
                    (found ? std::to_string(*found) : "missing") + " (expected " + std::to_string(value) + ")";
  };
  expect("block_count", geometry.layers + file.unsignedValue(arch + ".nextn_predict_layers").value_or(0));
  expect("embedding_length", geometry.hiddenSize);
  expect("attention.head_count", geometry.attentionWidth / geometry.attentionHeadDimension);
  expect("attention.head_count_kv", geometry.attentionKvHeads);
  expect("attention.key_length", geometry.attentionHeadDimension);
  expect("attention.value_length", geometry.attentionHeadDimension);
  expect("full_attention_interval", geometry.fullAttentionPeriod);
  expect("ssm.conv_kernel", kGdnConvolutionTaps);
  expect("ssm.group_count", geometry.gdnKeyHeads);
  expect("ssm.time_step_rank", geometry.gdnValueHeads);
  expect("ssm.state_size", geometry.gdnHeadDimension);
  expect("ssm.inner_size", uint64_t{geometry.gdnValueHeads} * geometry.gdnHeadDimension);
  if (geometry.sparseMoe()) {
    expect("expert_count", geometry.experts);
    expect("expert_used_count", geometry.expertsPerToken);
    expect("expert_feed_forward_length", geometry.expertIntermediateSize);
    expect("expert_shared_feed_forward_length", geometry.expertIntermediateSize);
  } else {
    expect("feed_forward_length", geometry.intermediateSize);
  }
  if (!mismatched.empty()) throw GgufError("GGUF metadata does not match the target: " + mismatched);
  // Whole-file type check first so one error names every unsupported tensor.
  std::string unsupported;
  const auto reject = [&](const std::string &name, const GgufTensor *tensor) {
    unsupported += (unsupported.empty() ? "" : ", ") + name +
                   (tensor ? " (" + ggmlTypeName(tensor->type) + ")" : " (missing)");
  };
  auto check = [&](const std::string &name, bool embedding = false) {
    const GgufTensor *tensor = file.find(name);
    const bool ok = tensor && (embedding ? tensor->type == ggml::kQ4_K || tensor->type == ggml::kQ6_K ||
                                               tensor->type == ggml::kQ8_0
                                         : gguf_format_of(tensor->type) != GGUF_FMT_COUNT);
    if (!ok) reject(name, tensor);
  };
  // Tensors llama.cpp keeps in F32, which run unrounded.
  auto checkFloat = [&](const std::string &name) {
    const GgufTensor *tensor = file.find(name);
    if (!tensor || tensor->type != ggml::kF32) reject(name, tensor);
  };
  for (uint32_t layer = 0; layer < geometry.layers; ++layer) {
    const std::string p = prefix(layer);
    if (geometry.isFullAttentionLayer(layer)) {
      for (const char *name : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
        check(p + name);
    } else {
      for (const char *name : {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight"}) check(p + name);
      // alpha/beta run in their stored format: both Q8_0 (one repacked
      // segment) or both F32 (one float segment).
      const GgufTensor *alpha = file.find(p + "ssm_alpha.weight"), *beta = file.find(p + "ssm_beta.weight");
      if (!alpha || !beta || alpha->type != beta->type ||
          (alpha->type != ggml::kQ8_0 && alpha->type != ggml::kF32)) {
        reject(p + "ssm_alpha.weight", alpha);
        reject(p + "ssm_beta.weight", beta);
      }
    }
    if (geometry.sparseMoe()) {
      for (const char *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                               "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"})
        check(p + name);
      checkFloat(p + "ffn_gate_inp.weight");
      checkFloat(p + "ffn_gate_inp_shexp.weight");
    } else {
      for (const char *name : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) check(p + name);
    }
  }
  check("output.weight");
  check("token_embd.weight", true);
  if (!unsupported.empty()) throw GgufError("GGUF tensors this build cannot load: " + unsupported);
}

Image ImagePlanner::layer(uint32_t index) const {
  const TargetGeometry &g = geometry_;
  const std::string p = prefix(index);
  const bool full = g.isFullAttentionLayer(index);
  Builder b(file_, g, "layer-" + std::to_string(index) + ".bin", index, full ? 1u : 0u);
  b.floatNorm(p + "attn_norm.weight", g.hiddenSize);
  if (full) {
    b.quantized(file_.require(p + "attn_q.weight"), 2ull * g.attentionHeadDimension * (g.attentionWidth / g.attentionHeadDimension), g.hiddenSize);
    const uint64_t kvRows = uint64_t{g.attentionKvHeads} * g.attentionHeadDimension;
    b.quantized(file_.require(p + "attn_k.weight"), kvRows, g.hiddenSize);
    b.quantized(file_.require(p + "attn_v.weight"), kvRows, g.hiddenSize);
    b.floatNorm(p + "attn_q_norm.weight", g.attentionHeadDimension);
    b.floatNorm(p + "attn_k_norm.weight", g.attentionHeadDimension);
    b.quantized(file_.require(p + "attn_output.weight"), g.hiddenSize, g.attentionWidth);
  } else {
    const uint32_t valueRows = g.gdnValueHeads * g.gdnHeadDimension;
    const uint32_t keyRows = g.convolutionDimension - valueRows; // q and k
    const RowOrder qkvOrder{keyRows, g.gdnHeadDimension, g.gdnKeyHeads, g.gdnValueHeads / g.gdnKeyHeads};
    b.quantized(file_.require(p + "attn_qkv.weight"), g.convolutionDimension, g.hiddenSize, qkvOrder);
    RowOrder gateOrder = qkvOrder;
    gateOrder.from = 0;
    b.quantized(file_.require(p + "attn_gate.weight"), valueRows, g.hiddenSize, gateOrder);
    const GgufTensor &beta = file_.require(p + "ssm_beta.weight"), &alpha = file_.require(p + "ssm_alpha.weight");
    if (beta.type == ggml::kF32) b.floatAlphaBeta(beta, alpha);
    else b.alphaBeta(beta, alpha);
    b.convolution(file_.require(p + "ssm_conv1d.weight"), keyRows);
    b.headVector(p + "ssm_a", false);
    b.headVector(p + "ssm_dt.bias", true);
    b.floatNorm(p + "ssm_norm.weight", g.gdnHeadDimension);
    b.quantized(file_.require(p + "ssm_out.weight"), g.hiddenSize, valueRows);
  }
  b.floatNorm(p + "post_attention_norm.weight", g.hiddenSize);
  if (g.sparseMoe()) {
    const uint64_t routed = g.experts, width = g.expertIntermediateSize;
    b.floatTensor(file_.require(p + "ffn_gate_inp.weight"), routed, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_gate_exps.weight"), routed * width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up_exps.weight"), routed * width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down_exps.weight"), routed * g.hiddenSize, width);
    b.quantized(file_.require(p + "ffn_gate_shexp.weight"), width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up_shexp.weight"), width, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down_shexp.weight"), g.hiddenSize, width);
    b.floatTensor(file_.require(p + "ffn_gate_inp_shexp.weight"), 1, g.hiddenSize);
  } else {
    b.quantized(file_.require(p + "ffn_gate.weight"), g.intermediateSize, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_up.weight"), g.intermediateSize, g.hiddenSize);
    b.quantized(file_.require(p + "ffn_down.weight"), g.hiddenSize, g.intermediateSize);
  }
  return b.finish();
}

Image ImagePlanner::head() const {
  Builder b(file_, geometry_, "head.bin", geometry_.layers, 2);
  b.floatNorm("output_norm.weight", geometry_.hiddenSize);
  b.quantized(file_.require("output.weight"), geometry_.vocabularySize, geometry_.hiddenSize);
  return b.finish();
}

Image ImagePlanner::embedding() const {
  Builder b(file_, geometry_, "embedding.bin", geometry_.vocabularySize, geometry_.hiddenSize);
  b.embeddingRows(file_.require("token_embd.weight"));
  return b.finish();
}

uint64_t ImagePlanner::totalBytes() const {
  uint64_t total = head().bytes + embedding().bytes;
  for (uint32_t i = 0; i < geometry_.layers; ++i) total += layer(i).bytes;
  return total;
}

} // namespace splash::model::gguf
