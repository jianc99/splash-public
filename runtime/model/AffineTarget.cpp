#include "WeightPreparationIdentity.hpp"
#include "model/AffineTarget.hpp"
#include "model/Qwen3_8.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <sstream>
#include <variant>

namespace splash::model {
namespace {

constexpr uint32_t kTileRows = 256;
constexpr uint64_t kChunkBytes = 16 * 1024 * 1024;
uint64_t align(uint64_t bytes) { return (bytes + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1); }

enum class SectionKind { Copy, Decay, Projection };
struct ProjectionPart {
  std::string name;
  uint32_t rows = 0;
  std::array<const SourceTensor *, 3> fields{};
};
struct Section {
  SectionKind kind = SectionKind::Copy;
  uint64_t offset = 0, bytes = 0;
  const SourceTensor *tensor = nullptr;
  std::vector<ProjectionPart> parts;
  uint32_t rows = 0, columns = 0, experts = 1, bits = 4;
};
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = kWeightFileAlignment;
  std::vector<Section> sections;
  Image(std::string name, std::string magic, uint32_t layer, uint32_t type)
      : name(std::move(name)), magic(std::move(magic)), layer(layer), type(type) {}
  void append(Section section) {
    section.offset = bytes;
    bytes = align(bytes + section.bytes);
    sections.push_back(std::move(section));
  }
};

const SourceTensor &requireTensor(const SafetensorsCheckpoint &source, const std::string &name,
                                 const std::string &dtype, const std::vector<uint64_t> &shape) {
  const auto &tensor = source.require(name);
  if (tensor.dtype != dtype || tensor.shape != shape)
    throw WeightStoreError("source tensor type or shape does not match: " + name);
  return tensor;
}

void copy(Image &image, const SafetensorsCheckpoint *source, const std::string &name,
           const std::vector<uint64_t> &shape) {
  Section section;
  section.bytes = 2;
  for (uint64_t dimension : shape) section.bytes = checkedWeightMultiply(section.bytes, dimension, "affine tensor");
  if (source) section.tensor = &requireTensor(*source, name, "BF16", shape);
  image.append(std::move(section));
}

void projection(Image &image, const SafetensorsCheckpoint *source,
                 std::initializer_list<std::pair<std::string, uint32_t>> parts,
                 uint32_t rows, uint32_t columns, uint32_t bits = 4, uint32_t experts = 1) {
  validateQ4Layout(rows, columns);
  Section section;
  section.kind = SectionKind::Projection;
  section.rows = rows;
  section.columns = columns;
  section.bits = bits;
  section.experts = experts;
  section.bytes = checkedWeightMultiply(uint64_t(rows) * columns * bits / 8 +
                                        uint64_t(rows) * columns / 16, experts, "affine projection");
  uint64_t sourceRows = 0;
  for (const auto &[name, count] : parts) {
    if (source) source->requireQuantization(name, bits);
    ProjectionPart part{name, count, {}};
    for (size_t field = 0; field < part.fields.size(); ++field) {
      std::vector<uint64_t> shape{count, field ? columns / 64 : columns * bits / 32};
      if (experts > 1) shape.insert(shape.begin(), experts);
      if (source) part.fields[field] = &requireTensor(*source, name + (field == 0 ? ".weight" : field == 1 ? ".scales" : ".biases"),
                                          field ? "BF16" : "U32", shape);
    }
    sourceRows += count;
    section.parts.push_back(std::move(part));
  }
  if (!experts || sourceRows > rows || rows - sourceRows >= kTileRows)
    throw WeightStoreError("invalid affine projection padding");
  image.append(std::move(section));
}

template<class Layout>
void validateConfiguration(const SafetensorsCheckpoint &source, const Layout &layout) {
  const std::pair<const char *, double> fields[] = {
      {"num_hidden_layers", layout.layers}, {"hidden_size", layout.hiddenSize},
      {"vocab_size", layout.vocabularySize}, {"head_dim", layout.attentionHeadDimension},
      {"num_attention_heads", layout.attentionQueryHeads}, {"num_key_value_heads", layout.attentionKvHeads},
      {"linear_num_key_heads", layout.gdnKeyHeads}, {"linear_num_value_heads", layout.gdnValueHeads},
      {"linear_key_head_dim", layout.gdnHeadDimension}, {"linear_value_head_dim", layout.gdnHeadDimension},
      {"linear_conv_kernel_dim", 4}, {"full_attention_interval", layout.fullAttentionPeriod},
      {"rms_norm_eps", 1e-6}, {"attention_bias", 0}, {"attn_output_gate", 1},
      {"tie_word_embeddings", 0}, {"rope_parameters.rope_theta", layout.rotaryTheta},
      {"rope_parameters.partial_rotary_factor", double(layout.rotaryPairs * 2) / layout.attentionHeadDimension}};
  for (const auto &[key, value] : fields) source.requireConfigNumber(key, value);
  source.requireConfigString("hidden_act", "silu");
  source.requireConfigString("rope_parameters.rope_type", "default");
  source.requireLayerTypes(layout.layers, layout.fullAttentionPeriod);
  if constexpr (requires { layout.experts; }) {
    source.requireConfigString("model_type", "qwen3_5_moe_text");
    source.requireConfigNumber("num_experts", layout.experts);
    source.requireConfigNumber("num_experts_per_tok", layout.expertsPerToken);
    source.requireConfigNumber("moe_intermediate_size", layout.expertIntermediateSize);
    source.requireConfigNumber("shared_expert_intermediate_size", layout.expertIntermediateSize);
  } else {
    source.requireConfigString("model_type", "qwen3_5_text");
    source.requireConfigNumber("intermediate_size", layout.intermediateSize);
  }
}

template<class Layout>
Image layerImage(const SafetensorsCheckpoint *source, const Layout &layout, uint32_t layer) {
  if (layer >= layout.layers) throw WeightStoreError("target layer is out of range");
  const bool full = layout.isFullAttentionLayer(layer);
  Image image{"layer-" + std::to_string(layer) + ".bin", std::string(Layout::layerMagic), layer, full ? 1u : 0u};
  const std::string prefix = "language_model.model.layers." + std::to_string(layer) + ".";
  copy(image, source, prefix + "input_layernorm.weight", {layout.hiddenSize});
  if (full) {
    const std::string attention = prefix + "self_attn.";
    projection(image, source, {{attention + "q_proj", 2 * layout.attentionWidth},
                                {attention + "k_proj", layout.attentionKvHeads * layout.attentionHeadDimension},
                                {attention + "v_proj", layout.attentionKvHeads * layout.attentionHeadDimension}},
               layout.packedFullWidth, layout.hiddenSize);
    copy(image, source, attention + "q_norm.weight", {layout.attentionHeadDimension});
    copy(image, source, attention + "k_norm.weight", {layout.attentionHeadDimension});
    projection(image, source, {{attention + "o_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth);
  } else {
    const std::string gdn = prefix + "linear_attn.";
    projection(image, source, {{gdn + "in_proj_qkv", layout.convolutionDimension},
                                {gdn + "in_proj_z", layout.attentionWidth},
                                {gdn + "in_proj_b", layout.gdnValueHeads},
                                {gdn + "in_proj_a", layout.gdnValueHeads}},
               layout.packedGdnWidth, layout.hiddenSize);
    copy(image, source, gdn + "conv1d.weight", {layout.convolutionDimension, 4, 1});
    Section decay;
    decay.kind = SectionKind::Decay;
    if (source) {
      const auto &a = source->require(gdn + "A_log");
      if (a.shape != std::vector<uint64_t>{layout.gdnValueHeads} || (a.dtype != "BF16" && a.dtype != "F32"))
        throw WeightStoreError("invalid GDN decay source");
      decay.tensor = &a;
    }
    decay.bytes = uint64_t(layout.gdnValueHeads) * sizeof(float);
    image.append(std::move(decay));
    copy(image, source, gdn + "dt_bias", {layout.gdnValueHeads});
    copy(image, source, gdn + "norm.weight", {layout.gdnHeadDimension});
    projection(image, source, {{gdn + "out_proj", layout.hiddenSize}}, layout.hiddenSize, layout.attentionWidth);
  }
  copy(image, source, prefix + "post_attention_layernorm.weight", {layout.hiddenSize});
  const std::string mlp = prefix + "mlp.";
  if constexpr (requires { layout.experts; }) {
    projection(image, source, {{mlp + "gate", layout.experts}}, 256, layout.hiddenSize, 8);
    for (const std::string name : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = name == "down_proj";
      const uint32_t n = down ? layout.hiddenSize : layout.expertIntermediateSize;
      const uint32_t k = down ? layout.expertIntermediateSize : layout.hiddenSize;
      projection(image, source, {{mlp + "switch_mlp." + name, n}}, n, k, 4, layout.experts);
    }
    for (const std::string name : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = name == "down_proj";
      const uint32_t n = down ? layout.hiddenSize : layout.expertIntermediateSize;
      const uint32_t k = down ? layout.expertIntermediateSize : layout.hiddenSize;
      projection(image, source, {{mlp + "shared_expert." + name, n}}, n, k);
    }
    projection(image, source, {{mlp + "shared_expert_gate", 1}}, 256, layout.hiddenSize, 8);
  } else {
    for (const std::string name : {"gate_proj", "up_proj", "down_proj"}) {
      const bool down = name == "down_proj";
      const uint32_t n = down ? layout.hiddenSize : layout.intermediateSize;
      const uint32_t k = down ? layout.intermediateSize : layout.hiddenSize;
      projection(image, source, {{mlp + name, n}}, n, k);
    }
  }
  return image;
}

template<class Layout>
Image headImage(const SafetensorsCheckpoint *source, const Layout &layout) {
  constexpr bool moe = requires { layout.experts; };
  Image image{"head.bin", moe ? "MDFM0002" : "MDFL0002", layout.layers, 2};
  copy(image, source, "language_model.model.norm.weight", {layout.hiddenSize});
  projection(image, source, {{"language_model.lm_head", layout.vocabularySize}}, layout.vocabularySize, layout.hiddenSize);
  return image;
}

template<class Layout>
Image embeddingImage(const SafetensorsCheckpoint *source, const Layout &layout) {
  Image image{"embedding.bin", "MDFE0001", layout.vocabularySize, layout.hiddenSize};
  const std::string prefix = "language_model.model.embed_tokens";
  if (source) source->requireQuantization(prefix, 4);
  for (const std::string field : {"weight", "scales", "biases"}) {
    const bool weight = field == "weight";
    Section section;
    section.bytes = uint64_t(layout.vocabularySize) * layout.hiddenSize / (weight ? 2 : 32);
    if (source) section.tensor = &requireTensor(*source, prefix + "." + field, weight ? "U32" : "BF16",
        {layout.vocabularySize, layout.hiddenSize / (weight ? 8 : 64)});
    image.append(std::move(section));
  }
  return image;
}

std::string imageKey(const SafetensorsCheckpoint &source, const Image &image) {
  std::ostringstream identity;
  identity << "splash-affine64-preparation-v1\n" SPLASH_AFFINE_PREPARATION_ID "\n" << source.digest() << '\n'
           << image.magic << ' ' << image.layer << ' ' << image.type << ' ' << image.bytes << '\n';
  for (const auto &section : image.sections) {
    identity << int(section.kind) << ' ' << section.offset << ' ' << section.bytes << ' '
             << section.rows << ' ' << section.columns << ' ' << section.experts << ' ' << section.bits << '\n';
    if (section.tensor) identity << section.tensor->file->digest() << ' ' << section.tensor->offset << '\n';
    for (const auto &part : section.parts) identity << part.name << ' ' << part.rows << '\n';
  }
  const std::string text = identity.str();
  return weightDigest({reinterpret_cast<const uint8_t *>(text.data()), text.size()});
}

void writeProjection(int destination, const Section &section, const PreparationCheck &check) {
  const uint32_t groups = section.columns / 64;
  const std::array<uint32_t, 3> unit{8 * section.bits, 2, 2};
  const uint64_t expertBytes = section.bytes / section.experts;
  for (uint32_t expert = 0; expert < section.experts; ++expert) {
    uint64_t fieldBase = section.offset + expert * expertBytes;
    for (size_t field = 0; field < unit.size(); ++field) {
      if (check) check();
      const uint32_t chunkGroups = std::min<uint64_t>(groups, kChunkBytes / (kTileRows * unit[field]));
      const uint64_t maximumBytes = uint64_t(kTileRows) * chunkGroups * unit[field];
      std::vector<uint8_t> input(maximumBytes), output(maximumBytes);
      for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kTileRows) {
        for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += chunkGroups) {
          if (check) check();
          const uint32_t count = std::min(chunkGroups, groups - firstGroup);
          const uint32_t rowBytes = count * unit[field];
          std::fill(input.begin(), input.end(), 0);
          uint32_t partBegin = 0;
          for (const auto &part : section.parts) {
            const uint32_t begin = std::max(firstRow, partBegin);
            const uint32_t end = std::min(firstRow + kTileRows, partBegin + part.rows);
            if (begin < end) {
              const uint64_t offset = (uint64_t(expert) * part.rows + begin - partBegin) * groups * unit[field] +
                                      uint64_t(firstGroup) * unit[field];
              if (count == groups)
                part.fields[field]->read(offset, {input.data() + uint64_t(begin - firstRow) * rowBytes,
                                                 uint64_t(end - begin) * rowBytes});
              else
                for (uint32_t row = begin; row < end; ++row)
                  part.fields[field]->read(offset + uint64_t(row - begin) * groups * unit[field],
                      {input.data() + uint64_t(row - firstRow) * rowBytes, rowBytes});
            }
            partBegin += part.rows;
          }
          for (uint32_t group = 0; group < count; ++group)
            for (uint32_t row = 0; row < kTileRows; ++row)
              std::memcpy(output.data() + (uint64_t(group) * kTileRows + row) * unit[field],
                          input.data() + uint64_t(row) * rowBytes + group * unit[field], unit[field]);
          const uint64_t offset = (uint64_t(firstRow / kTileRows) * groups + firstGroup) * kTileRows * unit[field];
          writeWeightBytes(destination, fieldBase + offset, std::span(output).first(uint64_t(kTileRows) * rowBytes));
        }
      }
      fieldBase += uint64_t(section.rows) * groups * unit[field];
    }
  }
}

void writeImage(int destination, const Image &image, const PreparationCheck &check) {
  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), image.magic.data(), 8);
  std::memcpy(header.data() + 8, &image.layer, 4);
  std::memcpy(header.data() + 12, &image.type, 4);
  writeWeightBytes(destination, 0, header);
  for (const auto &section : image.sections) {
    if (check) check();
    if (section.kind == SectionKind::Projection) {
      writeProjection(destination, section, check);
    } else if (section.kind == SectionKind::Decay) {
      if (section.bytes > 1024 * 1024) throw WeightStoreError("decay tensor exceeds preparation bound");
      std::vector<uint8_t> bytes(section.tensor->bytes);
      section.tensor->read(0, bytes);
      std::vector<float> values(section.bytes / sizeof(float));
      for (size_t i = 0; i < values.size(); ++i) {
        float logarithm;
        if (section.tensor->dtype == "BF16") {
          uint16_t bfloat;
          std::memcpy(&bfloat, bytes.data() + i * 2, 2);
          const uint32_t bits = uint32_t(bfloat) << 16;
          std::memcpy(&logarithm, &bits, 4);
        } else std::memcpy(&logarithm, bytes.data() + i * 4, 4);
        values[i] = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
        if (!std::isfinite(values[i])) throw WeightStoreError("non-finite GDN decay");
      }
      writeWeightBytes(destination, section.offset,
                       {reinterpret_cast<const uint8_t *>(values.data()), section.bytes});
    } else {
      std::vector<uint8_t> bytes(1024 * 1024);
      for (uint64_t at = 0; at < section.bytes; at += bytes.size()) {
        if (check) check();
        auto part = std::span(bytes).first(std::min<uint64_t>(bytes.size(), section.bytes - at));
        section.tensor->read(at, part);
        writeWeightBytes(destination, section.offset + at, part);
      }
    }
  }
}

template<class Layout>
uint64_t preparedBytes(const Layout &layout) {
  uint64_t bytes = headImage(nullptr, layout).bytes + embeddingImage(nullptr, layout).bytes;
  for (uint32_t layer = 0; layer < layout.layers; ++layer) bytes += layerImage(nullptr, layout, layer).bytes;
  return bytes;
}

} // namespace

struct AffineTargetLoader::Impl {
  metal::MetalBackend &backend;
  PreparationCheck check;
  SafetensorsCheckpoint source;
  std::filesystem::path sourceDirectory;
  std::variant<Qwen3_8Layout, Qwen3_6MoeLayout> layout;
  PreparedWeights cache;
  template<class Layout>
  Impl(metal::MetalBackend &backend, const std::filesystem::path &directory,
       const Layout &layout, PreparationCheck check, std::span<const PreparedWeight> alsoPrepared)
      : backend(backend), check(std::move(check)),
        source(directory, [&backend] { backend.checkOperation(); }), sourceDirectory(directory), layout(layout) {
    validateConfiguration(source, layout);
    std::vector<PreparedWeight> weights(alsoPrepared.begin(), alsoPrepared.end());
    const auto include = [&](const Image &image) {
      weights.push_back({imageKey(source, image), image.bytes});
    };
    for (uint32_t layer = 0; layer < layout.layers; ++layer) {
      backend.checkOperation();
      include(layerImage(&source, layout, layer));
    }
    include(headImage(&source, layout));
    include(embeddingImage(&source, layout));
    cache.requireSpace(weights, [&backend] { backend.checkOperation(); });
  }
  WeightFile load(const Image &image) {
    source.checkUnchanged();
    const auto key = imageKey(source, image);
    const auto path = cache.prepare({key, image.bytes, image.name, sourceDirectory.string()}, [&](int output) {
      writeImage(output, image, check);
      source.checkUnchanged();
    }, [&] { backend.checkOperation(); }, check);
    source.checkUnchanged();
    return WeightFile(backend, path, "target/" + image.name, image.magic, image.layer, image.type, key);
  }
};
AffineTargetLoader::AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                       const Qwen3_8Layout &layout, PreparationCheck check,
                                       std::span<const PreparedWeight> alsoPrepared)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(check), alsoPrepared)) {}
AffineTargetLoader::AffineTargetLoader(metal::MetalBackend &backend, const std::filesystem::path &directory,
                                       const Qwen3_6MoeLayout &layout, PreparationCheck check,
                                       std::span<const PreparedWeight> alsoPrepared)
    : impl_(std::make_unique<Impl>(backend, directory, layout, std::move(check), alsoPrepared)) {}
AffineTargetLoader::~AffineTargetLoader() = default;
WeightFile AffineTargetLoader::layer(uint32_t index, bool fullAttention) {
  return std::visit([&](const auto &layout) {
    if (layout.isFullAttentionLayer(index) != fullAttention) throw WeightStoreError("affine layer kind mismatch");
    return impl_->load(layerImage(&impl_->source, layout, index));
  }, impl_->layout);
}
WeightFile AffineTargetLoader::head(uint32_t layers) {
  return std::visit([&](const auto &layout) {
    if (layers != layout.layers) throw WeightStoreError("affine layer count mismatch");
    return impl_->load(headImage(&impl_->source, layout));
  }, impl_->layout);
}
WeightFile AffineTargetLoader::embedding(uint32_t vocabulary, uint32_t hidden) {
  return std::visit([&](const auto &layout) {
    if (vocabulary != layout.vocabularySize || hidden != layout.hiddenSize) throw WeightStoreError("affine embedding shape mismatch");
    return impl_->load(embeddingImage(&impl_->source, layout));
  }, impl_->layout);
}

uint64_t preparedAffineBytes(const Qwen3_8Layout &layout) { return preparedBytes(layout); }
uint64_t preparedAffineBytes(const Qwen3_6MoeLayout &layout) { return preparedBytes(layout); }

} // namespace splash::model
