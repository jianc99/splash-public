#include "model/VisionPreparation.hpp"
#include "WeightPreparationIdentity.hpp"
#include "model/GgufFile.hpp"
#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightStore.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <map>
#include <sstream>

namespace splash::model {
namespace {

struct Section {
  std::string mlx, gguf;
  uint32_t rows, columns, storedRows, storedColumns;
  bool patch = false;
  uint64_t offset = 0;
  uint32_t unit = 2;
};

std::vector<Section> sections(const ops::VisionLayout &l) {
  std::vector<Section> result;
  const auto add = [&](std::string mlx, std::string gguf, uint32_t rows,
                       uint32_t columns = 1, uint32_t paddedRows = 0,
                       uint32_t paddedColumns = 0) {
    result.push_back({"vision_tower." + mlx, std::move(gguf), rows, columns,
                      paddedRows ? paddedRows : rows,
                      paddedColumns ? paddedColumns : columns});
  };
  const auto affine = [&](const std::string &mlx, const std::string &gguf,
                          uint32_t rows, uint32_t columns,
                          uint32_t paddedRows = 0, uint32_t paddedColumns = 0) {
    add(mlx + ".weight", gguf + ".weight", rows, columns, paddedRows,
        paddedColumns);
    add(mlx + ".bias", gguf + ".bias", rows, 1, paddedRows);
  };
  const auto norm = [&](const std::string &mlx, const std::string &gguf) {
    add(mlx + ".weight", gguf + ".weight", l.hiddenSize);
    add(mlx + ".bias", gguf + ".bias", l.hiddenSize);
  };
  affine("patch_embed.proj", "v.patch_embd", l.hiddenSize, l.patchDimension);
  result.front().patch = true;
  add("pos_embed.weight", "v.position_embd.weight",
      l.positionGridSide * l.positionGridSide, l.hiddenSize);
  for (uint32_t i = 0; i < l.depth; ++i) {
    const auto mlx = "blocks." + std::to_string(i) + ".",
               gguf = "v.blk." + std::to_string(i) + ".";
    norm(mlx + "norm1", gguf + "ln1");
    affine(mlx + "attn.qkv", gguf + "attn_qkv", 3 * l.hiddenSize, l.hiddenSize);
    affine(mlx + "attn.proj", gguf + "attn_out", l.hiddenSize, l.hiddenSize);
    norm(mlx + "norm2", gguf + "ln2");
    affine(mlx + "mlp.linear_fc1", gguf + "ffn_up", l.intermediateSize,
           l.hiddenSize, l.paddedIntermediateSize);
    affine(mlx + "mlp.linear_fc2", gguf + "ffn_down", l.hiddenSize,
           l.intermediateSize, 0, l.paddedIntermediateSize);
  }
  norm("merger.norm", "v.post_ln");
  affine("merger.linear_fc1", "mm.0", l.mergedHiddenSize, l.mergedHiddenSize);
  affine("merger.linear_fc2", "mm.2", l.outputHiddenSize, l.mergedHiddenSize);
  return result;
}

uint64_t arrange(std::vector<Section> &plan) {
  uint64_t at = kWeightFileAlignment;
  for (auto &s : plan) {
    s.offset = at;
    at += uint64_t(s.storedRows) * s.storedColumns * s.unit;
    at = (at + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1);
  }
  return at;
}

uint32_t elementBytes(const SourceTensor &tensor) {
  if (tensor.dtype == "BF16" || tensor.dtype == "F16")
    return 2;
  if (tensor.dtype == "F32")
    return 4;
  throw WeightStoreError("vision requires BF16, F16 or F32 weights");
}

float scalar(const uint8_t *bytes, const std::string &type) {
  float value;
  if (type == "BF16") {
    uint16_t bits;
    std::memcpy(&bits, bytes, 2);
    value = std::bit_cast<float>(uint32_t(bits) << 16);
  } else if (type == "F16") {
    _Float16 half;
    std::memcpy(&half, bytes, 2);
    value = half;
  } else
    std::memcpy(&value, bytes, 4);
  return value;
}

void writeSection(int destination, const Section &s, const SourceTensor &a,
                  const SourceTensor *b, const ops::VisionLayout &layout,
                  const PreparationCheck &check) {
  const uint32_t unit = elementBytes(a);
  const uint32_t sourceColumns = s.patch && b ? s.columns / 2 : s.columns;
  const uint32_t batchRows = std::max(
      1u, uint32_t((8 * 1024 * 1024) / (uint64_t(s.columns) * unit +
                                        uint64_t(s.storedColumns) * s.unit)));
  std::vector<uint8_t> input(uint64_t(batchRows) * sourceColumns * unit),
      second;
  if (b)
    second.resize(input.size());
  std::vector<uint8_t> output(uint64_t(batchRows) * s.storedColumns * s.unit);
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    if (check)
      check();
    const uint32_t count = std::min(batchRows, s.rows - row);
    const uint64_t bytes = uint64_t(count) * sourceColumns * unit;
    a.read(uint64_t(row) * sourceColumns * unit, std::span(input).first(bytes));
    if (b)
      b->read(uint64_t(row) * sourceColumns * unit,
              std::span(second).first(bytes));
    std::fill(output.begin(), output.end(), 0);
    for (uint32_t r = 0; r < count; ++r) {
      if (!s.patch && ((s.unit == 2 && a.dtype == "BF16") ||
                       (s.unit == 4 && a.dtype == "F32"))) {
        std::memcpy(output.data() + uint64_t(r) * s.storedColumns * s.unit,
                    input.data() + uint64_t(r) * s.columns * unit,
                    uint64_t(s.columns) * unit);
        continue;
      }
      const uint32_t pixels = layout.patchSize * layout.patchSize;
      for (uint32_t c = 0; c < s.columns; ++c) {
        uint32_t source = c;
        const uint8_t *data = input.data();
        if (s.patch) {
          const uint32_t channel = c / (2 * pixels), time = (c / pixels) % 2,
                         pixel = c % pixels;
          if (b) {
            source = channel * pixels + pixel;
            if (time)
              data = second.data();
          } else
            source = (time * pixels + pixel) * 3 + channel;
        }
        const auto *from = data + (uint64_t(r) * sourceColumns + source) * unit;
        auto *to = output.data() + (uint64_t(r) * s.storedColumns + c) * s.unit;
        if (s.unit == 2)
          std::memcpy(to, from, 2);
        else {
          const float value = scalar(from, a.dtype);
          std::memcpy(to, &value, 4);
        }
      }
    }
    writeWeightBytes(destination,
                     s.offset + uint64_t(row) * s.storedColumns * s.unit,
                     {reinterpret_cast<const uint8_t *>(output.data()),
                      uint64_t(count) * s.storedColumns * s.unit});
  }
  // Preallocated files start zeroed, including padded output rows and
  // alignment.
}

} // namespace

uint64_t preparedVisionBytes(const std::filesystem::path &directory,
                             VisionSource format,
                             const ops::VisionLayout &layout) {
  auto plan = sections(layout);
  if (format == VisionSource::Gguf) {
    const GgufFile file(directory / "mmproj.gguf");
    for (auto &s : plan) {
      const auto &tensor = file.require(s.gguf);
      s.unit = tensor.type == ggml::kBF16 && s.columns > 1 &&
                       s.gguf != "v.position_embd.weight"
                   ? 2
                   : 4;
    }
  }
  return arrange(plan);
}

PreparedVision prepareVisionWeights(const std::filesystem::path &directory,
                                    VisionSource format,
                                    const ops::VisionLayout &layout,
                                    const PreparationCheck &check,
                                    const PreparationCheck &prepareCheck) {
  auto plan = sections(layout);
  if (format == VisionSource::Packed)
    return {directory / "model.bin",
            std::vector<ops::VisionPrecision>(plan.size(),
                                              ops::VisionPrecision::BFloat16),
            arrange(plan)};
  if (format != VisionSource::Safetensors && format != VisionSource::Gguf)
    throw WeightStoreError("vision source is disabled");
  std::unique_ptr<SafetensorsCheckpoint> mlx;
  std::unique_ptr<WeightSource> file;
  std::map<std::string, SourceTensor> tensors;
  std::string digest;
  if (format == VisionSource::Safetensors) {
    mlx = std::make_unique<SafetensorsCheckpoint>(directory, check);
    digest = mlx->digest();
    for (const auto &s : plan) {
      const auto &t = mlx->require(s.mlx);
      const std::vector<uint64_t> shape =
          s.patch          ? std::vector<uint64_t>{s.rows, 2, layout.patchSize,
                                                   layout.patchSize, 3}
          : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                           : std::vector<uint64_t>{s.rows, s.columns};
      if (t.shape != shape)
        throw WeightStoreError("vision tensor shape mismatch: " + s.mlx);
      if (t.dtype != "BF16")
        throw WeightStoreError("MLX vision currently requires BF16 tensors: " +
                               s.mlx);
      tensors.emplace(s.mlx, t);
    }
  } else {
    const auto path = directory / "mmproj.gguf";
    file = std::make_unique<WeightSource>(path, check);
    const GgufFile gguf(path);
    if (gguf.architecture() != "clip" ||
        gguf.stringValue("clip.projector_type") != "qwen3vl_merger")
      throw WeightStoreError("unsupported vision GGUF architecture");
    for (const auto &[key, expected] :
         std::initializer_list<std::pair<const char *, uint64_t>>{
             {"clip.vision.projection_dim", layout.outputHiddenSize},
             {"clip.vision.patch_size", layout.patchSize},
             {"clip.vision.embedding_length", layout.hiddenSize},
             {"clip.vision.feed_forward_length", layout.intermediateSize},
             {"clip.vision.block_count", layout.depth},
             {"clip.vision.attention.head_count", layout.heads},
             {"clip.vision.spatial_merge_size", layout.spatialMerge},
             {"clip.use_gelu", 1}})
      if (gguf.unsignedValue(key) != expected)
        throw WeightStoreError(std::string("vision metadata mismatch: ") + key);
    const auto epsilon =
        gguf.floatValue("clip.vision.attention.layer_norm_epsilon");
    if (!epsilon || !std::isfinite(*epsilon) ||
        std::abs(*epsilon - 1e-6) > 1e-12)
      throw WeightStoreError("vision LayerNorm epsilon mismatch");
    for (const char *key :
         {"clip.vision.image_mean", "clip.vision.image_std"}) {
      const auto values = gguf.numericArray(key);
      if (values.size() != 3 || !std::all_of(values.begin(), values.end(),
                                             [](double x) { return x == 0.5; }))
        throw WeightStoreError("vision image normalization mismatch");
    }
    const auto deepstack = gguf.numericArray("clip.vision.is_deepstack_layers");
    if (!std::all_of(deepstack.begin(), deepstack.end(),
                     [](double x) { return x == 0; }))
      throw WeightStoreError("vision deepstack layers are unsupported");
    digest = file->digest();
    for (auto &s : plan) {
      const uint32_t pieces = s.patch ? 2 : 1;
      for (uint32_t piece = 0; piece < pieces; ++piece) {
        const std::string name = s.gguf + (piece ? ".1" : "");
        const auto &t = gguf.require(name);
        const std::vector<uint64_t> shape =
            s.patch ? std::vector<uint64_t>{layout.patchSize, layout.patchSize,
                                            3, s.rows}
            : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                             : std::vector<uint64_t>{s.columns, s.rows};
        if (t.dims != shape)
          throw WeightStoreError("vision tensor shape mismatch: " + name);
        const std::string type = t.type == ggml::kBF16  ? "BF16"
                                 : t.type == ggml::kF16 ? "F16"
                                 : t.type == ggml::kF32 ? "F32"
                                                        : "unsupported";
        SourceTensor tensor{
            file.get(), type, {}, gguf.absoluteOffset(t), t.bytes};
        elementBytes(tensor);
        tensors.emplace(name, std::move(tensor));
      }
      s.unit = tensors.at(s.gguf).dtype == "BF16" && s.columns > 1 &&
                       s.gguf != "v.position_embd.weight"
                   ? 2
                   : 4;
      if (s.patch &&
          tensors.at(s.gguf).dtype != tensors.at(s.gguf + ".1").dtype)
        throw WeightStoreError("vision patch tensors have different dtypes");
    }
  }
  const uint64_t bytes = arrange(plan);
  const auto unchanged = [&] {
    if (mlx)
      mlx->checkUnchanged();
    else
      file->checkUnchanged();
  };
  unchanged();
  std::ostringstream identity;
  identity << SPLASH_VISION_PREPARATION_ID << '\n'
           << digest << '\n'
           << int(format) << '\n';
  for (const auto &s : plan)
    identity << s.mlx << ' ' << s.rows << ' ' << s.columns << ' '
             << s.storedRows << ' ' << s.storedColumns << ' ' << s.unit << '\n';
  const auto text = identity.str();
  PreparedWeight weight{
      weightDigest(
          {reinterpret_cast<const uint8_t *>(text.data()), text.size()}),
      bytes, "vision.bin", directory.string()};
  PreparedWeights cache;
  const auto path = cache.prepare(
      weight,
      [&](int output) {
        std::array<uint8_t, 16> header{};
        std::memcpy(header.data(), "MDFV0001", 8);
        std::memcpy(header.data() + 8, &layout.depth, 4);
        // Kind 1 uses source-derived tensor precisions; legacy kind 0 is all
        // BF16.
        const uint32_t kind = format == VisionSource::Gguf ? 1 : 0;
        std::memcpy(header.data() + 12, &kind, 4);
        writeWeightBytes(output, 0, header);
        for (const auto &s : plan) {
          const auto &name = mlx ? s.mlx : s.gguf;
          const SourceTensor *second =
              !mlx && s.patch ? &tensors.at(name + ".1") : nullptr;
          writeSection(output, s, tensors.at(name), second, layout,
                       prepareCheck);
        }
        unchanged();
      },
      check, prepareCheck);
  unchanged();
  PreparedVision result{path, {}, bytes};
  for (const auto &s : plan)
    result.precision.push_back(s.unit == 4 ? ops::VisionPrecision::Float32
                                           : ops::VisionPrecision::BFloat16);
  return result;
}

} // namespace splash::model
