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
#include <span>
#include <sstream>

namespace splash::model {
namespace {

// Staging for one batch of rows: source bytes, patch values and output rows.
constexpr uint64_t kBatchBytes = 8 * 1024 * 1024;

// Converts count values to BF16 bits. False when a value is not exactly a
// BF16: preparation never rounds a weight.
using Conversion = bool (*)(const uint8_t *source, uint16_t *destination,
                            uint64_t count);

bool copyBfloat16(const uint8_t *source, uint16_t *destination,
                  uint64_t count) {
  std::memcpy(destination, source, count * kBFloat16Bytes);
  return true;
}

template <class T>
bool exactBfloat16(const uint8_t *source, uint16_t *destination,
                   uint64_t count) {
  for (uint64_t i = 0; i < count; ++i) {
    T value;
    std::memcpy(&value, source + i * sizeof(T), sizeof(T));
    const auto bits = std::bit_cast<uint32_t>(static_cast<float>(value));
    if (bits & 0xFFFFu)
      return false;
    destination[i] = static_cast<uint16_t>(bits >> 16);
  }
  return true;
}

// A source tensor and the conversion of its dtype, resolved once.
struct Input {
  std::string name;
  SourceTensor tensor;
  uint32_t elementBytes = 2;
  Conversion convert = copyBfloat16;
};

Input input(std::string name, SourceTensor tensor) {
  Input result{std::move(name), std::move(tensor)};
  const std::string &dtype = result.tensor.dtype;
  if (dtype == "F16") {
    result.convert = exactBfloat16<_Float16>;
  } else if (dtype == "F32") {
    result.elementBytes = 4;
    result.convert = exactBfloat16<float>;
  } else if (dtype != "BF16") {
    throw WeightStoreError("vision tensor " + result.name + " in " +
                           result.tensor.file->path().string() + " is " +
                           dtype + "; preparation reads BF16, F16 or F32");
  }
  return result;
}

struct Section {
  std::string mlx, gguf;
  uint32_t rows, columns, storedRows, storedColumns;
  bool patch = false;
  uint64_t offset = 0;
  // One tensor, or the two temporal frames of a GGUF patch embedding.
  std::vector<Input> inputs{};
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
    at += uint64_t(s.storedRows) * s.storedColumns * kBFloat16Bytes;
    at = (at + kWeightFileAlignment - 1) & ~(kWeightFileAlignment - 1);
  }
  return at;
}

// MLX: vision_tower.*, whose patch embedding is one Conv3d weight [output,
// frame, patch-row, patch-col, channel].
void planCheckpoint(const SafetensorsCheckpoint &checkpoint,
                    const ops::VisionLayout &layout,
                    std::vector<Section> &plan) {
  const uint64_t p = layout.patchSize;
  for (auto &s : plan) {
    const auto &tensor = checkpoint.require(s.mlx);
    const std::vector<uint64_t> shape =
        s.patch          ? std::vector<uint64_t>{s.rows, 2, p, p, 3}
        : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                         : std::vector<uint64_t>{s.rows, s.columns};
    if (tensor.shape != shape)
      throw WeightStoreError("vision tensor shape mismatch: " + s.mlx);
    s.inputs.push_back(input(s.mlx, tensor));
  }
}

// GGUF: a qwen3vl_merger mmproj describing this tower. The patch embedding is
// one [channel, patch-row, patch-col] weight per temporal frame
// (v.patch_embd.weight and .weight.1).
void planMmproj(const GgufFile &gguf, const WeightSource &file,
                const ops::VisionLayout &layout, std::vector<Section> &plan) {
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
  if (!epsilon || !std::isfinite(*epsilon) || std::abs(*epsilon - 1e-6) > 1e-12)
    throw WeightStoreError("vision LayerNorm epsilon mismatch");
  for (const char *key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto values = gguf.numericArray(key);
    if (values.size() != 3 || !std::all_of(values.begin(), values.end(),
                                           [](double x) { return x == 0.5; }))
      throw WeightStoreError("vision image normalization mismatch");
  }
  const auto deepstack = gguf.numericArray("clip.vision.is_deepstack_layers");
  if (std::any_of(deepstack.begin(), deepstack.end(),
                  [](double x) { return x != 0; }))
    throw WeightStoreError("vision deepstack layers are unsupported");
  const uint64_t p = layout.patchSize;
  for (auto &s : plan) {
    for (uint32_t frame = 0; frame < (s.patch ? 2u : 1u); ++frame) {
      const std::string name = s.gguf + (frame ? ".1" : "");
      const GgufTensor &t = gguf.require(name);
      const std::vector<uint64_t> shape =
          s.patch          ? std::vector<uint64_t>{p, p, 3, s.rows}
          : s.columns == 1 ? std::vector<uint64_t>{s.rows}
                           : std::vector<uint64_t>{s.columns, s.rows};
      if (t.dims != shape)
        throw WeightStoreError("vision tensor shape mismatch: " + name);
      const std::string dtype = t.type == ggml::kBF16  ? "BF16"
                                : t.type == ggml::kF16 ? "F16"
                                : t.type == ggml::kF32 ? "F32"
                                                       : ggmlTypeName(t.type);
      s.inputs.push_back(
          input(name, {&file, dtype, {}, gguf.absoluteOffset(t), t.bytes}));
    }
  }
}

// Writes a section in batches of whole rows converted to BF16. The packed
// patch embedding orders a row [channel, frame, patch-row, patch-col]; MLX
// stores [frame, patch-row, patch-col, channel] and GGUF one [channel,
// patch-row, patch-col] tensor per frame. Padded rows, columns and alignment
// stay zero: a prepared file starts zeroed.
void writeSection(int destination, const Section &s, uint32_t pixels,
                  const PreparationCheck &check) {
  const auto frames = static_cast<uint32_t>(s.inputs.size());
  const uint32_t columns = s.columns / frames;
  uint32_t elementBytes = 0;
  for (const auto &in : s.inputs)
    elementBytes = std::max(elementBytes, in.elementBytes);
  const uint64_t storedRowBytes = uint64_t(s.storedColumns) * kBFloat16Bytes;
  const uint64_t rowBytes =
      uint64_t(columns) * elementBytes +
      (s.patch ? uint64_t(s.columns) * kBFloat16Bytes : 0) + storedRowBytes;
  const auto batchRows = static_cast<uint32_t>(
      std::clamp<uint64_t>(kBatchBytes / rowBytes, 1, s.rows));
  std::vector<uint8_t> source(uint64_t(batchRows) * columns * elementBytes);
  std::vector<uint16_t> values(s.patch ? uint64_t(batchRows) * s.columns : 0);
  std::vector<uint16_t> output(uint64_t(batchRows) * s.storedColumns);
  // Converts count rows of one input to rows of stride BF16 values.
  const auto convert = [&](const Input &in, uint32_t row, uint32_t count,
                           uint16_t *to, uint32_t stride) {
    const uint64_t bytes = uint64_t(columns) * in.elementBytes;
    in.tensor.read(row * bytes, std::span(source).first(count * bytes));
    for (uint32_t r = 0; r < count; ++r)
      if (!in.convert(source.data() + r * bytes, to + uint64_t(r) * stride,
                      columns))
        throw WeightStoreError("vision tensor " + in.name + " in " +
                               in.tensor.file->path().string() +
                               " is not exactly representable in BF16");
  };
  for (uint32_t row = 0; row < s.rows; row += batchRows) {
    if (check)
      check();
    const uint32_t count = std::min(batchRows, s.rows - row);
    if (!s.patch) {
      convert(s.inputs.front(), row, count, output.data(), s.storedColumns);
    } else {
      for (uint32_t frame = 0; frame < frames; ++frame)
        convert(s.inputs[frame], row, count,
                values.data() + uint64_t(frame) * count * columns, columns);
      for (uint32_t r = 0; r < count; ++r)
        for (uint32_t c = 0; c < s.columns; ++c) {
          const uint32_t channel = c / (2 * pixels), frame = c / pixels % 2,
                         pixel = c % pixels;
          output[uint64_t(r) * s.storedColumns + c] =
              frames == 1 ? values[uint64_t(r) * columns +
                                   (frame * pixels + pixel) * 3 + channel]
                          : values[(uint64_t(frame) * count + r) * columns +
                                   channel * pixels + pixel];
        }
    }
    writeWeightBytes(destination, s.offset + row * storedRowBytes,
                     {reinterpret_cast<const uint8_t *>(output.data()),
                      count * storedRowBytes});
  }
}

} // namespace

struct VisionPreparation::Impl {
  ops::VisionLayout layout;
  PreparationCheck check;
  std::vector<Section> plan;
  std::unique_ptr<SafetensorsCheckpoint> checkpoint;
  std::unique_ptr<WeightSource> mmproj;
  PreparedWeight weight{};

  void checkUnchanged() const {
    if (checkpoint)
      checkpoint->checkUnchanged();
    else
      mmproj->checkUnchanged();
  }
};

VisionPreparation::VisionPreparation(const std::filesystem::path &directory,
                                     VisionSource source,
                                     const ops::VisionLayout &layout,
                                     PreparationCheck check)
    : impl_(std::make_unique<Impl>()) {
  auto &i = *impl_;
  i.layout = layout;
  i.check = std::move(check);
  // The writer relies on the packed patch width and on padding that only adds
  // rows or columns.
  if (layout.patchDimension != 6 * layout.patchSize * layout.patchSize ||
      layout.paddedIntermediateSize < layout.intermediateSize)
    throw WeightStoreError("Qwen vision layout is inconsistent");
  i.plan = sections(layout);
  std::string digest;
  if (source == VisionSource::Safetensors) {
    i.checkpoint = std::make_unique<SafetensorsCheckpoint>(directory, i.check);
    planCheckpoint(*i.checkpoint, layout, i.plan);
    digest = i.checkpoint->digest();
  } else if (source == VisionSource::Gguf) {
    const auto path = directory / "mmproj.gguf";
    i.mmproj = std::make_unique<WeightSource>(path, i.check);
    planMmproj(GgufFile(path), *i.mmproj, layout, i.plan);
    digest = i.mmproj->digest();
  } else {
    throw WeightStoreError("only MLX and GGUF vision sources are prepared");
  }
  i.checkUnchanged();
  std::ostringstream identity;
  identity << SPLASH_VISION_PREPARATION_ID << '\n'
           << digest << '\n'
           << int(source) << '\n';
  for (const auto &s : i.plan)
    identity << s.mlx << ' ' << s.rows << ' ' << s.columns << ' '
             << s.storedRows << ' ' << s.storedColumns << '\n';
  const auto text = identity.str();
  i.weight = {weightDigest({reinterpret_cast<const uint8_t *>(text.data()),
                            text.size()}),
              arrange(i.plan), "vision.bin", directory.string()};
}

VisionPreparation::~VisionPreparation() = default;

const ops::VisionLayout &VisionPreparation::layout() const noexcept {
  return impl_->layout;
}

const PreparedWeight &VisionPreparation::weight() const noexcept {
  return impl_->weight;
}

std::filesystem::path
VisionPreparation::prepare(const PreparationCheck &prepareCheck) const {
  const auto &i = *impl_;
  i.checkUnchanged();
  const auto path = PreparedWeights().prepare(
      i.weight,
      [&](int output) {
        // The packed header: magic, block count and file kind 0.
        std::array<uint8_t, 16> header{};
        std::memcpy(header.data(), "MDFV0001", 8);
        std::memcpy(header.data() + 8, &i.layout.depth, 4);
        writeWeightBytes(output, 0, header);
        for (const auto &s : i.plan)
          writeSection(output, s, i.layout.patchSize * i.layout.patchSize,
                       prepareCheck);
        i.checkUnchanged();
      },
      i.check, prepareCheck);
  i.checkUnchanged();
  return path;
}

uint64_t preparedVisionBytes(const ops::VisionLayout &layout) {
  auto plan = sections(layout);
  return arrange(plan);
}

} // namespace splash::model
