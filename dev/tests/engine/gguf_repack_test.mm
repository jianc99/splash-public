// The GGUF image planner and preparation against the CPU reference, and that
// reference's values against hashes of upstream GGML's dequantization: the
// alpha/beta tensor, the float tensors (norms as stored, exact bf16
// conversions refused when inexact), the sparse MoE layer (qwen35moe), every
// format's planes through the production executor, and golden hashes of the
// images the loader prepares.
//   gguf-repack --cpu        the golden dequantization hashes and the plans
//   gguf-repack <metallib>   also the prepared bytes
// With SPLASH_GGML_ORACLE=<libggml-base.dylib> the reference is also compared
// with GGML directly and GGML's hashes are printed; a build of llama.cpp
// 7ab4ee7 regenerates kGolden.
#import <Foundation/Foundation.h>

#include "GgufFormatReference.hpp"
#include "model/GgufImage.hpp"
#include "model/GgufTarget.hpp"
#include "model/GgufPreparation.hpp"
#include "model/Qwen3_8.hpp"
#include <fcntl.h>
#include <unistd.h>

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

using namespace gguf_reference;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("%-64s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  failures += !ok;
}

std::string sha256(const void *data, size_t bytes) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data, static_cast<CC_LONG>(bytes), digest);
  std::string hex;
  for (unsigned char byte : digest) {
    char text[3];
    std::snprintf(text, sizeof text, "%02x", byte);
    hex += text;
  }
  return hex;
}

// Every byte random and every half scale a random finite half: either sign,
// zero and subnormal included.
std::vector<uint8_t> fixture(Fmt f, uint32_t rows, uint32_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  return makeNative(f, rows, K, rng, [&] {
    uint16_t h;
    do h = static_cast<uint16_t>(rng());
    while ((h & 0x7C00) == 0x7C00);
    return h;
  });
}

// SHA-256 of GGML's fp32 dequantization of fixture(f, 256, 1024, kGoldenSeed + f).
constexpr uint32_t kGoldenRows = 256, kGoldenK = 1024, kGoldenSeed = 7;
const char *const kGolden[FMT_COUNT] = {
    "36905e2a1a87522067f1dd4a59e5fa658390d01338bb2639df70d622b97804f4", // q4k
    "39f9536d2c5efdfc8f566c176f9ffe8b2eefcc395caa9b2ef9111eb5fc76b453", // iq4xs
    "efe2505213961362b93459a7eb511d7bee39487a2d7f870c0fa972912d93df29", // iq4nl
    "8c2c2b9caf1e1831d516610e0bddf3acc7d736d2884bc1549dd9e579fdd61921", // q5k
    "6478ac742b3e4010323642d0f99b91ba9f62513838cdfc775ec564062d86f117", // q6k
    "8ecc6c113faf9fb8cc77d438dd595a146933c6c75394651cab377c41a55e5677", // q3k
    "1f11dbe883e04db7367fbc5df5c7d2c34a14bae5d772501687a9072f9237b605", // q80
    "dacc281a47cc2940e8352e3cd890283f69c8126d2894768ab7f179cf8fda34e8", // iq3s
};

void checkGoldens(void *ggml) {
  for (int f = 0; f < FMT_COUNT; ++f) {
    const std::vector<uint8_t> native = fixture(Fmt(f), kGoldenRows, kGoldenK, kGoldenSeed + f);
    std::vector<float> values;
    repack(Fmt(f), native, kGoldenRows, kGoldenK, &values);
    check(sha256(values.data(), values.size() * sizeof(float)) == kGolden[f],
          std::string("CPU reference matches the GGML golden hash: ") + fmtName(f));
    if (!ggml) continue;
    std::vector<float> official;
    std::string error;
    const bool loaded = ggmlDequantize(ggml, Fmt(f), native, official, error);
    check(loaded && official.size() == values.size() &&
              !memcmp(official.data(), values.data(), values.size() * sizeof(float)),
          std::string("CPU reference matches GGML: ") + fmtName(f) + (loaded ? "" : " (" + error + ")"));
    if (loaded)
      std::printf("GGML %s %s\n", fmtName(f), sha256(official.data(), official.size() * sizeof(float)).c_str());
  }
}

// Rows [from, rows) of the image come from llama.cpp's tiled value-head order:
// destination head h reads source head (h % groups) * groupHeads + h / groups.
uint64_t sourceRow(uint64_t n, uint64_t from, uint64_t headRows, uint64_t groupHeads, uint64_t groups) {
  if (n < from) return n;
  const uint64_t head = (n - from) / headRows;
  return from + ((head % groups) * groupHeads + head / groups) * headRows + (n - from) % headRows;
}

template <class T> void append(std::vector<uint8_t> &out, T value) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(&value);
  out.insert(out.end(), bytes, bytes + sizeof value);
}

void appendString(std::vector<uint8_t> &out, const std::string &value) {
  append<uint64_t>(out, value.size());
  out.insert(out.end(), value.begin(), value.end());
}

struct Tensor {
  std::string name;
  std::vector<uint64_t> dims; // dims[0] is the row length
  uint32_t type;
  std::vector<uint8_t> data;
};

// The architecture metadata a GGUF of the geometry declares.
std::vector<std::pair<std::string, uint32_t>> metadata(const splash::model::gguf::TargetGeometry &g) {
  std::vector<std::pair<std::string, uint32_t>> keys{
      {"block_count", g.layers}, {"embedding_length", g.hiddenSize},
      {"attention.head_count", g.attentionWidth / g.attentionHeadDimension}, {"attention.head_count_kv", g.attentionKvHeads},
      {"attention.key_length", g.attentionHeadDimension}, {"attention.value_length", g.attentionHeadDimension},
      {"full_attention_interval", g.fullAttentionPeriod}, {"ssm.conv_kernel", 4}, {"ssm.group_count", g.gdnKeyHeads},
      {"ssm.time_step_rank", g.gdnValueHeads}, {"ssm.state_size", g.gdnHeadDimension},
      {"ssm.inner_size", g.gdnValueHeads * g.gdnHeadDimension}};
  if (g.sparseMoe())
    keys.insert(keys.end(), {{"expert_count", g.experts}, {"expert_used_count", g.expertsPerToken},
                             {"expert_feed_forward_length", g.expertIntermediateSize},
                             {"expert_shared_feed_forward_length", g.expertIntermediateSize}});
  else
    keys.push_back({"feed_forward_length", g.intermediateSize});
  return keys;
}

// A version 3 GGUF of the tensors of the geometry's architecture, in order and
// 32-byte aligned.
std::vector<uint8_t> ggufFile(const std::vector<Tensor> &tensors, const splash::model::gguf::TargetGeometry &geometry,
                             const std::string &displaced = {},
                             const std::vector<std::pair<std::string, std::string>> &strings = {}) {
  constexpr uint32_t kString = 8, kUint32 = 4, kAlignment = 32;
  const std::string architecture = geometry.architecture();
  const auto keys = metadata(geometry);
  std::vector<uint8_t> out{'G', 'G', 'U', 'F'};
  append<uint32_t>(out, 3);
  append<uint64_t>(out, tensors.size());
  append<uint64_t>(out, keys.size() + 1 + strings.size());
  appendString(out, "general.architecture");
  append(out, kString);
  appendString(out, architecture);
  for (const auto &[key, value] : strings) {
    appendString(out, key);
    append(out, kString);
    appendString(out, value);
  }
  for (const auto &[key, value] : keys) {
    appendString(out, architecture + "." + key);
    append(out, kUint32);
    append(out, value);
  }
  uint64_t offset = 0;
  for (const Tensor &tensor : tensors) {
    appendString(out, tensor.name);
    append<uint32_t>(out, tensor.dims.size());
    for (uint64_t dim : tensor.dims) append(out, dim);
    append(out, tensor.type);
    append(out, offset + (tensor.name == displaced ? uint64_t{1} << 32 : 0));
    offset += (tensor.data.size() + kAlignment - 1) / kAlignment * kAlignment;
  }
  for (const Tensor &tensor : tensors) {
    out.resize((out.size() + kAlignment - 1) / kAlignment * kAlignment, 0);
    out.insert(out.end(), tensor.data.begin(), tensor.data.end());
  }
  return out;
}

// F32 values in [-1, -0.5] and [0.5, 1], optionally exact bf16 values (the
// low half of every word zero), as a bf16 checkpoint converted to F32 has.
std::vector<uint8_t> floatValues(uint64_t count, uint32_t seed, bool bfloat16 = false) {
  std::vector<uint8_t> bytes(count * 4);
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t bits = 0x3F000000u | uint32_t((i * 2654435761u + seed * 40503u) & 0x7FFFFFu);
    if ((i + seed) % 3 == 0) bits |= 0x80000000u;
    if (bfloat16) bits &= 0xFFFF0000u;
    std::memcpy(bytes.data() + i * 4, &bits, 4);
  }
  return bytes;
}

namespace model = splash::model;

// A temporary directory, removed with its contents.
class TemporaryDirectory {
public:
  TemporaryDirectory() {
    char path[] = "/tmp/splash-gguf-repack-XXXXXX";
    if (!mkdtemp(path)) throw std::runtime_error("cannot create a temporary directory");
    path_ = path;
  }
  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  TemporaryDirectory &operator=(const TemporaryDirectory &) = delete;
  [[nodiscard]] const std::filesystem::path &path() const noexcept { return path_; }

private:
  std::filesystem::path path_;
};

void writeGguf(const std::filesystem::path &path, const std::vector<Tensor> &tensors,
               const model::gguf::TargetGeometry &geometry) {
  const std::vector<uint8_t> file = ggufFile(tensors, geometry);
  std::ofstream(path, std::ios::binary | std::ios::trunc).write(reinterpret_cast<const char *>(file.data()), file.size());
}

// Adds a tensor, zero unless data is given: only the shapes and types of most
// tensors matter to a check.
struct TensorList {
  std::vector<Tensor> tensors;
  void add(std::string name, std::vector<uint64_t> dims, uint32_t type, std::vector<uint8_t> data = {}) {
    if (data.empty()) {
      const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
      uint64_t elements = 1;
      for (uint64_t dim : dims) elements *= dim;
      data.assign(elements / traits.blockElements * traits.blockBytes, 0);
    }
    tensors.push_back({std::move(name), std::move(dims), type, std::move(data)});
  }
  std::vector<uint8_t> &data(const std::string &name) {
    return std::find_if(tensors.begin(), tensors.end(), [&](const Tensor &t) { return t.name == name; })->data;
  }
};

// Every byte of the image the loader prepares for layer `index`, the head for
// index == layers.
std::vector<uint8_t> preparedImage(splash::metal::MetalBackend &backend, const std::filesystem::path &path,
                                   const model::gguf::TargetGeometry &geometry, uint32_t index) {
  model::GgufTargetLoader loader(backend, path, geometry);
  const model::WeightFile weights = index < geometry.layers ? loader.layer(index) : loader.head();
  std::ifstream stream(std::filesystem::path(std::getenv("SPLASH_WEIGHT_CACHE")) / weights.record().contentIdentity /
                           "weights",
                       std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), {}};
}

std::vector<uint8_t> slice(const std::vector<uint8_t> &bytes, uint64_t offset, uint64_t size) {
  if (offset > bytes.size() || size > bytes.size() - offset) return {};
  return {bytes.begin() + offset, bytes.begin() + offset + size};
}

// The plan's copy of the named tensor, or nullptr.
const model::gguf::Copy *copyOf(const model::gguf::Image &image, const std::string &name) {
  const auto found = std::find_if(image.copies.begin(), image.copies.end(),
                                  [&](const model::gguf::Copy &copy) { return copy.source.name == name; });
  return found == image.copies.end() ? nullptr : &*found;
}

bool grouped(const model::gguf::RowOrder &order, uint64_t from, uint32_t headRows,
             const model::gguf::TargetGeometry &g) {
  return order.from == from && order.headRows == headRows && order.groupHeads == g.gdnKeyHeads &&
         order.groups == g.gdnValueHeads / g.gdnKeyHeads;
}

// Rows of `bytes` (rowBytes each) in image order: row n reads source row
// sourceRow(n, from, headRows, ...), the tiled-to-grouped head order.
std::vector<uint8_t> groupedRows(const std::vector<uint8_t> &bytes, uint64_t rowBytes, uint64_t from,
                                 uint32_t headRows, const model::gguf::TargetGeometry &g) {
  std::vector<uint8_t> out(bytes.size());
  for (uint64_t n = 0; n < bytes.size() / rowBytes; ++n)
    std::memcpy(out.data() + n * rowBytes,
                bytes.data() + sourceRow(n, from, headRows, g.gdnKeyHeads, g.gdnValueHeads / g.gdnKeyHeads) * rowBytes,
                rowBytes);
  return out;
}

// The bf16 upper halves of F32 values.
std::vector<uint8_t> bfloat16Halves(const std::vector<uint8_t> &floats) {
  std::vector<uint8_t> out;
  for (size_t i = 0; i < floats.size(); i += 4) out.insert(out.end(), floats.begin() + i + 2, floats.begin() + i + 4);
  return out;
}

// The GDN alpha/beta projection: beta rows, alpha rows, then zero rows up to
// one 256-row tile, as one Q8_0 tensor with rows in grouped head order. Its
// prepared plane0 and meta must equal the reference's repack of those native
// rows. The layer's other tensors are zero; only their shapes and types
// matter.
void checkAlphaBeta(splash::metal::MetalBackend *backend) {
  using namespace model::ggml;
  model::gguf::TargetGeometry geometry;
  geometry.layers = 1;
  geometry.hiddenSize = 512;
  geometry.vocabularySize = 256;
  geometry.intermediateSize = 256;
  geometry.gdnKeyHeads = 4;
  geometry.gdnValueHeads = 12;
  geometry.gdnHeadDimension = 64;
  geometry.convolutionDimension = 1280; // q and k of 4 heads, v of 12
  const uint32_t hidden = geometry.hiddenSize, heads = geometry.gdnValueHeads;
  const uint32_t groupHeads = geometry.gdnKeyHeads, groups = heads / groupHeads;
  const uint64_t valueRows = uint64_t{heads} * geometry.gdnHeadDimension;
  const std::vector<uint8_t> beta = fixture(Q80, heads, hidden, 200), alpha = fixture(Q80, heads, hidden, 201);
  TensorList list;
  list.add("blk.0.attn_norm.weight", {hidden}, kF32);
  list.add("blk.0.attn_qkv.weight", {hidden, geometry.convolutionDimension}, kQ4_K);
  list.add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ4_K);
  list.add("blk.0.ssm_beta.weight", {hidden, heads}, kQ8_0, beta);
  list.add("blk.0.ssm_alpha.weight", {hidden, heads}, kQ8_0, alpha);
  list.add("blk.0.ssm_conv1d.weight", {4, geometry.convolutionDimension}, kF32);
  list.add("blk.0.ssm_a", {heads}, kF32);
  list.add("blk.0.ssm_dt.bias", {heads}, kF32);
  list.add("blk.0.ssm_norm.weight", {geometry.gdnHeadDimension}, kF32);
  list.add("blk.0.ssm_out.weight", {valueRows, hidden}, kQ4_K);
  list.add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  list.add("blk.0.ffn_gate.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  list.add("blk.0.ffn_up.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  list.add("blk.0.ffn_down.weight", {geometry.intermediateSize, hidden}, kQ4_K);
  list.add("output_norm.weight", {hidden}, kF32);
  list.add("output.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  list.add("token_embd.weight", {hidden, geometry.vocabularySize}, kQ4_K);

  const uint32_t stride = rowBytes(Q80, hidden);
  std::vector<uint8_t> rows(size_t{256} * stride, 0);
  for (uint32_t n = 0; n < 2 * heads; ++n)
    std::memcpy(rows.data() + size_t{n} * stride,
                (n < heads ? beta : alpha).data() + size_t{sourceRow(n % heads, 0, 1, groupHeads, groups)} * stride, stride);
  const Packed expected = repack(Q80, rows, 256, hidden, nullptr);
  try {
    const TemporaryDirectory directory;
    const auto path = directory.path() / "alpha-beta.gguf";
    writeGguf(path, list.tensors, geometry);
    const model::GgufFile gguf(path);
    const model::gguf::Image plan = model::gguf::ImagePlanner(gguf, geometry).layer(0);
    const auto repack = std::find_if(plan.repacks.begin(), plan.repacks.end(), [](const model::gguf::Repack &r) {
      return r.format == GGUF_FMT_Q80 && r.rows == 256;
    });
    const auto sourceRows = [&](const model::gguf::TensorRows &r, const char *name) {
      return r.name == name && r.rows == heads && grouped(r.order, 0, 1, geometry);
    };
    const bool planned = repack != plan.repacks.end() && repack->sources.size() == 2 &&
                         sourceRows(repack->sources[0], "blk.0.ssm_beta.weight") &&
                         sourceRows(repack->sources[1], "blk.0.ssm_alpha.weight");
    check(planned, "planner alpha/beta: one 256-row Q8_0 tensor of beta then alpha rows in grouped order");
    if (backend && planned) {
      const std::vector<uint8_t> image = preparedImage(*backend, path, geometry, 0);
      check(slice(image, repack->plane0, expected.w0.size()) == expected.w0 &&
                slice(image, repack->meta, expected.meta.size()) == expected.meta,
            "prepared alpha/beta tensor matches the CPU reference");
    }
  } catch (const std::exception &error) {
    check(false, std::string("alpha/beta: ") + error.what());
  }
}

// Every norm of a GDN layer, a full-attention layer and the head goes into
// its image as the GGUF stores it: F32 values that bf16 would round. The
// convolution and time-bias tensors, which the kernels read as bf16, convert
// when every value is bf16-exact and are refused by name otherwise.
void checkFloatTensors(splash::metal::MetalBackend *backend) {
  using namespace model::ggml;
  model::gguf::TargetGeometry geometry;
  geometry.layers = 2;
  geometry.hiddenSize = 512;
  geometry.vocabularySize = 256;
  geometry.intermediateSize = 256;
  geometry.gdnKeyHeads = 4;
  geometry.gdnValueHeads = 12;
  geometry.gdnHeadDimension = 64;
  geometry.convolutionDimension = 1280; // q and k of 4 heads, v of 12
  geometry.attentionWidth = 512;        // two query heads of 256
  geometry.attentionKvHeads = 2;
  geometry.fullAttentionPeriod = 2;     // layer 1
  const uint32_t hidden = geometry.hiddenSize, heads = geometry.gdnValueHeads, head = 256;
  const uint64_t valueRows = uint64_t{heads} * geometry.gdnHeadDimension;
  const uint32_t keyRows = geometry.convolutionDimension - uint32_t(valueRows);
  TensorList list;
  const std::vector<std::string> norms{"blk.0.attn_norm.weight", "blk.0.ssm_norm.weight",
                                       "blk.0.post_attention_norm.weight", "blk.1.attn_norm.weight",
                                       "blk.1.attn_q_norm.weight", "blk.1.attn_k_norm.weight",
                                       "blk.1.post_attention_norm.weight", "output_norm.weight"};
  uint32_t seed = 400;
  list.add(norms[0], {hidden}, kF32, floatValues(hidden, ++seed));
  list.add("blk.0.attn_qkv.weight", {hidden, geometry.convolutionDimension}, kQ4_K);
  list.add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ4_K);
  list.add("blk.0.ssm_beta.weight", {hidden, heads}, kQ8_0);
  list.add("blk.0.ssm_alpha.weight", {hidden, heads}, kQ8_0);
  list.add("blk.0.ssm_conv1d.weight", {4, geometry.convolutionDimension}, kF32,
           floatValues(uint64_t{4} * geometry.convolutionDimension, ++seed, true));
  list.add("blk.0.ssm_a", {heads}, kF32, floatValues(heads, ++seed));
  list.add("blk.0.ssm_dt.bias", {heads}, kF32, floatValues(heads, ++seed, true));
  list.add(norms[1], {geometry.gdnHeadDimension}, kF32, floatValues(geometry.gdnHeadDimension, ++seed));
  list.add("blk.0.ssm_out.weight", {valueRows, hidden}, kQ4_K);
  list.add(norms[2], {hidden}, kF32, floatValues(hidden, ++seed));
  list.add(norms[3], {hidden}, kF32, floatValues(hidden, ++seed));
  list.add("blk.1.attn_q.weight", {hidden, 2 * geometry.attentionWidth}, kQ4_K);
  list.add("blk.1.attn_k.weight", {hidden, geometry.attentionKvHeads * head}, kQ4_K);
  list.add("blk.1.attn_v.weight", {hidden, geometry.attentionKvHeads * head}, kQ4_K);
  list.add(norms[4], {head}, kF32, floatValues(head, ++seed));
  list.add(norms[5], {head}, kF32, floatValues(head, ++seed));
  list.add("blk.1.attn_output.weight", {geometry.attentionWidth, hidden}, kQ4_K);
  list.add(norms[6], {hidden}, kF32, floatValues(hidden, ++seed));
  for (uint32_t layer = 0; layer < 2; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    list.add(p + "ffn_gate.weight", {hidden, geometry.intermediateSize}, kQ4_K);
    list.add(p + "ffn_up.weight", {hidden, geometry.intermediateSize}, kQ4_K);
    list.add(p + "ffn_down.weight", {geometry.intermediateSize, hidden}, kQ4_K);
  }
  list.add(norms[7], {hidden}, kF32, floatValues(hidden, ++seed));
  list.add("output.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  list.add("token_embd.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  try {
    const TemporaryDirectory directory;
    const auto path = directory.path() / "floats.gguf";
    // The images of the file of `tensors`; the planner's error, empty if none.
    const auto plan = [&](const std::vector<Tensor> &tensors, std::vector<model::gguf::Image> &images) {
      writeGguf(path, tensors, geometry);
      try {
        const model::GgufFile gguf(path);
        const model::gguf::ImagePlanner planner(gguf, geometry);
        images = {planner.layer(0), planner.layer(1), planner.head()};
      } catch (const model::GgufError &error) {
        return std::string(error.what());
      }
      return std::string();
    };
    std::vector<model::gguf::Image> images, scratch;
    const std::string error = plan(list.tensors, images);
    if (!error.empty()) throw std::runtime_error(error);
    // image index and copy of every norm
    std::vector<std::pair<size_t, const model::gguf::Copy *>> copies;
    for (const std::string &name : norms)
      for (size_t i = 0; i < images.size(); ++i)
        if (const auto *copy = copyOf(images[i], name)) copies.push_back({i, copy});
    check(copies.size() == norms.size() && std::all_of(copies.begin(), copies.end(), [&](const auto &entry) {
            const model::gguf::Copy &copy = *entry.second;
            return !copy.bfloat16 && copy.source.rows * copy.source.rowBytes == list.data(copy.source.name).size() &&
                   copy.source.order.from == UINT64_MAX;
          }),
          "planner keeps every norm's F32 values as stored");
    const auto *conv = copyOf(images[0], "blk.0.ssm_conv1d.weight");
    const auto *decay = copyOf(images[0], "blk.0.ssm_a");
    const auto *bias = copyOf(images[0], "blk.0.ssm_dt.bias");
    check(conv && conv->bfloat16 && conv->source.rows == geometry.convolutionDimension &&
              grouped(conv->source.order, keyRows, geometry.gdnHeadDimension, geometry) && decay && !decay->bfloat16 &&
              grouped(decay->source.order, 0, 1, geometry) && bias && bias->bfloat16 &&
              grouped(bias->source.order, 0, 1, geometry),
          "planner narrows the convolution and time bias to bf16 and groups their value heads");
    for (const char *name : {"blk.1.attn_k.weight", "blk.1.attn_v.weight"}) {
      for (uint64_t rows : {uint64_t{head}, uint64_t{3 * head}}) {
        std::vector<Tensor> malformed = list.tensors;
        auto &t = *std::find_if(malformed.begin(), malformed.end(),
                                [&](const Tensor &value) { return value.name == name; });
        t.dims[1] = rows;
        t.data.resize(rows * hidden / 256 * 144);
        check(plan(malformed, scratch) == std::string("unexpected shape for ") + name,
              std::string("planner rejects mismatched KV rows: ") + name + " rows=" + std::to_string(rows));
      }
    }
    if (!backend) return;
    writeGguf(path, list.tensors, geometry);
    std::vector<std::vector<uint8_t>> prepared;
    for (uint32_t index = 0; index <= geometry.layers; ++index)
      prepared.push_back(preparedImage(*backend, path, geometry, index));
    bool stored = true;
    for (const auto &[index, copy] : copies) {
      const std::vector<uint8_t> &values = list.data(copy->source.name);
      stored &= slice(prepared[index], copy->destination, values.size()) == values;
    }
    check(stored, "prepared norms are the GGUF's F32 values as stored");
    const auto prepares = [&](const model::gguf::Copy &copy, const std::vector<uint8_t> &expected) {
      return slice(prepared[0], copy.destination, expected.size()) == expected;
    };
    check(prepares(*conv, bfloat16Halves(groupedRows(list.data(conv->source.name), 16, keyRows,
                                                     geometry.gdnHeadDimension, geometry))) &&
              prepares(*decay, groupedRows(list.data(decay->source.name), 4, 0, 1, geometry)) &&
              prepares(*bias, bfloat16Halves(groupedRows(list.data(bias->source.name), 4, 0, 1, geometry))),
          "prepared convolution, decay and time bias in grouped head order, exact bf16 where narrowed");
    for (const char *name : {"blk.0.ssm_conv1d.weight", "blk.0.ssm_dt.bias"}) {
      TensorList inexact = list;
      const float value = 1.0f + 0x1p-10f;
      std::memcpy(inexact.data(name).data() + 12, &value, 4);
      writeGguf(path, inexact.tensors, geometry);
      std::string refused;
      try {
        static_cast<void>(preparedImage(*backend, path, geometry, 0));
      } catch (const model::GgufError &error) {
        refused = error.what();
      }
      check(refused == std::string(name) + " is not bf16-exact; it needs an F32 path",
            std::string("preparation refuses to round ") + name + " to bf16");
    }
  } catch (const std::exception &error) {
    check(false, std::string("float tensors: ") + error.what());
  }
}

// A qwen35moe layer: F32 alpha/beta as one float tensor (beta rows, then alpha
// rows, in grouped head order, values as stored), the F32 router and
// shared-expert gate copied as stored, and each 3-D expert tensor repacked as
// one tensor of experts * N rows. The metadata and architecture must match.
void checkMoeLayer(splash::metal::MetalBackend *backend) {
  using namespace model::ggml;
  model::gguf::TargetGeometry geometry;
  geometry.layers = 1;
  geometry.hiddenSize = 512;
  geometry.vocabularySize = 256;
  geometry.intermediateSize = 0;
  geometry.gdnKeyHeads = 4;
  geometry.gdnValueHeads = 8;
  geometry.gdnHeadDimension = 64;
  geometry.convolutionDimension = 1024; // q and k of 4 heads, v of 8
  geometry.experts = 4;
  geometry.expertsPerToken = 2;
  geometry.expertIntermediateSize = 256;
  const uint32_t hidden = geometry.hiddenSize, heads = geometry.gdnValueHeads, experts = geometry.experts;
  const uint32_t width = geometry.expertIntermediateSize, valueRows = heads * geometry.gdnHeadDimension;
  TensorList list;
  list.add("output_norm.weight", {hidden}, kF32);
  list.add("blk.0.attn_norm.weight", {hidden}, kF32);
  list.add("blk.0.attn_qkv.weight", {hidden, geometry.convolutionDimension}, kQ8_0);
  list.add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ8_0);
  list.add("blk.0.ssm_beta.weight", {hidden, heads}, kF32, floatValues(uint64_t{heads} * hidden, 301));
  list.add("blk.0.ssm_alpha.weight", {hidden, heads}, kF32, floatValues(uint64_t{heads} * hidden, 302));
  list.add("blk.0.ssm_conv1d.weight", {4, geometry.convolutionDimension}, kF32);
  list.add("blk.0.ssm_a", {heads}, kF32);
  list.add("blk.0.ssm_dt.bias", {heads}, kF32);
  list.add("blk.0.ssm_norm.weight", {geometry.gdnHeadDimension}, kF32);
  list.add("blk.0.ssm_out.weight", {valueRows, hidden}, kQ8_0);
  list.add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  list.add("blk.0.ffn_gate_inp.weight", {hidden, experts}, kF32, floatValues(uint64_t{experts} * hidden, 303));
  list.add("blk.0.ffn_gate_exps.weight", {hidden, width, experts}, kQ4_K);
  list.add("blk.0.ffn_up_exps.weight", {hidden, width, experts}, kQ4_K);
  list.add("blk.0.ffn_down_exps.weight", {width, hidden, experts}, kQ5_K, fixture(Q5K, experts * hidden, width, 507));
  list.add("blk.0.ffn_gate_shexp.weight", {hidden, width}, kQ8_0);
  list.add("blk.0.ffn_up_shexp.weight", {hidden, width}, kQ8_0);
  list.add("blk.0.ffn_down_shexp.weight", {width, hidden}, kQ8_0);
  list.add("blk.0.ffn_gate_inp_shexp.weight", {hidden}, kF32, floatValues(hidden, 304));
  list.add("output.weight", {hidden, geometry.vocabularySize}, kQ6_K);
  list.add("token_embd.weight", {hidden, geometry.vocabularySize}, kQ8_0);
  try {
    const TemporaryDirectory directory;
    const auto path = directory.path() / "moe.gguf";
    // The error of planning a file that declares `declared`, empty if none.
    const auto planError = [&](const model::gguf::TargetGeometry &declared) {
      writeGguf(path, list.tensors, declared);
      try {
        const model::GgufFile gguf(path);
        static_cast<void>(model::gguf::ImagePlanner(gguf, geometry));
      } catch (const model::GgufError &error) {
        return std::string(error.what());
      }
      return std::string();
    };
    model::gguf::TargetGeometry wrongExperts = geometry;
    wrongExperts.experts = 5;
    check(planError(wrongExperts).find("expert_count 5 (expected 4)") != std::string::npos,
          "planner names a metadata mismatch");
    model::gguf::TargetGeometry dense = geometry;
    dense.experts = 0;
    dense.intermediateSize = 256;
    check(planError(dense).find("GGUF architecture is qwen35, but the package's target is qwen35moe") != std::string::npos,
          "planner checks the architecture against the package");
    writeGguf(path, list.tensors, geometry);
    const model::GgufFile gguf(path);
    const model::gguf::Image plan = model::gguf::ImagePlanner(gguf, geometry).layer(0);
    const auto *beta = copyOf(plan, "blk.0.ssm_beta.weight"), *alpha = copyOf(plan, "blk.0.ssm_alpha.weight");
    check(beta && alpha && alpha->destination == beta->destination + uint64_t{heads} * hidden * 4 &&
              grouped(beta->source.order, 0, 1, geometry) && grouped(alpha->source.order, 0, 1, geometry) &&
              !beta->bfloat16 && !alpha->bfloat16,
          "planner F32 alpha/beta tensor: beta then alpha rows in grouped order");
    const auto stored = [&](const char *name) {
      const auto *copy = copyOf(plan, name);
      return copy && !copy->bfloat16 && copy->source.order.from == UINT64_MAX &&
             copy->source.rows * copy->source.rowBytes == gguf.require(name).bytes;
    };
    check(stored("blk.0.ffn_gate_inp.weight") && stored("blk.0.ffn_gate_inp_shexp.weight"),
          "planner copies the F32 router and shared-expert gate as stored");
    std::vector<std::pair<uint64_t, uint64_t>> shapes;
    for (const model::gguf::Repack &repack : plan.repacks) shapes.push_back({repack.rows, repack.columns});
    const std::vector<std::pair<uint64_t, uint64_t>> expected{
        {geometry.convolutionDimension, hidden}, {valueRows, hidden}, {hidden, valueRows}, {experts * width, hidden},
        {experts * width, hidden}, {experts * hidden, width}, {width, hidden}, {width, hidden}, {hidden, width}};
    check(shapes == expected, "planner repacks each expert tensor as experts * N rows");
    if (!backend) return;
    // beta then alpha rows, each in grouped head order.
    std::vector<uint8_t> gates = groupedRows(list.data(beta->source.name), uint64_t{hidden} * 4, 0, 1, geometry);
    const auto alphaRows = groupedRows(list.data(alpha->source.name), uint64_t{hidden} * 4, 0, 1, geometry);
    gates.insert(gates.end(), alphaRows.begin(), alphaRows.end());
    bool allowPreparation = true;
    const auto load = [&] {
      model::GgufTargetLoader loader(*backend, path, geometry, [&] {
        if (!allowPreparation) throw std::runtime_error("conversion forbidden on warm load");
      });
      auto weights = loader.layer(0);
      const auto bytes = weights.section(weights.record().declaredBytes - model::kWeightFileAlignment);
      const auto *begin = static_cast<const uint8_t *>(bytes.contents());
      std::vector<uint8_t> image(model::kWeightFileAlignment, 0);
      image.insert(image.end(), begin, begin + bytes.sizeBytes());
      weights.finish();
      return image;
    };
    const auto image = load();
    check(slice(image, beta->destination, gates.size()) == gates,
          "prepared F32 alpha/beta tensor: beta then alpha rows in grouped order");
    allowPreparation = false;
    check(load() == image, "GGUF warm load does not require conversion headroom");
    // The model's other prepared files join the target's disk check.
    const model::PreparedWeight vision{std::string(64, 'a'), UINT64_MAX / 2};
    std::string budget;
    try {
      model::GgufTargetLoader loader(*backend, path, geometry, {}, {&vision, 1});
    } catch (const std::runtime_error &error) {
      budget = error.what();
    }
    check(budget.starts_with("not enough disk space to prepare weights"),
          "GGUF target disk check budgets the model's other prepared files: " + budget);
    // A layer may have tensors on opposite sides of the 4 GiB boundary. Keep
    // the file sparse and poison the old location so a truncated offset
    // cannot accidentally read the right data.
    allowPreparation = true;
    for (const char *name : {"blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_inp.weight"}) {
      writeGguf(path, list.tensors, geometry);
      const model::GgufFile original(path);
      const uint64_t offset = original.absoluteOffset(original.require(name));
      const std::vector<uint8_t> &data = list.data(name);
      auto bytes = ggufFile(list.tensors, geometry, name);
      std::fill_n(bytes.begin() + offset, data.size(), 0);
      {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        stream.seekp(static_cast<std::streamoff>(offset + (uint64_t{1} << 32)));
        stream.write(reinterpret_cast<const char *>(data.data()), data.size());
      }
      check(load() == image, std::string("loader keeps weights exact beyond 4 GiB: ") + name);
    }
  } catch (const std::exception &error) {
    check(false, std::string("MoE layer: ") + error.what());
  }
}

// A qwen35 target read from a GGUF through the production loader: every
// projection a block projection of the layout's sizes, a fused one a segment
// per tensor in the layout's padded width, the norms F32 and the GDN output in
// the GGUF's tiled head order; the head and token table as the GGUF stores them.
void checkDenseTarget(splash::metal::MetalBackend &backend) {
  namespace model = splash::model;
  namespace ops = splash::ops;
  using namespace model::ggml;
  model::Qwen3_8Layout layout;
  layout.layers = 4;
  layout.hiddenSize = 256;
  layout.vocabularySize = 256;
  layout.gdnKeyHeads = 1;
  layout.gdnValueHeads = 2;
  layout.gdnHeadDimension = 128;
  layout.convolutionDimension = 512; // q and k of one head, v of two
  layout.packedGdnWidth = 1280;      // qkv | z | alpha-beta and one padding tile
  layout.attentionWidth = 256;
  layout.attentionQueryHeads = 2;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 128;
  layout.packedFullWidth = 1024;     // q and its gate | k | v
  layout.intermediateSize = 512;
  const uint32_t hidden = layout.hiddenSize, heads = layout.gdnValueHeads;
  const uint32_t valueRows = heads * layout.gdnHeadDimension, kvRows = layout.attentionKvHeads * layout.attentionHeadDimension;
  std::vector<Tensor> tensors;
  const auto add = [&](std::string name, std::vector<uint64_t> dims, uint32_t type) {
    const model::GgmlTypeTraits &traits = *model::ggmlTypeTraits(type);
    uint64_t elements = 1;
    for (uint64_t dim : dims) elements *= dim;
    tensors.push_back({std::move(name), std::move(dims), type,
                       std::vector<uint8_t>(elements / traits.blockElements * traits.blockBytes, 0)});
  };
  add("output_norm.weight", {hidden}, kF32);
  for (uint32_t layer = 0; layer < layout.layers; ++layer) {
    const std::string p = "blk." + std::to_string(layer) + ".";
    add(p + "attn_norm.weight", {hidden}, kF32);
    if (layout.isFullAttentionLayer(layer)) {
      add(p + "attn_q.weight", {hidden, 2 * layout.attentionWidth}, kQ8_0);
      add(p + "attn_k.weight", {hidden, kvRows}, kQ4_K);
      add(p + "attn_v.weight", {hidden, kvRows}, kQ6_K);
      add(p + "attn_q_norm.weight", {layout.attentionHeadDimension}, kF32);
      add(p + "attn_k_norm.weight", {layout.attentionHeadDimension}, kF32);
      add(p + "attn_output.weight", {layout.attentionWidth, hidden}, kQ8_0);
    } else {
      add(p + "attn_qkv.weight", {hidden, layout.convolutionDimension}, kQ8_0);
      add(p + "attn_gate.weight", {hidden, valueRows}, kQ4_K);
      add(p + "ssm_beta.weight", {hidden, heads}, kQ8_0);
      add(p + "ssm_alpha.weight", {hidden, heads}, kQ8_0);
      add(p + "ssm_conv1d.weight", {4, layout.convolutionDimension}, kF32);
      add(p + "ssm_a", {heads}, kF32);
      add(p + "ssm_dt.bias", {heads}, kF32);
      add(p + "ssm_norm.weight", {layout.gdnHeadDimension}, kF32);
      add(p + "ssm_out.weight", {valueRows, hidden}, kQ8_0);
    }
    add(p + "post_attention_norm.weight", {hidden}, kF32);
    add(p + "ffn_gate.weight", {hidden, layout.intermediateSize}, kQ4_K);
    add(p + "ffn_up.weight", {hidden, layout.intermediateSize}, kQ4_K);
    add(p + "ffn_down.weight", {layout.intermediateSize, hidden}, kQ6_K);
  }
  add("output.weight", {hidden, layout.vocabularySize}, kQ6_K);
  add("token_embd.weight", {hidden, layout.vocabularySize}, kQ8_0);
  char directory[] = "/tmp/splash-gguf-dense-XXXXXX";
  if (!mkdtemp(directory)) {
    check(false, "create a temporary directory");
    return;
  }
  const std::vector<uint8_t> file = ggufFile(tensors, model::ggufTargetGeometry(layout));
  std::ofstream(std::filesystem::path(directory) / "target.gguf", std::ios::binary)
      .write(reinterpret_cast<const char *>(file.data()), file.size());
  // The segments of a block projection of `n` x `k` by output width.
  const auto blocks = [](const ops::Projection &p, uint32_t n, uint32_t k, std::vector<uint32_t> widths) {
    if (p.layout() != ops::WeightLayout::Block32 || p.outputSize != n || p.inputSize != k ||
        p.blocks().segments.size() != widths.size())
      return false;
    uint32_t offset = 0;
    for (size_t i = 0; i < widths.size(); ++i) {
      const ops::QuantizedSegment &s = p.blocks().segments[i];
      if (s.columnOffset != offset || s.outputSize != widths[i] || s.inputSize != k) return false;
      offset += widths[i];
    }
    return true;
  };
  bool read = false;
  try {
    const model::Qwen3_8Weights weights =
        model::loadQwen3_8Weights(backend, directory, layout, model::TargetSource::Gguf);
    read = weights.layers.size() == layout.layers && weights.finalNorm.float32 &&
           blocks(weights.logitsProjection, layout.vocabularySize, hidden, {layout.vocabularySize}) &&
           std::string_view(weights.logitsProjection.blocks().segments.front().format) == "q6k" &&
           weights.tokenEmbedding.layout() == ops::WeightLayout::Block32 &&
           weights.tokenEmbedding.outputSize == layout.vocabularySize &&
           weights.tokenEmbedding.inputSize == hidden && model::qwenTargetGeometry(weights).valid();
    for (const auto &layer : weights.layers) {
      read = read && layer.inputNorm.float32 && layer.postAttentionNorm.float32 &&
             blocks(layer.gateProjection, layout.intermediateSize, hidden, {layout.intermediateSize}) &&
             blocks(layer.upProjection, layout.intermediateSize, hidden, {layout.intermediateSize}) &&
             blocks(layer.downProjection, hidden, layout.intermediateSize, {hidden});
      if (const auto *gdn = std::get_if<model::QwenGdnWeights>(&layer.mixer))
        read = read && blocks(gdn->inputProjection, layout.packedGdnWidth, hidden,
                              {layout.convolutionDimension, valueRows, 256}) &&
               blocks(gdn->outputProjection, hidden, valueRows, {hidden}) && gdn->mixerNorm.float32 &&
               gdn->outputHeadOrder == ops::GdnHeadOrder::Tiled;
      else {
        const auto &attention = std::get<model::QwenAttentionWeights>(layer.mixer);
        read = read && blocks(attention.inputProjection, layout.packedFullWidth, hidden,
                              {2 * layout.attentionWidth, kvRows, kvRows}) &&
               blocks(attention.outputProjection, hidden, layout.attentionWidth, {hidden}) &&
               attention.queryNorm.float32 && attention.keyNorm.float32;
      }
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
  }
  std::filesystem::remove_all(directory);
  check(read, "the target loader reads a GGUF as block projections of the layout's sizes");
}

// Keys follow the tensor data the images read: a GGUF whose metadata alone
// changes (a chat template, so every tensor moves in the file) keeps every
// prepared image, and a changed tensor byte prepares the images again.
void checkSourceIdentity(splash::metal::MetalBackend &backend) {
  using namespace model::ggml;
  model::gguf::TargetGeometry geometry;
  geometry.layers = 1;
  geometry.hiddenSize = 512;
  geometry.vocabularySize = 256;
  geometry.intermediateSize = 256;
  geometry.gdnKeyHeads = 4;
  geometry.gdnValueHeads = 12;
  geometry.gdnHeadDimension = 64;
  geometry.convolutionDimension = 1280;
  const uint32_t hidden = geometry.hiddenSize, heads = geometry.gdnValueHeads;
  const uint64_t valueRows = uint64_t{heads} * geometry.gdnHeadDimension;
  TensorList list;
  list.add("blk.0.attn_norm.weight", {hidden}, kF32, floatValues(hidden, 601));
  list.add("blk.0.attn_qkv.weight", {hidden, geometry.convolutionDimension}, kQ4_K);
  list.add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ4_K);
  list.add("blk.0.ssm_beta.weight", {hidden, heads}, kQ8_0);
  list.add("blk.0.ssm_alpha.weight", {hidden, heads}, kQ8_0);
  list.add("blk.0.ssm_conv1d.weight", {4, geometry.convolutionDimension}, kF32);
  list.add("blk.0.ssm_a", {heads}, kF32);
  list.add("blk.0.ssm_dt.bias", {heads}, kF32);
  list.add("blk.0.ssm_norm.weight", {geometry.gdnHeadDimension}, kF32);
  list.add("blk.0.ssm_out.weight", {valueRows, hidden}, kQ4_K, fixture(Q4K, hidden, uint32_t(valueRows), 602));
  list.add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  list.add("blk.0.ffn_gate.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  list.add("blk.0.ffn_up.weight", {hidden, geometry.intermediateSize}, kQ4_K);
  list.add("blk.0.ffn_down.weight", {geometry.intermediateSize, hidden}, kQ4_K);
  list.add("output_norm.weight", {hidden}, kF32);
  list.add("output.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  list.add("token_embd.weight", {hidden, geometry.vocabularySize}, kQ4_K);
  try {
    const TemporaryDirectory directory;
    const auto path = directory.path() / "identity.gguf";
    bool allowPreparation = true;
    // The keys of layer 0 and the head.
    const auto keys = [&] {
      model::GgufTargetLoader loader(backend, path, geometry, [&] {
        if (!allowPreparation) throw std::runtime_error("conversion forbidden");
      });
      return std::array<std::string, 2>{loader.layer(0).record().contentIdentity,
                                        loader.head().record().contentIdentity};
    };
    writeGguf(path, list.tensors, geometry);
    const auto original = keys();
    std::vector<uint8_t> file = ggufFile(list.tensors, geometry, {}, {{"tokenizer.chat_template", std::string(100, 'x')}});
    std::ofstream(path, std::ios::binary | std::ios::trunc).write(reinterpret_cast<const char *>(file.data()), file.size());
    allowPreparation = false;
    check(keys() == original, "a GGUF metadata edit keeps every prepared image");
    list.data("blk.0.ssm_out.weight")[100] ^= 1;
    writeGguf(path, list.tensors, geometry);
    allowPreparation = true;
    check(keys()[0] != original[0], "a changed tensor byte prepares its image again");
  } catch (const std::exception &error) {
    check(false, std::string("source identity: ") + error.what());
  }
}

// SHA-256 of every image the loader prepares from two small GGUFs: a dense
// qwen35 target (both layer kinds, all eight formats, permuted value-head rows,
// Q8_0 alpha/beta, F32 norms, bf16-exact convolution and time bias, Q6_K token
// rows) and a qwen35moe layer (F32 alpha/beta, F32 router and shared-expert
// gate, 3-D expert tensors). A different hash means the prepared bytes
// changed; that needs a new preparation identity, so no cache entry of the
// old bytes is served.
struct GoldenImage {
  const char *model, *image, *sha256;
};
constexpr GoldenImage kGoldenImages[] = {
    {"dense", "target/layer-0.bin",
     "3cb6434604cdcae0d93b72844939421eff05f98cff74196392c2153a5c0b4ab3"},
    {"dense", "target/layer-1.bin",
     "f7e1f9a7dec53a1a2396898a302d39e7877ee13bcb493f7240b7c9479df91bfe"},
    {"dense", "target/head.bin",
     "568caf2ce1c20591bbb6da095f1e94b47f868addf0ca3865fede7bafe7486e36"},
    {"dense", "target/embedding.bin",
     "6d17df18962f4941a0ce6538aa0092dba1073c78848a62f577a666b81a89e159"},
    {"moe", "target/layer-0.bin",
     "70d9d2727f5a8883507b6f7c3680913f6f8c3c131fbb90b01bffa05e831b637e"},
    {"moe", "target/head.bin",
     "3f61a8a408a40624b12f9564472930c655a84cdf5e1782191990f678fe80305c"},
    {"moe", "target/embedding.bin",
     "86a749f41cab45849eb9a2753025779ef1c58e74842b65290534eff4e29403e9"},
};

void checkGoldenImages(splash::metal::MetalBackend &backend) {
  namespace model = splash::model;
  using namespace model::ggml;
  char directory[] = "/tmp/splash-gguf-golden-XXXXXX";
  if (!mkdtemp(directory)) {
    check(false, "create a temporary directory");
    return;
  }
  const std::filesystem::path cache(std::getenv("SPLASH_WEIGHT_CACHE"));
  uint32_t seed = 900;
  const auto prepared = [&](const char *name, const std::vector<Tensor> &tensors,
                            const model::gguf::TargetGeometry &geometry) {
    const auto path = std::filesystem::path(directory) / (std::string(name) + ".gguf");
    const std::vector<uint8_t> file = ggufFile(tensors, geometry);
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char *>(file.data()), file.size());
    try {
      model::GgufTargetLoader loader(backend, path, geometry);
      const auto hash = [&](model::WeightFile weights) {
        const auto &record = weights.record();
        std::ifstream stream(cache / record.contentIdentity / "weights", std::ios::binary);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)), {});
        const std::string actual = sha256(bytes.data(), bytes.size());
        const auto golden = std::find_if(std::begin(kGoldenImages), std::end(kGoldenImages), [&](const GoldenImage &g) {
          return name == std::string_view(g.model) && record.relativePath == g.image;
        });
        check(golden != std::end(kGoldenImages) && actual == golden->sha256,
              std::string("golden prepared bytes: ") + name + " " + record.relativePath + " " + actual);
      };
      for (uint32_t layer = 0; layer < geometry.layers; ++layer) hash(loader.layer(layer));
      hash(loader.head());
      hash(loader.embedding());
    } catch (const std::exception &error) {
      check(false, std::string("golden prepared images: ") + name + ": " + error.what());
    }
  };
  std::vector<Tensor> tensors;
  const auto add = [&](std::string name, std::vector<uint64_t> dims, uint32_t type, std::vector<uint8_t> data = {}) {
    if (data.empty()) {
      const uint32_t format = gguf_format_of(type);
      data = format == GGUF_FMT_COUNT ? floatValues(dims[0] * (dims.size() > 1 ? dims[1] : 1), ++seed)
                                      : fixture(Fmt(format), uint32_t(dims[1] * (dims.size() > 2 ? dims[2] : 1)),
                                                uint32_t(dims[0]), ++seed);
    }
    tensors.push_back({std::move(name), std::move(dims), type, std::move(data)});
  };

  model::gguf::TargetGeometry dense;
  dense.layers = 2;
  dense.hiddenSize = 512;
  dense.vocabularySize = 256;
  dense.intermediateSize = 256;
  dense.gdnKeyHeads = 4;
  dense.gdnValueHeads = 12;
  dense.gdnHeadDimension = 64;
  dense.convolutionDimension = 1280; // q and k of 4 heads, v of 12
  dense.attentionWidth = 512;        // two query heads of 256
  dense.attentionKvHeads = 2;
  dense.fullAttentionPeriod = 2;     // layer 1
  const uint32_t hidden = dense.hiddenSize, heads = dense.gdnValueHeads, head = dense.attentionHeadDimension;
  const uint64_t valueRows = uint64_t{heads} * dense.gdnHeadDimension;
  add("blk.0.attn_norm.weight", {hidden}, kF32);
  add("blk.0.attn_qkv.weight", {hidden, dense.convolutionDimension}, kQ4_K);
  add("blk.0.attn_gate.weight", {hidden, valueRows}, kQ5_K);
  add("blk.0.ssm_beta.weight", {hidden, heads}, kQ8_0);
  add("blk.0.ssm_alpha.weight", {hidden, heads}, kQ8_0);
  add("blk.0.ssm_conv1d.weight", {4, dense.convolutionDimension}, kF32,
      floatValues(uint64_t{4} * dense.convolutionDimension, ++seed, true));
  add("blk.0.ssm_a", {heads}, kF32);
  add("blk.0.ssm_dt.bias", {heads}, kF32, floatValues(heads, ++seed, true));
  add("blk.0.ssm_norm.weight", {dense.gdnHeadDimension}, kF32);
  add("blk.0.ssm_out.weight", {valueRows, hidden}, kIQ4_XS);
  add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  add("blk.0.ffn_gate.weight", {hidden, dense.intermediateSize}, kQ3_K);
  add("blk.0.ffn_up.weight", {hidden, dense.intermediateSize}, kIQ3_S);
  add("blk.0.ffn_down.weight", {dense.intermediateSize, hidden}, kIQ4_NL);
  add("blk.1.attn_norm.weight", {hidden}, kF32);
  add("blk.1.attn_q.weight", {hidden, 2 * dense.attentionWidth}, kQ6_K);
  add("blk.1.attn_k.weight", {hidden, dense.attentionKvHeads * head}, kQ8_0);
  add("blk.1.attn_v.weight", {hidden, dense.attentionKvHeads * head}, kQ4_K);
  add("blk.1.attn_q_norm.weight", {head}, kF32);
  add("blk.1.attn_k_norm.weight", {head}, kF32);
  add("blk.1.attn_output.weight", {dense.attentionWidth, hidden}, kQ5_K);
  add("blk.1.post_attention_norm.weight", {hidden}, kF32);
  add("blk.1.ffn_gate.weight", {hidden, dense.intermediateSize}, kQ4_K);
  add("blk.1.ffn_up.weight", {hidden, dense.intermediateSize}, kQ6_K);
  add("blk.1.ffn_down.weight", {dense.intermediateSize, hidden}, kQ8_0);
  add("output_norm.weight", {hidden}, kF32);
  add("output.weight", {hidden, dense.vocabularySize}, kQ6_K);
  add("token_embd.weight", {hidden, dense.vocabularySize}, kQ6_K);
  prepared("dense", tensors, dense);

  model::gguf::TargetGeometry moe;
  moe.layers = 1;
  moe.hiddenSize = 512;
  moe.vocabularySize = 256;
  moe.intermediateSize = 0;
  moe.gdnKeyHeads = 4;
  moe.gdnValueHeads = 8;
  moe.gdnHeadDimension = 64;
  moe.convolutionDimension = 1024; // q and k of 4 heads, v of 8
  moe.experts = 4;
  moe.expertsPerToken = 2;
  moe.expertIntermediateSize = 256;
  const uint32_t experts = moe.experts, width = moe.expertIntermediateSize;
  const uint32_t moeHeads = moe.gdnValueHeads, moeValueRows = moeHeads * moe.gdnHeadDimension;
  tensors.clear();
  add("output_norm.weight", {hidden}, kF32);
  add("blk.0.attn_norm.weight", {hidden}, kF32);
  add("blk.0.attn_qkv.weight", {hidden, moe.convolutionDimension}, kQ8_0);
  add("blk.0.attn_gate.weight", {hidden, moeValueRows}, kQ6_K);
  add("blk.0.ssm_beta.weight", {hidden, moeHeads}, kF32);
  add("blk.0.ssm_alpha.weight", {hidden, moeHeads}, kF32);
  add("blk.0.ssm_conv1d.weight", {4, moe.convolutionDimension}, kF32,
      floatValues(uint64_t{4} * moe.convolutionDimension, ++seed, true));
  add("blk.0.ssm_a", {moeHeads}, kF32);
  add("blk.0.ssm_dt.bias", {moeHeads}, kF32, floatValues(moeHeads, ++seed, true));
  add("blk.0.ssm_norm.weight", {moe.gdnHeadDimension}, kF32);
  add("blk.0.ssm_out.weight", {moeValueRows, hidden}, kQ4_K);
  add("blk.0.post_attention_norm.weight", {hidden}, kF32);
  add("blk.0.ffn_gate_inp.weight", {hidden, experts}, kF32);
  add("blk.0.ffn_gate_exps.weight", {hidden, width, experts}, kQ4_K);
  add("blk.0.ffn_up_exps.weight", {hidden, width, experts}, kQ4_K);
  add("blk.0.ffn_down_exps.weight", {width, hidden, experts}, kQ5_K);
  add("blk.0.ffn_gate_shexp.weight", {hidden, width}, kQ8_0);
  add("blk.0.ffn_up_shexp.weight", {hidden, width}, kQ8_0);
  add("blk.0.ffn_down_shexp.weight", {width, hidden}, kQ8_0);
  add("blk.0.ffn_gate_inp_shexp.weight", {hidden}, kF32);
  add("output.weight", {hidden, moe.vocabularySize}, kQ6_K);
  add("token_embd.weight", {hidden, moe.vocabularySize}, kQ8_0);
  prepared("moe", tensors, moe);
  std::filesystem::remove_all(directory);
}

constexpr uint64_t kSection = 16384;      // image section alignment, as the planner lays out images
constexpr uint64_t kSourceOffset = 96;    // the tensor data section's offset in the source file

uint64_t alignUp(uint64_t value) { return (value + kSection - 1) / kSection * kSection; }

// Row tiles are whole (the planner requires rows and K to be multiples of 256);
// the shapes cover several tiles, super-blocks and groups, and a permuted row range.
struct Shape {
  uint32_t rows, K;
  uint64_t permuteFrom;
  uint32_t headRows, groupHeads, groups;
};
constexpr uint64_t kNoPermute = UINT64_MAX;

// The narrowest K whose 256-row tile of source and plane bytes exceeds the
// preparation staging, so preparation must split the rows by columns.
uint32_t widerThanStaging(Fmt f) {
  const QuantFormat &layout = kQuantFormats[f];
  const uint64_t perBlock = uint64_t(layout.block_bytes) * (256 / layout.block_elements) +
      8 * (layout.plane0_bytes + layout.plane1_bytes) + (8 / layout.meta_groups) * layout.meta_bytes;
  return uint32_t((splash::model::kWeightPreparationStagingBytes / (256 * perBlock) + 1) * 256);
}

// One quantized tensor through the production executor, from a file whose
// data section starts at kSourceOffset, against the CPU reference's planes.
void checkRepack(splash::metal::MetalBackend &backend, Fmt f, const Shape &shape, uint32_t seed) {
  const QuantFormat &layout = kQuantFormats[f];
  const uint32_t rows = shape.rows, K = shape.K, stride = rowBytes(f, K);
  const std::vector<uint8_t> native = fixture(f, rows, K, seed);
  std::vector<uint8_t> ordered(native.size());
  for (uint32_t n = 0; n < rows; ++n)
    std::memcpy(ordered.data() + uint64_t{n} * stride,
                native.data() + sourceRow(n, shape.permuteFrom, shape.headRows, shape.groupHeads, shape.groups) * stride,
                stride);
  const Packed expected = repack(f, ordered, rows, K, nullptr);
  const uint64_t plane0 = kSection, plane1 = alignUp(plane0 + expected.w0.size());
  const uint64_t meta = layout.plane1_bytes ? alignUp(plane1 + expected.w1.size()) : plane1;
  const uint64_t bytes = alignUp(meta + expected.meta.size()) + kSection;
  char what[96];
  std::snprintf(what, sizeof what, "prepared %s rows=%u K=%u%s", fmtName(f), rows, K,
                shape.permuteFrom == kNoPermute ? "" : " permuted");
  char inputPath[] = "/tmp/splash-repack-source-XXXXXX";
  char outputPath[] = "/tmp/splash-repack-output-XXXXXX";
  const int inputFd = mkstemp(inputPath), outputFd = mkstemp(outputPath);
  try {
    if (inputFd < 0 || outputFd < 0 || ftruncate(outputFd, bytes))
      throw std::runtime_error("cannot create repack fixture");
    splash::model::writeWeightBytes(inputFd, kSourceOffset, native);
    splash::model::gguf::Repack step;
    step.format = f;
    step.rows = rows;
    step.columns = K;
    step.plane0 = plane0;
    step.plane1 = layout.plane1_bytes ? plane1 : 0;
    step.meta = meta;
    step.sources = {{"fixture", layout.ggml_type, 0, rows, stride,
                     {shape.permuteFrom, shape.headRows, shape.groupHeads, shape.groups}}};
    splash::model::gguf::Image plan;
    plan.bytes = bytes;
    plan.repacks.push_back(step);
    const uint64_t before = backend.memoryStats().allocatedBytes;
    splash::model::prepareGgufImage(backend, inputFd, kSourceOffset, outputFd, plan);
    check(backend.memoryStats().allocatedBytes == before, "repack releases its staging buffers");
    check(backend.memoryStats().peakAllocatedBytes <= splash::model::kWeightPreparationStagingBytes,
          "repack staging stays within the preparation staging bound");
    std::vector<uint8_t> actual(bytes), reference(bytes, 0);
    splash::model::readWeightBytes(outputFd, 0, actual);
    std::copy(expected.w0.begin(), expected.w0.end(), reference.begin() + plane0);
    if (layout.plane1_bytes) std::copy(expected.w1.begin(), expected.w1.end(), reference.begin() + plane1);
    std::copy(expected.meta.begin(), expected.meta.end(), reference.begin() + meta);
    check(actual == reference, what);
  } catch (const std::exception &error) {
    check(false, std::string(what) + ": " + error.what());
  }
  if (inputFd >= 0) close(inputFd);
  if (outputFd >= 0) close(outputFd);
  unlink(inputPath);
  unlink(outputPath);
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 2) {
      std::fprintf(stderr, "usage: gguf-repack --cpu | <metallib>\n");
      return 2;
    }
    void *ggml = nullptr;
    if (const char *oracle = std::getenv("SPLASH_GGML_ORACLE")) {
      ggml = dlopen(oracle, RTLD_NOW | RTLD_LOCAL);
      check(ggml, std::string("load ") + oracle + (ggml ? "" : std::string(": ") + dlerror()));
    }
    // Every run must exercise conversion, including the >4 GiB source offsets.
    char cachePath[] = "/tmp/splash-repack-cache-XXXXXX";
    if (!mkdtemp(cachePath)) return 1;
    struct CacheCleanup {
      const char *path;
      ~CacheCleanup() { std::filesystem::remove_all(path); }
    } cleanup{cachePath};
    setenv("SPLASH_WEIGHT_CACHE", cachePath, 1);
    checkGoldens(ggml);
    std::optional<splash::metal::MetalBackend> backend;
    if (std::string(argv[1]) != "--cpu") backend.emplace(argv[1]);
    splash::metal::MetalBackend *gpu = backend ? &*backend : nullptr;
    checkAlphaBeta(gpu);
    checkFloatTensors(gpu);
    checkMoeLayer(gpu);
    if (gpu) {
      checkGoldenImages(*gpu);
      checkSourceIdentity(*gpu);
      checkDenseTarget(*gpu);
      const Shape shapes[] = {{512, 1024, kNoPermute, 0, 0, 0},
                              {768, 1280, 256, 16, 8, 4},
                              {768, 8448, 128, 16, 8, 5}};
      for (int s = 0; s < 3; ++s)
        for (int f = 0; f < FMT_COUNT; ++f) checkRepack(*gpu, Fmt(f), shapes[s], 100 + 8 * s + f);
      // Several bounded row batches, and rows wider than one staging step.
      for (Fmt format : {Q3K, Q80}) {
        checkRepack(*gpu, format, {8704, 2048, kNoPermute, 0, 0, 0}, 741);
        checkRepack(*gpu, format, {256, widerThanStaging(format), kNoPermute, 0, 0, 0}, 742);
      }
    }
    std::printf("%s (%d failures)\n", failures ? "GGUF repack tests FAILED" : "GGUF repack tests passed", failures);
    return failures ? 1 : 0;
  }
}
