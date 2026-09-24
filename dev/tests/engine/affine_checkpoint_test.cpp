#include "model/SafetensorsCheckpoint.hpp"

#include <array>
#include <cstdlib>
#include <unistd.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace splash::model;
namespace {
void require(bool value, const char *message) {
  if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F run, const char *message) {
  bool rejected = false;
  try { run(); } catch (const std::exception &) { rejected = true; }
  require(rejected, message);
}
void shard(const std::filesystem::path &path, std::string_view header, size_t bytes = 16) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  const uint64_t length = header.size();
  file.write(reinterpret_cast<const char *>(&length), sizeof(length));
  file << header;
  for (size_t i = 0; i < bytes; ++i) file.put(static_cast<char>(i + 1));
}
constexpr auto valid = R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,16]}})";
}
int main() {
  char temporary[] = "/tmp/splash-affine-checkpoint-XXXXXX";
  if (!mkdtemp(temporary)) return 1;
  const std::filesystem::path root(temporary);
  setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
  try {
    std::ofstream(root / "config.json") << R"({"quantization":{"bits":4,"group_size":64,"router":{"bits":8}},"text_config":{"layers":2,"model_type":"fixture","layer_types":["linear_attention","full_attention"]}})";
    shard(root / "model.safetensors", valid);
    SafetensorsCheckpoint source(root);
    source.requireQuantization("projection", 4);
    source.requireQuantization("router", 8);
    source.requireConfigNumber("layers", 2);
    source.requireConfigString("model_type", "fixture");
    source.requireLayerTypes(2, 2);
    require(source.require("a").shape == std::vector<uint64_t>({2, 2}), "shape changed");
    std::array<uint8_t, 4> data{};
    source.require("a").read(7, data);
    require(data == std::array<uint8_t, 4>{8, 9, 10, 11}, "bounded slice differs");
    rejects([&] { source.require("a").read(14, data); }, "out-of-bounds read accepted");
    rejects([&] { (void)source.require("missing"); }, "missing tensor accepted");
    rejects([&] { source.requireQuantization("router", 4); }, "wrong quantization accepted");
    rejects([&] { source.requireLayerTypes(2, 1); }, "wrong layer schedule accepted");
    const auto first = source.digest();
    shard(root / "model.safetensors", valid);
    rejects([&] { source.checkUnchanged(); }, "changed source accepted");
    require(SafetensorsCheckpoint(root).digest() == first, "same content changed identity");
    shard(root / "extra.safetensors", valid);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "duplicate tensor accepted");
    std::filesystem::remove(root / "extra.safetensors");
    for (const auto header : {
        R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[0,15]}})",
        R"({"a":{"dtype":"U32","shape":[2,2],"data_offsets":[1,17]}})",
        R"({"a":{"dtype":"U32","shape":[true,4],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"U32","shape":[1,1,1,1,1,1,1,1,4],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"U32","shape":[4503599627370496,4096],"data_offsets":[0,16]}})",
        R"({"a":{"dtype":"U32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"U32","shape":[2],"data_offsets":[8,16]}})",
        R"({"a":{"dtype":"FP4","shape":[4],"data_offsets":[0,16]}})"}) {
      shard(root / "model.safetensors", header);
      rejects([&] { SafetensorsCheckpoint invalid(root); }, "malformed tensor accepted");
    }
    shard(root / "model.safetensors", valid, 8);
    rejects([&] { SafetensorsCheckpoint invalid(root); }, "truncated tensor accepted");
    std::filesystem::remove_all(root);
    std::cout << "affine checkpoint: bounded reads, metadata, quantization, identity and malformed sources PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(root);
    return 1;
  }
}
