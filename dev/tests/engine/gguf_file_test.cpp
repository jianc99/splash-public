#include "TestFiles.hpp"
#include "TestGguf.hpp"
#include "model/GgufFile.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gguf = splash::test::gguf;
using splash::model::ggml::kF32;
using splash::model::ggml::kQ4_K;

constexpr uint64_t kQ4KBlockBytes = 144;

// A file of one tensor, "weight", with one Q4_K block of data.
gguf::Bytes model(const std::vector<uint64_t> &dims, uint64_t displacement = 0, uint32_t type = kQ4_K) {
  return gguf::file({}, {{"weight", dims, type, gguf::Bytes(kQ4KBlockBytes), displacement}});
}

// A file whose only key, "unused", is an array: its element type and count,
// then the encoded elements.
gguf::Bytes array(const gguf::Bytes &value) { return gguf::file({{"unused", gguf::kArray, value}}, {}); }

void require(bool ok, const std::string &message) {
  if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
  try {
    const splash::test::TemporaryDirectory directory("splash-gguf-file");
    const auto path = directory.path() / "test.gguf";
    const auto rejects = [&](const char *name, const gguf::Bytes &data) {
      splash::test::writeFile(path, data);
      try {
        splash::model::WeightSource source(path);
        splash::model::GgufFile file(source);
      }
      catch (const splash::model::GgufError &) { return; }
      throw std::runtime_error(std::string("accepted invalid GGUF: ") + name);
    };
    splash::test::writeFile(path, gguf::file({gguf::stringKey("general.architecture", "fixture"),
                                              gguf::uint32Key("fixture.block_count", 2)},
                                             {{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)}}));
    splash::model::WeightSource source(path);
    splash::model::GgufFile valid(source);
    require(valid.architecture() == "fixture" && valid.unsignedValue("fixture.block_count") == 2,
            "metadata values changed");
    const auto &weight = valid.require("weight");
    require(weight.bytes == kQ4KBlockBytes && weight.elements() == 256 && weight.rows() == 1,
            "valid Q4_K shape/size changed");
    require(source.dataOffset() + weight.offset + weight.bytes == source.bytes(),
            "valid data must end exactly at EOF");

    rejects("shape overflow", model({256, uint64_t{1} << 60}));
    rejects("row product overflow", model({256, uint64_t{1} << 63, 2}));
    rejects("byte size overflow", model({uint64_t{1} << 62}, 0, kF32));
    rejects("offset addition overflow", model({256, 1}, std::numeric_limits<uint64_t>::max() - 31));
    rejects("offset past EOF", model({256, 1}, 160));
    rejects("zero dimension", model({256, 0}));
    rejects("unaligned block", model({255, 1}));
    rejects("unaligned offset", model({256, 1}, 1));
    auto truncated = model({256, 1});
    truncated.pop_back();
    rejects("truncated tensor", truncated);
    const gguf::Tensor block{"weight", {256, 1}, kQ4_K, gguf::Bytes(kQ4KBlockBytes)};
    rejects("duplicate name", gguf::file({}, {block, block}));
    for (auto count : {uint64_t{1} << 62, uint64_t{100}}) {
      gguf::Bytes elements;
      gguf::append(elements, gguf::kUint64);
      gguf::append(elements, count);
      rejects("overflowing/truncated skipped array", array(elements));
    }
    gguf::Bytes nested;
    for (int i = 0; i < 18; ++i) {
      gguf::append(nested, gguf::kArray);
      gguf::append<uint64_t>(nested, 1);
    }
    rejects("excessive nesting", array(nested));
    std::cout << "GGUF parser bounds tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
