#include "model/VisionPreparation.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splash;
int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5)
      throw std::runtime_error(
          "usage: vision-preparation mlx|gguf DIRECTORY tiny|35b [EXPECTED]");
    model::VisionSource source = std::string(argv[1]) == "mlx"
                                     ? model::VisionSource::Safetensors
                                     : model::VisionSource::Gguf;
    ops::VisionLayout layout;
    layout.outputHiddenSize = 2048;
    if (std::string(argv[3]) == "tiny") {
      layout.depth = 2;
      layout.hiddenSize = 8;
      layout.patchSize = 2;
      layout.patchDimension = 24;
      layout.intermediateSize = 10;
      layout.paddedIntermediateSize = 16;
      layout.mergedHiddenSize = 32;
      layout.outputHiddenSize = 8;
      layout.heads = 2;
      layout.headDimension = 4;
      layout.positionGridSide = 2;
    }
    auto prepared = model::prepareVisionWeights(argv[2], source, layout);
    if (prepared.bytes != model::preparedVisionBytes(argv[2], source, layout))
      throw std::runtime_error("vision size estimate differs");
    const auto warm =
        model::prepareVisionWeights(argv[2], source, layout, {}, [] {
          throw std::runtime_error("unexpected warm conversion");
        });
    if (warm.path != prepared.path)
      throw std::runtime_error("warm cache miss");
    if (argc == 5) {
      std::ifstream actual(prepared.path, std::ios::binary),
          expected(argv[4], std::ios::binary);
      if (!expected || std::filesystem::file_size(argv[4]) != prepared.bytes)
        throw std::runtime_error("oracle size differs");
      std::vector<char> a(1024 * 1024), b(a.size());
      uint64_t offset = 0;
      while (actual) {
        actual.read(a.data(), a.size());
        const auto count = actual.gcount();
        expected.read(b.data(), count);
        if (!std::equal(a.begin(), a.begin() + count, b.begin()))
          throw std::runtime_error("oracle bytes differ at chunk " +
                                   std::to_string(offset));
        offset += count;
      }
    }
    std::cout << prepared.path << '\n';
    for (auto type : prepared.precision)
      std::cout << (type == ops::VisionPrecision::Float32 ? 4 : 2);
    std::cout << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
