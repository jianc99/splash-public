// Prepares one vision source and prints the prepared file and its cache key.
//
//   vision-preparation mlx|gguf DIRECTORY tiny|27b|35b cold|warm [EXPECTED]
//
// warm requires a cache hit. EXPECTED is an independently serialized file the
// prepared bytes must equal.

#include "model/VisionLoader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;

int main(int argc, char **argv) {
  try {
    if (argc < 5 || argc > 6)
      throw std::runtime_error("usage: vision-preparation mlx|gguf DIRECTORY "
                               "tiny|27b|35b cold|warm [EXPECTED]");
    const std::string format = argv[1], size = argv[3], mode = argv[4];
    if ((format != "mlx" && format != "gguf") ||
        (size != "tiny" && size != "27b" && size != "35b") ||
        (mode != "cold" && mode != "warm"))
      throw std::runtime_error("invalid vision-preparation arguments");
    const auto source = format == "mlx" ? model::VisionSource::Safetensors
                                        : model::VisionSource::Gguf;
    ops::VisionLayout layout;
    if (size == "35b")
      layout.outputHiddenSize = 2048;
    if (size == "tiny") {
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
    const auto forbidden = [] {
      throw std::runtime_error("unexpected warm conversion");
    };
    const model::VisionLoader preparation(argv[2], source, layout, {},
                                          mode == "warm" ? model::PreparationCheck(forbidden)
                                                         : model::PreparationCheck());
    if (preparation.weight().bytes != model::preparedVisionBytes(layout))
      throw std::runtime_error("vision size estimate differs");
    const auto path = preparation.prepare();
    if (model::VisionLoader(argv[2], source, layout, {}, forbidden).prepare() !=
        path)
      throw std::runtime_error("warm cache miss");
    if (argc == 6) {
      std::ifstream actual(path, std::ios::binary),
          expected(argv[5], std::ios::binary);
      if (!expected || std::filesystem::file_size(argv[5]) !=
                           std::filesystem::file_size(path))
        throw std::runtime_error("oracle size differs");
      std::vector<char> a(1024 * 1024), b(a.size());
      for (uint64_t offset = 0; actual; offset += a.size()) {
        actual.read(a.data(), a.size());
        const auto count = actual.gcount();
        expected.read(b.data(), count);
        if (!std::equal(a.begin(), a.begin() + count, b.begin()))
          throw std::runtime_error("oracle bytes differ in the MiB at " +
                                   std::to_string(offset));
      }
    }
    std::cout << path.string() << '\n' << preparation.weight().key << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
