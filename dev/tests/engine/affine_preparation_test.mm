#include "model/AffineTarget.hpp"
#include "model/Qwen3_8.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace splash;
int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 3) throw std::runtime_error("usage: affine-preparation METALLIB FIXTURE");
      const std::filesystem::path root(argv[2]);
      setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
      metal::MetalBackend backend(argv[1]);
      model::Qwen3_8Layout layout;
      layout.layers = 2;
      layout.hiddenSize = 256;
      layout.vocabularySize = 256;
      layout.packedGdnWidth = 1024;
      layout.packedFullWidth = 768;
      layout.convolutionDimension = 512;
      layout.gdnKeyHeads = 2;
      layout.gdnValueHeads = 4;
      layout.gdnHeadDimension = 64;
      layout.attentionWidth = 256;
      layout.intermediateSize = 512;
      layout.attentionQueryHeads = 4;
      layout.attentionKvHeads = 2;
      layout.attentionHeadDimension = 64;
      layout.rotaryPairs = 8;
      layout.fullAttentionPeriod = 2;
      bool cold = true;
      const auto admission = [&] { if (!cold) throw std::runtime_error("conversion forbidden on warm load"); };
      for (unsigned pass = 0; pass < 2; ++pass) {
        model::AffineTargetLoader loader(backend, root, layout, admission);
        const auto check = [&](model::WeightFile weights) {
          const auto &record = weights.record();
          std::ifstream expected(root / "expected" / std::filesystem::path(record.relativePath).filename(), std::ios::binary);
          std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(expected)), {});
          if (bytes.size() != record.declaredBytes) throw std::runtime_error("wrong image size");
          const auto contents = weights.section(record.declaredBytes - model::kWeightFileAlignment);
          if (std::memcmp(contents.contents(), bytes.data() + model::kWeightFileAlignment, contents.sizeBytes()))
            throw std::runtime_error("affine fixture differs: " + record.relativePath);
          weights.finish();
        };
        check(loader.layer(0, false));
        check(loader.layer(1, true));
        check(loader.head(2));
        check(loader.embedding(256, 256));
        cold = false;
      }
      // The model's other prepared files join the target's disk check.
      const model::PreparedWeight vision{std::string(64, 'a'), UINT64_MAX / 2};
      std::string budget;
      try {
        model::AffineTargetLoader loader(backend, root, layout, admission, {&vision, 1});
      } catch (const std::runtime_error &error) {
        budget = error.what();
      }
      if (!budget.starts_with("not enough disk space to prepare weights: need " +
                              std::to_string(vision.bytes) + " bytes"))
        throw std::runtime_error("other prepared files escaped the disk check: " + budget);
      std::cout << "affine preparation: exact independent fixture, padding, fused order, gate/up, warm admission and "
                   "model disk budget PASS\n";
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
