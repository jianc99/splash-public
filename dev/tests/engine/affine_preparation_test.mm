// Prepares a tiny dense or MoE affine checkpoint twice (cold, then warm
// without conversion headroom) and prints each prepared image's SHA-256.
// Dense images must also equal the independently serialized files in
// FIXTURE/expected.
//
//   affine-preparation METALLIB FIXTURE dense|moe
#include "model/AffineTarget.hpp"
#include "model/Qwen3_6Moe.hpp"
#include "model/Qwen3_8.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace splash;
namespace {

std::vector<uint8_t> fileBytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}

// The dimensions both fixtures share: two layers (GDN, then full attention)
// of width 256.
template <class Layout> Layout tinyLayout() {
  Layout layout;
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
  layout.attentionQueryHeads = 4;
  layout.attentionKvHeads = 2;
  layout.attentionHeadDimension = 64;
  layout.rotaryPairs = 8;
  layout.fullAttentionPeriod = 2;
  return layout;
}

template <class Layout>
void prepare(metal::MetalBackend &backend, const std::filesystem::path &root, const Layout &layout, bool oracle) {
  const std::filesystem::path cache(std::getenv("SPLASH_WEIGHT_CACHE"));
  bool cold = true;
  const auto admission = [&] { if (!cold) throw std::runtime_error("conversion forbidden on warm load"); };
  for (unsigned pass = 0; pass < 2; ++pass) {
    model::AffineTargetLoader loader(backend, root, layout, admission);
    const auto check = [&](model::WeightFile weights) {
      const auto &record = weights.record();
      const auto prepared = fileBytes(cache / record.contentIdentity / "weights");
      if (prepared.size() != record.declaredBytes) throw std::runtime_error("wrong image size");
      if (oracle && prepared != fileBytes(root / "expected" / std::filesystem::path(record.relativePath).filename()))
        throw std::runtime_error("affine fixture differs: " + record.relativePath);
      static_cast<void>(weights.section(record.declaredBytes - model::kWeightFileAlignment));
      weights.finish();
      if (pass == 0) std::cout << "prepared " << record.relativePath << ' ' << model::weightDigest(prepared) << '\n';
    };
    check(loader.layer(0, false));
    check(loader.layer(1, true));
    check(loader.head(2));
    check(loader.embedding(256, 256));
    cold = false;
  }
}

} // namespace

int main(int argc, char **argv) {
  @autoreleasepool {
    try {
      if (argc != 4 || (std::string_view(argv[3]) != "dense" && std::string_view(argv[3]) != "moe"))
        throw std::runtime_error("usage: affine-preparation METALLIB FIXTURE dense|moe");
      const std::filesystem::path root(argv[2]);
      setenv("SPLASH_WEIGHT_CACHE", (root / "cache").c_str(), 1);
      metal::MetalBackend backend(argv[1]);
      if (std::string_view(argv[3]) == "moe") {
        auto layout = tinyLayout<model::Qwen3_6MoeLayout>();
        layout.experts = 256;
        layout.expertsPerToken = 8;
        layout.expertIntermediateSize = 256;
        prepare(backend, root, layout, false);
        std::cout << "affine preparation: MoE experts, 8-bit router and shared-expert gate, warm admission PASS\n";
        return 0;
      }
      auto layout = tinyLayout<model::Qwen3_8Layout>();
      layout.intermediateSize = 512;
      prepare(backend, root, layout, true);
      // The target loader reads the prepared files as affine Q4 projections of
      // the layout's sizes with bf16 norms.
      const model::Qwen3_8Weights weights =
          model::loadQwen3_8Weights(backend, root, layout, model::TargetSource::Affine);
      const auto affine = [](const ops::Projection &p, uint32_t n, uint32_t k) {
        return p.layout() == ops::WeightLayout::Affine64 && p.outputSize == n && p.inputSize == k;
      };
      bool read = weights.layers.size() == layout.layers && !weights.finalNorm.float32 &&
                  affine(weights.logitsProjection, layout.vocabularySize, layout.hiddenSize) &&
                  weights.tokenEmbedding.layout() == ops::WeightLayout::Affine64;
      for (const auto &layer : weights.layers) {
        read = read && !layer.inputNorm.float32 && !layer.postAttentionNorm.float32 &&
               affine(layer.gateProjection, layout.intermediateSize, layout.hiddenSize) &&
               affine(layer.downProjection, layout.hiddenSize, layout.intermediateSize);
        if (const auto *gdn = std::get_if<model::QwenGdnWeights>(&layer.mixer))
          read = read && affine(gdn->inputProjection, layout.packedGdnWidth, layout.hiddenSize) &&
                 gdn->outputHeadOrder == ops::GdnHeadOrder::Grouped && !gdn->mixerNorm.float32;
        else
          read = read && affine(std::get<model::QwenAttentionWeights>(layer.mixer).inputProjection,
                                layout.packedFullWidth, layout.hiddenSize);
      }
      if (!read) throw std::runtime_error("the target loader misread the prepared affine files");
      // The model's other prepared files join the target's disk check.
      const model::PreparedWeight vision{std::string(64, 'a'), UINT64_MAX / 2, "vision/model.bin", std::string(64, 'b'), "/vision"};
      std::string budget;
      try {
        model::AffineTargetLoader loader(backend, root, layout, {}, {&vision, 1});
      } catch (const std::runtime_error &error) {
        budget = error.what();
      }
      if (!budget.starts_with("not enough disk space to prepare weights: need " +
                              std::to_string(vision.bytes) + " bytes"))
        throw std::runtime_error("other prepared files escaped the disk check: " + budget);
      std::cout << "affine preparation: exact independent fixture, padding, fused order, gate/up, warm admission and "
                   "model disk budget, target read PASS\n";
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
}
