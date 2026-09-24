#pragma once

// The images of an MLX affine checkpoint in the existing packed target ABI,
// as model/AffineTarget.cpp plans them: their identity and their writer.
// Codes, scales and biases are reordered into 256-row tiles without
// requantization, the GDN decay becomes float(-exp(double(A_log))), and
// every other tensor is copied as stored.

#include "model/PreparedWeights.hpp"

#include <array>
#include <string>
#include <vector>

namespace splash::model::affine {

enum class SectionKind { Copy, Decay, Projection };

// A checkpoint tensor a section reads: its name, the dtypes it is read in and
// its shape, planned from the layout; tensor is bound once the checkpoint is
// opened.
struct Input {
  std::string name;
  std::vector<std::string> dtypes;
  std::vector<uint64_t> shape;
  const SourceTensor *tensor = nullptr;
};

// Source rows of a fused projection: weight, scales and biases.
struct ProjectionPart {
  uint32_t rows = 0;
  std::array<Input, 3> fields{};
};

struct Section {
  SectionKind kind = SectionKind::Copy;
  uint64_t offset = 0, bytes = 0;
  Input input;                       // Copy and Decay
  std::vector<ProjectionPart> parts; // Projection: parts in row order, then zero rows
  uint32_t rows = 0, columns = 0, experts = 1, bits = 4;
};

// A 16-byte header (magic, layer, type) in a 16 KiB block, then 16 KiB-aligned
// sections; quantized lists the affine modules it reads and their bits.
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
  std::vector<std::pair<std::string, uint32_t>> quantized;
};

// The identity of an image planned from a checkpoint at `source`.
[[nodiscard]] PreparedWeight affineImageWeight(const Image &image, const std::string &source);

// Writes an image into its preallocated, zeroed destination within the
// preparation staging bound; admit runs before each chunk.
void writeAffineImage(int destination, const Image &image, const PreparationCheck &admit);

} // namespace splash::model::affine
