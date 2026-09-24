#pragma once

// The images of an MLX affine checkpoint in the existing packed target ABI,
// as model/AffineTarget.cpp plans them: their identity and their writer.
// Codes, scales and biases are reordered into 256-row tiles without
// requantization, the GDN decay becomes float(-exp(double(A_log))), and
// every other tensor is copied as stored.

#include "model/PreparedWeights.hpp"
#include "model/SafetensorsCheckpoint.hpp"

#include <array>
#include <string>
#include <vector>

namespace splash::model::affine {

enum class SectionKind { Copy, Decay, Projection };

// Source rows of a fused projection: weight, scales and biases.
struct ProjectionPart {
  uint32_t rows = 0;
  std::array<const SourceTensor *, 3> fields{};
};

struct Section {
  SectionKind kind = SectionKind::Copy;
  uint64_t offset = 0, bytes = 0;
  const SourceTensor *tensor = nullptr; // Copy and Decay
  std::vector<ProjectionPart> parts;    // Projection: parts in row order, then zero rows
  uint32_t rows = 0, columns = 0, experts = 1, bits = 4;
};

// A 16-byte header (magic, layer, type) in a 16 KiB block, then 16 KiB-aligned sections.
struct Image {
  std::string name, magic;
  uint32_t layer = 0, type = 0;
  uint64_t bytes = 0;
  std::vector<Section> sections;
};

// The identity of an image planned from a checkpoint at `source`.
[[nodiscard]] PreparedWeight imageWeight(const Image &image, const std::string &source);

// Writes an image into its preallocated, zeroed destination within the
// preparation staging bound; admit runs before each chunk.
void writeImage(int destination, const Image &image, const PreparationCheck &admit);

} // namespace splash::model::affine
