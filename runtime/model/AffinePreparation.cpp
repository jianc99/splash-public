// Editing this file re-prepares every affine model.
#include "WeightPreparationIdentity.hpp"
#include "model/AffinePreparation.hpp"
#include "model/WeightLayout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace splash::model::affine {
namespace {

// Input and output staging: each half of the preparation bound.
constexpr uint64_t kChunkBytes = kWeightPreparationStagingBytes / 2;

// Bytes per row of a 64-column group: codes, scales, biases.
std::array<uint32_t, 3> groupBytes(const Section &section) { return {8 * section.bits, 2, 2}; }

// Groups of one field a 256-row tile step converts.
uint32_t chunkGroups(const Section &section, uint32_t unit) {
  return static_cast<uint32_t>(std::min<uint64_t>(section.columns / kQ4GroupElements, kChunkBytes / (kQ4StorageN * unit)));
}

// Gathers into input rows [firstRow, firstRow + 256) of groups [firstGroup,
// firstGroup + count) of one field of one expert: each part's rows as
// stored, rows past the parts zero. A step of whole rows reads a part's rows
// at once, as they are contiguous in the source.
void gatherRows(const Section &section, uint32_t expert, size_t field, uint32_t firstRow, uint32_t firstGroup,
                uint32_t count, std::span<uint8_t> input) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const uint32_t unit = groupBytes(section)[field];
  const uint64_t rowBytes = uint64_t(count) * unit;
  const uint64_t sourceRowBytes = uint64_t(groups) * unit;
  uint32_t partStart = 0;
  for (const auto &part : section.parts) {
    const uint32_t begin = std::max(firstRow, partStart);
    const uint32_t end = std::min(firstRow + kQ4StorageN, partStart + part.rows);
    if (begin < end) {
      const uint64_t offset = (uint64_t(expert) * part.rows + begin - partStart) * sourceRowBytes +
                              uint64_t(firstGroup) * unit;
      uint8_t *to = input.data() + (begin - firstRow) * rowBytes;
      if (count == groups) {
        part.fields[field].tensor->read(offset, {to, (end - begin) * rowBytes});
      } else {
        for (uint32_t row = 0; row < end - begin; ++row)
          part.fields[field].tensor->read(offset + row * sourceRowBytes, {to + row * rowBytes, rowBytes});
      }
    }
    partStart += part.rows;
  }
  if (firstRow + kQ4StorageN > partStart) {
    const uint32_t gathered = std::max(firstRow, partStart) - firstRow;
    std::fill(input.begin() + gathered * rowBytes, input.begin() + kQ4StorageN * rowBytes, 0);
  }
}

// Transposes a gathered tile of 256 rows of count groups into count groups of
// 256 rows, unit bytes each.
void tile(std::span<const uint8_t> input, std::span<uint8_t> output, uint32_t count, uint32_t unit) {
  const uint64_t rowBytes = uint64_t(count) * unit;
  for (uint32_t group = 0; group < count; ++group)
    for (uint32_t row = 0; row < kQ4StorageN; ++row)
      std::memcpy(output.data() + (uint64_t(group) * kQ4StorageN + row) * unit,
                  input.data() + row * rowBytes + uint64_t(group) * unit, unit);
}

// Reorders each field of each expert into [rows / 256][groups][256] tiles of
// its group bytes; rows past the parts are zero.
void writeProjection(int destination, const Section &section, std::vector<uint8_t> &input,
                     std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint64_t expertBytes = section.bytes / section.experts;
  for (uint32_t expert = 0; expert < section.experts; ++expert) {
    uint64_t fieldBase = section.offset + expert * expertBytes;
    for (size_t field = 0; field < unit.size(); ++field) {
      const uint32_t stepGroups = chunkGroups(section, unit[field]);
      for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
        for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
          admit();
          const uint32_t count = std::min(stepGroups, groups - firstGroup);
          gatherRows(section, expert, field, firstRow, firstGroup, count, input);
          tile(input, output, count, unit[field]);
          const uint64_t offset = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN * unit[field];
          writeWeightBytes(destination, fieldBase + offset,
                           std::span(output).first(uint64_t(kQ4StorageN) * count * unit[field]));
        }
      }
      fieldBase += uint64_t(section.rows) * groups * unit[field];
    }
  }
}

// float(-exp(double(A_log))) of a BF16 or F32 vector. The A_log it reads and
// the decay it writes are staged together, within the staging bound of every
// conversion step.
void writeDecay(int destination, const Section &section) {
  const SourceTensor &tensor = *section.input.tensor;
  if (tensor.bytes + section.bytes > kWeightPreparationStagingBytes)
    throw std::runtime_error("decay tensor exceeds the preparation staging bound");
  std::vector<uint8_t> bytes(tensor.bytes);
  tensor.read(0, bytes);
  std::vector<float> values(section.bytes / sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    float logarithm;
    if (tensor.dtype == "BF16") {
      uint16_t bfloat;
      std::memcpy(&bfloat, bytes.data() + i * 2, 2);
      const uint32_t bits = uint32_t(bfloat) << 16;
      std::memcpy(&logarithm, &bits, 4);
    } else std::memcpy(&logarithm, bytes.data() + i * 4, 4);
    values[i] = static_cast<float>(-std::exp(static_cast<double>(logarithm)));
    if (!std::isfinite(values[i])) throw std::runtime_error("non-finite GDN decay");
  }
  writeWeightBytes(destination, section.offset, {reinterpret_cast<const uint8_t *>(values.data()), section.bytes});
}

} // namespace

// The key: this code's identity, the plan and the bytes, dtype and shape of
// every tensor read. config.json is only validated against the layout, whose
// dimensions and quantization the plan records; nothing else in it changes
// these bytes.
PreparedWeight affineImageWeight(const Image &image, const std::string &source) {
  WeightIdentity identity("splash-affine-preparation-v2 " SPLASH_AFFINE_PREPARATION_ID);
  identity.record("image", image.magic, image.layer, image.type, image.bytes);
  for (const Section &section : image.sections) {
    identity.record("section", int(section.kind), section.offset, section.bytes, section.rows, section.columns,
                    section.experts, section.bits);
    if (section.kind != SectionKind::Projection) section.input.tensor->identify(identity);
    for (const ProjectionPart &part : section.parts) {
      identity.record("part", part.rows);
      for (const Input &field : part.fields) field.tensor->identify(identity);
    }
  }
  return identity.weight(image.bytes, "target/" + image.name, source);
}

void writeAffineImage(int destination, const Image &image, const PreparationCheck &admit) {
  // One input and one output buffer serve every section.
  uint64_t inputBytes = 0, outputBytes = 0;
  for (const Section &section : image.sections) {
    switch (section.kind) {
    case SectionKind::Projection:
      for (uint32_t unit : groupBytes(section)) {
        const uint64_t bytes = uint64_t(kQ4StorageN) * chunkGroups(section, unit) * unit;
        inputBytes = std::max(inputBytes, bytes);
        outputBytes = std::max(outputBytes, bytes);
      }
      break;
    case SectionKind::Copy:
      inputBytes = std::max(inputBytes, std::min(section.bytes, kChunkBytes));
      break;
    case SectionKind::Decay:
      break;
    }
  }
  std::vector<uint8_t> input(inputBytes), output(outputBytes);
  writeWeightBytes(destination, 0, weightFileHeader(image.magic, image.layer, image.type));
  for (const Section &section : image.sections) {
    admit();
    switch (section.kind) {
    case SectionKind::Projection:
      writeProjection(destination, section, input, output, admit);
      break;
    case SectionKind::Decay:
      writeDecay(destination, section);
      break;
    case SectionKind::Copy:
      section.input.tensor->copy(destination, section.offset, input, admit);
      break;
    }
  }
}

} // namespace splash::model::affine
