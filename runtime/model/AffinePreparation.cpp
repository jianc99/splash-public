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

// Reorders each field of each expert into [rows / 256][groups][256] tiles of
// its group bytes; rows past the parts are zero.
void writeProjection(int destination, const Section &section, std::vector<uint8_t> &input,
                     std::vector<uint8_t> &output, const PreparationCheck &admit) {
  const uint32_t groups = section.columns / kQ4GroupElements;
  const auto unit = groupBytes(section);
  const uint64_t expertBytes = section.bytes / section.experts;
  uint32_t sourceRows = 0;
  for (const auto &part : section.parts) sourceRows += part.rows;
  for (uint32_t expert = 0; expert < section.experts; ++expert) {
    uint64_t fieldBase = section.offset + expert * expertBytes;
    for (size_t field = 0; field < unit.size(); ++field) {
      const uint32_t stepGroups = chunkGroups(section, unit[field]);
      for (uint32_t firstRow = 0; firstRow < section.rows; firstRow += kQ4StorageN) {
        for (uint32_t firstGroup = 0; firstGroup < groups; firstGroup += stepGroups) {
          if (admit) admit();
          const uint32_t count = std::min(stepGroups, groups - firstGroup);
          const uint32_t rowBytes = count * unit[field];
          if (firstRow + kQ4StorageN > sourceRows) {
            const uint32_t padding = std::max(firstRow, sourceRows) - firstRow;
            std::fill(input.begin() + uint64_t(padding) * rowBytes, input.begin() + uint64_t(kQ4StorageN) * rowBytes, 0);
          }
          uint32_t partBegin = 0;
          for (const auto &part : section.parts) {
            const uint32_t begin = std::max(firstRow, partBegin);
            const uint32_t end = std::min(firstRow + kQ4StorageN, partBegin + part.rows);
            if (begin < end) {
              const uint64_t offset = (uint64_t(expert) * part.rows + begin - partBegin) * groups * unit[field] +
                                      uint64_t(firstGroup) * unit[field];
              if (count == groups)
                part.fields[field]->read(offset, {input.data() + uint64_t(begin - firstRow) * rowBytes,
                                                 uint64_t(end - begin) * rowBytes});
              else
                for (uint32_t row = begin; row < end; ++row)
                  part.fields[field]->read(offset + uint64_t(row - begin) * groups * unit[field],
                      {input.data() + uint64_t(row - firstRow) * rowBytes, rowBytes});
            }
            partBegin += part.rows;
          }
          for (uint32_t group = 0; group < count; ++group)
            for (uint32_t row = 0; row < kQ4StorageN; ++row)
              std::memcpy(output.data() + (uint64_t(group) * kQ4StorageN + row) * unit[field],
                          input.data() + uint64_t(row) * rowBytes + group * unit[field], unit[field]);
          const uint64_t offset = (uint64_t(firstRow / kQ4StorageN) * groups + firstGroup) * kQ4StorageN * unit[field];
          writeWeightBytes(destination, fieldBase + offset,
                           std::span(output).first(uint64_t(kQ4StorageN) * rowBytes));
        }
      }
      fieldBase += uint64_t(section.rows) * groups * unit[field];
    }
  }
}

// float(-exp(double(A_log))) of a BF16 or F32 vector.
void writeDecay(int destination, const Section &section) {
  if (section.bytes > 1024 * 1024) throw std::runtime_error("decay tensor exceeds preparation bound");
  std::vector<uint8_t> bytes(section.tensor->bytes);
  section.tensor->read(0, bytes);
  std::vector<float> values(section.bytes / sizeof(float));
  for (size_t i = 0; i < values.size(); ++i) {
    float logarithm;
    if (section.tensor->dtype == "BF16") {
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
PreparedWeight imageWeight(const Image &image, const std::string &source) {
  WeightIdentity identity("splash-affine-preparation-v2 " SPLASH_AFFINE_PREPARATION_ID);
  identity.record("image", image.magic, image.layer, image.type, image.bytes);
  for (const Section &section : image.sections) {
    identity.record("section", int(section.kind), section.offset, section.bytes, section.rows, section.columns,
                    section.experts, section.bits);
    if (section.tensor) section.tensor->identify(identity);
    for (const ProjectionPart &part : section.parts) {
      identity.record("part", part.rows);
      for (const SourceTensor *field : part.fields) field->identify(identity);
    }
  }
  return identity.weight(image.bytes, "target/" + image.name, source);
}

void writeImage(int destination, const Image &image, const PreparationCheck &admit) {
  // One input and one output buffer serve every section.
  uint64_t inputBytes = 0, outputBytes = 0;
  for (const Section &section : image.sections) {
    if (section.kind == SectionKind::Projection) {
      for (uint32_t unit : groupBytes(section)) {
        const uint64_t bytes = uint64_t(kQ4StorageN) * chunkGroups(section, unit) * unit;
        inputBytes = std::max(inputBytes, bytes);
        outputBytes = std::max(outputBytes, bytes);
      }
    } else if (section.kind == SectionKind::Copy) {
      inputBytes = std::max(inputBytes, std::min(section.bytes, kChunkBytes));
    }
  }
  std::vector<uint8_t> input(inputBytes), output(outputBytes);
  std::array<uint8_t, 16> header{};
  std::memcpy(header.data(), image.magic.data(), 8);
  std::memcpy(header.data() + 8, &image.layer, 4);
  std::memcpy(header.data() + 12, &image.type, 4);
  writeWeightBytes(destination, 0, header);
  for (const Section &section : image.sections) {
    if (admit) admit();
    if (section.kind == SectionKind::Projection) {
      writeProjection(destination, section, input, output, admit);
    } else if (section.kind == SectionKind::Decay) {
      writeDecay(destination, section);
    } else {
      if (section.bytes != section.tensor->bytes) throw std::runtime_error("affine tensor size differs from its plan");
      copyWeightBytes(section.tensor->file->descriptor(), section.tensor->offset, destination, section.offset,
                      section.bytes, input, admit);
    }
  }
}

} // namespace splash::model::affine
