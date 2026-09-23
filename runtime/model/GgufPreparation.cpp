#include "model/GgufPreparation.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace splash::model {
namespace {

constexpr uint32_t kRows = 256, kColumns = 8192;

uint32_t sourceRow(uint32_t row, const GgufRepackParams &p) {
  if (row < p.permute_from_row) return row;
  if (!p.permute_head_rows || !p.permute_group_heads || !p.permute_groups)
    throw GgufError("invalid weight row permutation");
  const uint32_t head = (row - p.permute_from_row) / p.permute_head_rows;
  const uint64_t source = uint64_t(head % p.permute_groups) * p.permute_group_heads + head / p.permute_groups;
  const uint64_t result = p.permute_from_row + source * p.permute_head_rows +
                          (row - p.permute_from_row) % p.permute_head_rows;
  if (result >= p.rows) throw GgufError("weight row permutation is out of bounds");
  return static_cast<uint32_t>(result);
}

void requireRange(uint64_t offset, uint64_t bytes, uint64_t available) {
  if (offset > available || bytes > available - offset)
    throw GgufError("prepared weight section is out of bounds");
}

} // namespace

std::string ggufImageKey(const std::string &sourceDigest, const gguf::Image &image) {
  // Bump only when the transformation or stored bytes change, never for a
  // compute tile, core count or unrelated application release.
  const std::string identity = "splash-block32-preparation-v1\n" + sourceDigest;
  std::vector<uint8_t> bytes(identity.begin(), identity.end());
  const auto append = [&](const auto &value) {
    const auto *begin = reinterpret_cast<const uint8_t *>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
  };
  append(image.bytes);
  append(uint64_t(image.fills.size()));
  append(uint64_t(image.repacks.size()));
  append(uint64_t(image.copies.size()));
  for (const auto &fill : image.fills) {
    append(fill.offset);
    append(uint64_t(fill.bytes.size()));
    bytes.insert(bytes.end(), fill.bytes.begin(), fill.bytes.end());
  }
  for (const auto &repack : image.repacks) {
    append(repack.params);
    append(repack.sourceOffset);
    append(repack.sourceBytes);
  }
  for (const auto &copy : image.copies) {
    append(copy.params);
    append(copy.sourceOffset);
    append(copy.sourceBytes);
  }
  return weightDigest(bytes);
}

void prepareGgufImage(metal::MetalBackend &backend, int source, int destination,
                      const gguf::Image &image, const PreparationCheck &check) {
  const auto guard = [&] { backend.checkOperation(); if (check) check(); };
  guard();
  for (const auto &fill : image.fills) {
    requireRange(fill.offset, fill.bytes.size(), image.bytes);
    writeWeightBytes(destination, fill.offset, fill.bytes);
  }
  // Copies include raw embedding rows and F32 projections. They retain their
  // source precision and never pass through a quantization operation.
  {
    std::vector<uint8_t> bytes(1024 * 1024);
    for (const auto &copy : image.copies) {
      if (copy.params.bytes != copy.sourceBytes || copy.params.src_offset)
        throw GgufError("invalid prepared weight copy");
      requireRange(copy.params.dst_offset, copy.sourceBytes, image.bytes);
      for (uint64_t at = 0; at < copy.sourceBytes; at += bytes.size()) {
        guard();
        auto part = std::span(bytes).first(std::min<uint64_t>(bytes.size(), copy.sourceBytes - at));
        readWeightBytes(source, copy.sourceOffset + at, part);
        writeWeightBytes(destination, copy.params.dst_offset + at, part);
      }
    }
  }
  for (const auto &repack : image.repacks) {
    const auto &p = repack.params;
    if (p.fmt >= GGUF_FMT_COUNT || !p.rows || p.rows % kRows ||
        !p.input_size || p.input_size % 256 || p.src_offset)
      throw GgufError("invalid prepared weight repack");
    const auto &format = kQuantFormats[p.fmt];
    const uint32_t groups = p.input_size / 32;
    const uint64_t rowBytes = uint64_t(p.input_size / format.block_elements) * format.block_bytes;
    if (p.src_row_bytes != rowBytes || repack.sourceBytes != rowBytes * p.rows)
      throw GgufError("invalid prepared weight source size");
    const std::array<uint32_t, 3> unitBytes{format.plane0_bytes, format.plane1_bytes, format.meta_bytes};
    const std::array<uint32_t, 3> divisor{1, 1, format.meta_groups};
    const std::array<uint32_t, 3> base{p.dst_plane0, p.dst_plane1, p.dst_meta};
    for (size_t plane = 0; plane < base.size(); ++plane)
      if (unitBytes[plane]) requireRange(base[plane], uint64_t(p.rows) * (groups / divisor[plane]) * unitBytes[plane], image.bytes);
    const uint32_t maximumColumns = std::min(kColumns, p.input_size);
    const uint64_t inputBytes = uint64_t(kRows) * (maximumColumns / format.block_elements) * format.block_bytes;
    const uint64_t outputBytes = uint64_t(kRows) * (maximumColumns / 32) *
        (format.plane0_bytes + format.plane1_bytes) +
        uint64_t(kRows) * (maximumColumns / 32 / format.meta_groups) * format.meta_bytes;
    guard();
    auto input = backend.allocateBuffer(inputBytes, metal::BufferStorage::Shared, "prepare/source");
    auto output = backend.allocateBuffer(outputBytes, metal::BufferStorage::Shared, "prepare/planes");
    for (uint32_t firstRow = 0; firstRow < p.rows; firstRow += kRows) {
      for (uint32_t firstColumn = 0; firstColumn < p.input_size; firstColumn += kColumns) {
        guard();
        const uint32_t columns = std::min(kColumns, p.input_size - firstColumn);
        const uint32_t chunkRowBytes = columns / format.block_elements * format.block_bytes;
        auto *host = static_cast<uint8_t *>(input.contents());
        for (uint32_t row = 0; row < kRows; ++row)
          readWeightBytes(source, repack.sourceOffset + uint64_t(sourceRow(firstRow + row, p)) * rowBytes +
                          uint64_t(firstColumn / format.block_elements) * format.block_bytes,
                          {host + row * chunkRowBytes, chunkRowBytes});
        GgufRepackParams chunk{};
        chunk.rows = kRows;
        chunk.input_size = columns;
        chunk.fmt = p.fmt;
        chunk.src_row_bytes = chunkRowBytes;
        chunk.permute_from_row = std::numeric_limits<uint32_t>::max();
        const uint32_t chunkGroups = columns / 32;
        std::array<uint32_t, 3> lengths{};
        for (size_t plane = 0; plane < lengths.size(); ++plane)
          lengths[plane] = kRows * (chunkGroups / divisor[plane]) * unitBytes[plane];
        chunk.dst_plane1 = lengths[0];
        chunk.dst_meta = lengths[0] + lengths[1];
        const metal::ComputeDispatch dispatch{"gguf_repack", {{0, input}, {1, output}},
            {{2, &chunk, sizeof(chunk)}}, {uint64_t(kRows) * chunkGroups / 256, 1, 1}, {256, 1, 1}};
        static_cast<void>(backend.submitCommand(std::span(&dispatch, 1)));
        const auto *prepared = static_cast<const uint8_t *>(output.contents());
        uint32_t offset = 0;
        for (size_t plane = 0; plane < lengths.size(); ++plane) {
          const uint64_t position = (uint64_t(firstRow / kRows) * (groups / divisor[plane]) +
                                    firstColumn / 32 / divisor[plane]) * kRows * unitBytes[plane];
          writeWeightBytes(destination, base[plane] + position, {prepared + offset, lengths[plane]});
          offset += lengths[plane];
        }
      }
    }
  }
  guard();
}

} // namespace splash::model
