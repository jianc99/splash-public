#include "WeightPreparationIdentity.hpp"
#include "model/GgufPreparation.hpp"

#include "metal/abi/GgufRepack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

namespace splash::model {
namespace {

constexpr uint64_t kTileRows = 256; // rows of a plane tile (quant_tile_index)

// The source row image row `row` of `rows` is read from.
uint64_t sourceRow(const gguf::TensorRows &rows, uint64_t row) {
  const gguf::RowOrder &order = rows.order;
  if (row < order.from) return row;
  if (!order.headRows || !order.groupHeads || !order.groups)
    throw GgufError("invalid weight row permutation");
  const uint64_t head = (row - order.from) / order.headRows;
  const uint64_t source = order.from +
      ((head % order.groups) * order.groupHeads + head / order.groups) * order.headRows +
      (row - order.from) % order.headRows;
  if (source >= rows.rows) throw GgufError("weight row permutation is out of bounds");
  return source;
}

void requireRange(uint64_t offset, uint64_t bytes, uint64_t available) {
  if (offset > available || bytes > available - offset)
    throw GgufError("prepared weight section is out of bounds");
}

// Image rows [first, first + count) of `rows`, bytes [column, column + span)
// of each, back to back. Consecutive source rows are read together.
void readRows(int source, uint64_t dataOffset, const gguf::TensorRows &rows, uint64_t first,
              uint64_t count, uint64_t column, uint64_t span, uint8_t *to) {
  if (first > rows.rows || count > rows.rows - first || column > rows.rowBytes || span > rows.rowBytes - column)
    throw GgufError("prepared weight rows are out of bounds");
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (span == rows.rowBytes)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    readWeightBytes(source, dataOffset + rows.offset + start * rows.rowBytes + column, {to + row * span, run * span});
    row += run;
  }
}

// F32 values, narrowed in place to the bf16 values they equal: the kernels
// read these tensors as bf16, and preparation never rounds a weight.
std::span<uint8_t> narrowToBfloat16(std::span<uint8_t> bytes, const std::string &name) {
  const uint64_t count = bytes.size() / 4;
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t bits;
    std::memcpy(&bits, bytes.data() + 4 * i, 4);
    if (bits & 0xFFFFu) throw GgufError(name + " is not bf16-exact; it needs an F32 path");
    const auto upper = static_cast<uint16_t>(bits >> 16);
    std::memcpy(bytes.data() + 2 * i, &upper, 2);
  }
  return bytes.first(2 * count);
}

uint64_t copyBytes(const gguf::Copy &copy) {
  return copy.source.rows * copy.source.rowBytes / (copy.bfloat16 ? 2 : 1);
}

void writeCopy(int source, uint64_t dataOffset, int destination, const gguf::Copy &copy,
               std::span<uint8_t> staging, const PreparationCheck &admit) {
  const gguf::TensorRows &rows = copy.source;
  // Whole rows per read where they fit; a wider row in pieces of whole values.
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, staging.size() & ~uint64_t{3});
  const uint64_t batch = span == rows.rowBytes ? staging.size() / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      if (admit) admit();
      const uint64_t width = std::min(span, rows.rowBytes - column);
      auto bytes = staging.first(count * width);
      readRows(source, dataOffset, rows, first, count, column, width, bytes.data());
      if (copy.bfloat16) bytes = narrowToBfloat16(bytes, rows.name);
      writeWeightBytes(destination, copy.destination + (first * rows.rowBytes + column) / (copy.bfloat16 ? 2 : 1),
                       bytes);
    }
  }
}

// The rows and columns of one repack step and its staging: complete rows
// where they fit, so one read and one submission cover many 256-row tiles;
// very wide rows split within the same bound.
struct RepackChunk {
  uint64_t rows = 0, columns = 0, inputBytes = 0, outputBytes = 0;
};

RepackChunk repackChunk(const gguf::Repack &repack) {
  const QuantFormat &format = kQuantFormats[repack.format];
  const uint64_t planeBytes = format.plane0_bytes + format.plane1_bytes;
  const uint64_t bytesPerBlock = uint64_t(format.block_bytes) * (256 / format.block_elements) +
      8 * planeBytes + (8 / format.meta_groups) * format.meta_bytes;
  RepackChunk chunk;
  chunk.columns = std::min<uint64_t>(repack.columns,
      kWeightPreparationStagingBytes / (kTileRows * bytesPerBlock) * 256);
  if (!chunk.columns) throw GgufError("weight row exceeds preparation bound");
  const auto input = [&](uint64_t rows) { return rows * (chunk.columns / format.block_elements) * format.block_bytes; };
  const auto output = [&](uint64_t rows) {
    return rows * (chunk.columns / 32) * planeBytes + rows * (chunk.columns / 32 / format.meta_groups) * format.meta_bytes;
  };
  chunk.rows = chunk.columns == repack.columns
      ? std::min<uint64_t>(repack.rows, kWeightPreparationStagingBytes / (input(kTileRows) + output(kTileRows)) * kTileRows)
      : kTileRows;
  chunk.inputBytes = input(chunk.rows);
  chunk.outputBytes = output(chunk.rows);
  return chunk;
}

void writeRepack(metal::MetalBackend &backend, int source, uint64_t dataOffset, int destination,
                 const gguf::Repack &repack, uint64_t imageBytes, metal::MetalBuffer &input,
                 metal::MetalBuffer &output, const PreparationCheck &admit) {
  if (repack.format >= GGUF_FMT_COUNT || !repack.rows || repack.rows % kTileRows || !repack.columns ||
      repack.columns % 256 || repack.rows > std::numeric_limits<uint32_t>::max() ||
      repack.columns > std::numeric_limits<uint32_t>::max())
    throw GgufError("invalid prepared weight repack");
  const QuantFormat &format = kQuantFormats[repack.format];
  const uint64_t groups = repack.columns / 32;
  const uint64_t rowBytes = repack.columns / format.block_elements * format.block_bytes;
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &rows : repack.sources) {
    if (rows.rowBytes != rowBytes) throw GgufError("invalid prepared weight source size");
    sourceRows += rows.rows;
  }
  if (sourceRows > repack.rows) throw GgufError("invalid prepared weight source size");
  const std::array<uint64_t, 3> unitBytes{format.plane0_bytes, format.plane1_bytes, format.meta_bytes};
  const std::array<uint64_t, 3> divisor{1, 1, format.meta_groups};
  const std::array<uint64_t, 3> base{repack.plane0, repack.plane1, repack.meta};
  for (size_t plane = 0; plane < base.size(); ++plane)
    if (unitBytes[plane]) requireRange(base[plane], repack.rows * (groups / divisor[plane]) * unitBytes[plane], imageBytes);
  const RepackChunk chunk = repackChunk(repack);
  auto *host = static_cast<uint8_t *>(input.contents());
  const auto *prepared = static_cast<const uint8_t *>(output.contents());
  for (uint64_t firstRow = 0; firstRow < repack.rows; firstRow += chunk.rows) {
    const uint64_t rows = std::min(chunk.rows, repack.rows - firstRow);
    for (uint64_t firstColumn = 0; firstColumn < repack.columns; firstColumn += chunk.columns) {
      if (admit) admit();
      const uint64_t columns = std::min(chunk.columns, repack.columns - firstColumn);
      const uint64_t chunkRowBytes = columns / format.block_elements * format.block_bytes;
      const uint64_t column = firstColumn / format.block_elements * format.block_bytes;
      // The sources' rows in image order, then zero rows.
      uint64_t start = 0;
      for (const gguf::TensorRows &source_ : repack.sources) {
        const uint64_t begin = std::max(firstRow, start), end = std::min(firstRow + rows, start + source_.rows);
        if (begin < end)
          readRows(source, dataOffset, source_, begin - start, end - begin, column, chunkRowBytes,
                   host + (begin - firstRow) * chunkRowBytes);
        start += source_.rows;
      }
      if (start < firstRow + rows) {
        const uint64_t zero = std::max(start, firstRow);
        std::memset(host + (zero - firstRow) * chunkRowBytes, 0, (firstRow + rows - zero) * chunkRowBytes);
      }
      const uint64_t chunkGroups = columns / 32;
      std::array<uint64_t, 3> lengths{};
      for (size_t plane = 0; plane < lengths.size(); ++plane)
        lengths[plane] = rows * (chunkGroups / divisor[plane]) * unitBytes[plane];
      GgufRepackParams params{};
      params.rows = static_cast<uint32_t>(rows);
      params.input_size = static_cast<uint32_t>(columns);
      params.fmt = repack.format;
      params.src_row_bytes = static_cast<uint32_t>(chunkRowBytes);
      params.dst_plane1 = static_cast<uint32_t>(lengths[0]);
      params.dst_meta = static_cast<uint32_t>(lengths[0] + lengths[1]);
      const metal::ComputeDispatch dispatch{"gguf_repack", {{0, input}, {1, output}},
          {{2, &params, sizeof(params)}}, {rows * chunkGroups / 256, 1, 1}, {256, 1, 1}};
      static_cast<void>(backend.submitCommand(std::span(&dispatch, 1)));
      uint64_t offset = 0;
      for (size_t plane = 0; plane < lengths.size(); ++plane) {
        const uint64_t position = (firstRow / kTileRows * (groups / divisor[plane]) +
                                   firstColumn / 32 / divisor[plane]) * kTileRows * unitBytes[plane];
        writeWeightBytes(destination, base[plane] + position, {prepared + offset, lengths[plane]});
        offset += lengths[plane];
      }
    }
  }
}

} // namespace

std::string ggufImageKey(const std::string &sourceDigest, const gguf::Image &image) {
  // The envelope version covers identity serialization. Conversion code and
  // its storage ABI are fingerprinted at build time, independently of tuning.
  std::ostringstream identity;
  identity << "splash-gguf-preparation-v2\n" SPLASH_GGUF_PREPARATION_ID "\n" << sourceDigest << '\n'
           << "image " << image.layer << ' ' << image.type << ' ' << image.bytes << '\n';
  const auto rows = [&](const gguf::TensorRows &rows) {
    identity << "rows " << rows.type << ' ' << rows.offset << ' ' << rows.rows << ' ' << rows.rowBytes << ' '
             << rows.order.from << ' ' << rows.order.headRows << ' ' << rows.order.groupHeads << ' '
             << rows.order.groups << '\n';
  };
  for (const gguf::Fill &fill : image.fills)
    identity << "fill " << fill.offset << ' ' << weightDigest(fill.bytes) << '\n';
  for (const gguf::Copy &copy : image.copies) {
    identity << "copy " << copy.destination << ' ' << copy.bfloat16 << '\n';
    rows(copy.source);
  }
  for (const gguf::Repack &repack : image.repacks) {
    identity << "repack " << repack.format << ' ' << repack.rows << ' ' << repack.columns << ' '
             << repack.plane0 << ' ' << repack.plane1 << ' ' << repack.meta << '\n';
    for (const gguf::TensorRows &source : repack.sources) rows(source);
  }
  return weightDigest(identity.str());
}

void prepareGgufImage(metal::MetalBackend &backend, int source, uint64_t dataOffset, int destination,
                      const gguf::Image &image, const PreparationCheck &admit) {
  if (admit) admit();
  for (const gguf::Fill &fill : image.fills) {
    requireRange(fill.offset, fill.bytes.size(), image.bytes);
    writeWeightBytes(destination, fill.offset, fill.bytes);
  }
  // Copies keep their source precision and never pass through a
  // quantization operation. One staging buffer serves every copy.
  uint64_t copyStaging = 0;
  for (const gguf::Copy &copy : image.copies) {
    if (!copy.source.rows || !copy.source.rowBytes || (copy.bfloat16 && copy.source.rowBytes % 4))
      throw GgufError("invalid prepared weight copy");
    requireRange(copy.destination, copyBytes(copy), image.bytes);
    copyStaging = std::max(copyStaging, std::min(copy.source.rows * copy.source.rowBytes, kWeightPreparationStagingBytes));
  }
  {
    std::vector<uint8_t> staging(copyStaging);
    for (const gguf::Copy &copy : image.copies) writeCopy(source, dataOffset, destination, copy, staging, admit);
  }
  // One pair of staging buffers serves every repack of the image.
  RepackChunk largest;
  for (const gguf::Repack &repack : image.repacks) {
    if (repack.format >= GGUF_FMT_COUNT) throw GgufError("invalid prepared weight repack");
    const RepackChunk chunk = repackChunk(repack);
    largest.inputBytes = std::max(largest.inputBytes, chunk.inputBytes);
    largest.outputBytes = std::max(largest.outputBytes, chunk.outputBytes);
  }
  if (!image.repacks.empty()) {
    if (admit) admit();
    auto input = backend.allocateBuffer(largest.inputBytes, metal::BufferStorage::Shared, "prepare/source");
    auto output = backend.allocateBuffer(largest.outputBytes, metal::BufferStorage::Shared, "prepare/planes");
    for (const gguf::Repack &repack : image.repacks)
      writeRepack(backend, source, dataOffset, destination, repack, image.bytes, input, output, admit);
  }
  if (admit) admit();
}

} // namespace splash::model
