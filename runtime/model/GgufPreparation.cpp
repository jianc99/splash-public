// Editing this file re-prepares every GGUF model.
#include "WeightPreparationIdentity.hpp"
#include "model/GgufPreparation.hpp"

#include "metal/abi/GgufRepack.h"
#include "model/Bfloat16.hpp"
#include "model/GgufImageLayout.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

namespace splash::model {
namespace {

// A chunk's rows, columns and plane bytes are within the staging bound, so
// they fit the repack kernel's 32-bit parameters.
static_assert(kWeightPreparationStagingBytes <= std::numeric_limits<uint32_t>::max());

// The source row image row `row` of `rows` is read from.
uint64_t sourceRow(const gguf::TensorRows &rows, uint64_t row) {
  const gguf::RowOrder &order = rows.order;
  if (row < order.from) return row;
  if (!order.headRows || !order.keyHeads || !order.valueHeadsPerKey)
    throw GgufError("invalid weight row permutation");
  const uint64_t head = (row - order.from) / order.headRows;
  const uint64_t source = order.from +
      ((head % order.valueHeadsPerKey) * order.keyHeads + head / order.valueHeadsPerKey) * order.headRows +
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
void readRows(const WeightSource &source, const gguf::TensorRows &rows, uint64_t first,
              uint64_t count, uint64_t column, uint64_t span, uint8_t *to) {
  if (first > rows.rows || count > rows.rows - first || column > rows.rowBytes || span > rows.rowBytes - column)
    throw GgufError("prepared weight rows are out of bounds");
  for (uint64_t row = 0; row < count;) {
    const uint64_t start = sourceRow(rows, first + row);
    uint64_t run = 1;
    if (span == rows.rowBytes)
      while (row + run < count && sourceRow(rows, first + row + run) == start + run) ++run;
    source.readData(rows.offset + start * rows.rowBytes + column, {to + row * span, run * span});
    row += run;
  }
}

// F32 values, narrowed in place to the bf16 values they equal: the kernels
// read these tensors as bf16, and preparation never rounds a weight.
std::span<uint8_t> narrowToBfloat16(std::span<uint8_t> bytes, const std::string &name) {
  const uint64_t count = bytes.size() / 4;
  for (uint64_t i = 0; i < count; ++i) {
    float value;
    std::memcpy(&value, bytes.data() + 4 * i, 4);
    const std::optional<uint16_t> narrowed = exactBfloat16(value);
    if (!narrowed) throw GgufError(name + " is not bf16-exact; it needs an F32 path");
    std::memcpy(bytes.data() + 2 * i, &*narrowed, 2);
  }
  return bytes.first(2 * count);
}

uint64_t copyBytes(const gguf::Copy &copy) {
  return copy.source.rows * copy.source.rowBytes / (copy.bfloat16 ? 2 : 1);
}

void writeCopy(const WeightSource &source, int destination, const gguf::Copy &copy,
               std::span<uint8_t> staging, const PreparationCheck &admit) {
  const gguf::TensorRows &rows = copy.source;
  // Whole rows per read where they fit; a wider row in pieces of whole values.
  const uint64_t span = std::min<uint64_t>(rows.rowBytes, staging.size() & ~uint64_t{3});
  const uint64_t batch = span == rows.rowBytes ? staging.size() / rows.rowBytes : 1;
  for (uint64_t first = 0; first < rows.rows; first += batch) {
    const uint64_t count = std::min(batch, rows.rows - first);
    for (uint64_t column = 0; column < rows.rowBytes; column += span) {
      admit();
      const uint64_t width = std::min(span, rows.rowBytes - column);
      auto bytes = staging.first(count * width);
      readRows(source, rows, first, count, column, width, bytes.data());
      if (copy.bfloat16) bytes = narrowToBfloat16(bytes, rows.name);
      writeWeightBytes(destination, copy.destination + (first * rows.rowBytes + column) / (copy.bfloat16 ? 2 : 1),
                       bytes);
    }
  }
}

// One of a repack's three planes: where it starts in the image, its bytes
// per unit of a row and the 32-column groups a unit covers.
struct Plane {
  uint64_t offset, unitBytes, unitGroups;
  // Bytes of `rows` rows of `groups` 32-column groups.
  [[nodiscard]] uint64_t bytes(uint64_t rows, uint64_t groups) const { return rows * (groups / unitGroups) * unitBytes; }
};

std::array<Plane, 3> planes(const gguf::Repack &repack) {
  const QuantFormat &format = kQuantFormats[repack.format];
  return {{{repack.plane0, format.plane0_bytes, 1},
           {repack.plane1, format.plane1_bytes, 1},
           {repack.meta, format.meta_bytes, format.meta_groups}}};
}

// The rows and columns of one repack step and its staging: complete rows
// where they fit, so one read and one submission cover many 256-row tiles;
// very wide rows split within the same bound.
struct RepackChunk {
  uint64_t rows = 0, columns = 0, inputBytes = 0, outputBytes = 0;
};

RepackChunk repackChunk(const gguf::Repack &repack) {
  const QuantFormat &format = kQuantFormats[repack.format];
  // Input and output staging of 256 columns of one row.
  const uint64_t stagingPer256Columns = ggufRowBytes(format, 256) +
      ggufPlaneBytes(format, 1, 256).plane0 + ggufPlaneBytes(format, 1, 256).plane1 +
      ggufPlaneBytes(format, 1, 256).meta;
  RepackChunk chunk;
  chunk.columns = std::min<uint64_t>(repack.columns,
      kWeightPreparationStagingBytes / (kGgufTileRows * stagingPer256Columns) * 256);
  if (!chunk.columns) throw GgufError("weight row exceeds preparation bound");
  const auto input = [&](uint64_t rows) { return rows * ggufRowBytes(format, chunk.columns); };
  const auto output = [&](uint64_t rows) {
    const GgufPlaneBytes bytes = ggufPlaneBytes(format, rows, chunk.columns);
    return bytes.plane0 + bytes.plane1 + bytes.meta;
  };
  chunk.rows = chunk.columns == repack.columns
      ? std::min<uint64_t>(repack.rows,
            kWeightPreparationStagingBytes / (input(kGgufTileRows) + output(kGgufTileRows)) * kGgufTileRows)
      : kGgufTileRows;
  chunk.inputBytes = input(chunk.rows);
  chunk.outputBytes = output(chunk.rows);
  return chunk;
}

// A repack's plan within an image of imageBytes: its format, tile-aligned
// shape, sources of its row width and planes inside the image.
void requireRepack(const gguf::Repack &repack, uint64_t imageBytes) {
  if (repack.format >= GGUF_FMT_COUNT || !repack.rows || repack.rows % kGgufTileRows || !repack.columns ||
      repack.columns % kGgufBlockColumns)
    throw GgufError("invalid prepared weight repack");
  const uint64_t rowBytes = ggufRowBytes(kQuantFormats[repack.format], repack.columns);
  uint64_t sourceRows = 0;
  for (const gguf::TensorRows &rows : repack.sources) {
    if (rows.rowBytes != rowBytes) throw GgufError("invalid prepared weight source size");
    sourceRows += rows.rows;
  }
  if (sourceRows > repack.rows) throw GgufError("invalid prepared weight source size");
  for (const Plane &plane : planes(repack))
    if (plane.unitBytes) requireRange(plane.offset, plane.bytes(repack.rows, repack.columns / 32), imageBytes);
}

void writeRepack(metal::MetalBackend &backend, const WeightSource &source, int destination,
                 const gguf::Repack &repack, const RepackChunk &chunk, metal::MetalBuffer &input,
                 metal::MetalBuffer &output, const PreparationCheck &admit) {
  const QuantFormat &format = kQuantFormats[repack.format];
  const std::array<Plane, 3> plane = planes(repack);
  const uint64_t groups = repack.columns / 32;
  auto *host = static_cast<uint8_t *>(input.contents());
  const auto *prepared = static_cast<const uint8_t *>(output.contents());
  for (uint64_t firstRow = 0; firstRow < repack.rows; firstRow += chunk.rows) {
    const uint64_t rows = std::min(chunk.rows, repack.rows - firstRow);
    for (uint64_t firstColumn = 0; firstColumn < repack.columns; firstColumn += chunk.columns) {
      admit();
      const uint64_t columns = std::min(chunk.columns, repack.columns - firstColumn);
      const uint64_t chunkRowBytes = ggufRowBytes(format, columns);
      const uint64_t column = ggufRowBytes(format, firstColumn);
      // The sources' rows in image order, then zero rows.
      uint64_t start = 0;
      for (const gguf::TensorRows &tensor : repack.sources) {
        const uint64_t begin = std::max(firstRow, start), end = std::min(firstRow + rows, start + tensor.rows);
        if (begin < end)
          readRows(source, tensor, begin - start, end - begin, column, chunkRowBytes,
                   host + (begin - firstRow) * chunkRowBytes);
        start += tensor.rows;
      }
      if (start < firstRow + rows) {
        const uint64_t zero = std::max(start, firstRow);
        std::memset(host + (zero - firstRow) * chunkRowBytes, 0, (firstRow + rows - zero) * chunkRowBytes);
      }
      const uint64_t chunkGroups = columns / 32;
      std::array<uint64_t, 3> lengths{};
      for (size_t i = 0; i < lengths.size(); ++i) lengths[i] = plane[i].bytes(rows, chunkGroups);
      GgufRepackParams params{};
      params.rows = static_cast<uint32_t>(rows);
      params.input_size = static_cast<uint32_t>(columns);
      params.fmt = repack.format;
      params.src_row_bytes = static_cast<uint32_t>(chunkRowBytes);
      params.dst_plane1 = static_cast<uint32_t>(lengths[0]);
      params.dst_meta = static_cast<uint32_t>(lengths[0] + lengths[1]);
      const metal::ComputeDispatch dispatch{"gguf_repack", {{0, input}, {1, output}},
          {{2, &params, sizeof(params)}}, {rows * chunkGroups / 256, 1, 1}, {256, 1, 1}};
      static_cast<void>(backend.submit(dispatch));
      uint64_t offset = 0;
      for (size_t i = 0; i < lengths.size(); ++i) {
        const uint64_t position = (firstRow / kGgufTileRows * (groups / plane[i].unitGroups) +
                                   firstColumn / 32 / plane[i].unitGroups) * kGgufTileRows * plane[i].unitBytes;
        writeWeightBytes(destination, plane[i].offset + position, {prepared + offset, lengths[i]});
        offset += lengths[i];
      }
    }
  }
}

} // namespace

PreparedWeight ggufImageWeight(const WeightSource &source, const gguf::Image &image) {
  // The envelope version covers identity serialization. Conversion code and
  // its storage ABI are fingerprinted at build time, independently of tuning.
  WeightIdentity identity("splash-gguf-preparation-v2 " SPLASH_GGUF_PREPARATION_ID);
  identity.record("image", image.layer, image.type, image.bytes);
  const auto input = [&](const gguf::TensorRows &rows) {
    const uint64_t shape[] = {rows.rows, rows.rowBytes};
    identity.input(source, rows.offset, rows.rows * rows.rowBytes, ggmlTypeName(rows.type), shape);
    identity.record("order", rows.order.from, rows.order.headRows, rows.order.keyHeads,
                    rows.order.valueHeadsPerKey);
  };
  for (const gguf::Fill &fill : image.fills) identity.record("fill", fill.offset, weightDigest(fill.bytes));
  for (const gguf::Copy &copy : image.copies) {
    identity.record("copy", copy.destination, copy.bfloat16);
    input(copy.source);
  }
  for (const gguf::Repack &repack : image.repacks) {
    identity.record("repack", repack.format, repack.rows, repack.columns, repack.plane0, repack.plane1, repack.meta);
    for (const gguf::TensorRows &rows : repack.sources) input(rows);
  }
  return identity.weight(image.bytes, "target/" + image.name, source.path().string());
}

void writeGgufImage(metal::MetalBackend &backend, const WeightSource &source, int destination,
                    const gguf::Image &image, const PreparationCheck &admit) {
  admit();
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
    for (const gguf::Copy &copy : image.copies) writeCopy(source, destination, copy, staging, admit);
  }
  // One pair of staging buffers serves every repack of the image.
  std::vector<RepackChunk> chunks;
  RepackChunk largest;
  for (const gguf::Repack &repack : image.repacks) {
    requireRepack(repack, image.bytes);
    chunks.push_back(repackChunk(repack));
    largest.inputBytes = std::max(largest.inputBytes, chunks.back().inputBytes);
    largest.outputBytes = std::max(largest.outputBytes, chunks.back().outputBytes);
  }
  if (!image.repacks.empty()) {
    admit();
    auto input = backend.allocateBuffer(largest.inputBytes, metal::BufferStorage::Shared, "prepare/source");
    auto output = backend.allocateBuffer(largest.outputBytes, metal::BufferStorage::Shared, "prepare/planes");
    for (size_t i = 0; i < image.repacks.size(); ++i)
      writeRepack(backend, source, destination, image.repacks[i], chunks[i], input, output, admit);
  }
  admit();
}

} // namespace splash::model
