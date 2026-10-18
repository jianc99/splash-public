#include "ops/KvSlotFormat.hpp"

#include <stdexcept>

namespace splash::kv {

// Stubs, which the SSD I/O foundation replaces with version 1, the
// block records: version 0 is the format the tier has always written.

namespace {

[[noreturn]] void blockRecordsMissing() {
  throw std::logic_error("the block-record slot format is not built yet");
}

} // namespace

SlotFormat::SlotFormat(const Layout &layout) : layout_(layout) {
  if (!layout_.valid()) throw std::invalid_argument("KV slot format layout is invalid");
}

uint32_t SlotFormat::version() const noexcept { return layout_.spillable() ? 1 : 0; }

uint64_t SlotFormat::payloadBytes() const noexcept { return version() ? 0 : layout_.bytesPerModelPage(); }

std::vector<uint64_t> SlotFormat::segments() const {
  if (version()) blockRecordsMissing();
  return {payloadBytes()};
}

SlotRange SlotFormat::indexRange() const { blockRecordsMissing(); }

uint64_t SlotFormat::recordBytes() const { blockRecordsMissing(); }

uint64_t SlotFormat::recordPitch() const { blockRecordsMissing(); }

SlotRange SlotFormat::recordRange(uint32_t, uint32_t) const { blockRecordsMissing(); }

SlotRange SlotFormat::layerRange(uint32_t) const { blockRecordsMissing(); }

std::vector<std::span<std::byte>> SlotFormat::transferSpans(const PageStorage &pages, uint32_t page) const {
  if (version()) blockRecordsMissing();
  return pages.spans(page);
}

void SlotFormat::scatterIndex(std::span<const std::byte>, std::span<const std::span<std::byte>>) const {
  blockRecordsMissing();
}

void SlotFormat::scatterRecord(std::span<const std::byte>, const std::array<std::span<std::byte>, 4> &,
                               uint32_t) const {
  blockRecordsMissing();
}

void SlotFormat::scatterLayer(std::span<const std::byte>, const std::array<std::span<std::byte>, 4> &) const {
  blockRecordsMissing();
}

} // namespace splash::kv
