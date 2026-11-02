#pragma once

#include "TestKvTier.hpp"
#include "engine/Cache.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace splash::test {

// A test tier that says how many demotions it admits now, as KvPageTier
// does (KvTier::demotionRoom): those its transfer limit and its share of
// the class's writes leave. It is the tier it wraps otherwise.
class RoomTier final : public engine::KvTier {
public:
  explicit RoomTier(TestKvTier &tier) : tier_(tier) {}
  uint64_t slotBytes() const noexcept override { return tier_.slotBytes(); }
  bool writable() const noexcept override { return tier_.writable(); }
  bool persistent() const noexcept override { return tier_.persistent(); }
  void label(const std::shared_ptr<engine::KvDiskSlot> &slot, std::vector<std::byte> label) override {
    tier_.label(slot, std::move(label));
  }
  bool canDemote(engine::WriteClass write) const noexcept override { return tier_.canDemote(write); }
  std::optional<uint32_t> demotionRoom(engine::WriteClass write) const noexcept override {
    if (!tier_.canDemote(write))
      return 0u;
    const auto ofClass = static_cast<uint32_t>(std::ranges::count_if(
        tier_.transfers, [&](const auto &transfer) { return !transfer->finished && transfer->write == write; }));
    return std::min(tier_.transferLimit - tier_.inFlight(), tier_.writeLimits[static_cast<size_t>(write)] - ofClass);
  }
  bool canRestore() const noexcept override { return tier_.canRestore(); }
  std::shared_ptr<engine::KvDiskSlot> acquireSlot(model::DiskBudget::Draw draw) override {
    return tier_.acquireSlot(draw);
  }
  std::unique_ptr<engine::KvTransfer> demote(uint32_t page, std::shared_ptr<engine::KvDiskSlot> slot,
                                             engine::WriteClass write, std::function<void()> completion) override {
    return tier_.demote(page, std::move(slot), write, std::move(completion));
  }
  std::unique_ptr<engine::KvTransfer> restore(std::shared_ptr<engine::KvDiskSlot> slot, uint32_t page,
                                              std::function<void()> completion) override {
    return tier_.restore(std::move(slot), page, std::move(completion));
  }
  void submitRestores() override { tier_.submitRestores(); }
  std::shared_ptr<const model::KvCopy> copy(const std::shared_ptr<engine::KvDiskSlot> &slot) const override {
    return tier_.copy(slot);
  }
  void poll() override { tier_.poll(); }

private:
  TestKvTier &tier_;
};

// Publishes the request's committed blocks and returns the deepest one, 0
// while no block is complete.
inline uint64_t publishBlocks(engine::Cache &cache, uint64_t requestId,
                              std::span<const uint32_t> tokens,
                              uint32_t committedTokens,
                              engine::PromptIdentity identity = {}) {
  cache.publishCommittedBlocks(requestId, tokens, committedTokens, identity);
  constexpr uint32_t pageTokens = engine::KvCache::pageTokens;
  return committedTokens < pageTokens
             ? 0
             : cache.blockAt(requestId, committedTokens / pageTokens * pageTokens);
}

// One round of what the engine does between commands to spill, with the
// tier's IO done at once: starts writing the Spill requests' published pages
// through (Cache::writeThrough), lands every write (TestKvTier::complete) and
// moves the eligible pages to the index form (Cache::spillWritten), as far
// as the pool admits, releasing the Full extents they empty but the runway.
// Returns the writes it started.
inline uint32_t spillRound(engine::Cache &cache, TestKvTier &tier) {
  const uint32_t written = cache.writeThrough();
  tier.complete();
  static_cast<void>(cache.pollTransfers());
  static_cast<void>(cache.spillWritten(engine::SpilledExtents::KeepRunway));
  return written;
}

// The blocks the request's next command reads from their copies, by page,
// 0 for a Full page: its page sources, cut as the engine cuts them
// (Cache::cutPageTable).
inline std::vector<uint64_t> indexBlocks(engine::Cache &cache, uint64_t requestId) {
  const engine::PageTableView table = cache.cutPageTable(requestId);
  std::vector<uint64_t> blocks(table.pages.size());
  if (table.sources) {
    for (uint32_t page = 0; page < blocks.size(); ++page)
      blocks[page] = (*table.sources)[page].block;
  }
  return blocks;
}

} // namespace splash::test
