#pragma once

#include "engine/KvTier.hpp"
#include "model/Model.hpp"
#include "model/SlotFile.hpp"
#include "model/SsdReaders.hpp"
#include "ops/KvSlotFormat.hpp"
#include "ops/PageStorage.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace splash::engine {

// Moves KV pages between their extents and a slot file, each page laid out
// in its slot as kv::SlotFormat says, so the tier holds no memory of its own
// beyond what reads land in. The file's IO worker writes a page straight from
// its extent (SlotFormat::transferSpans), a bounded number at once: reclaim
// may use all of them, persistent copies a quarter and write-through half, so
// that one kind never starves another and reclaim always keeps a quarter.
// Reads go to the SSD readers beside the worker (SlotFile::readAt): a
// restore is a task of a background batch that reads what its page's form
// holds, the index alone or all of it, into a buffer of its reader and
// scatters it into the page; the restores asked for since the last
// submitRestores() go out as one batch. Copies of slots are read the same way
// by whoever holds them (copy()).
class KvPageTier final : public KvTier {
public:
  // A slot of the tier, which is also the copy reads of spilled K/V reach
  // (model::KvCopy).
  struct DiskSlot final : KvDiskSlot, model::KvCopy {
    DiskSlot(std::shared_ptr<model::SlotFile::Slot> held,
             std::shared_ptr<const model::SlotFile> owner)
        : slot(std::move(held)), file(std::move(owner)) {}
    [[nodiscard]] uint64_t id() const noexcept override { return slot->id(); }
    [[nodiscard]] std::optional<std::span<const std::byte>>
    readAt(uint64_t offset, uint64_t bytes, std::span<std::byte> buffer) const noexcept override {
      return file->readAt(*slot, offset, bytes, buffer);
    }
    std::shared_ptr<model::SlotFile::Slot> slot;
    std::shared_ptr<const model::SlotFile> file;
  };
  // Writes in flight at once, by default, and restores at most.
  static constexpr uint32_t kWrites = 64;
  static constexpr uint32_t kRestores = 4096;

  KvPageTier(kv::PageStorage &pages, kv::SlotFormat format,
             std::shared_ptr<model::SlotFile> file, std::shared_ptr<model::SsdReaders> readers,
             uint32_t writes = kWrites);
  // How the tier lays a page of the layout out in its slot, as a persistent
  // cache's namespace names it (persistentCacheNamespace's kv line): empty
  // for the order of the page's spans (kv::SlotFormat version 0), which every
  // cache before block records was written in, and the block records' version
  // for a spillable layout. Whatever changes what the tier writes changes
  // this with it, so that no process adopts a slot of another layout.
  [[nodiscard]] static std::string_view slotFormatToken(const kv::Layout &layout) noexcept;
  ~KvPageTier() override;
  KvPageTier(const KvPageTier &) = delete;
  KvPageTier &operator=(const KvPageTier &) = delete;

  [[nodiscard]] uint64_t slotBytes() const noexcept override;
  [[nodiscard]] bool writable() const noexcept override;
  [[nodiscard]] bool persistent() const noexcept override;
  void label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) override;
  // Takes back a slot an earlier process recorded in a persistent tier's
  // file (model::SlotFile::adopt).
  [[nodiscard]] std::shared_ptr<KvDiskSlot> adopt(const model::SlotRecord &record);
  [[nodiscard]] bool canDemote(WriteClass write) const noexcept override;
  [[nodiscard]] bool canRestore() const noexcept override;
  [[nodiscard]] std::shared_ptr<KvDiskSlot> acquireSlot(model::DiskBudget::Draw draw) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot, WriteClass write,
         std::function<void()> completion) override;
  [[nodiscard]] std::unique_ptr<KvTransfer>
  restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
          std::function<void()> completion) override;
  void submitRestores() override;
  [[nodiscard]] std::shared_ptr<const model::KvCopy>
  copy(const std::shared_ptr<KvDiskSlot> &slot) const override;
  void poll() override;

private:
  // What a transfer's ticket reports once the tier retired it.
  struct Outcome final {
    bool ready = false;
    bool success = false;
  };
  class Ticket;
  struct Write final {
    WriteClass write;
    std::shared_ptr<model::SlotFile::Operation> io;
    std::shared_ptr<Outcome> outcome;
  };
  // A restore: its slot, its page and whether it reads the page's index
  // alone (an Index page) or all of it.
  struct Restore final {
    std::shared_ptr<const DiskSlot> copy;
    uint32_t page = 0;
    bool index = false;
    std::function<void()> completion;
    std::shared_ptr<Outcome> outcome;
  };
  // Restores submitted together, until the tier retired each of them.
  struct RestoreBatch final {
    std::shared_ptr<std::vector<Restore>> restores;
    std::shared_ptr<model::SsdReaders::Batch> batch;
    std::vector<bool> retired;
    size_t left = 0;
  };
  // A reader's task: reads one restore's range and scatters it.
  void read(const Restore &restore) const;
  [[nodiscard]] uint32_t share(WriteClass write) const noexcept;

  kv::PageStorage &pages_;
  const kv::SlotFormat format_;
  std::shared_ptr<model::SlotFile> file_;
  std::shared_ptr<model::SsdReaders> readers_;
  uint32_t writeLimit_;
  std::array<uint32_t, 3> classWrites_{};
  // Every write whose IO the worker may still run.
  std::vector<Write> writes_;
  // Restores asked for and not yet submitted, and those submitted.
  std::vector<Restore> queued_;
  std::vector<RestoreBatch> batches_;
  uint32_t restores_ = 0;
};

} // namespace splash::engine
