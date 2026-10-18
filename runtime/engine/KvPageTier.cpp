#include "engine/KvPageTier.hpp"

#include <algorithm>
#include <cstddef>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace splash::engine {

class KvPageTier::Ticket final : public KvTransfer {
public:
  explicit Ticket(std::shared_ptr<Outcome> outcome) : outcome_(std::move(outcome)) {}
  bool ready() const noexcept override { return outcome_->ready; }
  bool finish() override { return outcome_->success; }

private:
  std::shared_ptr<Outcome> outcome_;
};

namespace {

std::shared_ptr<KvPageTier::DiskSlot> diskSlot(const std::shared_ptr<KvDiskSlot> &slot) {
  auto disk = std::dynamic_pointer_cast<KvPageTier::DiskSlot>(slot);
  if (!disk) throw std::invalid_argument("KV disk slot belongs to another tier");
  return disk;
}

// A reader's memory for what it reads, aligned for uncached IO and kept for
// the reader's next task: at most a slot each.
std::span<std::byte> readBuffer(uint64_t bytes) {
  constexpr std::align_val_t alignment{kv::SlotFormat::kReadAlignmentBytes};
  struct Free {
    void operator()(std::byte *memory) const noexcept { ::operator delete(memory, alignment); }
  };
  thread_local std::unique_ptr<std::byte, Free> memory;
  thread_local uint64_t capacity = 0;
  if (capacity < bytes) {
    memory.reset(static_cast<std::byte *>(::operator new(bytes, alignment)));
    capacity = bytes;
  }
  return {memory.get(), capacity};
}

} // namespace

KvPageTier::KvPageTier(kv::PageStorage &pages, kv::SlotFormat format,
                       std::shared_ptr<model::SlotFile> file,
                       std::shared_ptr<model::SsdReaders> readers, uint32_t writes)
    : pages_(pages), format_(std::move(format)), file_(std::move(file)),
      readers_(std::move(readers)), writeLimit_(writes) {
  if (!file_ || !readers_ || !writeLimit_)
    throw std::invalid_argument("KV tier needs a slot file, readers and room for a write");
  if (format_.layout() != pages_.layout())
    throw std::invalid_argument("KV tier's slot format is another layout's");
  // Tracking a submitted IO must not allocate: the tier drains every IO it
  // tracks before it goes, and an IO it lost would outlive the page.
  writes_.reserve(writeLimit_);
}

// A transfer in flight reads or writes its page in place, and the cache that
// held the page for it is gone before the tier is: the worker and the
// readers must be done with every page first. Nothing will read the slots a
// cancelled write leaves behind, nor the pages an unread restore leaves,
// because the tier that maps pages to them is going away too.
KvPageTier::~KvPageTier() {
  for (const Write &write : writes_)
    write.io->drain();
  for (RestoreBatch &batch : batches_)
    readers_->stop(*batch.batch);
}

std::string_view KvPageTier::slotFormatToken(const kv::Layout &layout) noexcept {
  // kv::SlotFormat's version 1, the block records of a spillable layout.
  return layout.spillable() ? " slots block-records-v1" : "";
}

uint64_t KvPageTier::slotBytes() const noexcept { return file_->slotBytes(); }

bool KvPageTier::writable() const noexcept { return file_->writable(); }

bool KvPageTier::persistent() const noexcept { return file_->persistent(); }

void KvPageTier::label(const std::shared_ptr<KvDiskSlot> &slot, std::vector<std::byte> label) {
  file_->label(diskSlot(slot)->slot, std::move(label));
}

std::shared_ptr<KvDiskSlot> KvPageTier::adopt(const model::SlotRecord &record) {
  return std::make_shared<DiskSlot>(file_->adopt(record), file_);
}

uint32_t KvPageTier::share(WriteClass write) const noexcept {
  switch (write) {
  case WriteClass::Reclaim: return writeLimit_;
  case WriteClass::Persist: return std::max(1u, writeLimit_ / 4);
  case WriteClass::Spill: return std::max(1u, writeLimit_ / 2);
  }
  return 0;
}

bool KvPageTier::canDemote(WriteClass write) const noexcept {
  return writes_.size() < writeLimit_ && classWrites_[static_cast<size_t>(write)] < share(write) &&
         writable();
}

bool KvPageTier::canRestore() const noexcept { return restores_ < kRestores; }

std::shared_ptr<KvDiskSlot> KvPageTier::acquireSlot(model::DiskBudget::Draw draw) {
  auto slot = file_->acquire(draw);
  if (!slot) return {};
  return std::make_shared<DiskSlot>(std::move(slot), file_);
}

std::unique_ptr<KvTransfer> KvPageTier::demote(uint32_t page, std::shared_ptr<KvDiskSlot> slot,
                                               WriteClass write, std::function<void()> completion) {
  if (!canDemote(write))
    return {};
  if (pages_.form(page) != kv::PageForm::Full)
    throw std::logic_error("only a page that holds its K/V is written to the KV tier");
  auto disk = diskSlot(slot);
  auto outcome = std::make_shared<Outcome>();
  auto ticket = std::make_unique<Ticket>(outcome);
  auto io = file_->write(disk->slot, format_.transferSpans(pages_, page), std::move(completion));
  // A file that stopped taking writes since canDemote() refuses this one as
  // canDemote() does from now on.
  if (!io)
    return {};
  ++classWrites_[static_cast<size_t>(write)];
  writes_.push_back({write, std::move(io), std::move(outcome)});
  return ticket;
}

std::unique_ptr<KvTransfer> KvPageTier::restore(std::shared_ptr<KvDiskSlot> slot, uint32_t page,
                                                std::function<void()> completion) {
  if (!canRestore())
    return {};
  Restore restore{diskSlot(slot), page, pages_.form(page) == kv::PageForm::Index,
                  std::move(completion), std::make_shared<Outcome>()};
  auto ticket = std::make_unique<Ticket>(restore.outcome);
  queued_.push_back(std::move(restore));
  ++restores_;
  return ticket;
}

void KvPageTier::submitRestores() {
  if (queued_.empty())
    return;
  // A batch that does not start leaves its restores queued for the next
  // call; one that starts is tracked without another allocation.
  batches_.reserve(batches_.size() + 1);
  auto restores = std::make_shared<std::vector<Restore>>();
  RestoreBatch batch{restores, nullptr, std::vector<bool>(queued_.size()), queued_.size()};
  restores->swap(queued_);
  try {
    // A failed read throws, and the readers try it once more. The batch
    // wakes the engine once every restore in it has ended.
    batch.batch = readers_->start(
        restores->size(), [this, restores](size_t index) { read((*restores)[index]); },
        readers_->readers(), [restores](const model::SsdReaders::Batch &) {
          if (const std::function<void()> &completion = restores->front().completion)
            completion();
        });
  } catch (...) {
    restores->swap(queued_);
    throw;
  }
  batches_.push_back(std::move(batch));
}

void KvPageTier::read(const Restore &restore) const {
  // The page's extent stays while its restore is in flight, and its memory
  // is found beside the engine thread's changes to other extents.
  const kv::SlotRange range =
      restore.index ? format_.indexRange() : kv::SlotRange{0, format_.payloadBytes()};
  const std::optional<std::span<const std::byte>> bytes =
      restore.copy->readAt(range.offset, range.bytes, readBuffer(file_->slotBytes()));
  if (!bytes)
    throw std::runtime_error("a KV tier slot could not be read");
  if (restore.index)
    format_.scatterIndex(*bytes, pages_.indexSpans(restore.page));
  else
    format_.scatterPage(*bytes, pages_, restore.page);
}

std::shared_ptr<const model::KvCopy> KvPageTier::copy(const std::shared_ptr<KvDiskSlot> &slot) const {
  return diskSlot(slot);
}

void KvPageTier::poll() {
  for (size_t index = 0; index < writes_.size();) {
    Write &write = writes_[index];
    if (!write.io->ready()) {
      ++index;
      continue;
    }
    --classWrites_[static_cast<size_t>(write.write)];
    write.outcome->success = write.io->wait();
    write.outcome->ready = true;
    writes_.erase(writes_.begin() + static_cast<std::ptrdiff_t>(index));
  }
  // A restore that ended lets its copy go: the cache may give the copy up
  // once the restore has landed.
  for (auto batch = batches_.begin(); batch != batches_.end();) {
    for (size_t index = 0; index < batch->retired.size(); ++index) {
      if (batch->retired[index] || !batch->batch->ended(index))
        continue;
      Restore &restore = (*batch->restores)[index];
      restore.outcome->success = !batch->batch->failed(index);
      restore.outcome->ready = true;
      restore.copy.reset();
      batch->retired[index] = true;
      --batch->left;
      --restores_;
    }
    batch = batch->left ? batch + 1 : batches_.erase(batch);
  }
}

} // namespace splash::engine
