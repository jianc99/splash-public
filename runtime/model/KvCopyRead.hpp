#pragma once

// One read of a tier copy's bytes (model::KvCopy::readAt) on the calling
// thread, as the reads of spilled K/V make them on the SSD readers: into an
// aligned buffer of the thread's own, which its next read reuses. A task
// whose read failed throws, and the readers try it once more (SsdReaders).

#include "model/Model.hpp"
#include "ops/KvSlotFormat.hpp"

#include <cstddef>
#include <optional>
#include <span>

namespace splash::model {

// The bytes of `range` of the copy, valid until the thread's next read, or
// nothing when the read failed.
[[nodiscard]] std::optional<std::span<const std::byte>> readKvCopy(const KvCopy &copy, kv::SlotRange range);

} // namespace splash::model
