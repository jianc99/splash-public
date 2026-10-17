#pragma once

// The order the slots of an expert arena are recycled in (model::
// ExpertStore): which slot a load's missing expert takes, never one of an
// expert the same load routes. A load is a number that grows from load to
// load; a long prefill chunk's (`chunk`) routes nearly every expert of its
// block.
//
// Two recency orders, each from the least recently used slot to the most:
// Chunk, the experts a long chunk loaded that no other load routed since,
// which are recycled first, and the free slots; Recent, the others. A long
// chunk routes more experts than the arena holds (nearly every expert of
// every block): its experts join Chunk, and those it routes keep their
// order, so the experts decode steps routed stay, and each chunk of a prompt
// reads what the last left out, where a single order would have each chunk
// evict the blocks the next one reads first and the decode steps after the
// prompt read their experts again. A short chunk's experts, which its decode
// steps route again, are recent as a decode step's.
//
// The store's own thread uses it alone (no locking).

#include <array>
#include <cstdint>
#include <vector>

namespace splash::model {

class ExpertSlots final {
public:
  // `slots` slots, every one free.
  explicit ExpertSlots(uint32_t slots);

  // Load `load` routes the expert in `slot`.
  void touch(uint32_t slot, uint64_t load, bool chunk);
  // The slot load `load`'s missing expert takes: free, or the one whose
  // expert to evict; then it is the load's. Throws std::runtime_error when
  // every slot holds an expert the load routes.
  [[nodiscard]] uint32_t claim(uint64_t load, bool chunk);
  // `slot` holds no expert (a failed load claimed it): the first claimed.
  void free(uint32_t slot);
  // Every slot free.
  void reset();

  [[nodiscard]] uint32_t size() const noexcept { return static_cast<uint32_t>(order_.size()); }

private:
  enum Order : uint8_t { Chunk, Recent };
  void unlink(uint32_t slot);
  void append(uint32_t slot, Order order);
  void prepend(uint32_t slot, Order order);
  void touch(uint32_t slot, uint64_t load, Order order);

  std::vector<uint32_t> previous_, next_;
  std::vector<uint8_t> order_;
  std::array<uint32_t, 2> head_{}, tail_{};
  // The load that last routed each slot's expert, 0 for none.
  std::vector<uint64_t> lastLoad_;
};

} // namespace splash::model
