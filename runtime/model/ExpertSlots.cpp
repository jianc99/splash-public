#include "model/ExpertSlots.hpp"

#include <stdexcept>

namespace splash::model {
namespace {

constexpr uint32_t kNone = UINT32_MAX;

} // namespace

ExpertSlots::ExpertSlots(uint32_t slots)
    : previous_(slots), next_(slots), order_(slots), lastLoad_(slots) {
  reset();
}

void ExpertSlots::unlink(uint32_t slot) {
  const uint8_t o = order_[slot];
  (previous_[slot] == kNone ? head_[o] : next_[previous_[slot]]) = next_[slot];
  (next_[slot] == kNone ? tail_[o] : previous_[next_[slot]]) = previous_[slot];
}

void ExpertSlots::append(uint32_t slot, Order o) {
  order_[slot] = o;
  previous_[slot] = tail_[o];
  next_[slot] = kNone;
  (tail_[o] == kNone ? head_[o] : next_[tail_[o]]) = slot;
  tail_[o] = slot;
}

void ExpertSlots::prepend(uint32_t slot, Order o) {
  order_[slot] = o;
  next_[slot] = head_[o];
  previous_[slot] = kNone;
  (head_[o] == kNone ? tail_[o] : previous_[head_[o]]) = slot;
  head_[o] = slot;
}

void ExpertSlots::touch(uint32_t slot, uint64_t load, Order o) {
  lastLoad_[slot] = load;
  unlink(slot);
  append(slot, o);
}

void ExpertSlots::touch(uint32_t slot, uint64_t load, bool chunk) {
  touch(slot, load, chunk ? Order(order_[slot]) : Recent);
}

uint32_t ExpertSlots::claim(uint64_t load, bool chunk) {
  // The slots a load routes are at their order's most recent end: an order
  // whose head the load routed holds only the load's.
  const auto takes = [&](uint32_t slot) { return slot != kNone && lastLoad_[slot] != load; };
  const uint32_t slot = takes(head_[Chunk]) ? head_[Chunk] : head_[Recent];
  if (!takes(slot)) throw std::runtime_error("the expert arena holds fewer slots than one load's experts");
  touch(slot, load, chunk ? Chunk : Recent);
  return slot;
}

void ExpertSlots::free(uint32_t slot) {
  lastLoad_[slot] = 0;
  unlink(slot);
  prepend(slot, Chunk);
}

void ExpertSlots::reset() {
  head_ = tail_ = {kNone, kNone};
  std::fill(lastLoad_.begin(), lastLoad_.end(), 0);
  for (uint32_t slot = 0; slot < size(); ++slot) append(slot, Chunk);
}

} // namespace splash::model
