// The order an expert arena's slots are recycled in (model::ExpertSlots): a
// load never takes a slot of an expert it routes, and takes none when every
// slot is its own; free slots are taken first, a freed one before them; the
// experts a long prefill chunk loaded go before those other loads routed,
// each order least recently used first, and a long chunk's routes keep their
// order.
#include "TestChecks.hpp"
#include "model/ExpertSlots.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

using splash::model::ExpertSlots;
using splash::test::rejects;
using splash::test::require;

namespace {

void neverTakesTheLoadsOwn() {
  ExpertSlots slots(8);
  std::set<uint32_t> taken;
  for (int expert = 0; expert < 8; ++expert) taken.insert(slots.claim(1, false));
  require(taken.size() == 8, "one load took a slot twice");
  rejects([&] { static_cast<void>(slots.claim(1, false)); }, "fewer slots", "a load took a slot of its own");
  // The next load takes the least recently used slot it does not route.
  for (uint32_t slot = 0; slot < 4; ++slot) slots.touch(slot, 2, false);
  const uint32_t slot = slots.claim(2, false);
  require(slot >= 4, "a load took a slot of an expert it routes");
}

void freeSlotsFirst() {
  ExpertSlots slots(4);
  for (uint64_t load = 1; load <= 4; ++load) static_cast<void>(slots.claim(load, false));
  slots.free(2);
  require(slots.claim(5, false) == 2, "a freed slot was not taken first");
  slots.reset();
  std::set<uint32_t> taken;
  for (int expert = 0; expert < 4; ++expert) taken.insert(slots.claim(6, false));
  require(taken.size() == 4, "a reset left a slot held");
}

void chunkLoadsGoFirst() {
  ExpertSlots slots(6);
  // Load 1, a decode step, takes slots for three experts; load 2, a long
  // chunk, three more.
  std::vector<uint32_t> decode, chunk;
  for (int expert = 0; expert < 3; ++expert) decode.push_back(slots.claim(1, false));
  for (int expert = 0; expert < 3; ++expert) chunk.push_back(slots.claim(2, true));
  // Another chunk routes the first chunk expert again (it keeps its order,
  // now the most recent of Chunk) and takes two slots: the chunk's others.
  slots.touch(chunk[0], 3, true);
  const uint32_t first = slots.claim(3, true), second = slots.claim(3, true);
  require(std::set<uint32_t>{first, second} == std::set<uint32_t>{chunk[1], chunk[2]},
          "a long chunk's load took a decode step's slot before another chunk's");
  // A decode step routes the first chunk expert: it is recent now, and the
  // next miss takes the least recent chunk slot, then the decode step's.
  slots.touch(chunk[0], 4, false);
  require(slots.claim(5, false) == first && slots.claim(5, false) == second,
          "a chunk slot routed by a decode step was recycled before the chunk's");
  require(slots.claim(5, false) == decode[0], "a decode step's slots were not recycled least recently used first");
}

} // namespace

int main() {
  try {
    neverTakesTheLoadsOwn();
    freeSlotsFirst();
    chunkLoadsGoFirst();
  } catch (const std::exception &error) {
    std::cerr << "expert_slots_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "expert_slots_test: PASS\n";
  return 0;
}
