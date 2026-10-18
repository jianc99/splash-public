// The order a store's slots are recycled in (model::SlotRecycler): a load
// never takes a slot of a key it uses, and takes none when every slot is its
// own; free slots are taken first, a freed one before them; a slot's key is
// found until it is recycled, freed or resized away; a long prefill chunk's
// scan of more experts than the slots leaves the experts decode steps routed
// again in place; and when more lanes add experts to each cycle until a cycle
// routes more than the slots hold, the experts every cycle routes keep
// hitting (recycling the least recently used slot keeps a fifth of them). A
// claim allocates nothing.
#include "TestChecks.hpp"
#include "model/SlotRecycler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Every allocation of the test's thread, which a claim must not add to.
thread_local uint64_t allocations = 0;
void *operator new(size_t size) {
  ++allocations;
  if (void *memory = std::malloc(size ? size : 1)) return memory;
  throw std::bad_alloc();
}
void operator delete(void *memory) noexcept { std::free(memory); }

using splash::model::SlotRecycler;
using splash::test::rejects;
using splash::test::require;

namespace {

// The slots of a store's experts, as the store keeps them: a load routes
// experts, those resident are touched, the others claim slots.
struct Arena final {
  explicit Arena(uint32_t slots) : order(slots), owner(slots, kFree) {}
  static constexpr uint64_t kFree = UINT64_MAX;
  SlotRecycler order;
  std::map<uint64_t, uint32_t> slotOf;
  std::vector<uint64_t> owner;
  uint64_t load = 0;
  // Routes `experts` in one load; returns how many were resident.
  uint32_t route(const std::vector<uint64_t> &experts, bool chunk) {
    ++load;
    uint32_t hits = 0;
    std::vector<uint64_t> missing;
    for (const uint64_t expert : experts)
      if (const auto found = slotOf.find(expert); found != slotOf.end()) {
        order.touch(found->second, load, chunk);
        ++hits;
      } else {
        missing.push_back(expert);
      }
    for (const uint64_t expert : missing) {
      const uint32_t slot = order.claim(expert, load, chunk);
      if (owner[slot] != kFree) slotOf.erase(owner[slot]);
      owner[slot] = expert;
      slotOf[expert] = slot;
    }
    return hits;
  }
};

void neverTakesTheLoadsOwn() {
  SlotRecycler slots(8);
  std::set<uint32_t> taken;
  for (uint64_t expert = 0; expert < 8; ++expert) taken.insert(slots.claim(expert, 1, false));
  require(taken.size() == 8, "one load took a slot twice");
  rejects([&] { static_cast<void>(slots.claim(8, 1, false)); }, "fewer slots", "a load took a slot of its own");
  // The next load takes a slot of an expert it does not route.
  for (uint32_t slot = 0; slot < 8; ++slot)
    if (slot % 2) slots.touch(slot, 2, false);
  for (uint32_t slot = 0; slot < 8; ++slot)
    require(slots.pinned(slot, 2) == (slot % 2 == 1), "a touch did not pin the slot to its load");
  for (uint64_t expert = 9; expert < 13; ++expert)
    require(slots.claim(expert, 2, false) % 2 == 0, "a load took a slot of an expert it routes");
  rejects([&] { static_cast<void>(slots.claim(13, 2, false)); }, "fewer slots", "a load took a slot of its own");
}

void freeSlotsFirst() {
  SlotRecycler slots(4);
  for (uint64_t load = 1; load <= 4; ++load) static_cast<void>(slots.claim(load, load, false));
  slots.free(2);
  require(slots.claim(5, 5, false) == 2, "a freed slot was not taken first");
  slots.reset();
  std::set<uint32_t> taken;
  for (uint64_t expert = 0; expert < 4; ++expert) taken.insert(slots.claim(expert, 6, false));
  require(taken.size() == 4, "a reset left a slot held");
}

void findsWhatItHolds() {
  SlotRecycler slots(2);
  const uint32_t first = slots.claim(10, 1, false);
  slots.touch(first, 2, false);
  const uint32_t second = slots.claim(11, 3, false);
  require(slots.find(10) == first && slots.find(11) == second && !slots.find(12), "a held key was not found");
  // T1's key goes for a third, as a ghost, whose claim grows T1 and takes
  // T2's slot.
  require(slots.claim(12, 4, false) == second && !slots.find(11) && slots.find(12) == second,
          "a recycled slot's key was still found");
  require(slots.claim(11, 5, false) == first && !slots.find(10) && slots.ghostHits() == 1,
          "a ghost's claim was not counted");
  slots.free(first);
  require(!slots.find(11), "a freed slot's key was still found");
  // A scan's claim of T2's ghost adapts nothing.
  static_cast<void>(slots.claim(10, 6, true));
  require(slots.ghostHits() == 1, "a scan's claim of a ghost counted as a ghost's");
}

void resizes() {
  SlotRecycler slots(4);
  for (uint64_t key = 0; key < 4; ++key) static_cast<void>(slots.claim(key, 1, false));
  // More slots: the load that holds every slot takes the new ones, and the
  // old keys stay.
  slots.resize(8);
  std::set<uint32_t> added;
  for (uint64_t key = 4; key < 8; ++key) added.insert(slots.claim(key, 1, false));
  require(added == std::set<uint32_t>{4, 5, 6, 7}, "a grown recycler did not take its new slots");
  for (uint64_t key = 0; key < 8; ++key) require(slots.find(key).has_value(), "growing lost a key");
  // Ghosts survive growth: with keys 0-3 frequent, 8 and 9 recycle 4 and 5,
  // and 4 comes back as a ghost.
  for (uint64_t key = 0; key < 4; ++key) slots.touch(*slots.find(key), 2, false);
  static_cast<void>(slots.claim(8, 3, false));
  static_cast<void>(slots.claim(9, 4, false));
  require(!slots.find(4) && !slots.find(5), "T1's oldest keys were not recycled");
  slots.resize(12);
  static_cast<void>(slots.claim(4, 5, false));
  require(slots.ghostHits() == 1, "growing forgot a ghost");
  // Fewer slots: the keys of the slots past them go, with every ghost.
  std::vector<uint64_t> kept, dropped;
  for (uint64_t key = 0; key < 10; ++key)
    if (const auto slot = slots.find(key)) (*slot < 5 ? kept : dropped).push_back(key);
  slots.resize(5);
  require(slots.size() == 5 && !dropped.empty(), "a shrunk recycler kept its slots");
  for (const uint64_t key : kept) require(slots.find(key).has_value(), "shrinking lost a key it kept");
  for (const uint64_t key : dropped) require(!slots.find(key), "shrinking kept a key past its slots");
  static_cast<void>(slots.claim(5, 6, false));
  require(slots.ghostHits() == 1, "shrinking kept a ghost");
  for (uint64_t load = 7; load < 40; ++load)
    require(slots.claim(100 + load, load, false) < 5, "a shrunk recycler took a slot past its size");
}

void aScanSparesTheFrequent() {
  Arena arena(6);
  const std::vector<uint64_t> frequent{0, 1, 2};
  static_cast<void>(arena.route(frequent, false));
  require(arena.route(frequent, false) == 3, "a decode step's experts were not resident for the next");
  // A prompt's chunks route forty experts, two a load.
  for (uint64_t expert = 100; expert < 140; expert += 2) static_cast<void>(arena.route({expert, expert + 1}, true));
  require(arena.route(frequent, false) == 3, "a long chunk's scan evicted experts decode steps route again");
}

void aRoutedCoreOutlivesMoreLanes() {
  // Six layers, each routing its ten core experts every cycle besides fresh
  // ones: two a cycle (one lane), then eight (more lanes), when a cycle
  // routes 108 experts and the arena holds 100.
  Arena arena(100);
  uint64_t fresh = 1000;
  double rate = 0;
  for (const uint32_t extra : {2u, 8u}) {
    uint64_t hits = 0, routes = 0;
    for (int cycle = 0; cycle < 30; ++cycle)
      for (uint64_t layer = 0; layer < 6; ++layer) {
        std::vector<uint64_t> experts;
        for (uint64_t core = 0; core < 10; ++core) experts.push_back(layer << 32 | core);
        for (uint32_t other = 0; other < extra; ++other) experts.push_back(layer << 32 | fresh++);
        const uint32_t resident = arena.route(experts, false);
        if (cycle >= 5) {
          hits += resident;
          routes += 10;
        }
      }
    rate = double(hits) / double(routes);
    require(rate > 0.95, "the experts every cycle routes stopped hitting: " + std::to_string(rate));
  }
}

void claimsAllocateNothing() {
  // Loads of 32 keys used again and again and 16 of 56 others in turn: the
  // 64 slots recycle the others, which come back as ghosts; every seventh
  // load is a scan, and every fiftieth frees a slot.
  std::vector<std::vector<uint64_t>> loads(400);
  for (size_t load = 0; load < loads.size(); ++load) {
    for (uint64_t key = 0; key < 32; ++key) loads[load].push_back(key);
    for (uint64_t other = 0; other < 16; ++other) loads[load].push_back(1000 + (load * 16 + other) % 56);
  }
  SlotRecycler slots(64);
  uint64_t before = 0;
  for (uint64_t load = 1; load <= loads.size(); ++load) {
    if (load == 100) before = allocations;
    const std::vector<uint64_t> &keys = loads[load - 1];
    const bool chunk = load % 7 == 0;
    for (const uint64_t key : keys)
      if (const auto slot = slots.find(key)) slots.touch(*slot, load, chunk);
      else static_cast<void>(slots.claim(key, load, chunk));
    if (load % 50 == 0) slots.free(*slots.find(keys.front()));
  }
  require(allocations == before, "a claim allocated: " + std::to_string(allocations - before) + " allocations");
  require(slots.ghostHits() > 0, "the loads hit no ghost");
}

} // namespace

int main() {
  try {
    neverTakesTheLoadsOwn();
    freeSlotsFirst();
    findsWhatItHolds();
    resizes();
    aScanSparesTheFrequent();
    aRoutedCoreOutlivesMoreLanes();
    claimsAllocateNothing();
  } catch (const std::exception &error) {
    std::cerr << "slot_recycler_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "slot_recycler_test: PASS\n";
  return 0;
}
