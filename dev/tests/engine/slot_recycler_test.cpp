// The order a store's slots are recycled in (model::SlotRecycler): a load
// never takes a slot of a key it uses, and takes none when every slot is its
// own; free slots are taken first, a freed one before them; a slot's key is
// found until it is recycled, freed or resized away; a long prefill chunk's
// scan of more experts than the slots leaves the experts decode steps routed
// again in place; and when more lanes add experts to each cycle until a cycle
// routes more than the slots hold, the experts every cycle routes keep
// hitting (recycling the least recently used slot keeps a fifth of them). Of
// the list adaptive replacement recycles from, the slot that goes is the one
// whose key missed the most loads of its position of the cycle, of those the
// one at the position the loads reach last, as a reference model of the
// policy over lists of keys recycles them, load by load, with one position
// (adaptive replacement alone) and with six; and loads that loop over more
// keys than the slots keep hitting, where one position hits none. A claim
// allocates nothing. On the decode selections of real requests
// (dev/tests/fixtures/kv-selection-traces, the directory the test takes), the
// hot block cache's resolves miss at most 2% more than recycling the least
// recently used block would.
#include "TestChecks.hpp"
#include "model/SlotRecycler.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <new>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

// The policy as SlotRecycler's description reads, over lists of keys: T1 and
// T2, a least-recently-used order of keys at each position (a scan's first),
// and B1 and B2, the keys each gave up lately; each key counts the loads of
// its position it missed since its last use (a scan's, all of them). A claim
// recycles, of the list adaptive replacement picks, the key the load does
// not use that missed the most, of those the first from the load's own
// position back around the cycle, and the least recently used of a position.
class Reference final {
public:
  Reference(uint32_t slots, uint32_t cycle) : slots_(slots), lists_{Lists(cycle), Lists(cycle)}, loads_(cycle) {}

  [[nodiscard]] bool holds(uint64_t key) const { return held_.contains(key); }
  [[nodiscard]] uint64_t ghostHits() const { return ghostHits_; }

  void touch(uint64_t key, uint64_t load, bool chunk) {
    Held &held = held_.at(key);
    count(load, held.position);
    held.load = load;
    if (chunk) return;
    held.visit = loads_[held.position];
    lists_[held.list][held.position].remove(key);
    held.list = kFrequent;
    held.scan = false;
    lists_[kFrequent][held.position].push_back(key);
  }

  void claim(uint64_t key, uint64_t load, bool chunk, uint32_t position) {
    count(load, position);
    const bool recentGhost = gaveUp(kRecent, key), frequentGhost = gaveUp(kFrequent, key);
    if (!chunk && (recentGhost || frequentGhost)) {
      const size_t mine = ghosts_[frequentGhost].size(), other = ghosts_[!frequentGhost].size();
      const uint64_t step = std::max<uint64_t>(1, other / mine);
      target_ = recentGhost ? std::min<uint64_t>(slots_, target_ + step) : target_ > step ? target_ - step : 0;
      ++ghostHits_;
      ghosts_[frequentGhost].remove(key);
      if (held_.size() == slots_) replace(position, load, frequentGhost);
      hold(key, kFrequent, position, load, false);
      return;
    }
    ghosts_[kRecent].remove(key);
    ghosts_[kFrequent].remove(key);
    if (held_.size() == slots_) {
      const size_t recent = count(kRecent) + ghosts_[kRecent].size();
      if (recent >= slots_ && (count(kRecent) == slots_ || ghosts_[kRecent].empty())) {
        evict(victim(kRecent, position, load), false);
      } else {
        if (recent >= slots_) ghosts_[kRecent].pop_front();
        else if (recent + count(kFrequent) + ghosts_[kFrequent].size() >= 2 * slots_ && !ghosts_[kFrequent].empty())
          ghosts_[kFrequent].pop_front();
        replace(position, load, false);
      }
    }
    hold(key, kRecent, position, load, chunk);
  }

  void free(uint64_t key) {
    const Held &held = held_.at(key);
    lists_[held.list][held.position].remove(key);
    held_.erase(key);
  }
  void grow(uint32_t slots) { slots_ = slots; }
  void reset() {
    held_.clear();
    for (Lists &list : lists_)
      for (std::list<uint64_t> &at : list) at.clear();
    for (std::list<uint64_t> &ghosts : ghosts_) ghosts.clear();
    target_ = 0;
    std::ranges::fill(loads_, 0);
    last_ = 0;
  }

private:
  static constexpr uint64_t kNoKey = UINT64_MAX;
  static constexpr int kRecent = 0, kFrequent = 1;
  using Lists = std::vector<std::list<uint64_t>>;
  struct Held final {
    int list = kRecent;
    uint32_t position = 0;
    uint64_t load = 0;
    // Its position's loads at its last use, 0 for a scan's.
    uint64_t visit = 0;
    bool scan = false;
  };

  // A load (each at one position) counts once.
  void count(uint64_t load, uint32_t position) {
    if (load == last_) return;
    last_ = load;
    ++loads_[position];
  }

  [[nodiscard]] bool gaveUp(int list, uint64_t key) const {
    return std::ranges::find(ghosts_[list], key) != ghosts_[list].end();
  }
  [[nodiscard]] size_t count(int list) const {
    size_t keys = 0;
    for (const std::list<uint64_t> &at : lists_[list]) keys += at.size();
    return keys;
  }
  [[nodiscard]] uint64_t victim(int list, uint32_t position, uint64_t load) const {
    const uint32_t cycle = static_cast<uint32_t>(lists_[list].size());
    uint64_t victim = kNoKey, missed = 0;
    for (uint32_t back = 0; back < cycle; ++back) {
      const uint32_t place = (position + cycle - back) % cycle;
      for (const uint64_t key : lists_[list][place]) {
        const Held &held = held_.at(key);
        if (held.load == load) continue;
        if (const uint64_t loads = loads_[place] - held.visit; victim == kNoKey || loads > missed) {
          victim = key;
          missed = loads;
        }
      }
    }
    return victim;
  }
  void replace(uint32_t position, uint64_t load, bool frequentGhost) {
    const bool fromRecent =
        count(kRecent) && (count(kRecent) > target_ || (frequentGhost && count(kRecent) == target_));
    uint64_t key = victim(fromRecent ? kRecent : kFrequent, position, load);
    if (key == kNoKey) key = victim(fromRecent ? kFrequent : kRecent, position, load);
    evict(key, true);
  }
  void evict(uint64_t key, bool ghost) {
    require(key != kNoKey, "the reference found no key to recycle");
    const Held held = held_.at(key);
    lists_[held.list][held.position].remove(key);
    if (ghost && !held.scan) ghosts_[held.list].push_back(key);
    held_.erase(key);
  }
  // A scan's key enters T1 at its least recently used end.
  void hold(uint64_t key, int list, uint32_t position, uint64_t load, bool scan) {
    held_[key] = {list, position, load, scan ? 0 : loads_[position], scan};
    if (scan) lists_[list][position].push_front(key);
    else lists_[list][position].push_back(key);
  }

  uint64_t slots_;
  std::array<Lists, 2> lists_;
  std::array<std::list<uint64_t>, 2> ghosts_;
  std::map<uint64_t, Held> held_;
  uint64_t target_ = 0, ghostHits_ = 0;
  // Each position's loads, and the last load counted.
  std::vector<uint64_t> loads_;
  uint64_t last_ = 0;
};

// Loads of six positions' keys, skewed to each position's first: mostly
// short loads in the cycle's order, a long chunk's scans every fifth cycle,
// now and then a load at another position, a slot a load claimed freed (its
// read failed), more slots once and a reset once; after every load the
// recycler holds the keys the reference holds, and it never took a slot of a
// key the load uses.
void followsItsReference(uint32_t cycle) {
  constexpr uint32_t kPositions = 6, kKeys = 24, kLoads = 6000;
  SlotRecycler slots(40, cycle);
  Reference reference(40, cycle);
  std::mt19937 random(cycle);
  std::vector<uint64_t> owner(56, Arena::kFree);
  std::map<uint64_t, uint32_t> slotOf;
  const std::string where = "cycle " + std::to_string(cycle) + ": ";
  for (uint64_t load = 1; load <= kLoads; ++load) {
    if (load == kLoads / 3) {
      slots.resize(56);
      reference.grow(56);
    }
    if (load == 2 * kLoads / 3) {
      slots.reset();
      reference.reset();
      slotOf.clear();
      std::ranges::fill(owner, Arena::kFree);
    }
    const uint32_t at = load % 11 == 0 ? random() % kPositions : load % kPositions;
    const bool chunk = load / kPositions % 5 == 0;
    std::set<uint64_t> keys;
    for (uint32_t route = 0, routes = chunk ? 16 + random() % 8 : 2 + random() % 9; route < routes; ++route)
      keys.insert(uint64_t{at} << 32 | std::min(random() % kKeys, random() % kKeys));
    const uint32_t position = at % cycle;
    std::vector<uint64_t> missing;
    for (const uint64_t key : keys)
      if (const auto found = slotOf.find(key); found != slotOf.end()) {
        slots.touch(found->second, load, chunk);
        reference.touch(key, load, chunk);
      } else {
        missing.push_back(key);
      }
    for (const uint64_t key : missing) {
      const uint32_t slot = slots.claim(key, load, chunk, position);
      reference.claim(key, load, chunk, position);
      require(owner[slot] == Arena::kFree || !keys.contains(owner[slot]), where + "a load took a slot of its own");
      if (owner[slot] != Arena::kFree) slotOf.erase(owner[slot]);
      owner[slot] = key;
      slotOf[key] = slot;
    }
    if (load % 37 == 0 && !missing.empty()) {
      const uint64_t failed = missing.back();
      slots.free(slotOf.at(failed));
      reference.free(failed);
      owner[slotOf.at(failed)] = Arena::kFree;
      slotOf.erase(failed);
    }
    for (uint64_t key = 0; key < kKeys; ++key)
      for (uint64_t place = 0; place < kPositions; ++place) {
        const uint64_t held = place << 32 | key;
        require(slots.find(held).has_value() == reference.holds(held),
                where + "load " + std::to_string(load) + " left key " + std::to_string(key) + " of position " +
                    std::to_string(place) + (reference.holds(held) ? " out, which the reference holds"
                                                                    : " in, which the reference recycled"));
      }
    require(slots.ghostHits() == reference.ghostHits(), where + "the ghosts' claims differ from the reference's");
  }
  require(slots.ghostHits() > 0, where + "the loads hit no ghost");
}

void positionsOfTheCycle() {
  rejects([] { static_cast<void>(SlotRecycler(4, 0)); }, "no position", "a cycle of no position was taken");
  SlotRecycler slots(4, 3);
  rejects([&] { static_cast<void>(slots.claim(1, 1, false, 3)); }, "past", "a position past the cycle was taken");
  // Keys at positions 0, 1 and 2 fill T1, and a load at 2 claims three more:
  // the keys of its own position it does not use go first (each missed a
  // load of it), then of the others, which missed none, that of the position
  // before it (1), not the least recently used (0).
  static_cast<void>(slots.claim(10, 1, false, 0));
  static_cast<void>(slots.claim(11, 2, false, 1));
  static_cast<void>(slots.claim(20, 3, false, 2));
  static_cast<void>(slots.claim(21, 4, false, 2));
  static_cast<void>(slots.claim(22, 5, false, 2));
  require(!slots.find(20) && slots.find(10) && slots.find(11), "a load recycled a key of another position first");
  static_cast<void>(slots.claim(23, 5, false, 2));
  require(!slots.find(21) && slots.find(10) && slots.find(11), "a load recycled a key of another position first");
  static_cast<void>(slots.claim(24, 5, false, 2));
  require(!slots.find(11) && slots.find(10), "a load passed over the position before its own");
  // A key that missed a load of its position goes before one that missed
  // none, though the loads reach it sooner: X at 2 missed the load at 2 that
  // claimed W, Q at 0 missed nothing, and a long chunk at 1 uses Z and claims
  // V.
  SlotRecycler cycle(4, 3);
  const uint32_t x = cycle.claim(30, 1, false, 2);
  static_cast<void>(cycle.claim(31, 2, false, 0));
  const uint32_t z = cycle.claim(32, 3, false, 1);
  static_cast<void>(cycle.claim(33, 4, false, 2));
  cycle.touch(z, 5, true);
  require(cycle.claim(34, 5, true, 1) == x && cycle.find(31), "a load gave up a key that missed no load of its "
                                                                  "position before one that missed one");
}

void aLoopOfMoreKeysThanSlotsKeepsHitting() {
  // Six positions route their ten keys every cycle, sixty keys for forty
  // slots: recycling by recency alone gives up each position's keys before
  // the cycle comes back to them; by the cycle, it gives up those it just
  // passed.
  constexpr uint32_t kPositions = 6, kKeys = 10, kSlots = 40;
  std::array<double, 2> rates{};
  for (const uint32_t cycle : {1u, kPositions}) {
    SlotRecycler slots(kSlots, cycle);
    uint64_t hits = 0, routes = 0, load = 0;
    for (int pass = 0; pass < 30; ++pass)
      for (uint32_t position = 0; position < kPositions; ++position) {
        ++load;
        std::vector<uint64_t> missing;
        for (uint64_t key = 0; key < kKeys; ++key) {
          const uint64_t held = uint64_t{position} << 32 | key;
          if (const auto slot = slots.find(held)) {
            slots.touch(*slot, load, false);
            hits += pass >= 5;
          } else {
            missing.push_back(held);
          }
          routes += pass >= 5;
        }
        for (const uint64_t key : missing) static_cast<void>(slots.claim(key, load, false, position % cycle));
      }
    rates[cycle > 1] = double(hits) / double(routes);
  }
  std::cout << "slot_recycler_test: a loop of 60 keys over 40 slots hits " << rates[0] << " by recency, " << rates[1]
            << " by the cycle\n";
  require(rates[0] < 0.1 && rates[1] > 0.4, "recycling by the cycle did not keep the loop hitting: " +
                                                 std::to_string(rates[0]) + " vs " + std::to_string(rates[1]));
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
  // The same with the loads at four positions in turn.
  for (const uint32_t cycle : {1u, 4u}) {
    SlotRecycler slots(64, cycle);
    uint64_t before = 0;
    for (uint64_t load = 1; load <= loads.size(); ++load) {
      if (load == 100) before = allocations;
      const std::vector<uint64_t> &keys = loads[load - 1];
      const bool chunk = load % 7 == 0;
      for (const uint64_t key : keys)
        if (const auto slot = slots.find(key)) slots.touch(*slot, load, chunk);
        else static_cast<void>(slots.claim(key, load, chunk, static_cast<uint32_t>(load % cycle)));
      if (load % 50 == 0) slots.free(*slots.find(keys.front()));
    }
    require(allocations == before, "a claim allocated: " + std::to_string(allocations - before) + " allocations");
    require(slots.ghostHits() > 0, "the loads hit no ghost");
  }
}

// The decode selections of a trace (dev/tools/kv_selection_traces.py): per
// request, its steps; per step and layer, its first position and blocks.
struct TraceStep final {
  uint32_t first = 0;
  std::vector<std::vector<uint32_t>> layers;
};
using Trace = std::vector<std::vector<TraceStep>>;

Trace readTrace(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  size_t at = 4;
  const auto varint = [&] {
    uint64_t value = 0;
    for (uint32_t shift = 0;; shift += 7) {
      require(at < bytes.size(), "a selection trace ends early: " + path.string());
      value |= uint64_t{bytes[at] & 0x7fu} << shift;
      if (!(bytes[at++] & 0x80)) return static_cast<uint32_t>(value);
    }
  };
  require(bytes.size() >= 4 && std::string(bytes.begin(), bytes.begin() + 4) == "KVS1",
          "not a selection trace: " + path.string());
  Trace trace(varint());
  const uint32_t layers = varint();
  for (std::vector<TraceStep> &request : trace) {
    request.resize(varint());
    for (TraceStep &step : request) {
      step.layers.resize(layers);
      for (std::vector<uint32_t> &blocks : step.layers) {
        step.first = varint();
        blocks.resize(varint());
        uint32_t block = 0;
        for (uint32_t &entry : blocks) entry = block += varint();
      }
    }
  }
  require(at == bytes.size(), "a selection trace has bytes past its steps: " + path.string());
  return trace;
}

// The misses of one layer of a request's decode as the hot block cache's
// resolves place its blocks in `slots` slots (model::KvBlockCache): those
// of the open page and after it are the page table's; a step touches the
// blocks it holds, then claims slots for the others, never one it touched;
// the first step, of an empty cache, does not count. `recency` recycles the
// least recently used slot instead.
uint64_t misses(const std::vector<TraceStep> &request, uint32_t layer, uint32_t slots, bool recency) {
  constexpr uint32_t kPageBlocks = 8, kPageTokens = 32;
  SlotRecycler order(slots);
  std::list<uint32_t> used;
  std::unordered_map<uint32_t, std::pair<std::list<uint32_t>::iterator, uint64_t>> lru;
  uint64_t total = 0;
  for (uint64_t load = 1; load <= request.size(); ++load) {
    const TraceStep &step = request[load - 1];
    const uint32_t paged = step.first / kPageTokens * kPageBlocks;
    std::vector<uint32_t> missing;
    for (const uint32_t block : step.layers[layer]) {
      if (block >= paged) continue;
      if (!recency) {
        if (const auto slot = order.find(block)) order.touch(*slot, load, false);
        else missing.push_back(block);
      } else if (const auto held = lru.find(block); held != lru.end()) {
        used.splice(used.end(), used, held->second.first);
        held->second.second = load;
      } else {
        missing.push_back(block);
      }
    }
    if (load > 1) total += missing.size();
    for (const uint32_t block : missing) {
      if (!recency) {
        static_cast<void>(order.claim(block, load, false));
        continue;
      }
      if (used.size() == slots) {
        auto oldest = used.begin();
        while (lru.at(*oldest).second == load) ++oldest;
        lru.erase(*oldest);
        used.erase(oldest);
      }
      lru[block] = {used.insert(used.end(), block), load};
    }
  }
  return total;
}

void adaptiveReplacementKeepsUpWithRecency(const std::filesystem::path &traces) {
  // One lane's floor, two lanes' selections a layer (the knee of the
  // HiSparse measurements).
  constexpr uint32_t kSlots = 8192;
  std::vector<std::filesystem::path> paths;
  for (const auto &entry : std::filesystem::directory_iterator(traces))
    if (entry.path().extension() == ".kvst") paths.push_back(entry.path());
  std::ranges::sort(paths);
  require(!paths.empty(), "no selection trace in " + traces.string());
  uint64_t adaptive = 0, recency = 0;
  for (const std::filesystem::path &path : paths) {
    const Trace trace = readTrace(path);
    uint64_t traceAdaptive = 0, traceRecency = 0;
    for (const std::vector<TraceStep> &request : trace)
      for (uint32_t layer = 0; layer < request.front().layers.size(); ++layer) {
        traceAdaptive += misses(request, layer, kSlots, false);
        traceRecency += misses(request, layer, kSlots, true);
      }
    std::cout << "slot_recycler_test: " << path.filename().string() << ": " << traceAdaptive << " misses, "
              << traceRecency << " least recently used\n";
    adaptive += traceAdaptive;
    recency += traceRecency;
  }
  require(adaptive * 100 <= recency * 102, "adaptive replacement missed " + std::to_string(adaptive) +
                                               " blocks, recency " + std::to_string(recency));
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: slot_recycler_test SELECTION_TRACES");
    neverTakesTheLoadsOwn();
    freeSlotsFirst();
    findsWhatItHolds();
    resizes();
    aScanSparesTheFrequent();
    aRoutedCoreOutlivesMoreLanes();
    positionsOfTheCycle();
    followsItsReference(1);
    followsItsReference(6);
    aLoopOfMoreKeysThanSlotsKeepsHitting();
    claimsAllocateNothing();
    adaptiveReplacementKeepsUpWithRecency(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "slot_recycler_test: FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "slot_recycler_test: PASS\n";
  return 0;
}
