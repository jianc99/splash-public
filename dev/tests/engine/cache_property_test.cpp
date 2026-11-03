#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Cache.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::require;

// A state the tier takes whenever reclaim or persist writes it, as long as
// the quota has room for its copy and the file takes writes: the restore
// points requests resume from. Its copies draw on the quota the KV tier's
// copies draw on, as the state file's slots do (openDiskTier).
struct Shelf final {
  std::shared_ptr<model::DiskBudget> budget;
  bool takesWrites = true;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {}
  ~ShelvedState() override {
    if (disk_) shelf_->budget->release(bytes());
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool canOffload(bool) const noexcept override { return !disk_ && shelf_->takesWrites; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override { return shelve(); }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override { return shelve(); }

private:
  std::unique_ptr<StateOffload> shelve() const {
    if (!shelf_->budget->reserve(bytes())) return {};
    struct Written final : StateOffload {
      std::shared_ptr<const CompositeState> disk;
      bool ready() const noexcept override { return true; }
      bool finish() override { return true; }
      const std::shared_ptr<const CompositeState> &state() const noexcept override { return disk; }
    };
    auto written = std::make_unique<Written>();
    written->disk = std::make_shared<ShelvedState>(shelf_, true);
    return written;
  }

  std::shared_ptr<Shelf> shelf_;
  bool disk_;
};

// One randomized run of the tiered block store. Requests of both modes on
// prompts that share prefixes publish, spill, restore, convert, read
// unreadable copies and end; RAM reclaim beside the requests and the disk
// quota take what they may, a persistent
// tier keeps restore points, the pool's growth and the
// tier's writes of every class fail now and then, and the tier's transfers
// land at random. The cache checks every invariant spilling relies on after
// each step (Cache::checkInvariants), and active KV never stays past its
// limit after the quota was kept. A windowed run's requests do not spill,
// and between commands the pages every window has passed give up their
// windowed layers' K/V: each page a request's next command reads from its
// window on stays Full (its cuts check it), and each state's tail too, in
// RAM and on disk, which it is published only with (Cache::stateTailFull).
class Run final {
public:
  // With peeks, the walk's units for a device or a page need are weighed
  // and taken too (checkPeek), and in half the runs the tier says how many
  // demotions it admits (test::RoomTier), which the walk then counts on.
  Run(uint32_t seed, bool windowed, bool peeks = false)
      : random_(seed), windowed_(windowed), peeks_(peeks), tier_(chance(50)), saysRoom_(peeks && chance(50)),
        budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        cache_(pool_, saysRoom_ ? static_cast<engine::KvTier *>(&room_) : &tier_, budget_,
               windowed ? std::nullopt : std::optional(CacheSpillPolicy{pick(0, 2), pick(1, 2)}),
               windowed ? std::optional(CacheWindowPolicy{pick(1, 3)}) : std::nullopt) {
    tier_.budget = budget_;
    tier_.transferLimit = 64;
    tier_.writeLimits = {8, 4, 8};
    storage_.budgetPages = 96;
    for (uint32_t base = 0; base < bases_.size(); ++base) {
      bases_[base].resize(kMaximumPages * KvCache::pageTokens + 1);
      for (uint32_t row = 0; row < bases_[base].size(); ++row)
        bases_[base][row] = 1000 * (base + 1) + row;
    }
  }

  void step() {
    switch (pick(0, peeks_ ? 14 : 13)) {
    case 0: start(); break;
    case 1: case 2: grow(); break;
    case 3: static_cast<void>(cache_.writeThrough()); break;
    case 4: land(); break;
    case 5:
      static_cast<void>(windowed_ ? cache_.trimWindows(chance(80))
                                  : cache_.spillWritten(chance(50) ? SpilledExtents::KeepRunway
                                                                   : SpilledExtents::Released)
                                        .additionalPages);
      break;
    case 6: reclaim(); break;
    case 7: keepQuota(); break;
    case 8: convert(); break;
    case 9: failRead(); break;
    case 10: finish(); break;
    case 11: command(); break;
    case 12: persist(); break;
    case 13: failWrites(); break;
    case 14: checkPeek(); break;
    }
    cache_.checkInvariants();
  }

  uint64_t exactUnits = 0, unsureUnits = 0, landingUnits = 0;

  // Every request ends, every transfer lands, and nothing leaks: the pool's
  // pages and the tier's slots belong to the cache's blocks alone.
  void drain() {
    endRequests();
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
    const KvPoolSnapshot pool = pool_.snapshot();
    require(pool.pagesActive == 0 && budget_->committedBytes() == 0 && cache_.spillSnapshot().heldBytes == 0,
            "a finished run left pages held, quota committed or copies held");
  }

  // Every request ends; the transfers in flight stay so.
  void endRequests() {
    while (!requests_.empty()) {
      sources_.erase(requests_.begin()->first);
      cache_.endRequest(requests_.begin()->first);
      requests_.erase(requests_.begin());
    }
  }

  // The quota is whole again, and in half the runs each the KV tier and the
  // state file take no more writes; what was in flight has landed.
  void settle() {
    budget_->setLimitBytes(kSlots * 100);
    budget_->setActiveLimitBytes(kSlots * 100);
    cache_.keepDiskLimit();
    tier_.writableFile = chance(50);
    shelf_->takesWrites = chance(50);
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
  }

  // What the walk weighs its victims to free never exceeds what taking them
  // frees (testPeekNeverOverstates): for a device or a page need of a drawn
  // class, reach and timing, a page need met in place or while the host
  // refuses growth, the bytes of the first units the walk gives, taken in
  // order, are at most those the pool releases and those take() says the
  // copies it started complete as they land (laterBytes), and their pages
  // at most the pages of the need's form that free, now or as demotions
  // land (Cache::demotedPages: a page that spills once its copy lands frees
  // only as spillWritten() moves it, which no retry does), with what was
  // started landed and the query's units taken again,
  // as a waiting claim's retries take a unit's later steps; an Immediate
  // claim, as free() does, passes over those that land later. Each unit of
  // a device need releases exactly what it was weighed
  // at while none so far lands later and the takes so far started no
  // transfer and wrote no state: what the tier refuses, and the copies a
  // write gives up for room, the walk cannot know (exactUnits, the others
  // unsureUnits).
  void checkPeek() {
    const ClaimSpec claim{.cls = std::array{ClaimClass::InService, ClaimClass::Start, ClaimClass::Optional,
                                            ClaimClass::Growth, ClaimClass::Host}[pick(0, 4)],
                          .reach = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary,
                          .timing = chance(50) ? NeedTiming::Immediate : NeedTiming::CanWait};
    const kv::PageForm form = !chance(50) ? kv::PageForm::Full : windowed_ ? kv::PageForm::Slim : kv::PageForm::Index;
    const uint32_t stateBuffers = chance(30) ? 1u : 0u;
    const bool device = chance(50);
    const VictimQuery query{.claim = &claim,
                            .need = device ? Need{.bytes = pick(1, 8) * 400ull, .stateBuffers = stateBuffers}
                                           : Need{.hostRefused = chance(50), .pages = pick(1, 8), .form = form,
                                                  .stateBuffers = stateBuffers},
                            .critical = chance(10)};
    std::vector<engine::Victim> victims;
    uint64_t predicted = 0;
    uint32_t predictedPages = 0;
    for (std::optional<engine::Victim> victim = cache_.peek(query, nullptr); victim && victims.size() < 4;
         victim = cache_.peek(query, &*victim)) {
      if (victim->later && claim.timing == NeedTiming::Immediate)
        continue;
      predicted += victim->bytes;
      predictedPages += victim->form == form ? victim->pages : 0;
      victims.push_back(*victim);
    }
    const auto pagesOf = [&] {
      return pool_.freePageCount(form) + cache_.demotedPages()[static_cast<size_t>(form)];
    };
    const size_t writes = tier_.writes.size();
    const uint64_t offloads = cache_.snapshot().stateCache.offloads;
    bool sure = true;
    uint64_t freed = 0;
    uint32_t freedPages = 0;
    for (const engine::Victim &victim : victims) {
      sure = sure && !victim.later;
      // A windowed cache's trims may take Slim extents for the Full pages
      // they free.
      const uint64_t before = pool_.allocatedBytes();
      const uint32_t pagesBefore = pagesOf();
      const FreeResult taken = cache_.take(query, victim);
      const uint64_t released = before - std::min(before, pool_.allocatedBytes());
      require(taken.nowBytes <= released, "a take released " + std::to_string(released) + " bytes, reported " +
                                              std::to_string(taken.nowBytes));
      freed += released + taken.laterBytes;
      const uint32_t pagesAfter = pagesOf();
      freedPages += pagesAfter - std::min(pagesAfter, pagesBefore);
      sure = sure && tier_.writes.size() == writes && cache_.snapshot().stateCache.offloads == offloads;
      if (!device)
        continue;
      landingUnits += taken.laterBytes != 0;
      ++(sure ? exactUnits : unsureUnits);
      require(!sure || released == victim.bytes,
              "a unit released " + std::to_string(released) + " bytes, weighed at " + std::to_string(victim.bytes));
    }
    // A unit's later steps free once the transfers before them land, which
    // the claim's retries take: what was started lands, the pages that frees
    // past the landing pages counted when it started count, and the query's
    // units are taken again until they free what was weighed or none goes.
    const auto landingOf = [&] { return cache_.demotedPages()[static_cast<size_t>(form)]; };
    for (uint32_t retry = 0; retry < 64 && (freed < predicted || freedPages < predictedPages); ++retry) {
      const uint64_t before = pool_.allocatedBytes();
      const uint32_t freeBefore = pool_.freePageCount(form), landingBefore = landingOf();
      tier_.complete();
      static_cast<void>(cache_.pollTransfers());
      const uint32_t freeAfter = pool_.freePageCount(form), landingAfter = landingOf();
      const uint32_t landed = landingBefore - std::min(landingBefore, landingAfter);
      freedPages += freeAfter - std::min(freeAfter, freeBefore + landed);
      std::optional<engine::Victim> unit = cache_.peek(query, nullptr);
      while (unit && unit->later && claim.timing == NeedTiming::Immediate)
        unit = cache_.peek(query, &*unit);
      const uint32_t pagesBefore = pagesOf();
      const FreeResult taken = unit ? cache_.take(query, *unit) : FreeResult{};
      freed += before - std::min(before, pool_.allocatedBytes()) + taken.laterBytes;
      const uint32_t pagesAfter = pagesOf();
      freedPages += pagesAfter - std::min(pagesAfter, pagesBefore);
      if (taken.outcome != FreeOutcome::Met)
        break;
    }
    require(freed >= predicted && freedPages >= predictedPages,
            "the walk weighed its victims to free more than taking them freed: weighed " + std::to_string(predicted) +
                " bytes and " + std::to_string(predictedPages) + " pages, freed " + std::to_string(freed) +
                " bytes and " + std::to_string(freedPages) + " pages");
  }

private:
  static constexpr uint64_t kSlots = 160;
  static constexpr uint32_t kMaximumPages = 12;

  struct Request final {
    uint32_t base = 0;
    uint32_t pages = 0;
    KvMode mode = KvMode::Resident;
  };

  uint32_t pick(uint32_t first, uint32_t last) {
    return std::uniform_int_distribution<uint32_t>(first, last)(random_);
  }
  bool chance(uint32_t percent) { return pick(1, 100) <= percent; }
  std::span<const uint32_t> prompt(const Request &request) const {
    return std::span<const uint32_t>(bases_[request.base]).first(request.pages * KvCache::pageTokens + 1);
  }
  Request *anyRequest() {
    if (requests_.empty()) return nullptr;
    auto entry = requests_.begin();
    std::advance(entry, pick(0, static_cast<uint32_t>(requests_.size()) - 1));
    return &entry->second;
  }
  uint64_t idOf(const Request &request) const {
    for (const auto &[id, each] : requests_)
      if (&each == &request) return id;
    throw std::logic_error("unknown request");
  }
  bool waiting(uint64_t id) const { return cache_.kvRestoreStatus(id) == KvRestoreStatus::Pending; }

  // Admits as the engine does: a denial makes room in reclaim steps, of
  // the form denied, until the pages fit or nothing more goes.
  template <class Attempt> TokenAdmission admit(Attempt attempt) {
    TokenAdmission admission = attempt();
    while (admission.failure == TokenAdmissionFailure::Denied &&
           test::takeFirst(cache_, test::kInServiceGrowth,
                           {.pages = admission.additionalPages, .form = admission.form})
                   .outcome == FreeOutcome::Met)
      admission = attempt();
    return admission;
  }

  // A new request on a prefix of a base prompt, restored from what the
  // cache holds of it, a Spill request committing quota for its pages
  // whether the tier takes writes or not, as a resume does.
  void start() {
    if (requests_.size() >= 6) return;
    const uint64_t id = ++nextId_;
    Request request{pick(0, static_cast<uint32_t>(bases_.size()) - 1), pick(1, kMaximumPages - 2),
                    !windowed_ && chance(60) ? KvMode::Spill : KvMode::Resident};
    CacheLookup lookup = cache_.lookup(prompt(request), {});
    cache_.beginRequest(id, request.mode);
    const uint32_t copied = lookup.state ? cache_.copiedBlocks(lookup.state->kvBlock()) : 0;
    if (request.mode == KvMode::Spill && !cache_.commitSpill(id, kMaximumPages - copied)) {
      cache_.endRequest(id);
      return;
    }
    if (!admit([&] { return cache_.restoreRequest(id, lookup); }).granted()) {
      cache_.endRequest(id);
      return;
    }
    request.pages = lookup.resumeBoundary() / KvCache::pageTokens;
    requests_.emplace(id, request);
  }

  // A request's command commits more pages, which it publishes; now and then
  // a state lands at its newest block.
  void grow() {
    Request *request = anyRequest();
    if (!request || request->pages >= kMaximumPages) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const uint32_t pages = std::min(kMaximumPages, request->pages + pick(1, 3));
    if (!admit([&] { return cache_.ensureTokens(id, pages * KvCache::pageTokens + 1); }).granted()) return;
    request->pages = pages;
    cache_.publishCommittedBlocks(id, prompt(*request), pages * KvCache::pageTokens, {});
    // As the engine does, a state at a block another request published one
    // at is reused, and one is published only with a tail it restores from.
    if (const uint64_t block = cache_.blockAt(id, pages * KvCache::pageTokens);
        chance(30) && !cache_.reuseStoredState(block) && cache_.stateTailFull(block))
      cache_.publishCompositeState(block, std::make_shared<ShelvedState>(shelf_, false), peeks_ && chance(30));
  }

  // The tier's transfers land, a few of them failing.
  void land() {
    tier_.complete(!chance(5));
    static_cast<void>(cache_.pollTransfers());
    for (auto entry = requests_.begin(); entry != requests_.end();) {
      if (cache_.kvRestoreStatus(entry->first) != KvRestoreStatus::Failed) {
        ++entry;
        continue;
      }
      sources_.erase(entry->first);
      cache_.endRequest(entry->first);
      entry = requests_.erase(entry);
    }
  }

  // Write-behind keeps a restore point at a request's newest block.
  void persist() {
    Request *request = anyRequest();
    if (!cache_.persistent() || !request || !request->pages) return;
    static_cast<void>(cache_.persist(cache_.blockAt(idOf(*request), request->pages * KvCache::pageTokens)));
  }

  // The writes of one class in flight fail, whichever requests hold their
  // blocks.
  void failWrites() {
    tier_.fail(std::array{WriteClass::Reclaim, WriteClass::Persist, WriteClass::Spill}[pick(0, 2)]);
    static_cast<void>(cache_.pollTransfers());
  }

  void reclaim() {
    const CacheReclaimMode mode = std::array{CacheReclaimMode::KeepExtents, CacheReclaimMode::ReleaseExtents,
                                             CacheReclaimMode::ReusePages}[pick(0, 2)];
    static_cast<void>(cache_.reclaimOne(mode, chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary));
  }

  // The volume's room changes, and the quota follows it, which the state
  // file shares: its writes then make room or are refused.
  void keepQuota() {
    budget_->setLimitBytes(pick(0, kSlots) * 100);
    budget_->setActiveLimitBytes(pick(0, kSlots) * 100);
    cache_.keepDiskLimit();
    const KvSpillSnapshot spilled = cache_.spillSnapshot();
    require(!spilled.committedBytes || spilled.heldBytes + spilled.committedBytes <= spilled.activeLimitBytes,
            "active KV stayed past its limit");
  }

  void convert() {
    Request *request = anyRequest();
    if (windowed_ || !request || request->mode == KvMode::Spill) return;
    if (cache_.convertToSpill(idOf(*request), kMaximumPages)) request->mode = KvMode::Spill;
  }

  // A read of a spilled block fails. The engine applies the results of the
  // command that read it while the command, and what it read, lives on: the
  // lane ends and the block's copy is given up.
  void failRead() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    if (waiting(id)) return;
    const PageTableView cut = cache_.cutPageTable(id);
    const std::shared_ptr<const model::KvPageSources> command = cut.sources;
    std::vector<uint64_t> blocks;
    for (uint32_t page = 0; command && page < cut.pages.size(); ++page)
      if ((*command)[page].indexed()) blocks.push_back((*command)[page].block);
    if (blocks.empty()) return;
    sources_.erase(id);
    cache_.endRequest(id);
    requests_.erase(id);
    cache_.poisonCopy(blocks[pick(0, static_cast<uint32_t>(blocks.size()) - 1)]);
    whileApplying();
  }

  // A request ends as the engine applies its last command, which holds what
  // it read until then.
  void finish() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    const auto command = sources_.extract(id);
    cache_.endRequest(id);
    requests_.erase(id);
    whileApplying();
  }

  // What else applying a command may do before the command goes: give up for
  // quota or memory what nothing holds any more.
  void whileApplying() {
    if (chance(50)) keepQuota();
    if (chance(50)) reclaim();
  }

  // A command cuts a request's page table and reads its sources until the
  // next one.
  void command() {
    Request *request = anyRequest();
    if (!request) return;
    const uint64_t id = idOf(*request);
    if (!waiting(id)) sources_[id] = cache_.cutPageTable(id).sources;
  }

  std::mt19937 random_;
  bool windowed_;
  bool peeks_;
  // Index pages of 10 bytes for a run that spills, Slim ones of 30 for a
  // windowed one.
  test::TestKvStorage storage_{1024, 100, 4, 10, 30};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  test::RoomTier room_{tier_};
  bool saysRoom_;
  std::shared_ptr<model::DiskBudget> budget_;
  engine::Cache cache_;
  std::shared_ptr<Shelf> shelf_ = std::make_shared<Shelf>(Shelf{.budget = budget_});
  std::array<std::vector<uint32_t>, 3> bases_;
  std::map<uint64_t, Request> requests_;
  // What the commands in flight read.
  std::map<uint64_t, std::shared_ptr<const model::KvPageSources>> sources_;
  uint64_t nextId_ = 0;
};

// A windowed run takes more seeds: the paths that leave a state's tail
// Slim, unless the cache keeps it Full, take a run of steps few seeds make.
// Windowed seed 9454 runs too: in it a poisoned block's own write lands
// after its subtree left (Cache::dropPoisoned). The seeds run are
// 1..plainSeeds plain and 1..windowedSeeds windowed.
void testRandomRuns(uint32_t plainSeeds, uint32_t windowedSeeds) {
  const auto check = [](uint32_t seed, bool windowed) {
    Run run(seed, windowed);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
      run.settle();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(windowed ? "windowed " : "") + "seed " + std::to_string(seed) + ": " +
                               error.what());
    }
  };
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
      check(seed, windowed);
  check(9454, true);
  std::cout << "cache property runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds
            << " windowed and windowed seed 9454\n";
}

// The same runs with the walk's units for a device need weighed and taken
// beside the other steps: no unit frees less than the walk weighed.
// Windowed seeds 2484 and 9349 run too: in them a unit's evictions take the
// last resident child of a block whose demotion is in flight, whose page
// then frees as it lands, for a claim that cannot wait (2484) and one that
// can (9349). So does windowed seed 25949: in it restores no request holds
// land with the demotion in flight, and the claim's first retry takes those
// leaves, with the first later step, before what the next landings free;
// the extent they empty leaves the next demotions out of reach. And windowed
// seed 28462: in it a unit takes the state on a block whose demotion is in
// flight and on which a request's chain ends, so its page frees as nothing
// lands. Plain seeds 11208, 99256 and 229505: in them a page the walk
// weighed frees at a landing that lets a poisoned subtree go, beside copies
// whose pages spill as they land, which free nothing until spillWritten().
// Windowed seed 1836083 and plain seed 396221: in them a leaf that goes
// passes its recency to its parent as the cache does, with what an Erase of
// its disk-only subtree gave it (1836083), and to a parent whose demotion
// then lands, which passes it on (396221). Plain seeds 10248, 15837, 24025,
// 105144, 386046, 470525, 482345, 247559 and 364835: in them the resume
// point lies below a poisoned block whose subtree goes as the first
// transfers land, and the retries keep the newest state left, most often
// one the walk left waiting for the write slot. Windowed seeds 55544, 63544,
// 546782, 1061965, 1748006 and 2429048, and plain seeds 415497, 478646 and
// 483524: in them a retry first releases a Full extent the plan placed
// elsewhere, as a leaf goes once the state write that kept it lands, a
// write the quota refuses drops its state, a demotion counted unsure
// starts, landings complete an extent with the free pages a unit leaves, or
// Spill writes land copies through which blocks no request holds go; the
// Slim and Index units after it reach as far as their own form's pages, and
// in 483524 the Full tail of a spilled chain goes, though no Full extent can
// empty any more, as the Index pages above it go only after it.
void testPeekNeverOverstates(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t exact = 0, unsure = 0, landing = 0;
  const auto check = [&](uint32_t seed, bool windowed) {
    Run run(seed, windowed, true);
    try {
      for (int step = 0; step < 600; ++step)
        run.step();
      run.endRequests();
      run.drain();
    } catch (const std::exception &error) {
      throw std::runtime_error(std::string(windowed ? "peek windowed " : "peek ") + "seed " + std::to_string(seed) +
                               ": " + error.what());
    }
    exact += run.exactUnits;
    unsure += run.unsureUnits;
    landing += run.landingUnits;
  };
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed)
      check(seed, windowed);
  check(2484, true);
  check(9349, true);
  check(25949, true);
  check(28462, true);
  for (const uint32_t seed : {11208u, 99256u, 229505u})
    check(seed, false);
  check(1836083, true);
  check(396221, false);
  for (const uint32_t seed : {10248u, 15837u, 24025u, 105144u, 386046u, 470525u, 482345u, 247559u, 364835u})
    check(seed, false);
  for (const uint32_t seed : {55544u, 63544u, 546782u, 1061965u, 1748006u, 2429048u})
    check(seed, true);
  for (const uint32_t seed : {415497u, 478646u, 483524u})
    check(seed, false);
  std::cout << "cache peek runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds
            << " windowed and the fixed seeds; " << exact
            << " units released what they were weighed at, "
            << unsure << " after a copy or a write, " << landing
            << " freeing extents as their copies land, never more than taken\n";
}

} // namespace

// cache-property [plainSeeds windowedSeeds]: 40 and 400 by default.
int main(int argc, char **argv) {
  if (argc != 1 && argc != 3) {
    std::cerr << "usage: cache-property [plainSeeds windowedSeeds]\n";
    return 2;
  }
  try {
    const uint32_t plainSeeds = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[1])) : 40u;
    const uint32_t windowedSeeds = argc == 3 ? static_cast<uint32_t>(std::stoul(argv[2])) : 400u;
    testRandomRuns(plainSeeds, windowedSeeds);
    testPeekNeverOverstates(plainSeeds, windowedSeeds);
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
