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
// `slots` leave room and the file takes writes: the restore points requests
// resume from.
struct Shelf final {
  uint32_t used = 0;
  uint32_t slots = 64;
  bool takesWrites = true;
};

class ShelvedState final : public CompositeState {
public:
  ShelvedState(std::shared_ptr<Shelf> shelf, bool disk) : shelf_(std::move(shelf)), disk_(disk) {
    if (disk_) ++shelf_->used;
  }
  ~ShelvedState() override {
    if (disk_) --shelf_->used;
  }
  uint64_t bytes() const noexcept override { return 100; }
  uint64_t residentBytes() const noexcept override { return disk_ ? 0 : bytes(); }
  bool canOffload(bool) const noexcept override { return !disk_ && shelf_->takesWrites; }
  std::unique_ptr<StateOffload> offload(std::function<void()>) const override { return shelve(); }
  std::unique_ptr<StateOffload> persist(std::function<void()>) const override { return shelve(); }

private:
  std::unique_ptr<StateOffload> shelve() const {
    if (shelf_->used >= shelf_->slots) return {};
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
// unreadable copies and end; RAM reclaim, the exchange for the expert store
// beside the requests, and the disk quota take what they may, a persistent
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
  // With peeks, the walk's units for a device need are weighed and taken
  // too (checkPeek). With entryPoints, every reclaim entry point is drawn
  // beside the other steps too, and with legacy the cache runs their old
  // bodies (testSourceMatchesTheEntryPoints).
  Run(uint32_t seed, bool windowed, bool peeks = false, bool entryPoints = false, bool legacy = false)
      : random_(seed), windowed_(windowed), peeks_(peeks), entryPoints_(entryPoints), tier_(chance(50)),
        budget_(std::make_shared<model::DiskBudget>(kSlots * 100)),
        cache_(pool_, &tier_, budget_,
               windowed ? std::nullopt : std::optional(CacheSpillPolicy{pick(0, 2), pick(1, 2)}),
               windowed ? std::optional(CacheWindowPolicy{pick(1, 3)}) : std::nullopt) {
    if (legacy)
      cache_.useLegacySteps();
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
    switch (pick(0, entryPoints_ ? 16 : peeks_ ? 15 : 14)) {
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
    case 14: checkExchange(false); break;
    case 15: checkPeek(); break;
    case 16: entryPoint(); break;
    }
    cache_.checkInvariants();
  }

  uint64_t exactUnits = 0, unsureUnits = 0;

  // One reclaim entry point, drawn with its arguments; its result is noted.
  void entryPoint() {
    const ReclaimClass upTo = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary;
    switch (pick(0, 19)) {
    case 0: note(cache_.evictAll(upTo)); break;
    case 1: case 2: case 3: case 4: case 5: note(cache_.reclaimForMemory(upTo, chance(50))); break;
    case 6: case 7: case 8: note(cache_.reclaimStateForLane(upTo)); break;
    case 9: case 10: case 11: {
      const StateRoom room = cache_.reclaimOneState(
          std::array{StateClass::Checkpoint, StateClass::Ordinary, StateClass::InUse}[pick(0, 2)], chance(50));
      results_.insert(results_.end(), {room.made, room.extentBytes});
      break;
    }
    case 12: results_.push_back(cache_.reclaimInUseState().made); break;
    default: {
      const CacheReclaimMode mode = std::array{CacheReclaimMode::KeepExtents, CacheReclaimMode::ReleaseExtents,
                                               CacheReclaimMode::ReusePages}[pick(0, 2)];
      const kv::PageForm form = windowed_ ? std::array{kv::PageForm::Full, kv::PageForm::Slim}[pick(0, 1)]
                                          : std::array{kv::PageForm::Full, kv::PageForm::Index}[pick(0, 1)];
      note(cache_.reclaimForPages(pick(1, 4), mode, upTo, form));
      break;
    }
    }
  }

  // What the run did and holds, for its twin to match: what each entry
  // point returned, the pool, the state cache, the tier and the quota, the
  // requests, and what the cache holds of each base prompt.
  std::vector<uint64_t> fingerprint() {
    const CacheSnapshot snapshot = cache_.snapshot();
    const KvPoolSnapshot &pool = snapshot.pool;
    std::vector<uint64_t> print = results_;
    print.insert(print.end(),
                 {pool.pagesAllocated, pool.pagesActive, pool.pagesPrefix, pool.pagesFree, pool.allocatedBytes,
                  pool.extentAllocations, pool.extentReleases, pool.extentCompactions, pool.pagesMoved,
                  snapshot.stateCache.entries, snapshot.stateCache.bytes, snapshot.stateCache.diskBytes,
                  snapshot.stateCache.offloads, snapshot.stateCache.evictions, snapshot.kvTier.demotions,
                  snapshot.kvTier.demotionsRefused, snapshot.kvTier.copies, snapshot.kvTier.diskBlocks,
                  cache_.uncopiedReached(), tier_.writes.size(), budget_->usedBytes(), budget_->committedBytes(),
                  shelf_->used});
    for (const auto &[id, request] : requests_)
      print.insert(print.end(), {id, request.pages, static_cast<uint64_t>(request.mode)});
    for (uint32_t base = 0; base < bases_.size(); ++base) {
      CacheProbe probe = cache_.probe(prompt({base, kMaximumPages - 1}), {});
      print.insert(print.end(), {probe.cachedTokens(), probe.restoredChain().size()});
    }
    return print;
  }

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
    shelf_->slots = 64;
    cache_.keepDiskLimit();
    tier_.writableFile = chance(50);
    shelf_->takesWrites = chance(50);
    tier_.complete();
    static_cast<void>(cache_.pollTransfers());
    cache_.checkInvariants();
  }

  // What the exchange for the expert store counts on (Cache::exchangeable)
  // reclaimForExchange releases at once, repeated until it makes no
  // progress, with no copy to the SSD cache started: no fewer extents and
  // RAM copies of states. With `exact`, once nothing is in flight, exactly
  // those, unless the KV tier takes writes and the state file none. The
  // chains that probes of base prompts would restore, as requests that have
  // not started hold them, stay: each still matches whole, and the state it
  // resumes from keeps its RAM copy.
  void checkExchange(bool exact) {
    struct Waiting final {
      Request request;
      CacheProbe probe;
      std::vector<uint64_t> chain;
      bool stateResident = false;
    };
    std::vector<Waiting> waiting;
    RestoredChains restored;
    for (uint32_t count = pick(0, 2); count; --count) {
      const Request request{pick(0, static_cast<uint32_t>(bases_.size()) - 1), pick(1, kMaximumPages - 2)};
      CacheProbe probe = cache_.probe(prompt(request), {});
      const std::span<const uint64_t> chain = probe.restoredChain();
      restored.add(chain);
      waiting.push_back({request, probe, {chain.begin(), chain.end()},
                         !chain.empty() && cache_.stateResident(chain.back())});
    }
    const CacheExchange predicted = cache_.exchangeable(restored);
    const uint64_t allocated = pool_.allocatedBytes();
    const uint64_t resident = cache_.snapshot().stateCache.bytes;
    const size_t writes = tier_.writes.size();
    const uint64_t offloads = cache_.snapshot().stateCache.offloads;
    for (uint32_t steps = 0;; ++steps) {
      const CacheReclaimResult step = cache_.reclaimForExchange(restored);
      note(step);
      if (!step.madeProgress)
        break;
      require(steps < 100'000, "the exchange's reclaim did not end");
    }
    cache_.checkInvariants();
    require(tier_.writes.size() == writes && cache_.snapshot().stateCache.offloads == offloads,
            "the exchange's reclaim started a copy to the SSD cache");
    const CacheExchange left = cache_.exchangeable(restored);
    require(!left.extentBytes && !left.states, "the exchange's reclaim left cache it counts on");
    for (Waiting &each : waiting) {
      cache_.refresh(each.probe, prompt(each.request), {});
      require(std::ranges::equal(each.probe.restoredChain(), each.chain) &&
                  (!each.stateResident || cache_.stateResident(each.chain.back())),
              "the exchange took a chain a request that has not started would restore");
    }
    const uint64_t released = allocated - pool_.allocatedBytes();
    const uint64_t freed = (resident - cache_.snapshot().stateCache.bytes) / 100;
    const std::string counts = "counted " + std::to_string(predicted.extentBytes) + " bytes and " +
                               std::to_string(predicted.states) + " states, released " +
                               std::to_string(released) + " bytes and " + std::to_string(freed) + " states";
    require(released >= predicted.extentBytes && freed >= predicted.states,
            "the exchange counted more than its reclaim releases: " + counts);
    if (exact && (!tier_.writableFile || shelf_->takesWrites))
      require(released == predicted.extentBytes && freed == predicted.states,
              "the exchange did not count what its reclaim releases: " + counts);
  }

  // What the walk weighs its victims to free never exceeds what taking them
  // frees (testPeekNeverOverstates): for a device need of a drawn class,
  // reach and timing, the bytes of the first units the walk gives, taken in
  // order, are at most the bytes the pool releases. Each unit releases
  // exactly what it was weighed at while none so far lands later and the
  // takes so far started no transfer and wrote no state: what the tier
  // refuses, and the copies a write gives up for room, the walk cannot know
  // (exactUnits, the others unsureUnits).
  void checkPeek() {
    const ClaimSpec claim{.cls = std::array{ClaimClass::InService, ClaimClass::Start, ClaimClass::Optional,
                                            ClaimClass::Growth, ClaimClass::Host}[pick(0, 4)],
                          .reach = chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary,
                          .timing = chance(50) ? NeedTiming::Immediate : NeedTiming::CanWait};
    const VictimQuery query{.claim = &claim,
                            .need = {.bytes = pick(1, 8) * 400ull, .stateBuffers = chance(30) ? 1u : 0u},
                            .critical = chance(10)};
    std::vector<engine::Victim> victims;
    uint64_t predicted = 0;
    for (std::optional<engine::Victim> victim = cache_.peek(query, nullptr); victim && victims.size() < 4;
         victim = cache_.peek(query, &*victim)) {
      predicted += victim->bytes;
      victims.push_back(*victim);
    }
    const uint64_t allocated = pool_.allocatedBytes();
    const size_t writes = tier_.writes.size();
    const uint64_t offloads = cache_.snapshot().stateCache.offloads;
    bool sure = true;
    for (const engine::Victim &victim : victims) {
      sure = sure && !victim.later;
      const uint64_t before = pool_.allocatedBytes();
      static_cast<void>(cache_.take(query, victim));
      const uint64_t freed = before - pool_.allocatedBytes();
      sure = sure && tier_.writes.size() == writes && cache_.snapshot().stateCache.offloads == offloads;
      ++(sure ? exactUnits : unsureUnits);
      require(!sure || freed == victim.bytes,
              "a unit released " + std::to_string(freed) + " bytes, weighed at " + std::to_string(victim.bytes));
    }
    require(allocated - pool_.allocatedBytes() >= predicted,
            "the walk weighed its victims to free more than taking them freed: weighed " +
                std::to_string(predicted) + ", freed " + std::to_string(allocated - pool_.allocatedBytes()));
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

  void note(const CacheReclaimResult &result) {
    results_.insert(results_.end(), {result.madeProgress, result.reclaimedBytes, result.pending,
                                     result.evictedState, result.evictedKv});
  }

  // Admits as the engine does: a denial makes room in reclaim steps, of
  // the form denied, until the pages fit or nothing more goes.
  template <class Attempt> TokenAdmission admit(Attempt attempt) {
    TokenAdmission admission = attempt();
    while (admission.failure == TokenAdmissionFailure::Denied) {
      const CacheReclaimResult step = cache_.reclaimForPages(admission.additionalPages, CacheReclaimMode::KeepExtents,
                                                             ReclaimClass::InUse, admission.form);
      note(step);
      if (!step.madeProgress)
        break;
      admission = attempt();
    }
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
      cache_.publishCompositeState(block, std::make_shared<ShelvedState>(shelf_, false), (peeks_ || entryPoints_) && chance(30));
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
    note(cache_.reclaimOne(mode, chance(50) ? ReclaimClass::InUse : ReclaimClass::Ordinary));
  }

  // The volume's room changes, and the quota follows it; so does the state
  // file's, whose writes then make room or are refused.
  void keepQuota() {
    budget_->setLimitBytes(pick(0, kSlots) * 100);
    budget_->setActiveLimitBytes(pick(0, kSlots) * 100);
    shelf_->slots = pick(0, 64);
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
  bool entryPoints_;
  // What the entry points returned, in order.
  std::vector<uint64_t> results_;
  // Index pages of 10 bytes for a run that spills, Slim ones of 30 for a
  // windowed one.
  test::TestKvStorage storage_{1024, 100, 4, 10, 30};
  KvPool pool_{storage_, 4};
  test::TestKvTier tier_;
  std::shared_ptr<model::DiskBudget> budget_;
  engine::Cache cache_;
  std::shared_ptr<Shelf> shelf_ = std::make_shared<Shelf>();
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
      run.checkExchange(false);
      run.drain();
      run.settle();
      run.checkExchange(true);
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
void testPeekNeverOverstates(uint32_t plainSeeds, uint32_t windowedSeeds) {
  uint64_t exact = 0, unsure = 0;
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed) {
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
    }
  std::cout << "cache peek runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds << " windowed; " << exact
            << " units released what they were weighed at, " << unsure << " after a copy or a write, never more than taken\n";
}

} // namespace

// The reclaim entry points as loops over the walk's units, as peek gives
// them and take takes them, do what their old bodies did: two runs from one
// seed, one cache stepping the old way, return the same results, take the
// same victims and leave the same pools, step after step, with every entry
// point drawn beside the other steps, checkpoints among the states, and
// device needs weighed and taken.
void testSourceMatchesTheEntryPoints(uint32_t plainSeeds, uint32_t windowedSeeds) {
  for (const bool windowed : {false, true})
    for (uint32_t seed = 1; seed <= (windowed ? windowedSeeds : plainSeeds); ++seed) {
      Run loops(seed, windowed, false, true), bodies(seed, windowed, false, true, true);
      const auto same = [&](const char *when, int step) {
        const std::vector<uint64_t> loop = loops.fingerprint(), body = bodies.fingerprint();
        if (loop != body)
          throw std::runtime_error(std::string("the loops and the old bodies differ ") + when + " " +
                                   std::to_string(step) + ", from fingerprint field " +
                                   std::to_string(std::ranges::mismatch(loop, body).in1 - loop.begin()));
      };
      try {
        for (int step = 0; step < 600; ++step) {
          loops.step();
          bodies.step();
          same("after step", step);
        }
        for (Run *run : {&loops, &bodies}) {
          run->endRequests();
          run->checkExchange(false);
          run->drain();
          run->settle();
          run->checkExchange(true);
        }
        same("at the end", 600);
      } catch (const std::exception &error) {
        throw std::runtime_error(std::string(windowed ? "differential windowed " : "differential ") + "seed " +
                                 std::to_string(seed) + ": " + error.what());
      }
    }
  std::cout << "cache differential runs ok: " << plainSeeds << " plain seeds, " << windowedSeeds << " windowed\n";
}

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
    testSourceMatchesTheEntryPoints(plainSeeds, windowedSeeds);
    std::cout << "cache property tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cache property tests failed: " << error.what() << '\n';
    return 1;
  }
}
