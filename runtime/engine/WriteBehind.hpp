#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_set>

namespace splash::engine {

struct WriteBehindSnapshot final {
  // Points waiting to fall due, for their request to end or for memory to
  // run short, or the one being written.
  uint32_t waiting = 0;
  // Points whose copies were all on disk when they fell due or once their
  // writes landed.
  uint64_t durable = 0;
  // Points their conversation went on from, or gone, before they fell due.
  uint64_t unneeded = 0;
  // Points the tier could not keep.
  uint64_t refused = 0;
};

// Gives the newest restore points their copies on the disk tier while they
// stay in RAM, ahead of need (Cache::persist): a reclaim then frees their
// memory at once instead of waiting for writes, and a persistent tier's next
// process takes them back. An ordinary point published or reused at least
// kMinimumTokens deep is copied once it has waited kDelayMilliseconds,
// unless its conversation has gone on from it by then: a point that is
// newest only briefly, like a step of a tool loop, costs no write, and a
// shorter one recomputes in less time than its state takes to write.
//
// A temporary tier's copies serve reclaims alone, so there a point is
// copied only while one is in sight: within kNeedMilliseconds of a reclaim
// last taking cache out of RAM to the disk tier (Cache::reclaimedToDisk),
// which shows memory ran short, and once its request has ended, since until
// then the request holds its chain and may go on from it, and reclaims take
// the point last. Cache that leaves through the copy it has counts as much
// as cache a reclaim writes, so that the copies go on while memory stays
// short, when reclaims find what they take copied and write nothing.
// Without memory pressure nothing is written ahead. Points that wait for
// need keep their turn while they stay the newest of their conversations: a
// publication meanwhile drops those that are not (Cache::newestPoint), so
// that need, once in sight, finds no backlog of superseded points. A
// persistent tier copies every point, in use or not, so that a later start,
// after a crash too, takes it back.
//
// Points are copied one at a time, in the order they were published, those
// that wait passed over, and none while a request waits for memory on its
// way back: what it waits for, a demotion, a restore or the state write,
// comes first, and a copy started as one lands could take the state write
// from it again. Like every write of the tier these draw on its
// write budget, when one is set (model::DiskBudget): a point waits while
// the budget holds its state back, which is written before its chain.
class WriteBehind final {
public:
  static constexpr double kDelayMilliseconds = 10'000.0;
  static constexpr uint32_t kMinimumTokens = 2048;
  // How long a reclaim that took cache to the disk tier keeps a temporary
  // tier copying points ahead. While a load keeps memory short, its reclaims
  // take cache again at least as often as its conversations take turns,
  // seconds apart for an agent and a minute or two for a person: five
  // minutes keeps the copies going between them, and ends them soon after
  // the load does.
  static constexpr double kNeedMilliseconds = 300'000.0;

  explicit WriteBehind(Cache &cache);
  WriteBehind(const WriteBehind &) = delete;
  WriteBehind &operator=(const WriteBehind &) = delete;

  // A point published or reused at this block, `tokens` deep. Nothing while
  // the disk tier takes no writes.
  void published(uint64_t block, uint32_t tokens, double now);
  // Steps the points that are due and not in use, while copies are needed
  // and no claim waits for memory on its way back (claimWaiting), which
  // needs the tier's writes and the state write first. True when one became
  // durable or was given up; a write it starts wakes the engine when it
  // lands. Whether copies are needed is the tier's and its reclaims' to say
  // (needed): the memory's demand for them, `demand`, is ignored.
  [[nodiscard]] bool run(double now, bool demand, bool claimWaiting);
  // The step with a request that waits for memory on its way back
  // (Engine::memoryAwaited) in place of a claim.
  [[nodiscard]] bool run(double now, bool memoryAwaited) { return run(now, false, memoryAwaited); }
  // At a clean stop: in a persistent tier, steps every waiting point, the
  // newest first, due or not and in use or not, whatever the write budget
  // (Cache::liftDiskWriteBudget). True once none is left, and at once in a
  // temporary tier, whose copies go with the process.
  [[nodiscard]] bool flush();
  // When run() next has something to do that no landing write will wake it
  // for: the next point not in use falling due, or, for a step the write
  // budget held back, when the budget admits it again; none while copies
  // wait for need, which a reclaim brings in a step of the engine.
  [[nodiscard]] std::optional<double> nextWakeup() const;
  [[nodiscard]] WriteBehindSnapshot snapshot() const noexcept;

private:
  struct Point final {
    uint64_t block = 0;
    double due = 0.0;
  };
  // Counts a step that settled its point; false while it is in progress.
  bool settle(PersistStatus status);
  // Whether points are copied now: always in a persistent tier, and in a
  // temporary one within kNeedMilliseconds of the last reclaim seen to take
  // cache to the disk tier, as needed_ keeps.
  [[nodiscard]] bool needed(double now);
  // A temporary tier's point whose request has not ended.
  [[nodiscard]] bool inUse(uint64_t block) const noexcept;

  Cache &cache_;
  std::deque<Point> waiting_;
  std::unordered_set<uint64_t> queued_;
  // The first point due and not in use waits for writes or restores to
  // land, its own or the tier's, or for the write budget.
  bool stalled_ = false;
  // Points were copied when last asked (needed).
  bool needed_;
  // What reclaims took to the disk tier as last seen
  // (Cache::reclaimedToDisk), and when it was last seen to grow.
  uint64_t reclaimedToDisk_ = 0;
  std::optional<double> shortAt_;
  WriteBehindSnapshot counters_;
};

} // namespace splash::engine
