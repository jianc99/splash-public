#pragma once

#include "engine/Cache.hpp"

#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <unordered_set>

namespace splash::engine {

struct WriteBehindSnapshot final {
  // Points waiting to fall due, for their request to end or for the
  // memory's demand for copies, or the one being written.
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
// A persistent tier copies every point, in use or not, so that a later
// start, after a crash too, takes it back. A temporary tier's copies serve
// reclaims alone, so there a point is copied only on the memory's demand for
// copies (run): while claims recently reached cache that a copy would have
// let go at once, and only as far as the claims ahead ask, which the cache's
// victims that go at once already cover otherwise. A point is copied once
// its request has ended, since until then the request holds its chain and
// may go on from it, and reclaims take the point last. Without that demand
// nothing is written ahead. Points that wait for demand keep their turn
// while they stay the newest of their conversations: a publication
// meanwhile drops those that are not (Cache::newestPoint), so that demand,
// once it comes, finds no backlog of superseded points.
//
// Points are copied one at a time, in the order they were published, those
// that wait passed over, and none while a claim waits for memory on its way
// back: what it waits for, a demotion, a restore or the state write, comes
// first, and a copy started as one lands could take the state write from it
// again. Like every write of the tier these draw on its write budget, when
// one is set (model::DiskBudget): a point waits while the budget holds its
// state back, which is written before its chain.
class WriteBehind final {
public:
  static constexpr double kDelayMilliseconds = 10'000.0;
  static constexpr uint32_t kMinimumTokens = 2048;

  explicit WriteBehind(Cache &cache);
  WriteBehind(const WriteBehind &) = delete;
  WriteBehind &operator=(const WriteBehind &) = delete;

  // A point published or reused at this block, `tokens` deep. Nothing while
  // the disk tier takes no writes. Whether copies wait for demand is asked
  // now (copying), of the demand the last step was asked with, which a
  // claim since changes at the next step.
  void published(uint64_t block, uint32_t tokens, double now);
  // Steps the points that are due and not in use, unless a claim waits for
  // memory on its way back (claimWaiting), which needs the tier's writes and
  // the state write first. A temporary tier copies only while `demandBytes`
  // (the memory's demand for copies: none while no recent claim reached
  // cache a copy would have let go at once, else twice what the claim ahead
  // will take) is more than the cache's victims that go at once, without a
  // write, the oldest first, hold, with what the copies ahead in flight free
  // at once after they land, those it started among them
  // (Cache::copyingAheadBytes). True when a point became durable or was
  // given up; a write it starts wakes the engine when it lands.
  [[nodiscard]] bool run(double now, uint64_t demandBytes, bool claimWaiting);
  // At a clean stop: in a persistent tier, steps every waiting point, the
  // newest first, due or not and in use or not, whatever the write budget
  // (Cache::liftDiskWriteBudget). True once none is left, and at once in a
  // temporary tier, whose copies go with the process.
  [[nodiscard]] bool flush();
  // When run() next has something to do that no landing write will wake it
  // for: the next point not in use falling due, or, for a step the write
  // budget held back, when the budget admits it again; none while copies
  // wait for demand, which a claim brings in a step of the engine, nor once
  // the last step stopped because a claim waits, whose landings wake the
  // engine, or because the demand was covered, which only a step changes.
  [[nodiscard]] std::optional<double> nextWakeup() const;
  [[nodiscard]] WriteBehindSnapshot snapshot() const noexcept;

private:
  struct Point final {
    uint64_t block = 0;
    double due = 0.0;
  };
  // Counts a step that settled its point; false while it is in progress.
  bool settle(PersistStatus status);
  // Steps the due points not in use, while copies are made (copying_) and
  // `copies` says the next one is.
  [[nodiscard]] bool step(double now, bool claimWaiting, const std::function<bool()> &copies);
  // Copies are made now: always in a persistent tier; in a temporary one
  // on the demand last asked with.
  [[nodiscard]] bool copying() const noexcept;
  // The cache's victims that go at once, without a write, the oldest first,
  // with what the copies ahead in flight free after they land, hold less
  // than `demandBytes`.
  [[nodiscard]] bool demanded(uint64_t demandBytes) const;
  // A temporary tier's point whose request has not ended.
  [[nodiscard]] bool inUse(uint64_t block) const noexcept;

  Cache &cache_;
  std::deque<Point> waiting_;
  std::unordered_set<uint64_t> queued_;
  // Why the last step stopped before the points due: a claim waits, the
  // demand was covered, or the first point due and not in use waits for
  // writes or restores to land, its own or the tier's, or for the write
  // budget.
  enum class Stop : uint8_t { None, ClaimWaiting, Covered, Stalled };
  Stop stopped_ = Stop::None;
  // Copies are made, as last asked (copying).
  bool copying_;
  // The demand the last step was asked with.
  uint64_t demandBytes_ = 0;
  WriteBehindSnapshot counters_;
};

} // namespace splash::engine
