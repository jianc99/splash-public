#include "engine/WriteBehind.hpp"

namespace splash::engine {

WriteBehind::WriteBehind(Cache &cache) : cache_(cache), needed_(cache.persistent()) {}

void WriteBehind::published(uint64_t block, uint32_t tokens, double now) {
  if (!cache_.kvTierWritable() || tokens < kMinimumTokens || queued_.contains(block))
    return;
  // While copies wait for need, the queue keeps only the points their
  // conversations still resume from.
  if (!needed(now))
    std::erase_if(waiting_, [&](const Point &point) {
      if (cache_.newestPoint(point.block))
        return false;
      queued_.erase(point.block);
      ++counters_.unneeded;
      return true;
    });
  queued_.insert(block);
  waiting_.push_back({block, now + kDelayMilliseconds});
}

bool WriteBehind::settle(PersistStatus status) {
  switch (status) {
  case PersistStatus::Durable:
    ++counters_.durable;
    return true;
  case PersistStatus::Unneeded:
    ++counters_.unneeded;
    return true;
  case PersistStatus::Refused:
    ++counters_.refused;
    return true;
  case PersistStatus::Started:
  case PersistStatus::Busy:
    return false;
  }
  return false;
}

bool WriteBehind::needed(double now) {
  if (cache_.persistent())
    return true;
  if (const uint64_t reclaimed = cache_.reclaimedToDisk(); reclaimed != reclaimedToDisk_) {
    reclaimedToDisk_ = reclaimed;
    shortAt_ = now;
  }
  needed_ = shortAt_ && now - *shortAt_ < kNeedMilliseconds;
  return needed_;
}

bool WriteBehind::inUse(uint64_t block) const noexcept {
  return !cache_.persistent() && cache_.stateInUse(block);
}

bool WriteBehind::run(double now, bool, bool claimWaiting) {
  stalled_ = false;
  if (!needed(now) || claimWaiting)
    return false;
  bool settled = false;
  for (auto point = waiting_.begin(); point != waiting_.end() && point->due <= now;) {
    if (inUse(point->block)) {
      ++point;
      continue;
    }
    if (!settle(cache_.persist(point->block))) {
      stalled_ = true;
      break;
    }
    queued_.erase(point->block);
    point = waiting_.erase(point);
    settled = true;
  }
  return settled;
}

bool WriteBehind::flush() {
  if (!cache_.persistent())
    return true;
  cache_.liftDiskWriteBudget();
  while (!waiting_.empty()) {
    const uint64_t block = waiting_.back().block;
    if (!settle(cache_.persist(block)))
      return false;
    queued_.erase(block);
    waiting_.pop_back();
  }
  return true;
}

std::optional<double> WriteBehind::nextWakeup() const {
  // A stalled step goes on when the writes or restores it waits for land,
  // which wakes the engine, or once the write budget has room again, which
  // does not. A point in use waits for its request's end, a step of the
  // engine, which runs this again.
  if (!needed_)
    return std::nullopt;
  for (const Point &point : waiting_)
    if (!inUse(point.block))
      return stalled_ ? cache_.diskWritesAdmittedAt() : point.due;
  return std::nullopt;
}

WriteBehindSnapshot WriteBehind::snapshot() const noexcept {
  WriteBehindSnapshot result = counters_;
  result.waiting = static_cast<uint32_t>(waiting_.size());
  return result;
}

} // namespace splash::engine
