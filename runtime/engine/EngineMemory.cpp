#include "engine/Engine.hpp"
#include "engine/EngineAllocate.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace splash::engine {

std::optional<double> Engine::memoryAwaitedUntil(const Request &waiter) const {
  const ResourceWait &wait = waiter.resourceWait;
  if (!wait.pending || !wait.startedMilliseconds)
    return std::nullopt;
  const bool retried = waiter.lane ? !belowRunnable(waiter)
                                   : admissionTries(waiter, suspendedTier(), drainingForRecovery()) &&
                                         !holdsBack(waiter.heldBehind);
  if (!retried)
    return std::nullopt;
  return *wait.startedMilliseconds + config_.resourceWaitTimeoutMilliseconds;
}

bool Engine::awaitsMemory(const Request &waiter, double now) const {
  const std::optional<double> until = memoryAwaitedUntil(waiter);
  return until && now < *until;
}

bool Engine::memoryAwaited(double now) const {
  return std::ranges::any_of(requests_,
                             [&](const auto &entry) { return awaitsMemory(entry.second, now); });
}

bool Engine::holdsBack(std::optional<uint64_t> refused) const {
  if (!refused)
    return false;
  const auto found = requests_.find(*refused);
  return found != requests_.end() && !found->second.finalized && found->second.refusedResources;
}

ResourceWaitSnapshot Engine::resourceWaitSnapshot(double now) const {
  ResourceWaitSnapshot result;
  result.draining = drainingForRecovery();
  for (const auto &[id, active] : requests_) {
    if (active.finalized)
      continue;
    // Admitted, it waits for disk reads, not for memory.
    if (active.restore) {
      ++result.restoring;
      continue;
    }
    const bool waiting = scheduler_.phase(id) == Phase::WaitingResources;
    // What the latest admission pass held back, in whatever phase it left
    // them, while the request they wait behind is still refused, and a
    // request refused resources while a pass keeps it out of its wait.
    if (holdsBack(active.heldBehind) || (active.refusedResources && !waiting))
      ++result.heldBehindRefusal;
    if (!waiting)
      continue;
    if (active.resourceWait.reason == StateFailure::ConcurrencyLimit)
      ++result.concurrency;
    else if (active.resourceWait.diskQuota)
      ++result.diskQuota;
    else
      ++result.memory;
    if (scheduler_.suspended(id))
      ++result.suspended;
    if (active.resourceWait.startedMilliseconds)
      result.oldestWaitMilliseconds = std::max(
          result.oldestWaitMilliseconds, now - *active.resourceWait.startedMilliseconds);
  }
  return result;
}

uint64_t Engine::projectedTokens(const Request &active) noexcept {
  const uint64_t generated = active.exactTokens.size() - active.promptTokens;
  const uint64_t remaining = active.outputLimit - std::min<uint64_t>(generated, active.outputLimit);
  return active.exactTokens.size() + std::min<uint64_t>(remaining, kProjectedOutputTokens);
}

KvMode Engine::kvModeFor(const Request &active) const {
  const EngineSpillConfig &spill = config_.spill;
  if (!spill.capable)
    return KvMode::Resident;
  if (spill.always)
    return KvMode::Spill;
  const uint64_t tokens = projectedTokens(active);
  if (tokens <= spill.denseTokens)
    return KvMode::Resident;
  uint64_t bytes = spill.geometry.residentBytes(tokens);
  for (const auto &[id, peer] : requests_) {
    if (!peer.lane || peer.pendingEnd || id == active.request.id)
      continue;
    const uint64_t projection = projectedTokens(peer);
    bytes += peer.kvMode == KvMode::Spill ? spill.geometry.spillFloorBytes(projection)
                                          : spill.geometry.residentBytes(projection);
  }
  return bytes <= spill.geometry.residentKvBytes ? KvMode::Resident : KvMode::Spill;
}

bool Engine::fitsResidentAlone(const Request &active) const {
  return projectedTokens(active) <= config_.spill.geometry.residentTokens;
}

uint64_t Engine::chainEnd(const CacheLookup &lookup) noexcept {
  return lookup.state ? lookup.state->kvBlock() : 0;
}

uint64_t Engine::chainEnd(const Request &active) const {
  const uint64_t id = active.request.id;
  const uint32_t held = scheduler_.phase(id) == Phase::Prefill ? scheduler_.promptProcessed(id)
                                                                : active.publishedTokens;
  return held < KvCache::pageTokens
             ? 0
             : cache_.blockAt(id, held / KvCache::pageTokens * KvCache::pageTokens);
}

uint32_t Engine::spillPages(const Request &active, uint64_t chainEnd) const {
  const uint64_t pages = projectedTokens(active) / KvCache::pageTokens;
  return static_cast<uint32_t>(pages - std::min<uint64_t>(pages, cache_.copiedBlocks(chainEnd)));
}

Engine::SpillAdmission Engine::admitSpill(Request &active, bool resuming, uint64_t chainEnd,
                                          double now) {
  const SpillGeometry &geometry = config_.spill.geometry;
  // What it would do without the SSD cache: start Resident if its
  // projection fits in memory alone, and wait for the other lanes there.
  const auto residentInstead = [&] {
    if (!fitsResidentAlone(active))
      return false;
    active.kvMode = KvMode::Resident;
    return true;
  };
  // A resume goes on spilling whatever the tier and the write budget: its
  // writes are in use, and pages a closed tier does not take stay Full.
  if (!resuming) {
    if (const SpillGate gate = cache_.spillGate(); gate != SpillGate::Open) {
      if (residentInstead())
        return SpillAdmission::Resident;
      const std::string reason = "requests past " + std::to_string(geometry.residentTokens) +
                                 " tokens keep their KV on the SSD cache, ";
      if (gate == SpillGate::WriteBudget) {
        ++counters_.spill.refusedWriteBudget;
        // It refills over hours: a retry may succeed once it covers again
        // what Spill requests committed, or sooner as they end.
        const double admitted = cache_.spillWritesAdmittedAt().value_or(now);
        settle(active, {LaneOutcome::KvWriteBudget,
                        reason + "whose daily write budget (--max-cache-disk-writes) is spent",
                        static_cast<uint32_t>(std::lround(std::max(0.0, admitted - now) / 1000.0))});
      } else {
        ++counters_.spill.refusedClosed;
        settle(active, {LaneOutcome::CapacityExhausted, reason + "which takes no writes"});
      }
      return SpillAdmission::Settled;
    }
  }
  const uint64_t room = cache_.spillRoom() / geometry.slotBytes;
  if (room >= spillPages(active, chainEnd))
    return SpillAdmission::Commit;
  // Short of its projection, a start that fits in memory alone starts
  // Resident, as without the SSD cache, unless every request spills.
  if (!resuming && !config_.spill.always && residentInstead())
    return SpillAdmission::Resident;
  // Otherwise the quota bounds its output to the pages it holds, if they
  // hold a token past the tokens the request holds; those of the chain it
  // restores that have copies need none. A resume counts among them the
  // anchor its model holds once it selected the request's first token
  // (Request::suspendedWithAnchor), which the model decodes only with
  // output left past it: it emits a last budgeted anchor as it selects it.
  const uint64_t held = active.exactTokens.size() + (active.suspendedWithAnchor ? 1 : 0);
  const uint64_t copied = cache_.copiedBlocks(chainEnd);
  const uint64_t needed = held / KvCache::pageTokens + 1 - copied;
  if (room >= needed) {
    active.outputLimit = static_cast<uint32_t>(std::min<uint64_t>(
        active.outputLimit, (copied + room) * KvCache::pageTokens - active.promptTokens));
    return SpillAdmission::Commit;
  }
  // What it lacks is other requests' active KV, the copies of the blocks
  // they hold and their commitments: it waits for them to end. The quota
  // cannot hold it even alone: a start is refused, and a resume that has no
  // room left for output ends.
  const uint64_t activeLimitBytes = cache_.spillLimit();
  if (activeLimitBytes / geometry.slotBytes >= needed) {
    waitForDiskQuota(active, now);
    return SpillAdmission::Waiting;
  }
  if (resuming) {
    endAtLength(active);
  } else {
    ++counters_.spill.refusedQuota;
    settle(active, {LaneOutcome::KvDiskQuota,
                    "this request's KV needs needed_bytes=" +
                        std::to_string(needed * geometry.slotBytes) +
                        " of the SSD cache, which may hold active_limit_bytes=" +
                        std::to_string(activeLimitBytes) +
                        " of active KV now (--max-cache-disk and the volume's free space)"});
  }
  return SpillAdmission::Settled;
}

void Engine::waitForDiskQuota(Request &active, double now) {
  active.refusedResources = true;
  scheduler_.waitForResources(active.request.id);
  deferResourceRetry(active, now, {.diskQuota = true});
}

bool Engine::convertToSpill(Request &active, const CacheLookup *restoring) {
  if (!config_.spill.capable || active.kvMode != KvMode::Resident || betweenPasses(active.request.id))
    return false;
  const uint64_t end = restoring ? chainEnd(*restoring) : chainEnd(active);
  if (!cache_.convertToSpill(active.request.id, spillPages(active, end), !commandShaped_))
    return false;
  active.kvMode = KvMode::Spill;
  // A start that converts starts as a Spill request, which completeAdmission
  // counts, and one refused all the same decides its mode afresh at its next
  // attempt; a lane that ran Resident before converts once.
  counters_.spill.converted += !restoring || scheduler_.suspended(active.request.id);
  return true;
}

bool Engine::spillFrees(const Request &active) const {
  const SpillGeometry &geometry = config_.spill.geometry;
  const uint64_t held = kvCapacity(active.request.id);
  return geometry.residentBytes(held) > geometry.spillFloorBytes(held) + geometry.laneReadBytes;
}

bool Engine::convertLargestResidentLane() {
  if (!config_.spill.capable)
    return false;
  // A lane being admitted, its chain's restore in flight or its start not
  // complete, holds the chain it restores, which only its admission's own
  // conversion counts (spillFallback's `restoring`).
  const auto admitted = [&](uint64_t id, const Request &active) {
    const Phase phase = scheduler_.phase(id);
    return active.lane && !active.restore && (phase == Phase::Prefill || phase == Phase::Decode);
  };
  std::vector<std::pair<uint64_t, Request *>> lanes;
  for (auto &[id, active] : requests_)
    if (admitted(id, active) && !active.finalized && active.kvMode == KvMode::Resident &&
        spillFrees(active))
      lanes.emplace_back(kvCapacity(id), &active);
  std::ranges::sort(lanes, std::greater{}, &std::pair<uint64_t, Request *>::first);
  return std::ranges::any_of(lanes, [&](const auto &lane) { return convertToSpill(*lane.second); });
}

std::function<bool(const Engine::Denial &)> Engine::spillFallback(Request &active,
                                                                  const CacheLookup *restoring) {
  if (!config_.spill.capable)
    return {};
  // Memory on its way back comes without it.
  return [this, &active, restoring](const Denial &denial) {
    return !denial.pending && convertToSpill(active, restoring);
  };
}

bool Engine::spillingPrompt() const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    const auto &[id, active] = entry;
    return active.lane && active.kvMode == KvMode::Spill && scheduler_.phase(id) == Phase::Prefill;
  });
}

bool Engine::laneInService(std::optional<KvMode> mode) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.second.lane && (!mode || entry.second.kvMode == *mode);
  });
}

SpilledExtents Engine::landedExtents() const {
  return spillingPrompt() || config_.growthPaused() ? SpilledExtents::KeepRunway
                                                    : SpilledExtents::Released;
}

void Engine::spillLanded() {
  if (!config_.spill.capable)
    return;
  // The Index runway the pages draw on refills if memory allows. The Index
  // pages, the runway's included, are memory the resident Spill lanes need,
  // as their Full pages are, and a fraction of what the moves free: the
  // host's pause holds them back only while no such lane is in service.
  const SpilledExtents extents = landedExtents();
  static_cast<void>(allocate([&] { return cache_.spillWritten(extents); },
                             laneInService(KvMode::Spill), ReclaimClass::Ordinary, {}, true));
}

void Engine::endAtLength(Request &active) {
  ++counters_.spill.lengthEnds;
  finish(active, EngineFinishReason::Length, {});
}

bool Engine::resourceRetryReady(const Request &active,
                                double now) const noexcept {
  return active.resourceWait.retryMilliseconds <= 0.0 ||
         active.resourceWait.epoch != resourceEpoch_ ||
         now >= active.resourceWait.retryMilliseconds;
}

void Engine::deferResourceRetry(Request &active, double now,
                                const Denial &denial, StateFailure reason) noexcept {
  auto &wait = active.resourceWait;
  if (!wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  // Waiting for a lane is not waiting for memory: the lanes that count are
  // those admitted before the first refusal of memory.
  if (!wait.admittedBefore && reason != StateFailure::ConcurrencyLimit)
    wait.admittedBefore = admissions_;
  noteRefusal(active, now, denial);
  wait.reason = reason;
  if (reason == StateFailure::ConcurrencyLimit)
    wait.deadlineMilliseconds = 0.0;
  else if (wait.deadlineMilliseconds <= 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  wait.retryMilliseconds = now + kResourceRetryBackoffMilliseconds;
}

void Engine::noteRefusal(Request &active, double now, const Denial &denial) noexcept {
  auto &wait = active.resourceWait;
  if (wait.pending && wait.epoch != resourceEpoch_ && wait.deadlineMilliseconds > 0.0)
    wait.deadlineMilliseconds = now + config_.resourceWaitTimeoutMilliseconds;
  if (denial.pending && !wait.startedMilliseconds)
    wait.startedMilliseconds = now;
  wait.allocationFailure = denial.allocationFailure;
  wait.pending = denial.pending;
  wait.diskQuota = denial.diskQuota;
  wait.epoch = resourceEpoch_;
}

void Engine::rewaitHeld(double now) noexcept {
  for (const uint64_t id : heldWaits_) {
    const auto found = requests_.find(id);
    if (found == requests_.end() || !found->second.lane)
      continue;
    Request &held = found->second;
    const ResourceWait &wait = held.resourceWait;
    if (wait.retryMilliseconds > 0.0)
      deferResourceRetry(held, now,
                         {.allocationFailure = wait.allocationFailure,
                          .pending = wait.pending,
                          .diskQuota = wait.diskQuota},
                         wait.reason);
  }
  heldWaits_.clear();
}

void Engine::deferWait(Request &active) noexcept {
  ResourceWait kept;
  if (active.refusedResources) {
    kept.startedMilliseconds = active.resourceWait.startedMilliseconds;
    kept.admittedBefore = active.resourceWait.admittedBefore;
  }
  active.resourceWait = kept;
}

double Engine::resourceDeadline(const Request &active) const noexcept {
  const ResourceWait &wait = active.resourceWait;
  if ((wait.pending && wait.epoch != resourceEpoch_) || wait.deadlineMilliseconds <= 0.0)
    return 0.0;
  // The limit restarts whenever a lane submitted before the request, or
  // admitted before it was refused memory or suspended, works, however long
  // that takes. Other lanes do not extend it: requests that keep arriving
  // would otherwise hold it until the request's deadline.
  return std::max(wait.deadlineMilliseconds,
                  wait.earlierLaneWorkMilliseconds + config_.resourceWaitTimeoutMilliseconds);
}

void Engine::signalResourceProgress() noexcept {
  if (resourceEpoch_ != std::numeric_limits<uint64_t>::max())
    ++resourceEpoch_;
}

Engine::LaneMemory Engine::growLane(Request &active, uint64_t workEnd, const PlanContext &plan) {
  const uint64_t id = active.request.id;
  if (workEnd > kvCapacity(id) && heldForWaiter(active, plan.now))
    return {.held = true};
  const auto kv = allocate([&] { return cache_.ensureTokens(id, workEnd); }, true,
                           growthClass(active), spillFallback(active));
  if (kv.admission.granted())
    return {};
  return {.denied = Denied{id, workEnd, kv.denial, kv.admission, {}}};
}

Engine::LaneMemory Engine::reserveReads(Request &active, const BatchItem &scheduled,
                                        const PlanContext &plan) {
  const uint64_t id = active.request.id;
  const auto pages = static_cast<uint32_t>(
      (uint64_t{scheduled.promptOffset} + scheduled.tokenCount + KvCache::pageTokens - 1) /
      KvCache::pageTokens);
  if (!model_.holdsSpilledReads(id, plan.prefill ? pages : 0) && heldForWaiter(active, plan.now))
    return {.held = true};
  const auto memory = allocate(
      [&] {
        return plan.prefill ? model_.reserveHistoryStaging(id, pages)
                            : model_.reserveSpilledReads(id);
      },
      true, growthClass(active));
  if (memory.admission.granted())
    return {};
  return {.denied = Denied{id, workEndOf(active, scheduled, plan.prefill), memory.denial,
                           std::nullopt,
                           plan.prefill ? "prefill history staging" : "hot KV cache"}};
}

Engine::Prepared Engine::refuse(std::span<const Denied> denied, bool heldBack,
                                const PlanContext &plan) {
  const bool spill = config_.spill.capable;
  // Partial admissions execute at their actual width. If no lane fits, the
  // lane to yield is chosen among all runnable residents: an unstarted peer
  // can release its lane before completed prefill is discarded. Lanes held
  // back for a lane that waits for memory only leave the plan for this tick.
  if (denied.empty()) {
    if (heldBack)
      return Prepared::Waiting;
    throw std::logic_error("empty resource admission result");
  }
  // No lane runs: the copies Spill requests owe and have not started, as a
  // chain a lane restored has before its first command or those the memory
  // governor's pass left when it collected the ones that landed, start now.
  // A lane alone waits while they are written (judge).
  if (spill)
    static_cast<void>(cache_.writeThrough());
  // Memory on its way back arrives without anyone yielding. The lanes still
  // take a retry deadline: the transfer's completion wakes the engine, and
  // the deadline is what makes the wait end if that wake is ever missed.
  const auto wait = [&] {
    for (const Denied &entry : denied)
      deferResourceRetry(request(entry.requestId), plan.now, entry.denial);
    return Prepared::Waiting;
  };
  if (std::any_of(denied.begin(), denied.end(),
                  [](const Denied &entry) { return entry.denial.pending; }))
    return wait();
  const Denied &victim = *std::min_element(
      denied.begin(), denied.end(),
      [&](const Denied &left, const Denied &right) {
        return yieldsBefore(request(left.requestId), request(right.requestId));
      });
  Request *const yielding = laneToYield();
  if (!yielding)
    throw std::logic_error("no resident lane can yield");
  Request &active = *yielding;
  const auto own = std::find_if(denied.begin(), denied.end(), [&](const Denied &entry) {
    return entry.requestId == active.request.id;
  });
  // A peer that has not failed a growth attempt resumes to its current KV
  // capacity, not to the blocked lane's requirement.
  const uint64_t resumeTarget =
      own != denied.end() ? own->workEnd : kvCapacity(active.request.id);
  switch (judge(victim.denial, active.request.id)) {
  case Verdict::Wait:
    // Alone, it waits for the copies owed, retrying as they land, within the
    // resource wait its first refusal started: copies that never land, or
    // keep failing, end it there.
    return wait();
  case Verdict::Yield:
    return yieldLane(active, resumeTarget, victim.denial.allocationFailure, plan.now);
  case Verdict::Fail:
    break;
  }
  // Alone with nothing left to reclaim or write. A Spill lane that decodes
  // ends at its length once the SSD cache can hold no more of its KV.
  Request &lone = request(victim.requestId);
  if (!victim.kv)
    settle(lone, capacityExhausted(victim.memory, victim.denial.allocationFailure));
  else if (lone.kvMode == KvMode::Spill && !plan.prefill)
    endAtLength(lone);
  else
    settle(lone, capacityExhausted("KV target", victim.denial.allocationFailure,
                                   pageShortfall(*victim.kv) +
                                       (spill ? "; the SSD cache took no more of its KV" : "")));
  return Prepared::Yielded;
}

ReclaimClass Engine::growthClass(const Request &active) const {
  return &active == laneToYield() && anotherResident(active.request.id) ? ReclaimClass::Ordinary
                                                                        : ReclaimClass::InUse;
}

bool Engine::heldForWaiter(const Request &active, double now) const {
  return std::ranges::any_of(requests_, [&](const auto &entry) {
    return &entry.second != &active && entry.second.lane &&
           awaitsMemory(entry.second, now) && yieldsBefore(active, entry.second);
  });
}

uint32_t Engine::reserveCaptures(Request &active, const BatchItem &item,
                                 std::vector<StateCapture> &states,
                                 std::vector<Capture> &captures, bool held) {
  const uint32_t start = item.promptOffset;
  // A request waiting for a prefix starts once the command that reaches its
  // state has ended.
  const uint32_t end = awaitedBoundary(active, start + item.tokenCount);
  const std::vector<Request::StateBoundary> &boundaries = active.stateBoundaries;
  const auto stateClass = [&](size_t index) {
    return boundaries[index].disposable                               ? StateClass::Checkpoint
           : boundaries[index].tokens == promptReplayBoundary(active) ? StateClass::InUse
                                                                      : StateClass::Ordinary;
  };
  // Takes the buffers of the state at boundaries[index]; false when refused.
  // Only the first state the command captures follows the lane's
  // checkpoint, and each later one the state before it.
  const auto reserve = [&](size_t index) {
    const uint32_t tokens = boundaries[index].tokens;
    const StateClass kind = stateClass(index);
    if (kind == StateClass::InUse && !states.empty() && boundaries[index - 1].disposable) {
      // The prompt's replay point keeps the checkpoint the command captures
      // before it beside buffers of its own, and takes that checkpoint's
      // when refused the pool and the admission, as it takes its lane's
      // after a command.
      metal::AllocationFailure refusal;
      if (std::shared_ptr<StateReservation> buffers =
              model_.reserveState(active.request.id, tokens, refusal)) {
        states.push_back({tokens, std::move(buffers)});
        captures.push_back({tokens});
      } else {
        states.back().tokens = tokens;
        captures.back().tokens = tokens;
      }
      return true;
    }
    Reservation reservation = reserveCapture(active, kind, tokens, states.empty());
    if (!reservation.buffers) {
      if (reservation.fromLane)
        captures.push_back({tokens, false, true});
      return false;
    }
    states.push_back({tokens, std::move(reservation.buffers)});
    captures.push_back({tokens, reservation.recycled});
    return true;
  };
  for (size_t index = active.stateBoundaryCursor;
       index < boundaries.size() && boundaries[index].tokens <= end; ++index) {
    const uint32_t tokens = boundaries[index].tokens;
    // Published, every state but the prompt's replay point retires the
    // checkpoint before it (apply): a checkpoint the command passes on its
    // way to such a state is not captured, and that state takes buffers of
    // its own class.
    const bool passed = boundaries[index].disposable && index + 1 < boundaries.size() &&
                        boundaries[index + 1].tokens <= end &&
                        stateClass(index + 1) != StateClass::InUse;
    if (!passed && !reserve(index))
      return tokens - start;
    if (tokens == end)
      break;
    // The KV up to the next boundary, as prepare() takes the first, but its
    // refusal only ends the command here, which then captures a checkpoint
    // it would have passed: the lane runs, and its next command asks again.
    // So it takes nothing in use, the host refusing it or not.
    const uint32_t next = index + 1 < boundaries.size() && boundaries[index + 1].tokens < end
                              ? boundaries[index + 1].tokens
                              : end;
    // A held lane (prepare) takes no KV pages past its capacity.
    if (held && next > kvCapacity(active.request.id)) {
      if (passed)
        static_cast<void>(reserve(index));
      return tokens - start;
    }
    const auto kv = allocate(
        [&] { return cache_.ensureTokens(active.request.id, next); }, true, ReclaimClass::Ordinary,
        {}, true);
    if (!kv.admission.granted()) {
      if (passed)
        static_cast<void>(reserve(index));
      return tokens - start;
    }
  }
  return end - start;
}

Engine::Reservation Engine::reserveCapture(Request &active, StateClass kind, uint32_t tokens,
                                           bool followsCheckpoint) {
  const uint64_t id = active.request.id;
  metal::AllocationFailure refusal = metal::AllocationFailure::None;
  Reservation result{model_.reserveState(id, tokens, refusal)};
  if (result.buffers)
    return result;
  // The lane's checkpoint is its recovery point until a state that retires
  // it is published, after a command that completed: one cancelled, expired
  // or failed leaves it to a retry. So no state takes its buffers before the
  // command, nor room a reclaim could make of it, since checkpoints go
  // first. Refused the pool and the admission while the lane holds one, the
  // state ends its command, and apply takes it from the lane after the
  // command: a junction or a checkpoint in the buffers of the checkpoint it
  // retires (publishReachedStateBoundaries). A checkpoint whose predecessor
  // a reader pins takes no room either: apply publishes it, from the lane,
  // only once the pin has gone, which keeps the predecessor the lane's
  // recovery point until then.
  const bool predecessorPinned =
      kind == StateClass::Checkpoint && followsCheckpoint && checkpointPinned(active);
  result.fromLane = predecessorPinned || checkpointReclaimable(active);
  if (result.fromLane)
    return result;
  // Room comes from what this state's class may take: cached KV unless it
  // is an optional checkpoint, and, only for the prompt's replay point,
  // which is in use itself, the cache that needs a copy on the SSD cache
  // first, escalate()'s rungs short of what is in use (makeStateRoom), then
  // states in use. A pin keeps the lane's checkpoint, if any, out of it.
  result.recycled = makeStateRoom(kind, [&] {
    result.buffers = model_.reserveState(id, tokens, refusal);
    return result.buffers != nullptr;
  });
  // The prompt's replay point is work of the request in service, the point
  // its follow-up resumes from: once nothing its class may take is left,
  // the host refuses it no more than that request's KV (allocate()). A
  // junction or a checkpoint stays optional.
  if (!result.buffers && kind == StateClass::InUse &&
      refusal == metal::AllocationFailure::HostPressure) {
    const Serving mark(config_.serving, true);
    result.buffers = model_.reserveState(id, tokens, refusal);
  }
  return result;
}

Engine::ClaimHandle::ClaimHandle(ClaimHandle &&other) noexcept
    : governor(other.governor), id(std::exchange(other.id, {})) {}

Engine::ClaimHandle::~ClaimHandle() {
  if (id)
    governor->ledger().close(id);
}

ClaimRank Engine::rankOf(const Request &active) const {
  const Phase phase = scheduler_.phase(active.request.id);
  return {.priority = static_cast<uint8_t>(static_cast<uint8_t>(RequestPriority::Background) -
                                           static_cast<uint8_t>(active.request.priority)),
          // At equal priority, prefer uninterrupted streaming over less
          // replay work.
          .decoding = phase == Phase::Decode || phase == Phase::WaitingMask,
          .completedTokens = completedTokens(active),
          .earliness = std::numeric_limits<uint64_t>::max() - scheduler_.submissionOrder(active.request.id)};
}

ClaimSpec Engine::claimFor(const Request &active, ClaimClass cls) const {
  return {.cls = cls,
          .requestId = active.request.id,
          .rank = rankOf(active),
          .reach = cls == ClaimClass::InService ? growthClass(active) : ReclaimClass::Ordinary,
          .priority = active.request.priority};
}

bool Engine::yieldsBefore(const Request &a, const Request &b) const {
  const ClaimRank left = rankOf(a);
  const ClaimRank right = rankOf(b);
  return std::tie(left.priority, left.decoding, left.completedTokens) <
         std::tie(right.priority, right.decoding, right.completedTokens);
}

const Engine::Request *Engine::laneToYield() const {
  const Request *yielding = nullptr;
  for (const auto &[id, candidate] : requests_) {
    if (!candidate.lane)
      continue;
    // Requests enter the scheduler before they can acquire a lane.
    const Phase phase = scheduler_.phase(id);
    if (phase != Phase::Prefill && phase != Phase::Decode)
      continue;
    if (!yielding || rankOf(candidate) < rankOf(*yielding))
      yielding = &candidate;
  }
  return yielding;
}

uint64_t Engine::kvCapacity(uint64_t requestId) const {
  return uint64_t{cache_.pageTable(requestId).pages.size()} * KvCache::pageTokens;
}

bool Engine::preemptBelow(RequestPriority priority, uint64_t start, double now) {
  // The lane that yields first has the lowest priority of all residents.
  Request *const victim = laneToYield();
  if (!victim || victim->request.priority <= priority)
    return false;
  if (yieldsEndedLanes(*victim))
    return true;
  // Suspended, a lane of the command between its passes would give that
  // command up all the same: for a start whose prompt would not supersede
  // it, the start waits for its end instead, which runs at its priority.
  if (betweenPasses(victim->request.id) &&
      !scheduler_.yieldsTo(priority, 1, {.shapes = laneShapes(), .passes = passProgress()})) {
    passCommand_->progress.heldStarts.push_back(start);
    return false;
  }
  suspendLane(*victim, kvCapacity(victim->request.id), metal::AllocationFailure::None,
              StateFailure::ConcurrencyLimit, now);
  // What the lane held is free: an earlier failure may fit now.
  allocationFailed_ = false;
  ++counters_.prioritySuspensions;
  signalResourceProgress();
  return true;
}

bool Engine::anotherResident(uint64_t requestId) const {
  return std::any_of(requests_.begin(), requests_.end(), [&](const auto &entry) {
    return entry.first != requestId && entry.second.lane;
  });
}

Engine::Verdict Engine::judge(const Denial &denial, uint64_t requestId) const {
  if (denial.pending)
    return Verdict::Wait;
  // Pages held by resident lanes come back when they finish, and memory the
  // host refuses when its pressure lifts; only a lane that cannot fit on its
  // own has hit the capacity.
  if (anotherResident(requestId) ||
      denial.allocationFailure == metal::AllocationFailure::HostPressure)
    return Verdict::Yield;
  // Alone, it waits for the copies Spill requests owe, the slowest memory
  // to come back: they free its Full pages as the SSD cache writes them.
  if (cache_.owedCopiesReturn(denial.pages, denial.form))
    return Verdict::Wait;
  return Verdict::Fail;
}

// The rungs a refusal of the engine's limit climbs once the cache nothing in
// use holds gave nothing: each is tried only when the ones below it gave
// nothing, the same for every allocation the limit refuses (a lane's state,
// its KV, the model's memory for reading spilled KV, the prompt's replay
// point) and for a pass the host is short for (reclaimMemory). They rank what
// a byte is worth where it is:
// - For work that supersedes the command between its passes (superseding_:
//   a plan that supersedes it, or a start whose prompt would), that command,
//   abandoned as it would be once that work runs: it gives back the buffers
//   of the states it would capture and the lanes of it that ended.
// - On a target whose KV can live on the SSD cache, a Resident lane converts
//   to Spill (convertToSpill): beyond its dense and window pages a page then
//   keeps only its index in memory, and the rest comes back as the copies
//   land. The requester's own lane goes first (`convertRequester`): that
//   allocates nothing, and the allocation is tried again. While the copies
//   Spill lanes owe return an extent's Full pages as they land
//   (Cache::owedCopiesReturn), that memory is on its way (pending), and
//   nothing more converts or goes for it. Otherwise the lane holding the
//   most KV converts (convertLargestResidentLane), its memory on its way
//   too: its pages leave memory as their copies land, at the next point
//   between commands (spillLanded), and an attempt that finds it fell
//   short climbs on, to the next lane and then the store, so each lane that
//   converts delays the store by one attempt at most. A target whose KV
//   stays in memory skips this rung, and so does one whose lanes would give
//   back less than spilling costs them.
// - Then one chunk of the expert store above its floor, which every decode
//   step of a streamed MoE target reads.
// - Then what is in use (`takeInUse`), which only work that would not yield
//   for it may take.
CacheReclaimResult Engine::escalate(const std::function<bool()> &convertRequester,
                                    const std::function<CacheReclaimResult()> &takeInUse) {
  if (superseding_ && passCommand_) {
    abandonPasses();
    return {.madeProgress = true};
  }
  if (convertRequester && convertRequester())
    return {.madeProgress = true};
  if (cache_.owedCopiesReturn(0, kv::PageForm::Full) || convertLargestResidentLane())
    return {.pending = true};
  if (const uint64_t chunk = releaseExpertChunk())
    return {.madeProgress = true, .reclaimedBytes = chunk};
  return takeInUse ? takeInUse() : CacheReclaimResult{};
}

// The reclaim step for a lane's state, or the model's memory for reading
// spilled KV, the engine's limit refused. The pooled buffers a lane starts
// from stay for its activation to take: idle model memory beyond them goes
// first (a pooled buffer, else the idle vision arena, else one rebuildable
// cache entry), then the cache nothing in use holds gives up what gives
// memory back: an extent, or a state (Cache::reclaimForMemory), starting the
// copies to the SSD cache what it takes needs, whose memory is then on its
// way back. The KV it evicts before then frees pages other KV may take, so it
// wakes the requests waiting for memory as progress does. Then the rungs
// above it (escalate), what is in use only up to InUse; memory on its way
// back holds them back.
// Memory for later (the hot block cache's growth toward its planned size,
// any optional allocation) takes no rebuildable cache entry, among which
// that very cache's slots past its floor are, nor the empty extent kept for
// the next request, nor any rung above the cache nothing in use holds.
// A host refusal comes to neither step but to the reuse path (allocate()):
// the pressure controller owns that shrink, and evicting for an allocator
// that refuses all the same would drain the cache, and the caches the model
// can rebuild, before macOS can acknowledge any reclaimed bytes.
CacheReclaimResult Engine::reclaimForState(ReclaimClass upTo, bool forLater) {
  if (reclaimIdleState(true, forLater ? model::IdleMemory::Buffers
                                      : model::IdleMemory::BuffersThenCaches))
    return {true, 0};
  CacheReclaimResult reclaimed = cache_.reclaimForMemory(ReclaimClass::Ordinary, forLater);
  if (!reclaimed.madeProgress && !reclaimed.pending && !forLater) {
    const bool evictedKv = reclaimed.evictedKv;
    reclaimed = escalate({}, [&] {
      return upTo == ReclaimClass::InUse ? cache_.reclaimForMemory(ReclaimClass::InUse, forLater)
                                         : CacheReclaimResult{};
    });
    reclaimed.evictedKv = reclaimed.evictedKv || evictedKv;
  }
  if (reclaimed.madeProgress || reclaimed.evictedKv)
    signalResourceProgress();
  return reclaimed;
}

// The reclaim step for KV pages the engine's limit refused. Allocated extents
// stay for the pages to reuse; idle state memory beyond the lane's pooled
// buffers goes first, then the cache nothing in use holds gives up what
// covers the shortfall, of the pages' form, in one step, starting the copies
// what it takes needs; then the rungs above it (escalate), the requester's
// own conversion to Spill ahead of another lane's (`convertRequester`), what
// is in use only up to InUse. Memory on its way back holds the rungs back,
// and so does an optional allocation (a capture's KV past its first
// boundary, whose refusal only ends the command early). The lane's buffers go last. Their draft ring
// is the one buffer no cached state holds: while it stays, a request that
// waits can start beside this one from cached states' cells and windows.
CacheReclaimResult Engine::reclaimForKv(const TokenAdmission &admission, ReclaimClass upTo,
                                        bool optional,
                                        const std::function<bool()> &convertRequester) {
  if (reclaimIdleState(true, model::IdleMemory::BuffersThenCaches))
    return {true, 0};
  const auto reclaimPages = [&](ReclaimClass reach) {
    return cache_.reclaimForPages(admission.additionalPages, CacheReclaimMode::KeepExtents, reach,
                                  admission.form);
  };
  CacheReclaimResult reclaimed = reclaimPages(ReclaimClass::Ordinary);
  if (!reclaimed.madeProgress && !reclaimed.pending && !optional)
    reclaimed = escalate(convertRequester, [&] {
      return upTo == ReclaimClass::InUse ? reclaimPages(ReclaimClass::InUse) : CacheReclaimResult{};
    });
  if (reclaimed.madeProgress)
    signalResourceProgress();
  if (!reclaimed.madeProgress && !reclaimed.pending &&
      reclaimIdleState(false, model::IdleMemory::Buffers))
    return {true, 0};
  return reclaimed;
}

uint64_t Engine::releaseExpertChunk() {
  const uint64_t bytes = model_.releaseExpertChunk();
  if (bytes)
    ++expertReleases_;
  return bytes;
}

uint64_t Engine::releaseExpertsToFloor() {
  const uint64_t bytes = model_.releaseExpertsToFloor();
  if (bytes)
    ++expertReleases_;
  return bytes;
}

bool Engine::growExperts(uint64_t headroomBytes) {
  if (pending_)
    throw std::logic_error("expert growth requested while a command is in flight");
  const uint64_t bytes = model_.expertGrowthBytes();
  // It takes only room the engine's limit has free, and asks for none it
  // lacks, which would count as a denied reservation for nothing. The
  // exchange alone makes that room (reclaimForExperts), which buys the
  // chunk or takes nothing.
  return bytes && headroomBytes >= bytes && model_.growExperts().granted();
}

uint64_t Engine::reclaimForExperts(uint64_t bytes) {
  if (pending_)
    throw std::logic_error("cache reclaim for the expert store requested while a command is in flight");
  // A request that holds no lane, yet to start or suspended, restores the
  // chain its probe finds once it starts: admission's, brought up to date,
  // or one made here where admission has made none. One that skips the cache
  // restores none.
  RestoredChains restored;
  for (auto &[_, active] : requests_) {
    if (active.lane || active.finalized || active.skipCache)
      continue;
    if (active.admissionProbe) {
      cache_.refresh(*active.admissionProbe, active.exactTokens, promptIdentity(active.request));
      restored.add(active.admissionProbe->restoredChain());
    } else {
      restored.add(cache_.probe(active.exactTokens, promptIdentity(active.request)).restoredChain());
    }
  }
  // Cache that cannot buy the chunk would go for nothing: as a lane takes a
  // cached state only when those in RAM cover what it lacks
  // (reuseCachedStateWhilePaused), nothing goes unless what leaves at once
  // covers the bytes, and then the bytes leave in this call.
  if (exchangeableBytes(restored) < bytes)
    return 0;
  uint64_t returned = 0;
  while (returned < bytes) {
    const CacheReclaimResult step = reclaimStepForExperts(restored);
    if (!step.madeProgress)
      break;
    returned += step.reclaimedBytes;
  }
  return returned;
}

uint64_t Engine::exchangeableBytes(const RestoredChains &restored) const {
  const CacheExchange cache = cache_.exchangeable(restored);
  return model_.idleBufferBytes(cache.states) + cache.extentBytes;
}

CacheReclaimResult Engine::reclaimStepForExperts(const RestoredChains &restored) {
  if (const uint64_t buffer = model_.reclaimIdleState(true, model::IdleMemory::Buffers)) {
    signalResourceProgress();
    return {.madeProgress = true, .reclaimedBytes = buffer};
  }
  return cache_.reclaimForExchange(restored);
}

bool Engine::reclaimIdleState(bool keepLane, model::IdleMemory scope) noexcept {
  if (!model_.reclaimIdleState(keepLane, scope))
    return false;
  signalResourceProgress();
  return true;
}

// While growth is paused a lane short of state buffers takes a cached
// state's, as a request short of pages takes idle cached pages below:
// evicting the state returns its cell and context window to the pool the
// lane draws from, and nothing is allocated. A state goes only when those in
// RAM cover what the pool lacks, which they never do for the draft rings no
// cached state holds; otherwise the cache survives, and the request grows if
// it is in service and waits if it is not. Idle buffers beyond the lane's go
// back to the host first, but no cache the model can rebuild: freeing one
// would not let the lane start, and the paced pass takes those caches toward
// what the host lacks.
CacheReclaimResult Engine::reuseCachedStateWhilePaused(ReclaimClass upTo) {
  if (reclaimIdleState(true, model::IdleMemory::Buffers))
    return {true, 0};
  const uint32_t lacked = model_.statesToActivate();
  if (!lacked || cache_.evictableStates(upTo) < lacked)
    return {};
  const CacheReclaimResult reused = cache_.reclaimStateForLane(upTo);
  if (reused.madeProgress)
    signalResourceProgress();
  return reused;
}

// Host pressure pauses growth, and the pressure controller owns the shrink.
// Extents that stay allocated are outside that accounting: a request short
// of pages takes idle cached pages before it grows or waits. Cache is only
// evicted when the pages of the shortfall's form the request's class may
// take can actually cover it: never those a request holds, nor, for a start
// a resident lane holds back, the idle KV that states in use restore through
// (Cache::reusablePages). Otherwise it survives for later hits, and the
// request grows if it is in service and waits if it is not. Only KV goes: a
// state's buffers give no page while the host refuses growth, unless the
// state sits on the leaf that goes next (CacheReclaimMode::ReusePages). A
// reclaim that must wait for the transfer in flight makes the request wait
// with it, as it does without the pause. Idle buffers go first, but neither
// the pooled buffers the next lane starts from nor the caches the model can
// rebuild: they would not let this request grow, and the paced pass keeps
// the buffers for the next lane and takes the caches toward what the host
// lacks.
CacheReclaimResult Engine::reuseCachedPagesWhilePaused(const TokenAdmission &admission,
                                                       ReclaimClass upTo) {
  if (reclaimIdleState(true, model::IdleMemory::Buffers))
    return {true, 0};
  if (cache_.reusablePages(upTo, admission.form) < admission.additionalPages)
    return {};
  const CacheReclaimResult reused = cache_.reclaimForPages(
      admission.additionalPages, CacheReclaimMode::ReusePages, upTo, admission.form);
  if (reused.madeProgress)
    signalResourceProgress();
  return reused;
}

void Engine::suspendLane(Request &active, uint64_t workEnd, metal::AllocationFailure failure,
                         StateFailure reason, double now) {
  if (!active.lane || scheduler_.suspended(active.request.id)) {
    throw std::logic_error("request cannot be suspended");
  }
  // A decoding lane holds an anchor unless its first mask has yet to select
  // one; a lane in prefill holds what it held when its replay began.
  if (scheduler_.phase(active.request.id) == Phase::Decode)
    active.suspendedWithAnchor =
        scheduler_.decodeStage(active.request.id) == DecodeStage::Regular;
  vacateLane(active, true);
  active.resumeKvTargetTokens = workEnd;
  active.replayTokens = static_cast<uint32_t>(active.exactTokens.size());
  scheduler_.suspendForResources(active.request.id);
  deferResourceRetry(active, now, {.allocationFailure = failure}, reason);
}

void Engine::suspendForGrowth(Request &active, uint64_t workEnd,
                              metal::AllocationFailure failure, double now) {
  suspendLane(active, workEnd, failure, StateFailure::MemoryPressure, now);
  // Without its lane it waits for the memory of every lane resident now,
  // also those admitted while it waited to grow.
  active.resourceWait.admittedBefore = admissions_;
  // Resident lanes drain before admission resumes. Growth the host refused
  // resumes when its pressure lifts; any other limit only once memory is
  // freed, so it counts as a failure the drain waits out.
  drainEndMilliseconds_ = now + config_.resourceWaitTimeoutMilliseconds;
  allocationFailed_ = failure != metal::AllocationFailure::HostPressure;
  ++counters_.resourceSuspensions;
}

MemoryReclaimResult Engine::reclaimMemory(const MemoryReclaimDirective &directive) {
  if (pending_)
    throw std::logic_error("memory reclaim requested while a command is in flight");
  if (cache_.pollTransfers())
    signalResourceProgress();

  const bool keep = !directive.critical;
  uint64_t released = 0;
  // A reclaim step may free memory that stays in the engine, pages of an
  // extent that stays or buffers that refill the lane's footprint, and a
  // waiting request may fit in it all the same.
  bool reclaimed = false;
  // Pages whose copies are being written count toward the target. A
  // critical directive has none: it takes everything it may.
  const auto targetUnmet = [&] {
    return directive.critical || released + cache_.pendingBytes() < directive.targetBytes;
  };
  // Evicted states park their buffers in the model's pool, which a pass
  // returns to the host at once, or uses to refill the buffers it keeps.
  const auto releaseIdle = [&] {
    while (const uint64_t idle = model_.reclaimIdleState(keep, model::IdleMemory::Buffers))
      released += idle;
  };
  releaseIdle();
  // The pages whose copies landed leave memory next, as before admission
  // (spillLanded), and what the KV pool gives back for them counts toward
  // the target. The pass spills them by its own rules: their Index pages
  // are memory a Spill lane in service needs, as there, but no other step
  // makes room for them, which would take what the pass keeps and count
  // none of it. Pages a refused extent leaves Full spill at the next tick.
  if (config_.spill.capable) {
    const uint64_t allocated = cache_.allocatedBytes();
    {
      const Serving mark(config_.serving, laneInService(KvMode::Spill));
      static_cast<void>(cache_.spillWritten(landedExtents()));
    }
    released += allocated - std::min(allocated, cache_.allocatedBytes());
  }
  // Caches the model can rebuild go only toward a byte target or under
  // critical pressure: an untargeted pass keeps the embedding rows and the
  // vision encoder.
  const auto reclaimModelCaches = [&] {
    while (targetUnmet()) {
      const uint64_t cache =
          model_.reclaimIdleState(keep, model::IdleMemory::BuffersThenCaches);
      if (!cache)
        break;
      released += cache;
    }
  };
  reclaimModelCaches();
  // Whether the pass stopped on memory on its way back rather than on nothing
  // left to take.
  bool pending = false;
  // The expert store's chunks above its floor go after what nothing in use
  // holds and before what is in use.
  if (directive.critical) {
    const auto evictAll = [&](ReclaimClass upTo) {
      const CacheReclaimResult evicted = cache_.evictAll(upTo);
      reclaimed = reclaimed || evicted.madeProgress;
      released += evicted.reclaimedBytes;
      releaseIdle();
    };
    evictAll(ReclaimClass::Ordinary);
    released += releaseExpertsToFloor();
    evictAll(ReclaimClass::InUse);
  } else {
    // Even a zero-byte directive releases completely empty KV extents.
    released += cache_.releaseEmptyExtents(keep);
    // True when the steps stopped on what waits for a transfer in flight
    // rather than on nothing left to take.
    const auto reclaimUpTo = [&](ReclaimClass upTo) {
      while (targetUnmet()) {
        const CacheReclaimResult step = cache_.reclaimOne(
            CacheReclaimMode::ReleaseExtents, upTo, directive.keepResumePoint, keep);
        if (!step.madeProgress)
          return step.pending;
        reclaimed = true;
        released += step.reclaimedBytes;
        releaseIdle();
      }
      return false;
    };
    // Cache nothing in use holds that waits for a transfer, an SSD demotion
    // or a state write, still ranks below the rungs above it: the pass ends
    // Pending, and the pass that continues it once the transfer landed takes
    // it first. A host that is really short then climbs escalate()'s rungs
    // short of what is in use, a Resident lane's conversion to Spill before
    // the expert store; what a conversion frees comes back as its copies
    // land, which the pass waits for as well: the pass that continues it
    // spills the pages whose copies landed before it climbs and counts what
    // they give back.
    pending = reclaimUpTo(ReclaimClass::Ordinary);
    if (!pending && directive.hostShort)
      while (targetUnmet()) {
        const CacheReclaimResult step = escalate({}, {});
        pending = step.pending;
        if (!step.madeProgress)
          break;
        released += step.reclaimedBytes;
      }
    if (!pending && directive.upTo == ReclaimClass::InUse)
      static_cast<void>(reclaimUpTo(ReclaimClass::InUse));
  }
  // Image rows only an evicted state held are ordinary cache entries now.
  reclaimModelCaches();
  if (released || reclaimed)
    signalResourceProgress();
  if (!directive.critical && !directive.targetBytes)
    return {released, ReclaimOutcome::Untargeted};
  if (!targetUnmet())
    return {released, ReclaimOutcome::Met};
  return {released, pending || cache_.transfersInFlight() ? ReclaimOutcome::Pending
                                                          : ReclaimOutcome::Exhausted};
}

} // namespace splash::engine
