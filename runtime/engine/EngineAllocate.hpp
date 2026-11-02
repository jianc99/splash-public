#pragma once

// The engine's allocation templates and the helpers both of its sources use:
// Engine.cpp and EngineMemory.cpp include it, nothing else does.

#include "engine/Engine.hpp"

#include <string>
#include <type_traits>

namespace splash::engine {

// What the request's KV blocks are cached under beside its tokens.
inline PromptIdentity promptIdentity(const EngineRequest &request) noexcept {
  return {request.images, request.cacheSalt};
}

// A refusal for memory, which reclaim or the host's recovery may end.
inline bool memoryDenied(const StateAdmission &admission) noexcept {
  return admission.failure == StateFailure::MemoryPressure;
}

inline bool memoryDenied(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Denied;
}

inline bool memoryDenied(const model::ModelMemoryAdmission &admission) noexcept {
  return !admission.granted();
}

// Memory a transfer in flight holds, which returns by itself.
inline bool transferPending(const StateAdmission &) noexcept { return false; }

inline bool transferPending(const TokenAdmission &admission) noexcept {
  return admission.failure == TokenAdmissionFailure::Pending;
}

inline bool transferPending(const model::ModelMemoryAdmission &) noexcept { return false; }

// What a KV target lacked when it could not be allocated.
inline std::string pageShortfall(const TokenAdmission &admission) {
  return "additional_pages=" + std::to_string(admission.additionalPages) +
         ", free_pages=" + std::to_string(admission.availablePages);
}

template <class Take> bool Engine::makeStateRoom(StateClass kind, Take &&take) {
  // No command is in flight, so KV that empties an extent releases it now.
  // A recycled state hands over its buffers; an extent may hold less than a
  // state, so room is made until the buffers fit, nothing more of the class
  // goes, or the extents given cover one state: a refusal after that is not
  // the budget's.
  const bool growth = !config_.growthPaused();
  const uint64_t needed = model_.snapshotBytes();
  uint64_t released = 0;
  // Steps of `reclaim` until `take` gets the buffers, nothing more of its
  // kind goes, or what they gave covers one state.
  enum class Room : uint8_t { Taken, Refused, Exhausted };
  const auto makeRoom = [&](const auto &reclaim) {
    StateRoom room;
    do {
      room = reclaim();
      released += room.extentBytes;
      if (room && take())
        return Room::Taken;
    } while (room.extentBytes && released < needed);
    return room ? Room::Refused : Room::Exhausted;
  };
  const Room made = makeRoom([&] { return cache_.reclaimOneState(kind, growth); });
  if (made != Room::Exhausted || kind != StateClass::InUse)
    return made == Room::Taken;
  // The prompt's replay point then climbs the ladder every refusal of the
  // engine's limit climbs, while the host lets the engine grow (the paced
  // reclaim makes room for the host): the cache nothing in use holds whose
  // memory needs its copy on the SSD cache first, which reclaimOneState
  // leaves, then escalate()'s rungs short of what is in use, a step at a
  // time until they cover its buffers. Memory on its way back refuses the
  // state's buffers: its command ends at the boundary, and the point is kept
  // as a refused one is, from the lane after the command
  // (publishReachedStateBoundaries).
  while (growth && released < needed) {
    CacheReclaimResult step = cache_.reclaimForMemory(ReclaimClass::Ordinary, false);
    if (!step.madeProgress && !step.pending) {
      const bool evictedKv = step.evictedKv;
      step = escalate({}, {});
      step.evictedKv = step.evictedKv || evictedKv;
    }
    if (step.madeProgress || step.evictedKv)
      signalResourceProgress();
    if (!step.madeProgress) {
      if (step.pending)
        return false;
      break;
    }
    released += step.reclaimedBytes;
    if (take())
      return true;
  }
  // Another request's state in use goes last.
  return makeRoom([&] { return cache_.reclaimInUseState(); }) == Room::Taken;
}

template <class Attempt>
auto Engine::allocate(Attempt &&attempt, bool inService, ReclaimClass upTo,
                      const std::function<bool(const Denial &)> &fallback, bool optional)
    -> Allocation<std::invoke_result_t<Attempt &>> {
  using Admission = std::invoke_result_t<Attempt &>;
  // Set once a request in service has nothing held left to reuse: from then
  // on only the engine's limit and critical pressure refuse it.
  bool serving = false;
  const auto tryOnce = [&] {
    const Serving mark(config_.serving, serving);
    return attempt();
  };
  Allocation<Admission> result{tryOnce(), {}};
  Admission &admission = result.admission;
  Denial &denial = result.denial;
  while (memoryDenied(admission)) {
    const bool paused =
        admission.allocationFailure == metal::AllocationFailure::HostPressure;
    CacheReclaimResult reclaimed;
    if constexpr (std::is_same_v<Admission, StateAdmission>)
      reclaimed = paused ? reuseCachedStateWhilePaused(upTo) : reclaimForState(upTo, optional);
    else if constexpr (std::is_same_v<Admission, TokenAdmission>)
      // A KV allocation's fallback converts its lane to Spill (spillFallback):
      // the first rung above the cache nothing in use holds.
      reclaimed = paused ? reuseCachedPagesWhilePaused(admission, upTo)
                         : reclaimForKv(admission, upTo, optional, [&] {
                             return fallback &&
                                    fallback({.allocationFailure = admission.allocationFailure,
                                              .pages = admission.additionalPages,
                                              .form = admission.form});
                           });
    else
      // The model's memory for reads of spilled KV (a hot block cache floor,
      // a history's staging) takes what a state takes; while the host
      // refuses growth, idle buffers alone, and memory for later (that
      // cache's optional growth) nothing.
      reclaimed = paused ? CacheReclaimResult{!optional &&
                                              reclaimIdleState(true, model::IdleMemory::Buffers)}
                         : reclaimForState(upTo, optional);
    if (reclaimed.madeProgress) {
      admission = tryOnce();
      continue;
    }
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = reclaimed.pending;
    if (paused && inService && !serving && !reclaimed.pending) {
      serving = true;
      admission = tryOnce();
      continue;
    }
    if (fallback && fallback(denial)) {
      admission = tryOnce();
      continue;
    }
    break;
  }
  if (!admission.granted()) {
    denial.allocationFailure = admission.allocationFailure;
    denial.pending = denial.pending || transferPending(admission);
    if constexpr (std::is_same_v<Admission, TokenAdmission>) {
      denial.pages = admission.additionalPages;
      denial.form = admission.form;
    }
    // Memory on its way back ends the shortage without the residents.
    if (memoryDenied(admission) && !denial.pending && !optional)
      allocationFailed_ = true;
  }
  return result;
}

} // namespace splash::engine
