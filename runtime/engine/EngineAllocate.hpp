#pragma once

// The engine's allocation template and the helpers both of its sources use:
// Engine.cpp and EngineMemory.cpp include it, nothing else does.

#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <type_traits>
#include <vector>

namespace splash::engine {

// A state's buffers, taken from the model's pool or one admission
// (Model::reserveState, snapshot), or the refusal's cause.
struct StateBuffers final {
  bool ok = false;
  metal::AllocationFailure allocationFailure = metal::AllocationFailure::None;
  [[nodiscard]] bool granted() const noexcept { return ok; }
};

// What the request's KV blocks are cached under beside its tokens.
inline PromptIdentity promptIdentity(const EngineRequest &request) noexcept {
  return {request.images, request.cacheSalt};
}

// A refusal for memory, which what the ledger frees or the host's recovery
// may end. A KV admission the pages on their way back cover is one too: the
// ledger decides whether the claim waits for them (MemoryLedger::free).
inline bool memoryDenied(const StateAdmission &admission) noexcept {
  return admission.failure == StateFailure::MemoryPressure;
}

inline bool memoryDenied(const TokenAdmission &admission) noexcept {
  return admission.failure != TokenAdmissionFailure::None;
}

inline bool memoryDenied(const model::ModelMemoryAdmission &admission) noexcept {
  return !admission.granted();
}

inline bool memoryDenied(const StateBuffers &buffers) noexcept { return !buffers.granted(); }

// The part of a refused need that memory already allocated meets in place:
// the pages of the refused form it lacks beyond the free ones not held for
// claims ranked above it (`held`), the pooled state buffers a lane's
// activation lacks, one state's buffers.
inline std::optional<Need> inPlace(const TokenAdmission &admission, const std::array<uint32_t, 3> &held) {
  const uint32_t heldPages = held[static_cast<size_t>(admission.form)];
  const uint32_t free = admission.availablePages > heldPages ? admission.availablePages - heldPages : 0;
  return Need{.pages = admission.additionalPages > free ? admission.additionalPages - free : 0,
              .form = admission.form};
}

inline std::optional<Need> inPlace(const model::ModelMemoryAdmission &, const std::array<uint32_t, 3> &) {
  return Need{};
}

inline std::optional<Need> inPlace(const StateBuffers &, const std::array<uint32_t, 3> &) {
  return Need{.stateBuffers = 1};
}

// What a KV target lacked when it could not be allocated.
inline std::string pageShortfall(const TokenAdmission &admission) {
  return "additional_pages=" + std::to_string(admission.additionalPages) +
         ", free_pages=" + std::to_string(admission.availablePages);
}

template <class Attempt>
auto Engine::allocate(Attempt &&attempt, const ClaimSpec &spec, bool frees)
    -> Allocation<std::invoke_result_t<Attempt &>> {
  using Admission = std::invoke_result_t<Attempt &>;
  MemoryGovernor &governor = this->governor();
  MemoryLedger &ledger = governor.ledger();
  // A request's allocations are its one claim's, the engine's own a claim
  // of this allocation alone.
  ClaimHandle transient;
  ClaimHandle &claim = spec.requestId ? request(spec.requestId).claim : transient;
  if (claim.id) {
    ledger.update(claim.id, spec);
  } else {
    claim.governor = &governor;
    claim.id = ledger.open(spec);
  }
  ledger.setLanes(laneDemandBytes(), highestLanePriority());
  // Each attempt runs in the claim's scope, from a need the scope's refusal
  // records afresh.
  const auto tryOnce = [&] {
    ledger.refused(claim.id, {});
    const MemoryGovernor::Scope scope(governor, claim.id);
    return attempt();
  };
  Allocation<Admission> result{tryOnce(), {}};
  Admission &admission = result.admission;
  Denial &denial = result.denial;
  // Each refusal asks the ledger to free what the claim lacks, and the
  // attempt runs again once that is met or memory left now. A refusal that
  // lacks nothing the ledger counts, that the host refuses after exempting
  // the claim (critical pressure), or that lacks what it lacked when the
  // ledger met a need of this allocation before, is not for want of what the
  // ledger frees.
  std::vector<Need> met;
  const auto metBefore = [&](const Need &need) {
    return std::ranges::any_of(met, [&](const Need &before) {
      return need.bytes == before.bytes && need.hostRefused == before.hostRefused && need.pages == before.pages &&
             need.form == before.form && need.stateBuffers == before.stateBuffers;
    });
  };
  while (frees && memoryDenied(admission)) {
    std::optional<Need> part;
    if constexpr (std::is_same_v<Admission, StateAdmission>)
      part = Need{.stateBuffers = model_.statesToActivate()};
    else
      part = inPlace(admission, ledger.heldPages(claim.id));
    const Need need = governor.need(claim.id, part);
    if ((!need.bytes && !need.hostRefused && !need.pages && !need.stateBuffers) ||
        (need.hostRefused && ledger.claim(claim.id).hostExempt) ||
        metBefore(need))
      break;
    const bool exempt = ledger.claim(claim.id).hostExempt;
    const FreeResult freed = governor.free(claim.id, need);
    ++ledgerChanges_;
    // What it freed now, the units that cost nothing among them, the
    // attempt may fit in.
    if (freed.outcome != FreeOutcome::Met && !freed.nowBytes) {
      denial.pending = freed.outcome == FreeOutcome::Pending;
      // A conversion frees later, and at once the pages whose copies landed
      // before (Cache::convertToSpill), which one more attempt may fit in.
      if (freed.laterBytes)
        admission = tryOnce();
      break;
    }
    // The host's exemption takes nothing for it.
    result.freed = result.freed || exempt || !ledger.claim(claim.id).hostExempt;
    met.push_back(need);
    signalResourceProgress();
    admission = tryOnce();
  }
  if (!admission.granted()) {
    denial.allocationFailure = admission.allocationFailure;
    if constexpr (std::is_same_v<Admission, TokenAdmission>) {
      denial.pages = admission.additionalPages;
      denial.form = admission.form;
    }
    // Memory on its way back ends the shortage without the residents, and a
    // refused optional allocation only shortens a command or leaves memory
    // for later.
    if (memoryDenied(admission) && !denial.pending && spec.cls != ClaimClass::Optional)
      allocationFailed_ = true;
  } else if (ledger.claim(claim.id).waiting) {
    ledger.setWaiting(claim.id, false);
  }
  return result;
}

} // namespace splash::engine
