#pragma once

// The control pass's host claim as a test drives it on its engine's
// governor (MemoryGovernor::controlPass): what free() gives a Host claim, up
// to what is there for it, the host lacking memory past its margin
// (`deficit`) or under macOS's critical pressure, which first drops every
// pool to its floor.

#include "engine/MemoryGovernor.hpp"
#include "model/SlotFile.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

namespace splash::test {

inline engine::FreeResult hostPass(engine::MemoryGovernor &governor,
                                   uint64_t bytes = std::numeric_limits<uint64_t>::max(), bool deficit = true,
                                   bool critical = false) {
  engine::MemoryLedger &ledger = governor.ledger();
  const engine::ClaimSpec spec{.cls = engine::ClaimClass::Host,
                               .reach = critical ? engine::ReclaimClass::InUse : engine::ReclaimClass::Ordinary,
                               .holdSeconds = engine::kValueHorizonSeconds};
  if (critical)
    static_cast<void>(ledger.dropToFloors());
  const uint64_t found = ledger.available(spec, bytes, 0, critical, deficit);
  const engine::ClaimId claim = ledger.open(spec);
  const engine::FreeResult result = ledger.free(claim, {.bytes = std::min(bytes, found)}, 0, critical, deficit);
  ledger.close(claim);
  return result;
}

// The pass under macOS's critical pressure: every pool to its floor, then
// everything the host claim may take.
inline engine::FreeResult criticalPass(engine::MemoryGovernor &governor) {
  return hostPass(governor, std::numeric_limits<uint64_t>::max(), true, true);
}

// The quota a test's cache with a disk tier counts its copies against, as
// every production cache with a tier has one (model::DiskBudget): far past
// what any test writes, so that a demotion the tier has room for is sure to
// start.
inline std::shared_ptr<model::DiskBudget> ampleDiskBudget() {
  return std::make_shared<model::DiskBudget>(uint64_t{1} << 40);
}

} // namespace splash::test
