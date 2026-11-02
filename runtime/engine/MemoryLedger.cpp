#include "engine/MemoryLedger.hpp"

#include <stdexcept>

namespace splash::engine {

AccountId MemoryLedger::addAccount(std::string, AccountRole, std::function<uint64_t()>) {
  throw std::logic_error("M1");
}

AccountId MemoryLedger::addPool(model::ElasticPool &) { throw std::logic_error("M1"); }

void MemoryLedger::addSource(VictimSource &) { throw std::logic_error("M1"); }

void MemoryLedger::setReturning(std::function<Returning()>) { throw std::logic_error("M1"); }

void MemoryLedger::observePools(double) { throw std::logic_error("M1"); }

ClaimId MemoryLedger::open(const ClaimSpec &) { throw std::logic_error("M1"); }

void MemoryLedger::update(ClaimId, const ClaimSpec &) { throw std::logic_error("M1"); }

void MemoryLedger::refused(ClaimId, const Need &) { throw std::logic_error("M1"); }

void MemoryLedger::setWaiting(ClaimId, bool) { throw std::logic_error("M1"); }

uint64_t MemoryLedger::heldBytes(ClaimId, std::optional<AccountId>) const { throw std::logic_error("M1"); }

std::array<uint32_t, 3> MemoryLedger::heldPages(ClaimId) const { throw std::logic_error("M1"); }

FreeResult MemoryLedger::free(ClaimId, const Need &, uint64_t, bool) { throw std::logic_error("M1"); }

uint64_t MemoryLedger::available(const ClaimSpec &, uint64_t, uint64_t) const { throw std::logic_error("M1"); }

bool MemoryLedger::grow(uint64_t, const std::function<metal::AllocationResult(AccountId, model::ElasticPool &)> &) {
  throw std::logic_error("M1");
}

uint64_t MemoryLedger::dropToFloors() { throw std::logic_error("M1"); }

void MemoryLedger::fill(MemoryGovernorSnapshot &, uint64_t) const { throw std::logic_error("M1"); }

} // namespace splash::engine
