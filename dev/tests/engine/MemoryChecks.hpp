#pragma once

// The memory checks the harness and the loop fixture run after every tick
// when the test's engine allocates through FakeMemory. I1 is here; the
// ledger's relations (I2) join it when the ledger lands.

#include "FakeMemory.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "engine/MemoryGovernor.hpp"

#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

namespace splash::test {

// What allocates through FakeMemory, as a check reads it.
struct MemoryHolders final {
  const FakeMemory &memory;
  const TestKvStorage &storage;
  const Executor &model;
  const Weights *weights = nullptr;
  const engine::MemoryGovernor *governor = nullptr;
};

// I1: nothing wrote the backend's statistics behind FakeMemory's back; per
// holder, FakeMemory agrees with the allocator's own count (the KV storage's
// extents, the model's state buffers, its reads' memory and its experts
// pool, the weights); and the governor charges at least what FakeMemory
// holds (MemoryGovernor charges the backend's buffers in full).
inline void checkMemory(const MemoryHolders &holders) {
  holders.memory.check();
  const auto agree = [&](std::string_view holder, uint64_t counted) {
    if (holders.memory.held(holder) != counted)
      throw std::logic_error("fake memory holds " + std::to_string(holders.memory.held(holder)) + " bytes for " +
                             std::string(holder) + ", its allocator " + std::to_string(counted));
  };
  agree("kv", holders.storage.allocatedBytes());
  agree("states", holders.model.heldStateBytes);
  agree("spilled_reads", std::accumulate(holders.model.spilledReadBytes.begin(), holders.model.spilledReadBytes.end(),
                                         uint64_t{0}, [](uint64_t sum, const auto &entry) { return sum + entry.second; }));
  if (holders.model.experts) {
    const model::PoolState pool = holders.model.experts->state();
    agree(pool.name, pool.currentBytes);
  }
  if (holders.weights)
    agree("weights", uint64_t{holders.weights->held()} * Weights::kImageBytes);
  if (holders.governor && holders.governor->snapshot().chargedBytes < holders.memory.total())
    throw std::logic_error("the governor charges less than fake memory holds");
}

} // namespace splash::test
