#pragma once

#include "engine/MemoryGovernor.hpp"
#include "engine/NativeRuntime.hpp"
#include "metal/MetalBackend.hpp"

#include <string>

namespace splash::engine {

// Emits only transitions; retry counts and queue depth do not produce logs.
class MemoryStatusReporter final {
public:
  [[nodiscard]] std::string update(const ResourceWaitSnapshot &wait,
                                    bool hostGrowthAllowed);
private:
  unsigned state_ = 0;
};

// The engine's memory control between commands: it releases the weights of
// an engine left without a request (NativeRuntime::releaseIdleWeights),
// records the system's pressure, lands what transfers finished and gives
// the ledger the lanes' demand, the wanted chains and the rates it prices
// units at (NativeRuntime::collectLandings, refreshMemoryRates), logs a
// change in what requests wait for, and runs the governor's control pass
// (MemoryGovernor::controlPass): under macOS's critical pressure every pool
// to its floor and the host claim, under its warning or in the host's band
// the host claim, otherwise the pools' growth.
class MemoryControl final {
public:
  MemoryControl(MemoryGovernor &governor, metal::MetalBackend &backend,
                NativeRuntime &loop) noexcept
      : governor_(governor), backend_(backend), loop_(loop) {}

  // One memory-control pass at a command-free point. True while the host
  // claim waits for memory on its way back: the transport runs it again at
  // the next command-free point.
  [[nodiscard]] bool run(MemoryPressure pressure);

private:
  MemoryGovernor &governor_;
  metal::MetalBackend &backend_;
  NativeRuntime &loop_;
  MemoryStatusReporter reporter_;
};

} // namespace splash::engine
