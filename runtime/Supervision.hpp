#pragma once

#include <chrono>

// Every time limit one Splash process holds another to, and those the engine
// holds itself to, in one place. server/protocol.py SUPERVISION mirrors the
// ones the server times the engine by.
namespace splash::supervision {

using std::chrono::milliseconds;

// Metal fails an event wait still unmet this long after its queue reached it
// ("Caused GPU Timeout Error"), and the command with it: the work between two
// waits of a command, with the host step's or the Neural Engine's part, must
// end within it (metal::EventStep). Metal's limit, not ours.
inline constexpr milliseconds kGpuWaitLimit{5'000};
// A command that makes no progress this long fails the engine
// (metal::CommandWatchdog): a GPU hang or a lost completion, since Metal fails
// a slow host step or Neural Engine wait first.
inline constexpr milliseconds kCommandTimeout{120'000};
// The longest one control pass plus one tick of the native loop take
// (FdTransport::run): a frame waits at most this long before the loop takes
// it, and a status request is answered within it.
inline constexpr milliseconds kLoopStall{30'000};
// What a clean stop gives the newest restore points of a persistent cache to
// reach the disk (main.mm).
inline constexpr milliseconds kFlush{6'000};
// From a stop signal to the engine's exit, its flush included; the server
// kills an engine still running after it.
inline constexpr milliseconds kStop{15'000};
// A start serves on probation this long: a persistent cache it took back is
// dropped if the engine fails within it (RuntimeResources::beginServing), and
// the server counts a failure within it as short-lived.
inline constexpr milliseconds kProbation{60'000};

static_assert(kGpuWaitLimit < kCommandTimeout);
static_assert(kFlush < kStop);
static_assert(kLoopStall < kProbation);

} // namespace splash::supervision
