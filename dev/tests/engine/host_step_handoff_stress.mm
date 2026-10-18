// Scratch stress test of the CPU/GPU handoff of host steps (not for merging):
// COMMANDS commands of STEPS host steps, each followed by a one-thread
// dispatch, submitted one after another until one fails. MODE idle: empty
// steps the GPU idles through (a near-empty prefill round); awake: empty
// steps the GPU spins through; sleep: idle steps that sleep 1 ms; zero: idle
// steps each followed by an indirect dispatch of no threadgroup (an empty
// expert repack); timeout: one idle step that sleeps 6 s (the GPU's wait on
// it should time out at 5 s). Reports
// failures with their message, the runner's re-reads of each event it raised
// (metal/HandoffDiag.hpp), and the time per step.
//
// usage: host-step-handoff-stress METALLIB MODE COMMANDS STEPS
#include "../../../runtime/metal/HandoffDiag.hpp"
#include "../../../runtime/metal/MetalBackend.hpp"
#include "TestBuffers.hpp"

#import <Foundation/Foundation.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using splash::metal::BufferBinding;
using splash::metal::Command;
using splash::metal::ComputeDispatch;
using splash::metal::EventStep;
using splash::metal::HostStepGpu;
using splash::metal::MetalBackend;
using splash::metal::MetalBackendError;
using splash::metal::MetalBuffer;

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      std::fprintf(stderr, "usage: host-step-handoff-stress METALLIB idle|awake|sleep|zero|timeout COMMANDS STEPS\n");
      return 2;
    }
    const std::string mode = argv[2];
    const uint32_t commands = static_cast<uint32_t>(std::atoi(argv[3]));
    const uint32_t steps = static_cast<uint32_t>(std::atoi(argv[4]));
    MetalBackend backend(argv[1]);
    MetalBuffer buffer = splash::test::sharedBuffer(backend, sizeof(uint32_t));
    auto *value = static_cast<volatile uint32_t *>(buffer.contents());
    *value = 0;
    const uint32_t count = 1, one = 1;
    MetalBuffer noGrid = splash::test::sharedBuffer(backend, 16);
    std::memset(noGrid.contents(), 0, 16);
    ComputeDispatch dispatch{"test_add_u32",
                             {{0, buffer}},
                             {{1, &count, sizeof(count)}, {2, &one, sizeof(one)}},
                             {1, 1, 1},
                             {1, 1, 1}};
    if (mode == "zero") dispatch.indirectThreadgroups = {noGrid, 0};
    const std::vector<ComputeDispatch> dispatches(steps + 1, dispatch);
    const bool sleeps = mode == "sleep", times = mode == "timeout";
    std::vector<EventStep> events;
    for (uint32_t step = 0; step < steps; ++step)
      events.push_back({step + 1, {}, 0, EventStep::Kind::Host,
                        [sleeps, times] {
                          if (sleeps) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                          if (times) std::this_thread::sleep_for(std::chrono::seconds(6));
                        },
                        {}, mode == "awake" ? HostStepGpu::Awake : HostStepGpu::Idle});
    const Command command{dispatches, events};
    const auto start = std::chrono::steady_clock::now();
    uint32_t done = 0, failures = 0;
    for (; done < commands; ++done) {
      const auto commandStart = std::chrono::steady_clock::now();
      try {
        static_cast<void>(backend.submitCommandAsync(command).wait());
      } catch (const MetalBackendError &error) {
        ++failures;
        const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - commandStart).count();
        std::printf("command %u FAILED after %.3f s: %s\n", done + 1, took, error.what());
        splash::handoff::log("stress command %u failed after %.3f s: %s", done + 1, took, error.what());
        break;
      }
      if ((done + 1) % 100 == 0)
        std::printf("  %u commands, %.1f s, value %u, re-reads %llu, regressions %llu\n", done + 1,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), *value,
                    static_cast<unsigned long long>(splash::handoff::checks.load()),
                    static_cast<unsigned long long>(splash::handoff::regressions.load()));
      std::fflush(stdout);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const uint64_t handoffs = uint64_t{done} * steps;
    std::printf("%s: %u of %u commands of %u steps, %llu handoffs in %.1f s (%.1f us a step), failures %u, "
                "value %u (expected %llu), re-reads %llu, regressions %llu\n",
                mode.c_str(), done, commands, steps, static_cast<unsigned long long>(handoffs), seconds,
                handoffs ? seconds * 1e6 / double(handoffs) : 0.0, failures, *value,
                static_cast<unsigned long long>(uint64_t{done} * (steps + 1)),
                static_cast<unsigned long long>(splash::handoff::checks.load()),
                static_cast<unsigned long long>(splash::handoff::regressions.load()));
    return failures ? 1 : 0;
  }
}
