#pragma once

// What the native loop tests of several areas share: the loop fixture, its
// requests and the loops that tick it.

#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestStatus.hpp"
#include "engine/Cache.hpp"
#include "engine/NativeRuntime.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace splash::test {

// A host whose spare memory is all its available memory.
inline std::optional<engine::HostMemory> hostMemory(std::optional<uint64_t> bytes) {
  if (!bytes)
    return std::nullopt;
  return engine::HostMemory{*bytes, *bytes};
}

// Every submitted request has ended and no command is in flight.
inline bool idle(const engine::NativeRuntime &loop) {
  const engine::EngineSnapshot counts = loop.statusSnapshot();
  return counts.submitted == counts.completed + counts.cancelled + counts.failed &&
         !loop.commandInFlight();
}

inline protocol::RequestFrame request(uint64_t id, uint32_t maxOutputTokens = 1) {
  protocol::RequestFrame result;
  result.requestId = id;
  result.priority = engine::RequestPriority::Foreground;
  result.constraint = ConstraintMode::None;
  result.absoluteDeadlineUnixMicros = 2'000'000;
  result.remainingDeadlineMicros = 1'000'000;
  result.logicalMaxOutputTokens = maxOutputTokens;
  result.promptTokens.resize(65);
  for (uint32_t i = 0; i < result.promptTokens.size(); ++i) {
    result.promptTokens[i] = i + 1;
  }
  return result;
}

inline void runUntilIdle(engine::NativeRuntime &loop) {
  for (uint32_t step = 0; step < 32 && !idle(loop); ++step) {
    static_cast<void>(loop.tick());
  }
  require(idle(loop), "native loop did not become idle");
}

// The KV pool under the loop: `pages` pages in extents of `extentPages`, of
// which the first `runwayPages` are allocated up front.
struct PoolShape {
  uint32_t pages = 32;
  uint32_t extentPages = 4;
  uint32_t runwayPages = pages;
};

// A native loop over the fake model, collecting the bytes it writes. Its
// config is completed with the engine's (test::engineConfig), and the
// fixture's metrics and weights, each weight allocation admitted, the
// optional growth too (test::admitAll), where the test gave none; the
// weights' idle release is production's unless the test gives one. Its unix
// clock stands still and its awake clock reads `monotonic`, advanced by
// `clockStep` on every read.
struct LoopFixture {
  explicit LoopFixture(engine::NativeLoopConfig config = {}, PoolShape shape = {},
                       protocol::ProtocolLimits limits = {},
                       double idleReleaseSeconds = metal::kResidencyKeepAliveSeconds)
      : storage(shape.pages, 4096, shape.extentPages),
        pool(storage, shape.runwayPages), cache(pool, nullptr, nullptr),
        loop(
            clocked(std::move(config)), idleReleaseSeconds, cache, executor,
            [this](std::span<const uint8_t> bytes) {
              if (writeFailure)
                throw *writeFailure;
              output.insert(output.end(), bytes.begin(), bytes.end());
            },
            [this] { return status(); }, limits) {
    storage.commandInFlight = [this] { return loop.commandInFlight(); };
  }

  // Every event the loop has written.
  std::vector<protocol::EngineEvent> events() const {
    return protocol::peer::decodeEvents(output);
  }

  engine::NativeLoopConfig clocked(engine::NativeLoopConfig config) {
    config.engine = test::engineConfig(std::move(config.engine));
    if (!config.metrics)
      config.metrics = &metrics;
    if (!config.weights)
      config.weights = &weights;
    if (!config.weightAdmission)
      config.weightAdmission = test::admitAll;
    if (!config.growthAdmission)
      config.growthAdmission = test::admitAll;
    config.unixMicros = [] { return uint64_t{1'000'000}; };
    config.monotonicMilliseconds = [this] { return monotonic += clockStep; };
    return config;
  }

  test::TestKvStorage storage;
  engine::KvPool pool;
  engine::Cache cache;
  Executor executor;
  std::vector<uint8_t> output;
  // Thrown by every write while set, as by a closed output pipe.
  std::optional<std::system_error> writeFailure;
  // What the loop answers a status request with.
  std::function<std::string()> status = test::readyStatusJson;
  double monotonic = 100.0;
  double clockStep = 0.0;
  engine::RuntimeMetrics metrics;
  Weights weights;
  engine::NativeRuntime loop;
};

inline bool answeredStatus(const LoopFixture &fixture, uint64_t correlationId) {
  const auto events = fixture.events();
  return std::any_of(events.begin(), events.end(), [&](const auto &message) {
    const auto *event = std::get_if<protocol::StatusJsonEvent>(&message);
    return event && event->correlationId == correlationId;
  });
}

// Sends a release request and returns the weights' snapshot its answer reports:
// the status document that answers it is written from the loop's snapshot of the
// moment (RuntimeBootstrap::statusJson).
inline engine::WeightsSnapshot releaseAnswer(LoopFixture &fixture, uint64_t correlationId) {
  std::optional<engine::WeightsSnapshot> reported;
  fixture.status = [&] {
    reported = fixture.loop.weightsSnapshot();
    return test::readyStatusJson();
  };
  require(fixture.loop.receive(
              protocol::peer::serialize(protocol::ReleaseRequestFrame{correlationId})) &&
              answeredStatus(fixture, correlationId) && reported,
          "a release request was not answered with a status document");
  fixture.status = test::readyStatusJson;
  return *reported;
}

} // namespace splash::test
