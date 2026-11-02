#pragma once

// What the engine tests of several areas share: the event sink they read,
// requests, the loops that tick an engine, and their common fixtures.

#include "TestCache.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "engine/Engine.hpp"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace splash::test {

class Events final : public engine::EngineEventSink {
public:
  void batchCompleted(WorkKind kind, uint32_t, uint32_t, uint32_t,
                      uint32_t drafted, uint32_t accepted, double,
                      double cycleMilliseconds) override {
    cycles.emplace_back(kind, cycleMilliseconds);
    draftedTokens += drafted;
    acceptedDraftTokens += accepted;
  }
  void started(uint64_t requestId, uint32_t matched, uint32_t) override {
    startIds.push_back(requestId);
    starts.push_back(matched);
  }
  void promptProgress(uint64_t id, uint32_t processed) override {
    progress[id].push_back(processed);
  }
  void tokens(uint64_t id, std::span<const uint32_t> values) override {
    emitted += values.size();
    if (clock)
      tokenTimes[id].push_back(clock());
    auto &transcript = outputs[id];
    transcript.insert(transcript.end(), values.begin(), values.end());
  }
  void maskRequested(uint64_t requestId,
                     std::span<const uint32_t> simulation) override {
    maskRequests.emplace_back(
        requestId, std::vector<uint32_t>(simulation.begin(), simulation.end()));
  }
  void completed(uint64_t id, engine::EngineFinishReason, bool bounded, uint32_t prompt,
                 uint32_t completion, uint32_t deduplicated,
                 std::span<const float>, const engine::SpeculativeCounts &speculation) override {
    ++completedCount;
    if (bounded)
      boundedIds.push_back(id);
    usage[id] = {prompt, completion};
    deduplicatedTokens[id] = deduplicated;
    speculativeCounts[id] = speculation;
  }
  void failed(uint64_t, engine::LaneOutcome outcome, std::string message,
              uint32_t retryAfterSeconds) override {
    ++failedCount;
    if (outcome == engine::LaneOutcome::CapacityExhausted)
      ++capacityExhaustedCount;
    const engine::LaneOutcomeWire wire = engine::laneOutcomeWire(outcome);
    failures.emplace_back(wire.code);
    failureDetails.emplace_back(std::move(message), wire.retryable);
    retryAfters.push_back(retryAfterSeconds);
  }

  // With a clock, the time of each request's token events.
  std::function<double()> clock;
  std::unordered_map<uint64_t, std::vector<double>> tokenTimes;
  std::unordered_map<uint64_t, std::vector<uint32_t>> progress;
  // The matched prompt tokens of each start, zero for a cold one.
  std::vector<uint32_t> starts;
  std::vector<uint64_t> startIds;
  std::unordered_map<uint64_t, std::vector<uint32_t>> outputs;
  std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> usage;
  // The requests that ended at a bound of the engine's own.
  std::vector<uint64_t> boundedIds;
  std::unordered_map<uint64_t, uint32_t> deduplicatedTokens;
  std::unordered_map<uint64_t, engine::SpeculativeCounts> speculativeCounts;
  std::vector<std::string> failures;
  std::vector<std::pair<std::string, bool>> failureDetails;
  // When each failure said a retry may succeed, in seconds.
  std::vector<uint32_t> retryAfters;
  uint32_t emitted = 0;
  uint32_t completedCount = 0;
  uint32_t failedCount = 0;
  uint32_t capacityExhaustedCount = 0;
  std::vector<std::pair<uint64_t, std::vector<uint32_t>>> maskRequests;
  // Each completed command's kind and engine cycle.
  std::vector<std::pair<WorkKind, double>> cycles;
  // The draft tokens of every completed command, as /status sums them.
  uint32_t draftedTokens = 0;
  uint32_t acceptedDraftTokens = 0;
};

// Every submitted request has ended and no command is in flight.
inline bool idle(const engine::Engine &engine) {
  const engine::EngineSnapshot counts = engine.snapshot();
  return counts.submitted == counts.completed + counts.cancelled + counts.failed &&
         !engine.commandInFlight();
}

inline engine::EngineRequest request(uint64_t id, const std::vector<uint32_t> &prompt) {
  engine::EngineRequest result;
  result.id = id;
  result.prompt = prompt;
  result.maxNewTokens = 1;
  result.deadlineMilliseconds = 10000;
  return result;
}

// Every engine test holds the engine to PageStorage's rule: no extent is
// released while a command is in flight.
inline void guardReleases(test::TestKvStorage &storage, const engine::Engine &engine) {
  storage.commandInFlight = [&engine] { return engine.commandInFlight(); };
}

// Ticks until every prompt (65 tokens of its request id) has published a
// state, then holds a lookup on each, which keeps its state resident.
inline std::vector<engine::CacheLookup> runUntilStatesHeld(engine::Engine &engine, engine::Cache &cache,
                                                           std::initializer_list<uint64_t> ids) {
  for (double now = 1; now < 100 && cache.snapshot().stateCache.entries < ids.size(); ++now)
    static_cast<void>(engine.tick(now));
  require(cache.snapshot().stateCache.entries == ids.size(), "the prompts did not publish states");
  std::vector<engine::CacheLookup> held;
  for (uint64_t id : ids)
    held.push_back(cache.lookup(std::vector<uint32_t>(65, static_cast<uint32_t>(id)), {}));
  return held;
}

inline void runUntilIdle(engine::Engine &engine) {
  for (uint32_t step = 0; step < 32 && !idle(engine); ++step) {
    static_cast<void>(engine.tick(step + 1));
  }
  require(idle(engine), "engine did not reach idle");
}

// Ticks on from `now` until done() holds.
inline void tickUntil(engine::Engine &engine, double &now, const std::function<bool()> &done,
                      const char *message) {
  for (uint32_t step = 0; step < 1000 && !done(); ++step)
    static_cast<void>(engine.tick(now++));
  require(done(), message);
}

// The states a request's prefill steps captured, as (boundary, end of the
// step) pairs.
inline std::vector<std::pair<uint32_t, uint64_t>> capturedStates(const Executor &executor,
                                                                 uint64_t requestId) {
  std::vector<std::pair<uint32_t, uint64_t>> result;
  for (const Executor::Capture &capture : executor.captures)
    if (capture.requestId == requestId)
      result.emplace_back(capture.tokens, capture.end);
  return result;
}

inline engine::EngineRequest constrainedRequest(uint64_t id, double deadline = 10'000.0) {
  engine::EngineRequest value = request(id, {1});
  value.maxNewTokens = 2;
  value.constraint = ConstraintMode::TokenMask;
  value.deadlineMilliseconds = deadline;
  return value;
}

// A target whose requests may spill (EngineConfig::spill), on a cache with a
// spill policy: Full pages of 100 bytes and Index pages of 10, four to an
// extent unless `extent` says, with the Index runway allocated; a tier of
// 100-byte slots whose quota, active KV's limit included, holds `slots` of
// them; and a dense history and a tail window of a page each. With
// `always`, every request spills (--kv-spill always); otherwise a request
// spills where its projection does not fit `residentKvBytes` of resident KV
// (1,000 pages by default) beside the other lanes' (SpillGeometry). One that
// cannot spill may start Resident within `residentTokens` alone, by default
// the tokens of resident KV's pages. The tier's transfers land as the
// fixture ticks on, and decode runs to each request's output limit. The
// tier's write budget, if a test sets one, refills on the fixture's clock.
// Its engine's governor is `governor` where the test gives one, else an
// ample one (test::engineConfig).
struct Spilling {
  // The pages of an extent, of a type of its own for a call to name.
  struct Extent final {
    uint32_t pages;
  };
  // `shape` sets what one of the model's prefill commands may hold
  // (Executor::groupRows, passes), which the engine takes as it is built.
  explicit Spilling(uint32_t slots, bool always = true, uint64_t residentKvBytes = 100'000,
                    uint32_t residentTokens = 0, Extent extent = {4}, uint64_t laneReadBytes = 0,
                    const std::function<void(Executor &)> &shape = {}, engine::MemoryGovernor *governor = nullptr)
      : storage(256, 100, extent.pages, 10),
        budget(std::make_shared<model::DiskBudget>(uint64_t{slots} * 100, [this] { return now; })),
        cache(pool, &tier, budget, engine::CacheSpillPolicy{1, 1}),
        settings((shape ? shape(model) : void(),
                  config(always, residentKvBytes,
                         residentTokens ? residentTokens
                                        : static_cast<uint32_t>(residentKvBytes / 100 * engine::KvCache::pageTokens),
                         &paused, &serving, laneReadBytes, governor))),
        engine(settings, cache, model, events) {
    tier.budget = budget;
    tier.transferLimit = 1024;
    require(static_cast<bool>(pool.fillRunway(kv::PageForm::Index)),
            "the Index runway was not allocated");
    guardReleases(storage, engine);
    model.decodeFinishes = false;
  }

  static engine::EngineConfig config(bool always, uint64_t residentKvBytes, uint32_t residentTokens,
                                     const bool *paused, bool *serving, uint64_t laneReadBytes,
                                     engine::MemoryGovernor *governor) {
    engine::EngineConfig result = test::engineConfig({.governor = governor});
    result.growthPaused = [paused] { return *paused; };
    result.serving = [serving](bool value) { return std::exchange(*serving, value); };
    result.spill = {.capable = true,
                    .always = always,
                    .denseTokens = engine::KvCache::pageTokens,
                    .geometry = {.fullPageBytes = 100,
                                 .indexPageBytes = 10,
                                 .slotBytes = 100,
                                 .residentKvBytes = residentKvBytes,
                                 .residentTokens = residentTokens,
                                 .densePages = 1,
                                 .windowPages = 1,
                                 .groupPages = 4,
                                 .laneReadBytes = laneReadBytes}};
    return result;
  }
  // A request whose prompt is `pages` whole pages of `token` and one token
  // more, which may generate `output` tokens.
  static engine::EngineRequest prompt(uint64_t id, uint32_t token, uint32_t pages, uint32_t output) {
    engine::EngineRequest result =
        request(id, std::vector<uint32_t>(pages * engine::KvCache::pageTokens + 1, token));
    result.maxNewTokens = output;
    return result;
  }
  // Ticks on, landing the tier's transfers after each tick, until done().
  void runUntil(const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 2000 && !done(); ++step) {
      static_cast<void>(engine.tick(now++));
      tier.complete();
    }
    require(done(), message);
  }
  void runUntilIdle() {
    runUntil([&] { return idle(engine); }, "the requests did not end");
  }
  // The slots the quota of active KV keeps for commitments now.
  [[nodiscard]] uint64_t committedSlots() const { return budget->committedBytes() / 100; }
  void allowSpillWrites(bool allowed) {
    tier.writeLimits[static_cast<size_t>(engine::WriteClass::Spill)] = allowed ? 32 : 0;
  }

  test::TestKvStorage storage;
  engine::KvPool pool{storage, 0};
  test::TestKvTier tier;
  std::shared_ptr<model::DiskBudget> budget;
  engine::Cache cache;
  Executor model;
  Events events;
  // Whether the host holds growth back (EngineConfig::growthPaused), and
  // whether the engine marks what it allocates as memory a request in
  // service needs (EngineConfig::serving).
  bool paused = false;
  bool serving = false;
  // The engine's config, its governor among it.
  engine::EngineConfig settings;
  engine::Engine engine;
  double now = 1;
};

// Caches `count` one-page blocks, each under a state written to disk: their
// pages stay resident, and evicting one starts a demotion whose page comes
// back when the copy lands.
inline void cacheBlocksUnderDiskStates(engine::Cache &cache, uint32_t count) {
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  for (uint64_t id = 900; id < 900 + count; ++id) {
    std::vector<uint32_t> prompt(32, static_cast<uint32_t>(id));
    cache.beginRequest(id);
    require(cache.ensureTokens(id, 32).granted(), "fixture KV failed");
    const auto block = test::publishBlocks(cache, id, prompt, 32);
    cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
    cache.endRequest(id);
    require(cache.reclaimOneState(engine::StateClass::Ordinary, false) && cache.pollTransfers(),
            "state was not demoted");
  }
}

// States reach the disk tier by demotion only: the fixture publishes a RAM
// state and reclaims it while nothing else is in RAM.
inline void demoteState(engine::Cache &cache, uint64_t block) {
  auto transfer = std::make_shared<OffloadControl>();
  transfer->ready = true;
  cache.publishCompositeState(block, std::make_shared<OffloadState>(transfer), false);
  require(cache.reclaimOneState(engine::StateClass::Ordinary, false) && cache.pollTransfers() &&
              cache.snapshot().stateCache.bytes == 0,
          "fixture state did not move to disk");
}

} // namespace splash::test
