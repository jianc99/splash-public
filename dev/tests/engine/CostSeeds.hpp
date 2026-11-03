#pragma once

// An engine's prefill price seeded as a streamed target's startup seeds it
// (RuntimeBootstrap: the two prefill warm-ups and the load of the weights,
// Scheduler::seedCost), for engine tests whose commands' shapes follow the
// price: groups, passes, joins.

#include "TestExecutor.hpp"
#include "TimingProfile.hpp"
#include "engine/Engine.hpp"

#include <array>
#include <functional>

namespace splash::test {

// A streamed target whose commands' fixed part, its expert reads, is small
// beside their chunks' work: the M6's Flash-Next with an SSD 17 times as
// fast (85 GB/s, a chunk's expert reads 0.19 s) and nothing hiding them, a
// chunk of 2.22 s and 0.3 ms a row. A group still saves a wait of G (three
// chunks' reads, 0.55 s), and as the engine tests' model runs a command in
// six passes at most, every command takes six: one to four agent turns of
// 300 rows cost 2.55-2.84 s seeded, six passes of G; a group of two or three
// chunks, 5.4-8.9 s, more passes of G than six.
inline const TimingProfile &passingTarget() {
  static const TimingProfile profile = [] {
    TimingProfile result = kM6FlashNextQ2;
    result.name = "passing-target";
    result.chunkMilliseconds = 2220.0;
    result.rowMilliseconds = 0.3;
    result.weightBytesPerMillisecond = 85e6;
    result.hostFloorMilliseconds = 0.0;
    result.hiddenShare = 0.0;
    return result;
  }();
  return profile;
}

// A target whose experts are all resident and whose K/V spills, a page of
// history costing a command 80 ms to read (1 GB at the Studio's 12.4 GB/s):
// a lane that holds a few Index pages saves more than a wait of G in a
// group, and one that holds none, whose commands read nothing, runs chunks.
inline const TimingProfile &heavyHistoryTarget() {
  static const TimingProfile profile = [] {
    TimingProfile result = kStoreLessSpillFlashNext;
    result.name = "heavy-history-target";
    result.spilledPageBytes = 1'000'000'000;
    return result;
  }();
  return profile;
}

// A dense target whose prefill takes 4 ms a row and nothing else, and whose
// decode commands take 10 ms: beside a decoding lane a command halves from a
// chunk to 64 rows, the first whose time is within G (128 rows take 512 ms).
inline const TimingProfile &fourMillisecondRows() {
  static const TimingProfile profile{
      .name = "four-millisecond-rows",
      .rowMilliseconds = 4.0,
      .decodeMilliseconds = {10.0, 10.0, 10.0, 10.0},
  };
  return profile;
}

// What one of the fake model's prefill commands may hold (Executor::groupRows,
// passes), for a fixture to set before it builds the engine, which takes it
// then (Spilling's shape).
inline std::function<void(Executor &)> commandShape(uint32_t groupRows, uint32_t passes = 1) {
  return [=](Executor &model) {
    model.groupRows = groupRows;
    model.passes = passes;
  };
}

// The fake model takes `profile`'s work, and its times unless a test gives
// its own (Executor::time, commandTimes), and its warm-ups, a prompt of one
// token of a chunk's rows and of 256, and a load at its SSD rate seed the
// engine's price. A test's own times carry no work, so they train nothing:
// its commands' shapes follow the seed.
inline void seedAsStartup(engine::Engine &engine, Executor &model, const TimingProfile &profile = passingTarget()) {
  model.time(profile);
  const TimedWork &work = *model.timing;
  const std::array warmups{sampleOf(work.warmupTimes(model::ExecutionLimits::prefillTokenBudget)),
                           sampleOf(work.warmupTimes(256))};
  const double rate = profile.weightBytesPerMillisecond;
  engine.seedCost(warmups, rate ? model::SsdReads{10'000'000'000, 1e10 / rate} : model::SsdReads{});
}

} // namespace splash::test
