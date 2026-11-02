#include "LoopFixture.hpp"
#include "ProtocolPeer.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestMetalStatistics.hpp"
#include "TestRegistry.hpp"
#include "engine/MemoryControl.hpp"
#include "engine/NativeRuntime.hpp"
#include "engine/ReleasableMemory.hpp"

#include <algorithm>
#include <limits>
#include <optional>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::answeredStatus;
using splash::test::hostMemory;
using splash::test::idle;
using splash::test::LoopFixture;
using splash::test::rejects;
using splash::test::releaseAnswer;
using splash::test::request;
using splash::test::require;
using splash::test::runUntilIdle;
using splash::test::Weights;

// The control pass between commands releases the weights once the engine has
// held no request for the idle release, counted from Ready or from the end of
// the last request however long it ran, never while one is there. A request
// waits while they are written back, an image per tick, so the loop answers
// frames between them, and the images all come back even when it is cancelled,
// while memory admits them (testRefusedRestoreWaitsForMemory). A restore that
// fails other than for memory stops the engine. What the engine gives back
// while idle without a Neural Engine split, ReleasableMemory over the images
// alone, is the images: the test runs through it when `releasable`.
void testIdleWeightsAreReleasedAndRestored(bool releasable) {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double idleRelease = 1000.0 * kIdleReleaseSeconds;
  Weights weights;
  ReleasableMemory memory({&weights}, std::nullopt);
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.weights = releasable ? static_cast<model::WeightMemory *>(&memory) : &weights;
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.beginObserver = [&] {
    require(!weights.released(), "the model began a request on released weights");
  };
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 40ULL << 30, 2ULL << 30, 2ULL << 30,
                          [] { return hostMemory(64ULL << 30); }, 0);
  MemoryControl control(governor, backend, loop);
  const auto idlePass = [&](double elapsed) {
    fixture.monotonic += elapsed;
    static_cast<void>(control.run(MemoryPressure::Normal));
    return weights.released();
  };
  // Warmup before Ready is not idleness.
  fixture.monotonic += 10 * idleRelease;
  loop.announceReady();
  require(!idlePass(idleRelease - 1), "weights were released before the idle release passed");
  require(idlePass(1), "idle weights were kept past the idle release");
  require(loop.weightsSnapshot().released && loop.weightsSnapshot().restores == 0 &&
              loop.weightsSnapshot().idleReleaseSeconds == kIdleReleaseSeconds,
          "the weights' snapshot did not report their release");

  require(loop.receive(protocol::peer::serialize(request(1))) && weights.restores == 0,
          "weights were written back before a tick");
  require(loop.tick() && weights.restores == 1 && weights.released(),
          "a tick did not write back one image");
  require(loop.receive(protocol::peer::serialize(protocol::StatusRequestFrame{77})) &&
              answeredStatus(fixture, 77),
          "the loop did not answer status while it restored the weights");
  while (weights.released())
    require(loop.tick(), "a tick did not write back an image");
  require(weights.restores == Weights::kImages, "an image was written back twice");
  require(!loop.weightsSnapshot().released && loop.weightsSnapshot().restores == 1,
          "the weights' snapshot did not count one restore");
  require(!idlePass(2 * idleRelease), "weights were released under a request");
  runUntilIdle(loop);

  require(loop.receive(protocol::peer::serialize(request(2))), "a second request failed");
  // It runs past the idle release, without a control pass between commands.
  // The clock moves before each tick alone: the snapshot idle() checks reads
  // it too, for the requests' ages.
  for (uint32_t step = 0; step < 32 && !idle(loop); ++step) {
    fixture.monotonic += idleRelease;
    static_cast<void>(loop.tick());
  }
  require(idle(loop) && !idlePass(0), "weights were released as soon as a long request ended");
  require(idlePass(idleRelease), "weights were kept past the idle release after a request");

  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              loop.receive(protocol::peer::serialize(protocol::CancelFrame{3})),
          "a request cancelled while the weights came back failed");
  while (weights.released())
    require(loop.tick(), "a cancelled request stopped its weights coming back");
  require(weights.restores == 2 * Weights::kImages, "an image was written back twice");
  // The transport ticks again after a tick that progressed: the engine then
  // lets the cancelled request go.
  static_cast<void>(loop.tick());

  require(idlePass(idleRelease), "weights were kept past the idle release after a cancel");
  weights.failRestore = true;
  require(loop.receive(protocol::peer::serialize(request(4))) && !loop.tick(),
          "a failed restore kept the engine running");
  const auto events = fixture.events();
  require(std::any_of(events.begin(), events.end(),
                      [](const auto &message) {
                        const auto *error = std::get_if<protocol::ErrorEvent>(&message);
                        return error &&
                               error->failureClass == protocol::FailureClass::EngineUnhealthy &&
                               error->message.find("weights restore test") != std::string::npos;
                      }) &&
              !loop.engineHealthy() && loop.connectionMustClose(),
          "a failed restore did not stop the engine");
}

// A store of streamed experts (model::ExpertStore), the part after the images,
// goes back with them and comes back in one restore after their last; a
// release during a restore gives back what came back.
void testReleasableMemoryRestoresTheExpertsAfterTheImages() {
  Weights weights;
  struct Experts final : model::WeightMemory {
    bool held = true;
    uint32_t restores = 0;
    [[nodiscard]] bool released() const noexcept override { return !held; }
    void release() override { held = false; }
    bool restore(const metal::AllocationAdmission &, const metal::AllocationAdmission &) override {
      require(!held, "a held arena was restored");
      held = true;
      ++restores;
      return true;
    }
  } experts;
  ReleasableMemory memory({&weights, &experts}, std::nullopt);
  const metal::AllocationAdmission admit = [](uint64_t, const std::function<void()> &operation) {
    operation();
    return metal::AllocationResult{};
  };
  memory.release();
  require(memory.released() && weights.released() && experts.released(),
          "the release kept the images or the arena");
  for (uint32_t image = 0; image < Weights::kImages; ++image)
    require(!memory.restore(admit, admit) && experts.released(),
            "the arena came back before the last image");
  require(!weights.released() && memory.released(), "the restore ended without the arena");
  require(memory.restore(admit, admit) && !memory.released() && experts.restores == 1,
          "the arena did not come back in one restore after the images");
  memory.release();
  require(!memory.restore(admit, admit) && weights.held() == 1,
          "a restore did not write the first image back");
  memory.release();
  require(weights.held() == 0 && experts.released() && experts.restores == 1,
          "a release during a restore kept what came back");
}

// With the Neural Engine split, the idle release gives back the images, then
// the split's program. A request waits while the images are written back, an
// image per tick, and the program is loaded in one tick more, before the
// request begins. A program that does not load again stops the split, never
// the engine (ops::AneFfn::restore): the request runs. The stopped split gives
// back what a load that ran late may have left once, at the next idle
// release, and loads nothing, so that the images alone come back after that.
void testIdleReleaseRestoresTheSplitLast() {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double idleRelease = 1000.0 * kIdleReleaseSeconds;
  Weights weights;
  // The split's program: whether it is loaded, whether the split released
  // it, the times it was loaded again, and whether the split stopped, as one
  // whose program does not load again (`failLoad`) does.
  bool loaded = true, released = false, stopped = false, failLoad = false;
  uint32_t loads = 0;
  ReleasableMemory memory(
      {&weights}, ReleasableMemory::Split{
                   [&] {
                     require(weights.released(),
                             "the split's program was unloaded before the images were released");
                     if (released)
                       return false;
                     released = true;
                     loaded = false;
                     return true;
                   },
                   [&] {
                     require(!weights.released() && !loaded,
                             "the split's program was loaded before every image was written back");
                     if (!released || stopped)
                       return;
                     released = false;
                     ++loads;
                     if (failLoad)
                       stopped = true;
                     else
                       loaded = true;
                   }});
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.weights = &memory;
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  fixture.executor.beginObserver = [&] {
    require(!weights.released() && (loaded || stopped),
            "the model began a request before the split's program was back");
  };
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  MemoryGovernor governor(backend, 40ULL << 30, 2ULL << 30, 2ULL << 30,
                          [] { return hostMemory(64ULL << 30); }, 0);
  MemoryControl control(governor, backend, loop);
  const auto idlePass = [&] {
    fixture.monotonic += idleRelease;
    static_cast<void>(control.run(MemoryPressure::Normal));
    return memory.released();
  };
  // Sends request `id`, which ends the idle, and ticks once per image, the
  // request waiting: true while it still waits after the last image.
  const auto imagesBack = [&](uint64_t id) {
    require(loop.receive(protocol::peer::serialize(request(id))), "a request was refused");
    for (uint32_t image = 0; image < Weights::kImages; ++image)
      require(loop.tick() && loop.statusSnapshot().completed == id - 1,
              "a request ran before every image was written back");
    return loop.weightsSnapshot().restores == id - 1;
  };
  loop.announceReady();

  require(idlePass() && !loaded, "the idle release kept the split's program");
  require(imagesBack(1) && !loaded && !memory.released(),
          "the restore ended with the last image, before the split's program");
  // The weights are no longer released, but the restore is still under way: a
  // release request leaves it alone, and its answer reports it.
  const WeightsSnapshot answer = releaseAnswer(fixture, 1);
  require(!answer.released && answer.restoring && !memory.released() && !loaded,
          "a release request before the split's program was back was answered as a release");
  require(loop.tick() && loaded && loads == 1 && loop.weightsSnapshot().restores == 1,
          "the tick after the last image did not load the split's program");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1,
          "the request did not run once the program was back");

  failLoad = true;
  require(idlePass() && !loaded, "the idle release kept the split's program");
  require(imagesBack(2) && loop.tick() && stopped && loads == 2 &&
              loop.weightsSnapshot().restores == 2 && loop.engineHealthy(),
          "a program that did not load again failed the restore");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 2, "the request did not run after the split stopped");

  require(idlePass() && released, "the stopped split kept what its program may hold");
  require(imagesBack(3) && loop.tick() && loads == 2 && loop.weightsSnapshot().restores == 3,
          "a stopped split's program was loaded again");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 3 && loop.engineHealthy(),
          "a request after the split stopped did not run");
  require(idlePass(), "the idle release kept the images once the split stopped");
  require(!imagesBack(4) && loads == 2 && loop.weightsSnapshot().restores == 4,
          "the images did not come back alone once the stopped split had given its program back");
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 4, "a later request did not run");
}

// A restore takes the weights' memory under the governor's admission as
// memory the requests waiting need: the host's margins do not refuse it, and
// critical pressure does. A refused restore keeps the parts that came back
// and makes no progress; the loop tries again after the retry backoff, and
// every request waits for memory, as their resource wait reports, for the
// resource wait limit from the first refusal after the last part that came
// back. Memory that comes back in time lets the restore go on. Once the limit
// passes, every request waiting fails retryably and the weights give back the
// parts that came back, the split's program still released for the next
// restore to load last; the engine stays up, and the next request starts
// over, its wait counted afresh. A refused restore that no request waits for
// any more takes no memory past the host's margins: it gives up at once and
// counts no failure.
void testRefusedRestoreWaitsForMemory() {
  constexpr double kIdleReleaseSeconds = 2.0;
  constexpr double kWaitLimit = 400.0;
  constexpr uint64_t kReserve = 2ULL << 30;
  constexpr uint64_t kHostMemory = 64ULL << 30;
  test::metalStatistics() = {};
  metal::MetalBackend backend("unused");
  uint64_t hostAvailable = kHostMemory;
  MemoryGovernor governor(backend, 40ULL << 30, kReserve, kReserve,
                          [&] { return hostMemory(hostAvailable); }, 0);
  Weights weights;
  uint32_t programReleases = 0, programLoads = 0;
  ReleasableMemory memory({&weights}, ReleasableMemory::Split{[&] {
                                                             ++programReleases;
                                                             return true;
                                                           },
                                                           [&] { ++programLoads; }});
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.engine.resourceWaitTimeoutMilliseconds = kWaitLimit;
  config.engine.serving = [&](bool serving) { return governor.setServing(serving); };
  config.weights = &memory;
  config.weightAdmission = governor.allocationAdmission();
  LoopFixture fixture(config, {}, {}, kIdleReleaseSeconds);
  engine::NativeRuntime &loop = fixture.loop;
  MemoryControl control(governor, backend, loop);
  const auto pass = [&](MemoryPressure pressure, double elapsed = 0.0) {
    fixture.monotonic += elapsed;
    static_cast<void>(control.run(pressure));
  };
  const auto errorOf = [&](uint64_t id) -> std::optional<protocol::ErrorEvent> {
    for (const auto &message : fixture.events())
      if (const auto *error = std::get_if<protocol::ErrorEvent>(&message);
          error && error->requestId == id)
        return *error;
    return std::nullopt;
  };
  loop.announceReady();

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && programReleases == 1, "idle weights were kept");
  hostAvailable = kReserve + (512ULL << 20);
  require(governor.allocationAdmission()(Weights::kImageBytes, [] {}).failure ==
              metal::AllocationFailure::HostPressure,
          "the host's margin admitted memory no request needs");
  require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick() &&
              weights.held() == 1,
          "the host's margin held back weights a request waits for");

  pass(MemoryPressure::Critical);
  require(!loop.tick() && weights.held() == 1 && weights.restores == 1,
          "critical pressure admitted the weights' memory");
  require(loop.millisecondsUntilNextWakeup() == engine::kResourceRetryBackoffMilliseconds &&
              loop.resourceWaitSnapshot().memory == 1,
          "the request behind a refused restore did not wait for memory");
  fixture.monotonic += kWaitLimit - 1.0;
  require(!loop.tick() && weights.held() == 1 && !errorOf(1) && loop.engineHealthy() &&
              loop.resourceWaitSnapshot().oldestWaitMilliseconds == kWaitLimit - 1.0,
          "a refused restore gave up before the resource wait limit");
  hostAvailable = kHostMemory;
  pass(MemoryPressure::Normal);
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 1 && weights.restores == Weights::kImages &&
              programLoads == 1 && loop.weightsSnapshot().restores == 1,
          "the restore did not go on once memory came back");

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && programReleases == 2 &&
              loop.receive(protocol::peer::serialize(request(2))) && loop.tick() &&
              weights.held() == 1,
          "the second restore did not write back its first image");
  pass(MemoryPressure::Critical);
  require(!loop.tick(), "critical pressure admitted the second image");
  require(loop.receive(protocol::peer::serialize(request(4))) && !loop.tick() &&
              loop.resourceWaitSnapshot().memory == 2,
          "a request that came during a refused restore did not wait for memory");
  fixture.monotonic += 0.75 * kWaitLimit;
  pass(MemoryPressure::Normal);
  require(loop.tick() && weights.held() == 2, "the second image did not come back");
  pass(MemoryPressure::Critical);
  require(!loop.tick(), "critical pressure admitted the third image");
  fixture.monotonic += 0.75 * kWaitLimit;
  require(!loop.tick() && weights.held() == 2 && !errorOf(2) && !errorOf(4),
          "the resource wait limit did not restart once a part came back");
  fixture.monotonic += 0.25 * kWaitLimit;
  require(loop.tick(), "a restore past the resource wait limit made no progress");
  for (uint64_t id : {2, 4}) {
    const std::optional<protocol::ErrorEvent> error = errorOf(id);
    require(error && error->failureClass == protocol::FailureClass::RequestError &&
                error->code == "resource_timeout" && error->retryable &&
                error->message.find("macOS is short of memory") != std::string::npos,
            "a request behind a restore that gave up did not fail retryably");
  }
  require(memory.released() && weights.held() == 0 && programReleases == 2 &&
              loop.weightsSnapshot().restoreFailures == 1 &&
              loop.resourceWaitSnapshot().memory == 0 && loop.engineHealthy() &&
              !loop.connectionMustClose(),
          "a restore that gave up kept what came back, released the program again or stopped "
          "the engine");

  pass(MemoryPressure::Normal);
  static_cast<void>(loop.tick());
  require(weights.held() == 0, "a restore went on after it gave up");

  pass(MemoryPressure::Critical);
  require(loop.receive(protocol::peer::serialize(request(3))) && !loop.tick() && !errorOf(3) &&
              loop.resourceWaitSnapshot().memory == 1,
          "the next request did not wait for memory afresh");
  pass(MemoryPressure::Normal);
  runUntilIdle(loop);
  require(loop.statusSnapshot().completed == 2 && weights.restores == 2 * Weights::kImages + 2 &&
              programLoads == 2 && loop.weightsSnapshot().restores == 2,
          "the next request did not restore the weights from the first image");

  pass(MemoryPressure::Normal, 1000.0 * kIdleReleaseSeconds);
  require(memory.released() && loop.receive(protocol::peer::serialize(request(5))) &&
              loop.tick() && weights.held() == 1,
          "the last restore did not write back its first image");
  pass(MemoryPressure::Critical);
  require(!loop.tick() && loop.receive(protocol::peer::serialize(protocol::CancelFrame{5})),
          "a refused restore's request could not be cancelled");
  hostAvailable = kReserve + (512ULL << 20);
  pass(MemoryPressure::Normal);
  require(loop.tick() && memory.released() && weights.held() == 0 &&
              loop.weightsSnapshot().restoreFailures == 1 &&
              !loop.millisecondsUntilNextWakeup(),
          "a refused restore no request waits for took memory past the host's margin, went "
          "on, or counted a failure");
}

// With --idle-release off the control pass never releases the weights,
// however long the engine holds no request, but a release request still does,
// and the next request restores them.
void testIdleReleaseOffReleasesOnlyOnRequest() {
  Weights weights;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.weights = &weights;
  LoopFixture fixture(config, {}, {}, std::numeric_limits<double>::infinity());
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  fixture.monotonic += 1e12;
  loop.releaseIdleWeights();
  require(!weights.released(), "weights were released with the idle release off");
  const WeightsSnapshot answer = releaseAnswer(fixture, 5);
  require(weights.released() && answer.released && !answer.restoring &&
              answer.idleReleaseSeconds == std::numeric_limits<double>::infinity(),
          "a release request did not release the weights with the idle release off");
  require(loop.receive(protocol::peer::serialize(request(1))), "the request after the release failed");
  runUntilIdle(loop);
  require(!weights.released() && weights.restores == Weights::kImages,
          "the request did not restore the weights released with the idle release off");
  // An idle release that is not positive is refused.
  for (double seconds : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
    rejects([&] { LoopFixture refusedFixture(config, {}, {}, seconds); },
            "the idle release must be positive",
            "an idle release that is not positive was accepted");
  }
}

// A release request gives the weights back now, as the idle release does, when
// the engine holds no request and no restore is under way, and otherwise leaves
// them as they are. Either way its answer reports what the engine did: released
// and not restoring only once the weights are given back; a restore under way,
// which leaves them released until its last image, as restoring. The next
// request restores them as after the idle release.
void testReleaseRequestGivesTheWeightsBackWhenIdle() {
  Weights weights;
  ReleasableMemory memory({&weights}, std::nullopt);
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.weights = &memory;
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  require(!weights.released(), "the weights were released before any release request");

  WeightsSnapshot answer = releaseAnswer(fixture, 5);
  require(weights.released() && answer.released && !answer.restoring && answer.restores == 0,
          "an idle engine did not release its weights on request");
  // A second release request changes nothing, and is answered as the first.
  answer = releaseAnswer(fixture, 6);
  require(weights.released() && answer.released && !answer.restoring,
          "releasing released weights again failed");

  // A request restores them, an image per tick. A release request during the
  // restore leaves it alone, and its answer reports the restore.
  require(loop.receive(protocol::peer::serialize(request(1))) && loop.tick() &&
              weights.held() == 1,
          "the request did not start the restore");
  answer = releaseAnswer(fixture, 7);
  require(weights.held() == 1 && answer.released && answer.restoring,
          "a release request during the restore released the weights, or its answer did not "
          "report the restore");
  while (weights.released())
    require(loop.tick(), "a tick did not write back an image");
  require(weights.restores == Weights::kImages, "the restore was disturbed by a release request");

  // With a request in service the weights stay.
  require(loop.receive(protocol::peer::serialize(request(2))), "a second request failed");
  answer = releaseAnswer(fixture, 8);
  require(!weights.released() && !answer.released && !answer.restoring,
          "the weights were released under a request");
  runUntilIdle(loop);
  answer = releaseAnswer(fixture, 9);
  require(weights.released() && answer.released && !answer.restoring,
          "the engine did not release once its request ended");
}

// The loop reports that it holds requests from the first it takes to the end
// of its last, the span the process keeps the Mac from idle sleep for.
void testHoldingRequestsSpansFirstToLastRequest() {
  std::vector<bool> reported;
  engine::NativeLoopConfig config;
  config.engine.maxContext = 1024;
  config.holdingRequests = [&](bool held) { reported.push_back(held); };
  LoopFixture fixture(config);
  engine::NativeRuntime &loop = fixture.loop;
  loop.announceReady();
  require(reported.empty(), "a loop without requests reported holding one");
  require(loop.receive(protocol::peer::serialize(request(1))) &&
              loop.receive(protocol::peer::serialize(request(2))),
          "two requests failed");
  require(reported == std::vector<bool>{true}, "two requests did not share one span");
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false},
          "the span did not end with the last request");
  // A cancelled request completes; a request whose deadline passes fails.
  require(loop.receive(protocol::peer::serialize(request(3))) && loop.tick() &&
              loop.receive(protocol::peer::serialize(protocol::CancelFrame{3})),
          "a cancelled request failed");
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false, true, false},
          "a cancelled request did not end its span");
  require(loop.receive(protocol::peer::serialize(request(4, 64))), "request 4 failed");
  fixture.monotonic += 5000.0;
  runUntilIdle(loop);
  require(reported == std::vector<bool>{true, false, true, false, true, false},
          "a request that failed did not end its span");
  // One refused on arrival is never held.
  protocol::RequestFrame expired = request(5);
  expired.absoluteDeadlineUnixMicros = 1'000'000;
  require(loop.receive(protocol::peer::serialize(expired)) && reported.size() == 6,
          "a request refused on arrival was held");
}

} // namespace

std::span<const test::TestCase> weightsTests() {
  static constexpr test::TestCase kTests[] = {
      {"testIdleWeightsAreReleasedAndRestored(false)", [] { testIdleWeightsAreReleasedAndRestored(false); }},
      {"testIdleWeightsAreReleasedAndRestored(true)", [] { testIdleWeightsAreReleasedAndRestored(true); }},
      {"testReleasableMemoryRestoresTheExpertsAfterTheImages", testReleasableMemoryRestoresTheExpertsAfterTheImages},
      {"testIdleReleaseRestoresTheSplitLast", testIdleReleaseRestoresTheSplitLast},
      {"testRefusedRestoreWaitsForMemory", testRefusedRestoreWaitsForMemory},
      {"testIdleReleaseOffReleasesOnlyOnRequest", testIdleReleaseOffReleasesOnlyOnRequest},
      {"testReleaseRequestGivesTheWeightsBackWhenIdle", testReleaseRequestGivesTheWeightsBackWhenIdle},
      {"testHoldingRequestsSpansFirstToLastRequest", testHoldingRequestsSpansFirstToLastRequest},
  };
  return kTests;
}
