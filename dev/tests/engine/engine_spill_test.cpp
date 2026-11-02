#include "EngineTestSupport.hpp"
#include "TestChecks.hpp"
#include "TestEngine.hpp"
#include "TestExecutor.hpp"
#include "TestKvPool.hpp"
#include "TestKvTier.hpp"
#include "TestRegistry.hpp"
#include "engine/Engine.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

using namespace splash;
using namespace splash::engine;
using splash::test::Events;
using splash::test::Executor;
using splash::test::guardReleases;
using splash::test::idle;
using splash::test::OffloadControl;
using splash::test::request;
using splash::test::runUntilIdle;
using splash::test::Spilling;

namespace {

using splash::test::require;

// On a target that can spill, a request that would spill starts Resident
// while the SSD cache takes no writes and its projection fits in memory
// alone; it is counted as a Resident start. One whose projection does not fit
// fails naming the SSD cache, not the memory limits.
void testUnwritableTierKeepsRequestsResident() {
  for (const uint32_t residentTokens : {262'144U, 64U}) {
    test::TestKvStorage storage(64, 4096, 4);
    KvPool pool(storage, 0);
    test::TestKvTier tier;
    tier.writableFile = false;
    engine::Cache cache(pool, &tier, nullptr);
    Executor model;
    Events events;
    EngineConfig config = test::engineConfig();
    config.spill = {.capable = true,
                    .always = true,
                    .denseTokens = 2048,
                    .geometry = {.fullPageBytes = 423'936,
                                 .indexPageBytes = 24'576,
                                 .slotBytes = 425'984,
                                 .residentKvBytes = uint64_t{1} << 40,
                                 .residentTokens = residentTokens,
                                 .densePages = 64,
                                 .windowPages = 64,
                                 .groupPages = 256}};
    engine::Engine engine(config, cache, model, events);
    guardReleases(storage, engine);
    engine.submit(request(1, std::vector<uint32_t>(100, 3)));
    runUntilIdle(engine);
    const SpillRequestsSnapshot spill = engine.snapshot().spill;
    if (residentTokens == 64) {
      require(events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                  spill.refusedClosed == 1 &&
                  events.failureDetails.front().first.find("SSD cache") != std::string::npos &&
                  events.failureDetails.front().first.find("--max-memory") == std::string::npos,
              "a request past the resident line did not fail naming the closed SSD cache");
      continue;
    }
    require(events.completedCount == 1 && events.failedCount == 0 && spill.resident == 1 &&
                spill.spill == 0 && spill.refusedClosed == 0,
            "a request did not start Resident while the SSD cache took no writes");
  }
}

// Admission keeps a request's KV in memory where its projection, its prompt
// and min(max_tokens, 4096) tokens of output, fits the memory the plan
// leaves resident KV beside the other lanes' (a Resident lane's projection, a
// Spill lane's floor), and spills it otherwise; a request within the dense
// history never spills. Resident KV holds 200 pages here.
void testAdmissionSpillsWhatMemoryCannotHold() {
  Spilling spill(1024, false, 200 * 100);
  // Eight pages of prompt and 32K tokens of output: the output is clipped at
  // 4,096 tokens, so it projects 137 pages, which memory holds.
  spill.engine.submit(Spilling::prompt(1, 1, 8, 32'768));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  // Four pages and 4,096 tokens project 133 pages more, which it does not.
  spill.engine.submit(Spilling::prompt(2, 2, 4, 4'096));
  spill.runUntil([&] { return spill.events.startIds.size() == 2; }, "the second request did not start");
  // Half a page of prompt and 8 tokens of output stay within the dense
  // history's page.
  EngineRequest dense = request(3, std::vector<uint32_t>(16, 3));
  dense.maxNewTokens = 8;
  spill.engine.submit(dense);
  spill.runUntil([&] { return spill.events.startIds.size() == 3; }, "the third request did not start");
  std::vector<KvMode> modes;
  for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
    modes.push_back(active.kvMode);
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(modes == std::vector<KvMode>{KvMode::Resident, KvMode::Spill, KvMode::Resident} &&
              counts.resident == 2 && counts.spill == 1,
          "admission did not keep in memory exactly the requests whose projection memory holds");
  spill.engine.cancel(1);
  spill.engine.cancel(2);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "a request kept its commitment");
}

// A Spill request's commitment covers each page of its projection that has
// no copy, those of the chain it restores included: a cached chain whose
// copies were never written takes a slot a page as its new pages do, and
// each of those pages draws one as it is written.
void testSpillCommitmentCoversTheChainWithoutCopies() {
  Spilling spill(64);
  // The first turn's pages are written nowhere: the tier takes no writes
  // of Spill requests' pages.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  require(spill.tier.writes.empty() && spill.committedSlots() == 0,
          "the first turn wrote copies or kept its commitment");
  spill.allowSpillWrites(true);
  // The second turn restores the first's four pages and the state at their
  // end, and its projection holds two pages more: six, none with a copy.
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  spill.runUntil([&] { return spill.events.startIds.size() == 2; },
                 "the second turn did not start");
  require(spill.events.starts.back() == 4 * KvCache::pageTokens && spill.committedSlots() == 6,
          "the commitment left out the restored pages that have no copy");
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.tier.writes.size() == 6 &&
              std::ranges::all_of(spill.tier.draws,
                                  [](model::DiskBudget::Draw draw) {
                                    return draw == model::DiskBudget::Draw::Commitment;
                                  }) &&
              spill.committedSlots() == 0 && spill.engine.snapshot().spill.spill == 2,
          "the restored and new pages were not written on the commitment");
}

// A Resident lane that memory refuses to grow converts to Spill: it commits
// a slot for each page of its projection that has no copy, its published
// history's included, starts writing its pages through, and waits, without
// allocating anything, for the pages their copies free, then grows into
// them; so it does at the engine's limit and while the host refuses growth.
// A quota that holds the pages it has still to publish, but not its history
// too, refuses the conversion, and the lane that could not grow fails as it
// would without the SSD cache.
void testConversionCommitsTheHistoryWithoutCopies() {
  for (const auto failure :
       {metal::AllocationFailure::EngineBudget, metal::AllocationFailure::HostPressure}) {
    for (const uint32_t slots : {5U, 8U}) {
      // The host's refusal makes a lone lane that cannot convert wait rather
      // than fail: the engine's limit shows the refusal.
      if (failure == metal::AllocationFailure::HostPressure && slots == 5)
        continue;
      Spilling spill(slots, false);
      // Two Full extents beside the Index runway, and one Index extent more.
      spill.storage.budgetPages = 9;
      spill.storage.allocationFailure = failure;
      // Seven pages of prompt and 40 tokens of output reach a ninth page past
      // 248 tokens; the projection then is eight whole pages, none copied.
      spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
      spill.runUntilIdle();
      const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
      if (slots == 5) {
        require(counts.converted == 0 && spill.tier.writes.empty() &&
                    spill.events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                    spill.events.failureDetails.front().first.find("SSD cache") !=
                        std::string::npos,
                "a conversion the quota could not hold with the history went ahead");
      } else {
        require(counts.converted == 1 && counts.resident == 1 &&
                    spill.events.completedCount == 1 && spill.events.usage.at(1).second == 40 &&
                    spill.tier.writes.size() == 8 && spill.storage.allocatedBytes() <= 900,
                "the converted lane did not wait for its spilled pages and grow into them");
      }
      require(spill.committedSlots() == 0, "the lane kept its commitment");
    }
  }
}

// On a target whose KV can live on the SSD cache, a Resident lane the
// engine's limit refuses to grow converts to Spill before a chunk of the
// expert store goes: its KV then gives memory back as its copies land, the
// store every decode step reads keeps its chunks, and the lane decodes its
// whole output. A lane that cannot convert, its quota short of its history,
// keeps all its KV in memory, as on a target without the SSD cache, and
// takes the chunk.
void testResidentLaneConvertsBeforeTheExpertStore() {
  for (const uint32_t slots : {8U, 5U}) {
    Spilling spill(slots, false);
    // Two Full extents beside the Index runway, and one Index extent more.
    spill.storage.budgetPages = 9;
    spill.model.expertChunks = {1000};
    // A chunk given back is room the engine's limit then gives the KV.
    spill.model.expertReleaseObserver = [&] { spill.storage.budgetPages += 8; };
    spill.engine.submit(Spilling::prompt(1, 1, 7, 40));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.failedCount == 0 &&
                spill.events.usage.at(1).second == 40,
            "the lane did not decode its whole output");
    if (slots == 8)
      require(counts.converted == 1 && spill.model.expertChunks.size() == 1 &&
                  spill.model.releasedExpertChunks.empty(),
              "a lane that could convert to Spill took a chunk of the expert store first");
    else
      require(counts.converted == 0 && spill.model.releasedExpertChunks.size() == 1,
              "a lane that could not convert did not take a chunk of the expert store");
    require(spill.committedSlots() == 0, "the lane kept its commitment");
  }
}

// A Resident lane whose history has copies already, as the chain it restored
// from an earlier Spill request's has them, converts at the memory wall as a
// lane without them does: its pages whose copies landed spill at once, and
// it decodes its whole output rather than ending at the wall.
void testConversionSpillsAHistoryWithCopies() {
  // Resident KV holds 20 pages.
  Spilling spill(400, false, 2'000);
  // Ten pages and 4,096 tokens of output do not fit: the first turn spills,
  // its pages written through, and is cancelled while it decodes.
  spill.engine.submit(Spilling::prompt(1, 1, 10, 4'096));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first turn did not decode");
  for (int step = 0; step < 50; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  spill.engine.cancel(1);
  spill.runUntilIdle();
  // With 120 tokens of output the second turn fits and starts Resident: it
  // restores the first's chain whole, with its copies, and meets the wall of
  // the 14 pages memory holds while it decodes (the first turn's spill gave
  // back the Full extents it emptied).
  spill.storage.budgetPages = 14;
  spill.engine.submit(Spilling::prompt(2, 1, 10, 120));
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.converted == 1 && counts.lengthEnds == 0 && spill.events.usage.contains(2) &&
              spill.events.usage.at(2).second == 120 && spill.committedSlots() == 0,
          "the lane whose history had copies did not spill them when it converted");
}

// A start that converts to Spill when memory refuses its restore starts as a
// Spill request, counted as one. Refused all the same, as while its writes
// have not landed beside a resident lane, it decides its mode afresh at each
// attempt and counts no conversion, however often it retries.
void testAStartThatConvertsCountsNoConversion() {
  Spilling spill(64, false);
  spill.model.deniedReservations = 1000;
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  // The first turn's state goes to the disk: the second restores it.
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  // States take memory from here on: refused, the resident lane's replay
  // point would start copies of the first turn's KV, which hold back the
  // writes of the conversion below.
  spill.model.deniedReservations = 0;
  // A resident lane decodes within its last page.
  spill.engine.submit(Spilling::prompt(3, 3, 3, 30));
  spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the resident lane did not decode");
  // The second turn's restore is refused memory; it converts, and the
  // writes of its chain do not land for 2 s.
  spill.storage.growthBlocked = true;
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  for (int step = 0; step < 20; ++step) {
    spill.now += 100;
    static_cast<void>(spill.engine.tick(spill.now));
  }
  require(spill.events.startIds.size() == 2 && spill.tier.writes.size() == 4 &&
              spill.engine.snapshot().spill.converted == 0,
          "the start that converted and waited counted its conversions, or did not convert");
  // Once memory grows again it starts Resident, its next attempt's mode.
  spill.storage.growthBlocked = false;
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(spill.events.completedCount == 3 && counts.converted == 0 && counts.spill == 0 &&
              counts.resident == 3,
          "a start was not counted once, by the mode it started in");
}

// Short of its projection, a Spill start that the quota of active KV holds
// with a page of output is bounded to what the quota holds, which the model
// is given, and ends there; one that other Spill requests' commitments keep
// out waits for them, and admits nothing that arrived after it meanwhile;
// one that the quota cannot hold even alone is refused, retrying would
// fail the same way.
void testShortQuotaBoundsWaitsOrRefusesASpillStart() {
  {
    // 3 pages of prompt and 200 tokens of output project 9 pages: 5 hold
    // the prompt and 63 tokens.
    Spilling spill(5);
    spill.engine.submit(Spilling::prompt(1, 1, 3, 200));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 63 &&
                spill.model.outputLimits.at(1) == 63 && counts.bounded == 1 &&
                counts.lengthEnds == 1 && spill.events.boundedIds == std::vector<uint64_t>{1} &&
                spill.tier.writes.size() == 5 && spill.committedSlots() == 0,
            "a start the quota holds with a page of output was not bounded to it");
  }
  {
    Spilling spill(10);
    // The first request commits 8 of the 10 slots.
    spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
    spill.runUntil([&] { return spill.events.outputs.contains(1); },
                   "the first request did not decode");
    // The second needs 4 for its prompt and a page of output, and waits; the
    // third, which arrives then, would fit in the 2 left.
    spill.engine.submit(Spilling::prompt(2, 2, 3, 200));
    spill.runUntil([&] { return spill.engine.resourceWaitSnapshot(spill.now).diskQuota == 1; },
                   "the second request did not wait for the quota");
    spill.engine.submit(Spilling::prompt(3, 3, 1, 31));
    for (int step = 0; step < 5; ++step) {
      static_cast<void>(spill.engine.tick(spill.now++));
      spill.tier.complete();
    }
    const ResourceWaitSnapshot waits = spill.engine.resourceWaitSnapshot(spill.now);
    require(spill.events.startIds == std::vector<uint64_t>{1} && waits.diskQuota == 1 &&
                waits.memory == 0 && waits.heldBehindRefusal == 1,
            "a start waiting for the quota let a later arrival take it");
    spill.runUntilIdle();
    require(spill.events.startIds == std::vector<uint64_t>{1, 2, 3} &&
                spill.events.completedCount == 3 && spill.events.usage.at(2).second == 200 &&
                spill.events.usage.at(3).second == 31 && spill.events.boundedIds.empty() &&
                spill.committedSlots() == 0,
            "the waiting start did not run once the quota it waited for came back, or ended bounded");
  }
  {
    Spilling spill(3);
    spill.engine.submit(Spilling::prompt(1, 1, 3, 200));
    spill.runUntilIdle();
    require(spill.events.failures == std::vector<std::string>{"kv_disk_quota"} &&
                !spill.events.failureDetails.front().second &&
                spill.events.failureDetails.front().first.find("--max-cache-disk") !=
                    std::string::npos &&
                spill.engine.snapshot().spill.refusedQuota == 1 && spill.committedSlots() == 0,
            "a start the quota cannot hold even alone was not refused naming it");
  }
}

// A start that cannot spill, the tier closed or the quota short of it,
// starts Resident where its projection fits in memory alone, as it would
// without the SSD cache: within the resident context, beyond the resident
// KV the plan leaves beside the readers of spilled KV, which it does not use.
void testAStartThatCannotSpillStartsResidentWithinTheResidentContext() {
  for (const bool closed : {true, false}) {
    // Resident KV beside other lanes holds 20 pages, one request alone 50.
    Spilling spill(8, false, 20 * 100, 50 * KvCache::pageTokens);
    spill.tier.writableFile = !closed;
    // 40 pages of prompt and 64 tokens of output project 43 pages; the 8
    // slots of the quota cannot hold them.
    spill.engine.submit(Spilling::prompt(1, 1, 40, 64));
    spill.runUntilIdle();
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    require(spill.events.completedCount == 1 && spill.events.usage.at(1).second == 64 &&
                counts.resident == 1 && counts.spill == 0 && counts.refusedClosed == 0 &&
                counts.refusedQuota == 0 && spill.tier.writes.empty(),
            closed ? "a start that fits the resident context was refused while the tier took no writes"
                   : "a start that fits the resident context was refused for a short quota");
  }
}

// A start that would spill while the write budget (--max-cache-disk-writes)
// does not cover what Spill requests committed, and that memory does not
// hold alone, is refused with kv_write_budget, which says when a retry may
// succeed: once the budget, refilling over hours, covers them.
void testWriteBudgetRefusalSaysWhenToRetry() {
  // Every start spills, and none fits in memory alone.
  Spilling spill(64, true, 100'000, KvCache::pageTokens);
  // 1,600 bytes a day, 400 at once: four slots.
  spill.budget->setWriteBudget(1'600);
  // The first request commits the eight pages of its projection and writes
  // none of them.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 2, 191));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the first request did not decode");
  require(spill.committedSlots() == 8, "the first request did not commit its projection");
  // The budget covers the 800 bytes committed once it refilled 400 more,
  // a quarter of a day's: in 6 hours.
  spill.engine.submit(Spilling::prompt(2, 2, 1, 31));
  spill.runUntil([&] { return spill.events.failedCount == 1; }, "the second request was not refused");
  require(spill.events.failures == std::vector<std::string>{"kv_write_budget"} &&
              spill.events.failureDetails.front().second &&
              spill.events.retryAfters == std::vector<uint32_t>{21'600} &&
              spill.engine.snapshot().spill.refusedWriteBudget == 1,
          "the write budget's refusal did not say when a retry may succeed");
  spill.engine.cancel(1);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the first request kept its commitment");
}

// On a target whose prefill groups save only a spilling lane's reads, its
// experts loaded whole, a Spill request's long prompt runs as groups and a
// Resident one's as chunks, as without the SSD cache: a request that arrives
// beside a Resident one waits for a chunk, not a group.
void testOnlySpillingLanesRunPrefillGroups() {
  for (const bool spills : {false, true}) {
    Spilling spill(256, spills);
    spill.model.groupRows = 4 * model::ExecutionLimits::prefillTokenBudget;
    spill.model.groupsOnlySpilling = true;
    // Groups run once prefill has a measured cost, as after the start's.
    spill.engine.observePrefill(model::ExecutionLimits::prefillTokenBudget, 1000.0, 0.0);
    // 128 pages and a token: three chunks, or one group.
    spill.engine.submit(Spilling::prompt(1, 1, 128, 1));
    spill.runUntilIdle();
    require(spill.events.completedCount == 1 && spill.model.prefillChunks.at(1) == (spills ? 1u : 3u),
            spills ? "a Spill request's prompt did not run as a prefill group"
                   : "a Resident request's prompt ran as a prefill group");
  }
}

// The Full extents a spill empties leave memory, but one, the runway, while
// a spilling prompt still grows into it or the host holds growth back.
void testSpilledFullExtentsLeaveOnceNoPromptGrows() {
  for (const bool paused : {false, true}) {
    Spilling spill(1024);
    const auto emptyFullExtents = [&] {
      const KvPoolSnapshot pool = spill.pool.snapshot();
      return (pool.reclaimableBytes - pool.index.reclaimableBytes) /
             (uint64_t{spill.pool.extentPages()} * spill.pool.bytesPerPage());
    };
    // 70 pages and a token: two prefill commands, the second after the
    // first's pages spilled, then 64 tokens of output.
    spill.paused = paused;
    spill.engine.submit(Spilling::prompt(1, 1, 70, 64));
    uint64_t prompting = 0;
    spill.runUntil(
        [&] {
          if (spill.events.outputs.contains(1)) return true;
          prompting = std::max(prompting, emptyFullExtents());
          return false;
        },
        "the request did not decode");
    spill.runUntil([&] { return spill.cache.spillSnapshot().spilledPages == 68; },
                   "the prompt's pages did not spill");
    require(spill.model.prefillChunks.at(1) == 2 && prompting <= 1 && emptyFullExtents() == (paused ? 1u : 0u),
            paused ? "the Full extents spills emptied did not keep the runway while the host held growth back"
                   : "the Full extents spills emptied stayed, or the runway went while the prompt grew");
    spill.paused = false;
    spill.runUntilIdle();
  }
}

// While the host holds growth back, a Spill prompt's landed pages move to
// Index pages past the runway too: the Index extents they take are memory
// the request in service needs, as its Full pages are, and free many times
// what they take. A prompt that the engine's limit holds with its pages
// spilled, but not as Full pages, completes with them spilled.
void testSpillMovesThroughTheHostPause() {
  Spilling spill(1024);
  // As the governor does, the host refuses growth unless a request in
  // service needs it, then the engine's limit: 90 Full pages.
  spill.storage.budgetPages = 90;
  spill.storage.growthAllowed = [&](uint32_t) {
    const bool host = spill.paused && !spill.serving;
    spill.storage.allocationFailure =
        host ? metal::AllocationFailure::HostPressure : metal::AllocationFailure::EngineBudget;
    return !host;
  };
  spill.paused = true;
  // 120 pages and a token: two prefill commands, the second after 62 of the
  // first's 64 pages spilled, 58 of them past the runway's four Index pages.
  // As Full pages, the prompt's 121 outgrow the limit.
  spill.engine.submit(Spilling::prompt(1, 1, 120, 8));
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 1 &&
              spill.events.usage.at(1).second == 8 && spill.model.prefillChunks.at(1) == 2 &&
              spill.cache.spillSnapshot().spilledPages == 118,
          "a Spill prompt kept its landed pages Full under the host's pause until the engine's limit");
}

// A Spill request suspended for a higher priority returns its commitment
// and commits again for what it still projects when it resumes; short of
// that, the quota bounds its output there too, and the model is given that
// bound. Each attempt bounds it afresh: a bounded attempt that could not
// resume leaves the next one the whole output.
void testSpillResumeCommitsAgain() {
  for (const bool roomLater : {false, true}) {
    Spilling spill(20);
    // No copies: the request holds its whole history without any.
    spill.allowSpillWrites(false);
    EngineRequest background = Spilling::prompt(1, 1, 2, 100);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                   "the request did not decode");
    // A Normal start is refused memory while it is resident, which
    // suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; },
                   "the Normal request did not run");
    require(spill.model.suspensions == 1 && spill.committedSlots() == 0,
            "the suspended request kept its commitment");
    // Its projection is 5 pages, none copied; 4 hold its history and a page
    // of output, 63 tokens in all.
    spill.allowSpillWrites(true);
    spill.budget->setActiveLimitBytes(4 * 100);
    if (roomLater) {
      const uint32_t attempts = spill.model.resumeAttempts;
      spill.runUntil([&] { return spill.model.resumeAttempts > attempts; },
                     "the request did not try to resume");
      require(spill.model.outputLimits.at(1) == 63, "the refused resume was not bounded");
      spill.budget->setActiveLimitBytes(20 * 100);
    }
    spill.model.resumeDenied = false;
    spill.runUntil([&] { return spill.model.resumptions == 1; }, "the request did not resume");
    require(spill.committedSlots() == (roomLater ? 5 : 4),
            "the resumed request did not commit for what it still projects");
    spill.runUntilIdle();
    const uint32_t output = roomLater ? 100 : 63;
    require(spill.events.completedCount == 2 && spill.events.usage.at(1).second == output &&
                spill.model.outputLimits.at(1) == output && spill.committedSlots() == 0,
            roomLater ? "a bound of a resume that did not start outlived it"
                      : "the resumed request was not bounded to what the quota holds");
  }
}

// A Spill request suspended with spilled pages before the SSD cache stopped
// taking writes resumes once it takes none: the tier's closure refuses new
// Spill starts alone, and the resume's commitment, which no write draws
// there, goes back as it ends.
void testSpillResumeOnAClosedTier() {
  Spilling spill(20);
  // Its prompt's second and third pages spill.
  EngineRequest background = Spilling::prompt(1, 1, 4, 100);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 10; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which
  // suspends it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  require(spill.model.suspensions == 1 && spill.cache.spillSnapshot().spilledPages != 0,
          "the request was not suspended with spilled pages");
  spill.tier.writableFile = false;
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 2 &&
              spill.model.resumptions == 1 && spill.events.usage.at(1).second == 100 &&
              spill.committedSlots() == 0,
          "a Spill request suspended before the tier closed did not resume");
}

// A Spill request resumed after a suspension holds the anchor its model
// selected under its whole output, which the model decodes only with output
// left past it: it emits a last budgeted anchor as it selects it. The quota
// bounds a resume's output only to pages that hold a token past that anchor:
// a resume whose quota holds the pages up to its anchor and none past them
// ends at its length; one more page bounds its output to that page's end.
void testSpillResumeBoundLeavesOutputPastItsAnchor() {
  for (const uint32_t slots : {3U, 4U}) {
    Spilling spill(20);
    // No copies: the request holds its whole history without any.
    spill.allowSpillWrites(false);
    EngineRequest background = Spilling::prompt(1, 1, 2, 100);
    background.priority = RequestPriority::Background;
    spill.engine.submit(std::move(background));
    // 65 tokens of prompt and 30 of output end a token short of the third
    // page, which the anchor ends.
    spill.runUntil([&] { return spill.events.outputs[1].size() == 30; },
                   "the request did not decode");
    // A Normal start is refused memory while it is resident, which
    // suspends it.
    spill.model.beginGrowthBlocked = [&] {
      return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
    };
    spill.model.resumeDenied = true;
    spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
    spill.runUntil([&] { return spill.events.completedCount == 1; },
                   "the Normal request did not run");
    spill.budget->setActiveLimitBytes(uint64_t{slots} * 100);
    spill.model.resumeDenied = false;
    spill.runUntilIdle();
    const bool ends = slots == 3;
    require(spill.model.suspensions == 1 && spill.model.resumptions == (ends ? 0u : 1u) &&
                spill.events.completedCount == 2 &&
                spill.events.usage.at(1).second == (ends ? 30u : 63u) &&
                spill.events.boundedIds == std::vector<uint64_t>{1} &&
                spill.engine.snapshot().spill.lengthEnds == 1 && spill.committedSlots() == 0,
            ends ? "a resume whose quota held no output past its anchor did not end at its length"
                 : "a resume was not bounded to the output past its anchor that the quota holds");
  }
}

// A Spill request suspended in its prompt holds no anchor: the step that
// completes its prompt selects its first token under the output limit its
// resume is given. A quota of exactly its prompt's pages bounds its output
// to that token, which it emits.
void testSpillResumeInItsPromptHoldsNoAnchor() {
  Spilling spill(20);
  // No copies: the request holds its whole history without any.
  spill.allowSpillWrites(false);
  // 95 tokens of prompt end a token short of the third page, and each
  // prefill command takes a page of them.
  spill.model.groupRows = KvCache::pageTokens;
  EngineRequest background = request(1, std::vector<uint32_t>(3 * KvCache::pageTokens - 1, 1));
  background.maxNewTokens = 100;
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.model.prefillChunks[1] == 1; }, "the prompt did not start");
  // A Normal start is refused memory while it is resident, which suspends
  // it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  require(spill.model.suspensions == 1 &&
              spill.engine.snapshot().activeRequests.front().promptProcessed <
                  3 * KvCache::pageTokens - 1,
          "the request was not suspended in its prompt");
  spill.budget->setActiveLimitBytes(3 * 100);
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  require(spill.model.resumptions == 1 && spill.model.outputLimits.at(1) == 1 &&
              spill.events.completedCount == 2 && spill.events.usage.at(1).second == 1 &&
              spill.events.boundedIds == std::vector<uint64_t>{1} && spill.committedSlots() == 0,
          "a resume in its prompt was not bounded to the token its prompt's pages leave");
}

// A Spill request suspended for a higher priority resumes with the chain it
// restores and its replay's first page, as a start that restores one does,
// not with all the pages it held, as Full pages at once: its replay takes
// Full pages a command at a time, which leave as its blocks are found cached
// or spill as their copies land, as its output's did. A replay longer than
// memory holds as Full pages resumes and completes.
void testSpillResumeTakesItsReplaysPagesAsItRuns() {
  Spilling spill(1024);
  // 100 Full pages: a prefill command's 64 fit beside the Index pages of the
  // request's spilled history, and its 125 pages do not.
  spill.storage.budgetPages = 100;
  spill.model.decodeTokens = 8;
  EngineRequest background = Spilling::prompt(1, 1, 4, 4'096);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 3'840; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which suspends
  // it.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntilIdle();
  require(spill.model.suspensions == 1 && spill.model.resumptions == 1 &&
              spill.events.failedCount == 0 && spill.events.completedCount == 2 &&
              spill.events.usage.at(1).second == 4'096,
          "a Spill request whose replay memory holds only a command at a time as Full pages "
          "did not resume");
}

// A Resident request suspended for a higher priority resumes with the KV it
// held, all Full, at once. Where memory no longer holds that, it converts to
// Spill there, and then takes its replay's first page, as a Spill request
// resumes, rather than ask for all of it as Full pages again.
void testAResumeThatConvertsTakesItsReplaysPagesAsItRuns() {
  // Requests spill only where memory refuses them.
  Spilling spill(1024, false);
  spill.model.decodeTokens = 8;
  EngineRequest background = Spilling::prompt(1, 1, 4, 4'096);
  background.priority = RequestPriority::Background;
  spill.engine.submit(std::move(background));
  // 2,400 tokens of output: 80 pages, all Full.
  spill.runUntil([&] { return spill.events.outputs[1].size() >= 2'400; },
                 "the request did not decode");
  // A Normal start is refused memory while it is resident, which suspends
  // it, and its resume is refused until memory has shrunk.
  spill.model.beginGrowthBlocked = [&] {
    return spill.model.lastBeginId == 2 && spill.model.requests.at(1).resident;
  };
  spill.model.resumeDenied = true;
  spill.engine.submit(Spilling::prompt(2, 2, 1, 1));
  spill.runUntil([&] { return spill.events.completedCount == 1; },
                 "the Normal request did not run");
  // Memory shrinks to 72 Full pages, which hold a prefill command's 64 and
  // not the 80 it held: a critical pass releases what the cache held.
  spill.storage.budgetPages = 72;
  static_cast<void>(spill.engine.reclaimMemory({.critical = true}));
  spill.model.resumeDenied = false;
  spill.runUntilIdle();
  const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
  require(counts.resident == 2 && counts.converted == 1 && spill.events.failedCount == 0 &&
              spill.events.completedCount == 2 && spill.events.usage.at(1).second == 4'096,
          "a Resident request that converted as it resumed asked for its whole replay as Full "
          "pages");
}

// A decoding lane whose history holds Index pages reads them through the hot
// block cache: the model refuses its floor there, and decode runs the other
// lanes without it, which waits rather than fails while they hold memory.
void testDecodeLeavesOutALaneRefusedItsHotFloor() {
  Spilling spill(64);
  spill.model.refusedSpilledReads = {2};
  spill.engine.submit(Spilling::prompt(1, 1, 5, 20));
  spill.engine.submit(Spilling::prompt(2, 2, 5, 20));
  spill.runUntil([&] { return spill.events.usage.contains(1); },
                 "the lane given its floor did not finish");
  require(spill.cache.holdsIndexPages(2) && !spill.events.outputs.contains(2) &&
              spill.events.failedCount == 0,
          "a lane refused its hot floor decoded or failed beside a lane that ran");
  spill.model.refusedSpilledReads.clear();
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.usage.at(2).second == 20,
          "the lane refused its hot floor did not run once it had it");
}

// A prefill lane holds memory between its commands that a Spill lane's
// first decode may need for its floor of the hot block cache, as a
// streaming prompt holds its history's staging. Refused it beside a lane
// that decodes, the lane takes it from the prefill lane, which yields before
// it, as when no lane fits: it decodes while the prompt waits, rather than
// once the decoding lane or the prompt has ended. It does not wait for the
// pages written through after each command either, which a prompt keeps in
// flight and which free too few pages to empty an extent: they return no
// memory.
void testARefusedDecodeTakesThePrefillLanesMemory() {
  for (const bool writesInFlight : {false, true}) {
    // With writes in flight, extents of 64 pages, which the requests share.
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{writesInFlight ? 64u : 4u});
    // Lane 3's floor fits only while the prompt's lane is not resident.
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      const auto prompt = spill.model.requests.find(2);
      return id != 3 || prompt == spill.model.requests.end() || !prompt->second.resident;
    };
    // Prefill takes 4 ms a row: beside a decoding lane, a command runs 64
    // rows, two pages.
    spill.model.commandTimes = [](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      uint32_t rows = 0;
      for (const BatchItem &item : plan.items)
        rows += item.tokenCount;
      return test::CommandTimes{plan.kind == WorkKind::Decode ? 10.0 : 4.0 * rows};
    };
    // With writes in flight, the tier lands every transfer but the writes of
    // the pages the prompt's lane holds, which stay in flight. No restore
    // runs: its writes and transfers pair up in order.
    std::unordered_set<uint32_t> promptPages;
    const auto run = [&](const std::function<bool()> &done, const char *message) {
      for (uint32_t step = 0; step < 2000 && !done(); ++step) {
        static_cast<void>(spill.engine.tick(spill.now++));
        const auto prompt = spill.model.requests.find(2);
        if (writesInFlight && prompt != spill.model.requests.end() && prompt->second.resident)
          for (const uint32_t page : spill.cache.pageTable(2).pages)
            promptPages.insert(page);
        for (size_t index = 0; index < spill.tier.writes.size(); ++index)
          spill.tier.transfers.at(index)->ready =
              !promptPages.contains(spill.tier.writes[index].page);
      }
      require(done(), message);
    };
    spill.engine.submit(Spilling::prompt(1, 1, 5, 200));
    run([&] { return spill.events.outputs.contains(1); }, "the decoding lane did not start");
    // The prompt runs three commands, whose pages but its first and last
    // the tier would spill.
    spill.engine.submit(Spilling::prompt(2, 2, 40, 1));
    run([&] { return spill.model.prefillChunks[2] >= 3; }, "the prompt did not run");
    spill.engine.submit(Spilling::prompt(3, 3, 5, 8));
    run(
        [&] {
          return spill.events.outputs.contains(3) || spill.events.usage.contains(1) ||
                 spill.events.usage.contains(2);
        },
        "no request decoded or ended");
    require(spill.events.outputs.contains(3) && !spill.events.usage.contains(1) &&
                !spill.events.usage.contains(2) &&
                spill.engine.snapshot().resourceSuspensions == 1,
            writesInFlight
                ? "a lane refused its hot floor waited for writes that return no memory"
                : "a lane refused its hot floor waited for the decoding lane or the prompt to end");
    spill.runUntilIdle();
    require(spill.events.completedCount == 3 && spill.events.failedCount == 0 &&
                spill.events.usage.at(3).second == 8,
            "the lanes did not all complete once the prompt had yielded");
  }
}

// The model's memory for a lane's reads of spilled KV is memory the
// governor funds: only an extent's bytes or a state's buffers make room for
// it. Refused a lane's floor of the hot block cache, the engine evicts no KV
// leaf whose page cannot empty an extent, as an idle spilled chain's Index
// pages cannot, and asks for the floor again only once memory has come
// back, not after each leaf.
void testRefusedModelMemoryTakesWhatGivesMemoryBack() {
  // Extents of 64 pages, which the requests share.
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{64});
  // An earlier conversation leaves its spilled chain cached, its pages
  // holding their index.
  const std::vector<uint32_t> earlier(40 * KvCache::pageTokens + 1, 1);
  spill.engine.submit(request(1, earlier));
  spill.runUntilIdle();
  const uint32_t indexPages = spill.cache.spillSnapshot().spilledPages;
  require(spill.cache.lookup(earlier, {}).kvBoundary == 40 * KvCache::pageTokens && indexPages != 0,
          "the earlier conversation did not leave its spilled chain cached");
  // A lane decodes beside one refused its floor.
  uint32_t attempts = 0;
  bool room = false;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    attempts += id == 3;
    return id != 3 || room;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 5, 200));
  spill.engine.submit(Spilling::prompt(3, 3, 5, 8));
  spill.runUntil([&] { return attempts != 0; }, "the lane did not ask for its floor");
  require(attempts <= 4 && spill.cache.spillSnapshot().spilledPages >= indexPages,
          "a refused floor evicted Index pages, asking for it again after each");
  room = true;
  spill.runUntilIdle();
  require(spill.events.completedCount == 3 && spill.events.failedCount == 0,
          "the lane refused its floor did not run once it had it");
}

// A cached chain leaves memory leaf first, and a leaf a state on the SSD
// needs is written before it goes: its block keeps its page until the copy
// lands, and only then is the block above it a leaf. So a chain whose newest
// blocks have no copy, as a finished lane's prompt past the chain it restored
// or a cancelled cold prompt, gives its memory back one landed write at a
// time, the blocks with a copy at once after them. A Spill lane refused its
// floor of the hot block cache, left alone by such a neighbour, waits for
// those writes rather than fail while the memory it lacks comes back.
void testALoneLaneWaitsForTheChainANeighbourLeft() {
  for (const bool restored : {true, false}) {
    // Extents of 16 pages, which the requests share. Every state goes
    // straight to the SSD. The quota holds the Spill lane's commitment but
    // not the neighbour's, which cannot convert to Spill then.
    Spilling spill(128, true, 100'000, 0, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    // An earlier turn leaves its chain spilled, its copies written through.
    if (restored) {
      spill.engine.submit(Spilling::prompt(1, 1, 40, 1));
      spill.runUntilIdle();
    }
    // The neighbour starts Resident, as a request does while the SSD cache
    // takes no writes: it restores that chain into Full pages, its copies
    // kept, or none, and runs eight pages of prompt past it.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(2, 1, 48, 10000));
    spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the neighbour did not decode");
    spill.tier.writableFile = true;
    // The Spill lane's floor fits once two extents have left memory since it
    // first asked for it.
    std::optional<uint64_t> limit;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 3)
        return true;
      if (!limit)
        limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
      return spill.storage.allocatedBytes() <= *limit;
    };
    spill.engine.submit(Spilling::prompt(3, 3, 20, 8));
    spill.runUntil([&] { return limit.has_value(); }, "the Spill lane did not ask for its floor");
    // The neighbour finishes, or is cancelled.
    if (restored)
      spill.model.decodeFinishes = true;
    else
      spill.engine.cancel(2);
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.outputs.contains(3),
            restored ? "a lone lane refused its floor failed while a finished neighbour's chain was written"
                     : "a lone lane refused its floor failed while a cancelled neighbour's chain was written");
    require(std::ranges::count_if(spill.tier.writes,
                                  [](const test::TestKvTier::Write &write) {
                                    return write.write == WriteClass::Reclaim;
                                  }) > 1,
            "the neighbour's chain left memory without writes one after another");
  }
}

// A lane that waits for memory on its way back keeps it from the lanes that
// yield before it. A Spill lane refused its floor of the hot block cache
// reclaims a cached chain whose leaves are written to the SSD cache before
// they go, a landed write at a time, and its floor fits once two extents
// have left memory. Beside it a Resident prompt, which yields before a
// decoding lane, would take the pages of each write as it lands, and the
// floor would wait for as long as the prompt grows: the prompt waits
// instead, and the lane decodes before the prompt has ended. Where the model
// runs a prompt's command in passes, a command between its passes takes
// nothing more and goes on; no new one starts while the lane waits.
void testAWaitingFloorKeepsTheMemoryOnItsWayBack() {
  for (const uint32_t passes : {1u, 4u}) {
    // Extents of 16 pages. Every state goes straight to the SSD. The quota
    // holds the Spill lane's commitment but not the prompt's, which cannot
    // convert to Spill then.
    Spilling spill(128, true, 100'000, 0, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    std::vector<std::pair<uint32_t, uint32_t>> promptPasses;
    spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem>) {
      if (plan.kind == WorkKind::Prefill && plan.items.front().requestId == 3)
        promptPasses.emplace_back(plan.pass, plan.passes);
      return test::CommandTimes{};
    };
    spill.model.passes = passes;
    spill.engine.observePrefill(model::ExecutionLimits::prefillTokenBudget, 4900.0, 3400.0);
    // While the SSD cache takes no writes, an earlier turn leaves its chain
    // cached without copies, and the prompt starts Resident.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(1, 1, 48, 1));
    spill.runUntilIdle();
    spill.engine.submit(Spilling::prompt(3, 3, 120, 4096));
    spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
    spill.tier.writableFile = true;
    // The Spill lane's first decode command reads; from its second on,
    // which follows the prompt's next command or its first pass, its floor
    // fits once two extents have left memory since it was first refused.
    uint32_t asks = 0;
    std::optional<uint64_t> limit;
    size_t refused = 0;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 2 || ++asks == 1)
        return true;
      if (!limit) {
        limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
        refused = promptPasses.size();
      }
      return spill.storage.allocatedBytes() <= *limit;
    };
    spill.engine.submit(Spilling::prompt(2, 2, 20, 8));
    spill.runUntil(
        [&] {
          return (spill.events.outputs.contains(2) && spill.events.outputs.at(2).size() > 1) ||
                 spill.events.outputs.contains(3);
        },
        "neither the Spill lane nor the prompt decoded");
    require(limit && spill.engine.snapshot().spill.converted == 0,
            "the fixture's floor or quota is off");
    require(spill.events.outputs.contains(2) && !spill.events.outputs.contains(3),
            "a prompt that yields before a lane waiting for its floor took the memory on its way "
            "back");
    const auto waited = std::span(promptPasses).subspan(refused);
    if (passes > 1)
      require(std::ranges::any_of(waited, [](const auto &each) { return each.first != 0; }) &&
                  std::ranges::none_of(waited, [](const auto &each) { return each.first == 0; }),
              "a prompt between its command's passes stopped, or started a command, while a lane it "
              "yields before waited for its floor");
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
            "the lanes did not all complete");
  }
}

// Two Spill lanes decode beside a cached chain whose leaves are written to
// the SSD cache before they go, a landed write at a time. Their pages spill
// at once, and from their next decode command on each needs its floor of the
// hot block cache, which takes memory, an extent, and fits only once that
// memory comes back. The lane with less history, which yields before the
// other and is planned first, takes no floor while the other waits for its
// own: the other decodes first. A held lane keeps its wait and retries once
// memory lands or after the backoff, so a tick that runs nothing leaves the
// loop a later wakeup. Where the lane that yields first holds its floor
// already, it decodes within what it holds while the other waits, and the
// hot block cache does not grow toward its target until the other has its
// floor.
void testAHeldSpillLaneLeavesTheFloorToTheWaiter() {
  for (const bool holdsFloor : {false, true}) {
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    const auto prompt = [](uint64_t id, uint32_t token, uint32_t pages) {
      EngineRequest result = Spilling::prompt(id, token, pages, 400);
      result.deadlineMilliseconds = 1e9;
      return result;
    };
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(1, 1, 48));
    spill.runUntilIdle();
    spill.tier.writableFile = true;
    // A floor takes an extent beside the KV pool. Once the floors run short,
    // the first needs two extents back and the second one more.
    constexpr uint64_t floorBytes = 16 * 100;
    std::optional<uint64_t> budget;
    const auto charged = [&] {
      return spill.storage.allocatedBytes() + floorBytes * spill.model.spilledReadFloors.size();
    };
    uint32_t refusals = 0;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (spill.model.spilledReadFloors.contains(id))
        return true;
      if (!budget) {
        if (holdsFloor && id == 2)
          return true;
        budget = charged() - floorBytes;
      }
      const bool fits = charged() + floorBytes <= *budget;
      refusals += !fits;
      return fits;
    };
    // The lane with less history is submitted, and so planned, first. Both
    // decode, their pages' writes in flight.
    spill.engine.submit(prompt(2, 2, 4));
    spill.engine.submit(prompt(3, 3, 6));
    for (uint32_t step = 0; step < 100 && !(spill.events.outputs.contains(2) &&
                                            spill.events.outputs.contains(3));
         ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(spill.events.outputs.contains(2) && spill.events.outputs.contains(3) && !budget,
            "the lanes did not decode before their pages spilled");
    std::optional<double> shortFloor;
    std::optional<double> longFloor;
    std::optional<uint32_t> growthAtRefusal;
    uint32_t growthWhileWaiting = 0;
    size_t shortOutputs = 0;
    size_t shortOutputsAtRefusal = 0;
    for (uint32_t step = 0; step < 4000 && !(shortFloor && longFloor); ++step) {
      // The growth of a tick that grants the floor follows it.
      const uint32_t growthBefore = spill.model.growthAttempts;
      const bool ran = spill.engine.tick(spill.now += 50);
      if (!ran && !spill.engine.commandInFlight()) {
        const std::optional<double> wakeup = spill.engine.nextWakeupMilliseconds();
        require(!wakeup || *wakeup > spill.now,
                "a tick that ran nothing left the loop a wakeup that was due");
      }
      // Writes land every eighth tick.
      if (step % 8 == 0)
        spill.tier.complete();
      // From the refusal that follows the first, a write is on its way.
      if (refusals > 1 && !growthAtRefusal) {
        growthAtRefusal = spill.model.growthAttempts;
        shortOutputsAtRefusal = spill.events.outputs.at(2).size();
      }
      if (!shortFloor && spill.model.spilledReadFloors.contains(2))
        shortFloor = spill.now;
      if (!longFloor && spill.model.spilledReadFloors.contains(3)) {
        longFloor = spill.now;
        shortOutputs = spill.events.outputs.at(2).size() - shortOutputsAtRefusal;
        growthWhileWaiting = growthBefore - growthAtRefusal.value_or(growthBefore);
      }
    }
    require(budget && shortFloor && longFloor && spill.events.failedCount == 0,
            "the lanes did not both get their floors");
    if (!holdsFloor) {
      require(*longFloor < *shortFloor,
              "a Spill lane that yields before one waiting for its floor took the memory on its "
              "way back");
    } else {
      require(shortOutputs > 1, "a lane that holds its floor did not decode while another waited");
      require(growthWhileWaiting == 0,
              "the hot block cache grew while a lane waited for memory on its way back");
    }
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
            "the lanes did not all complete");
  }
}

// A start refused memory on its way back holds no lane: only a resident
// lane's wait leaves its memory to the lanes that yield before it. Beside a
// start that waits for a cached chain's writes to land, a Resident prompt
// goes on growing.
void testAWaitingStartHoldsNoLane() {
  Spilling spill(128, true, 100'000, 0, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(1, 1, 48, 1));
  spill.runUntilIdle();
  spill.engine.submit(Spilling::prompt(3, 3, 120, 4096));
  spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
  spill.tier.writableFile = true;
  // The start's state fits once two extents have left memory since it was
  // first refused.
  std::optional<uint64_t> limit;
  spill.model.beginGrowthBlocked = [&] {
    if (spill.model.lastBeginId != 2)
      return false;
    if (!limit)
      limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
    return spill.storage.allocatedBytes() > *limit;
  };
  const size_t pages = spill.cache.pageTable(3).pages.size();
  const auto started = [&] {
    const auto found = spill.model.requests.find(2);
    return found != spill.model.requests.end() && found->second.resident;
  };
  spill.engine.submit(Spilling::prompt(2, 2, 20, 8));
  spill.runUntil([&] { return started() || spill.cache.pageTable(3).pages.size() > pages; },
                 "neither the start nor the prompt went on");
  require(limit && !started() && spill.cache.pageTable(3).pages.size() > pages,
          "a start waiting for memory on its way back held a resident prompt");
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 3,
          "the requests did not all complete");
}

// A held lane takes no KV pages past its capacity anywhere in its command:
// its command ends at the state boundary its pages reach. A Resident prompt
// whose first command, beside another prompt's rows, ended inside the page
// of its replay point holds the KV up to that point, and its next command,
// planned past it, runs while a Spill lane waits for its floor of the hot
// block cache: it reaches the point, takes no page past it, and goes on
// once the lane has its floor.
void testAHeldLaneEndsItsCommandAtItsCapacity() {
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  const auto prompt = [](uint64_t id, uint32_t token, uint32_t tokens) {
    EngineRequest result = request(id, std::vector<uint32_t>(tokens, token));
    result.maxNewTokens = 400;
    result.deadlineMilliseconds = 1e9;
    return result;
  };
  // An earlier turn leaves its chain cached without copies, its state on
  // the SSD; the states of the turns after it have buffers.
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(1, 1, 48 * KvCache::pageTokens + 1));
  spill.runUntilIdle();
  spill.tier.writableFile = true;
  spill.model.deniedReservations = 0;
  const auto processed = [&](uint64_t id) {
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      if (active.id == id)
        return active.promptProcessed;
    return 0u;
  };
  bool room = false;
  spill.model.spilledReadsRoom = [&](uint64_t id) { return id != 2 || room; };
  // The Spill lane decodes, its pages' writes in flight.
  spill.engine.submit(prompt(2, 2, 4 * KvCache::pageTokens + 1));
  for (uint32_t step = 0; step < 100 && !spill.events.outputs.contains(2); ++step)
    static_cast<void>(spill.engine.tick(spill.now += 50));
  require(spill.events.outputs.contains(2) && !spill.cache.holdsIndexPages(2),
          "the Spill lane did not decode before its pages spilled");
  // Two prompts start Resident: one command runs the short one's 30 rows and
  // 2018 of the long one's, inside the page that ends at its replay point,
  // 2048.
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(3, 3, 30));
  spill.engine.submit(prompt(4, 4, 2060));
  for (uint32_t step = 0; step < 100 && processed(4) == 0; ++step)
    static_cast<void>(spill.engine.tick(spill.now += 50));
  spill.tier.writableFile = true;
  require(processed(4) == 2018 &&
              spill.cache.pageTable(4).pages.size() == 64,
          "the long prompt's first command did not end inside its replay point's page");
  // The Spill lane's pages spill; from its next decode command it waits for
  // its floor while a cached leaf's write, which does not land, is in flight.
  spill.tier.complete();
  for (uint32_t step = 0; step < 100 && processed(4) == 2018; ++step) {
    static_cast<void>(spill.engine.tick(spill.now += 50));
    for (const auto &transfer : spill.tier.transfers)
      if (transfer->submitted && transfer->write != WriteClass::Reclaim)
        transfer->ready = true;
  }
  require(spill.cache.holdsIndexPages(2) && spill.events.outputs.at(2).size() == 1,
          "the Spill lane did not wait for its floor");
  require(processed(4) == 2048 &&
              spill.cache.pageTable(4).pages.size() == 64,
          "a held lane's command took KV pages past its capacity");
  room = true;
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 4,
          "the requests did not complete");
}

// A command that a lane's arrival would regroup, while a lane it yields before
// waits for memory on its way back, loses only the passes the regroup costs: a
// Resident prompt's command in passes, planned again with a short arrival,
// keeps the KV it holds and runs without the arrival, which takes no page while
// the Spill lane waits for its floor, and is not planned again at every
// boundary. The arrival runs once the lane has its floor. So it does where the
// command's lane is cancelled meanwhile, which ends it at once and leaves the
// memory to the waiting lane.
void testAJoinerHeldBackLeavesACommandItsPasses() {
  for (const bool cancelled : {false, true}) {
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    std::vector<std::tuple<uint64_t, uint32_t, uint32_t>> prefills;
    spill.model.commandTimes = [&](const BatchPlan &plan, std::span<const ModelBatchItem> items) {
      if (plan.kind == WorkKind::Prefill)
        for (const ModelBatchItem &item : items)
          prefills.emplace_back(item.requestId, plan.pass, plan.passes);
      return plan.kind == WorkKind::Decode
                 ? test::CommandTimes{.wallMilliseconds = 100.0}
                 : test::CommandTimes{4900.0 / plan.passes, 3400.0 / plan.passes};
    };
    spill.engine.observePrefill(model::ExecutionLimits::prefillTokenBudget, 4900.0, 3400.0);
    const auto prompt = [](uint64_t id, uint32_t token, uint32_t tokens) {
      EngineRequest result = request(id, std::vector<uint32_t>(tokens, token));
      result.maxNewTokens = 400;
      result.deadlineMilliseconds = 1e9;
      return result;
    };
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(1, 1, 48 * KvCache::pageTokens + 1));
    spill.runUntilIdle();
    spill.tier.writableFile = true;
    spill.model.deniedReservations = 0;
    bool room = false;
    spill.model.spilledReadsRoom = [&](uint64_t id) { return id != 2 || room; };
    spill.engine.submit(prompt(2, 2, 4 * KvCache::pageTokens + 1));
    for (uint32_t step = 0; step < 100 && !spill.events.outputs.contains(2); ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(spill.events.outputs.contains(2) && !spill.cache.holdsIndexPages(2),
            "the Spill lane did not decode before its pages spilled");
    // A short prompt starts Resident and runs its command's first pass.
    spill.model.passes = 4;
    spill.tier.writableFile = false;
    spill.engine.submit(prompt(3, 3, 60));
    const auto passesOf = [&](uint64_t id, uint32_t pass) {
      return std::ranges::count_if(prefills, [&](const auto &each) {
        return std::get<0>(each) == id && std::get<1>(each) == pass;
      });
    };
    for (uint32_t step = 0; step < 100 && !passesOf(3, 0); ++step)
      static_cast<void>(spill.engine.tick(spill.now += 50));
    require(std::get<2>(prefills.back()) == 4, "the prompt's command did not run in passes");
    spill.tier.writableFile = true;
    // The Spill lane's pages spill; from its next decode command it waits for
    // its floor while a cached leaf's write, which does not land, is in flight.
    spill.tier.complete();
    const size_t outputs = spill.events.outputs.at(2).size();
    spill.engine.submit(prompt(4, 4, 300));
    const auto step = [&] {
      static_cast<void>(spill.engine.tick(spill.now += 50));
      for (const auto &transfer : spill.tier.transfers)
        if (transfer->submitted && transfer->write != WriteClass::Reclaim)
          transfer->ready = true;
    };
    for (uint32_t tick = 0; tick < 200 && passesOf(3, 0) < 2; ++tick)
      step();
    require(passesOf(3, 0) == 2, "the arrival did not regroup the command");
    if (cancelled) {
      step();
      spill.engine.cancel(3);
      require(spill.engine.snapshot().cancelled == 1, "a command's only lane cancelled between its passes stayed");
      for (uint32_t tick = 0; tick < 8; ++tick)
        step();
    } else {
      for (uint32_t tick = 0; tick < 200 && !passesOf(3, 3); ++tick)
        step();
      require(passesOf(3, 0) == 2 && passesOf(3, 3) == 1,
              "a command planned again with an arrival held back did not run its passes to its last without it");
    }
    require(spill.cache.holdsIndexPages(2) && spill.events.outputs.at(2).size() == outputs,
            "the Spill lane did not wait for its floor");
    require(!passesOf(4, 0), "an arrival took pages while a lane it yields before waited for its floor");
    room = true;
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    const EngineSnapshot counts = spill.engine.snapshot();
    require(passesOf(4, 0) != 0 && counts.resourceSuspensions == 0 && counts.failed == 0 &&
                counts.cancelled == cancelled && spill.events.completedCount == 4,
            "the arrival did not run once the lane had its floor, or a lane yielded");
  }
}

// A lane whose memory on its way back never lands holds the lanes that yield
// before it no longer than the resource wait's limit from its first
// refusal. The Spill lane refused its floor waits for writes of a cached
// chain that do not land, while a lane admitted before it, which decodes
// with more of its history and yields after it, works and so keeps its wait
// open: the Resident prompt that yields before it waits within the limit,
// and runs past it, while the lane still waits.
void testAWaitThatNeverLandsHoldsLanesUpToItsLimit() {
  Spilling spill(128, true, 100'000, 0, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  // Prefill takes 4 ms a row: beside a decoding lane, a command runs 64
  // rows, two pages.
  spill.model.commandTimes = [](const BatchPlan &plan, std::span<const ModelBatchItem>) {
    uint32_t rows = 0;
    for (const BatchItem &item : plan.items)
      rows += item.tokenCount;
    return test::CommandTimes{plan.kind == WorkKind::Decode ? 10.0 : 4.0 * rows};
  };
  // No request reaches its deadline here.
  const auto prompt = [](uint64_t id, uint32_t token, uint32_t pages, uint32_t output) {
    EngineRequest result = Spilling::prompt(id, token, pages, output);
    result.deadlineMilliseconds = 1e9;
    return result;
  };
  spill.tier.writableFile = false;
  spill.engine.submit(prompt(1, 1, 40, 1));
  spill.runUntilIdle();
  spill.engine.submit(prompt(4, 4, 24, 100'000));
  spill.runUntil([&] { return spill.events.outputs.contains(4); }, "the decoding lane did not start");
  spill.engine.submit(prompt(3, 3, 120, 4096));
  spill.runUntil([&] { return spill.model.prefillChunks[3] != 0; }, "the prompt did not run");
  spill.tier.writableFile = true;
  // From the Spill lane's second decode command on, its floor fits only
  // once two extents have left memory, which only the cached chain's
  // writes, which never land, would give back.
  uint32_t asks = 0;
  std::optional<uint64_t> limit;
  std::optional<double> refusedAt;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    if (id != 2 || ++asks == 1)
      return true;
    if (!limit) {
      limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
      refusedAt = spill.now;
    }
    return spill.storage.allocatedBytes() <= *limit;
  };
  // Ticks 50 ms apart, landing every transfer but those writes.
  const auto run = [&](const std::function<bool()> &done, const char *message) {
    for (uint32_t step = 0; step < 4000 && !done(); ++step) {
      static_cast<void>(spill.engine.tick(spill.now += 50));
      for (const auto &transfer : spill.tier.transfers)
        if (transfer->submitted && transfer->write != WriteClass::Reclaim)
          transfer->ready = true;
    }
    require(done(), message);
  };
  spill.engine.submit(prompt(2, 2, 20, 8));
  run([&] { return refusedAt.has_value(); }, "the Spill lane was not refused its floor");
  const double wait = kResourceWaitTimeoutMilliseconds;
  const auto held = [&] { return spill.cache.pageTable(3).pages.size(); };
  const size_t pages = held();
  run([&] { return spill.now >= *refusedAt + wait - 100 || held() != pages; },
      "time did not pass");
  require(held() == pages && spill.events.outputs.at(2).size() == 1 &&
              spill.events.failedCount == 0,
          "the prompt grew, or the Spill lane decoded or failed, within the wait's limit");
  run([&] { return held() != pages; },
      "a wait that never landed held the prompt past its limit");
  require(spill.now <= *refusedAt + wait + 1000 && spill.events.outputs.at(2).size() == 1 &&
              spill.events.failedCount == 0,
          "the prompt grew late, or only once the Spill lane's wait had ended");
  spill.tier.complete();
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0 && spill.events.completedCount == 4,
          "the lanes did not all complete");
}

// A Spill lane refused its floor of the hot block cache converts the Resident
// neighbour that holds the most KV to Spill before a chunk of the expert
// store goes: the neighbour's pages leave as their copies land, the floor
// fits, and the store every decode step reads keeps its chunks. A neighbour
// that restored a chain whose copies landed keeps those pages Full in the
// command its conversion came in, whose lanes' reads were reserved before,
// and spills them from the next. A neighbour that cannot convert, its
// projection beyond the quota, keeps its KV in memory, and the chunk goes.
void testAFloorConvertsAResidentNeighbourBeforeTheExpertStore() {
  for (const uint32_t slots : {1024U, 128U})
  for (const bool restored : {false, true}) {
    Spilling spill(slots, true, 100'000, 0, Spilling::Extent{16});
    spill.model.expertChunks = {1000};
    // An earlier turn leaves its chain spilled, its copies written through.
    if (restored) {
      spill.engine.submit(Spilling::prompt(1, 1, 40, 1));
      spill.runUntilIdle();
    }
    // The neighbour starts Resident, as a request does while the SSD cache
    // takes no writes: it restores that chain into Full pages, its copies
    // kept, or none.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(2, 1, 48, 10000));
    spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the neighbour did not decode");
    spill.tier.writableFile = true;
    // The Spill lane's first decode command reads; from its second on, which
    // its neighbour leads, its floor fits once two extents have left memory
    // since it was first refused, or a chunk of the store went.
    uint32_t asked = 0;
    std::optional<uint64_t> limit;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 3 || ++asked == 1)
        return true;
      if (!limit)
        limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
      return spill.storage.allocatedBytes() <= *limit || !spill.model.releasedExpertChunks.empty();
    };
    spill.engine.submit(Spilling::prompt(3, 3, 20, 64));
    spill.runUntil(
        [&] {
          return limit && (spill.storage.allocatedBytes() <= *limit ||
                           !spill.model.releasedExpertChunks.empty());
        },
        "the Spill lane's floor did not fit");
    const SpillRequestsSnapshot counts = spill.engine.snapshot().spill;
    if (slots == 1024)
      require(counts.converted == 1 && spill.model.releasedExpertChunks.empty(),
              "a floor took a chunk of the expert store before a neighbour that could convert");
    else
      require(counts.converted == 0 && spill.model.releasedExpertChunks.size() == 1,
              "a floor beside a neighbour that could not convert did not take a chunk of the store");
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.committedSlots() == 0,
            "a request failed or kept its commitment");
  }
}

// A Spill lane refused its floor of the hot block cache converts the Resident
// lane holding the most KV, and, while that lane's copies bring an extent's
// pages back as they land, no other: the next lane keeps its KV in memory. A
// lane whose Full pages past those a Spill lane keeps would not outweigh the
// hot block cache its reads add is passed over, and the store gives a chunk
// back instead.
void testAFloorConvertsTheLargestLaneThatGivesMemoryBack() {
  for (const uint64_t laneReadBytes : {0UL, 10'000UL}) {
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16}, laneReadBytes);
    spill.model.expertChunks = {1000};
    // Two neighbours start Resident, as requests do while the SSD cache takes
    // no writes: 24 and 48 pages.
    spill.tier.writableFile = false;
    spill.engine.submit(Spilling::prompt(4, 4, 24, 10000));
    spill.engine.submit(Spilling::prompt(5, 5, 48, 10000));
    spill.runUntil([&] { return spill.events.outputs.contains(4) && spill.events.outputs.contains(5); },
                   "the neighbours did not decode");
    spill.tier.writableFile = true;
    // The Spill lane's floor fits once two extents have left memory since it
    // first asked for it, or a chunk of the store went.
    std::optional<uint64_t> limit;
    spill.model.spilledReadsRoom = [&](uint64_t id) {
      if (id != 3)
        return true;
      if (!limit)
        limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
      return spill.storage.allocatedBytes() <= *limit || !spill.model.releasedExpertChunks.empty();
    };
    spill.engine.submit(Spilling::prompt(3, 3, 20, 8));
    spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the Spill lane did not decode");
    std::unordered_map<uint64_t, KvMode> modes;
    for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
      modes[active.id] = active.kvMode;
    if (!laneReadBytes)
      require(modes.at(5) == KvMode::Spill && modes.at(4) == KvMode::Resident &&
                  spill.engine.snapshot().spill.converted == 1 && spill.model.releasedExpertChunks.empty(),
              "the floor did not convert the largest lane alone before the store");
    else
      require(modes.at(5) == KvMode::Resident && modes.at(4) == KvMode::Resident &&
                  spill.engine.snapshot().spill.converted == 0 &&
                  spill.model.releasedExpertChunks.size() == 1,
              "a lane whose reads would cost more than spilling frees converted");
    spill.model.decodeFinishes = true;
    spill.runUntilIdle();
    require(spill.events.failedCount == 0 && spill.committedSlots() == 0,
            "a request failed or kept its commitment");
  }
}

// The copies a converted lane owes free its Full pages as they land, a few
// writes at a time where an extent holds more pages than the tier writes at
// once, as the production pool's do: while they bring an extent's pages
// back, a Spill lane refused its floor again waits for them, and no other
// lane converts.
void testOwedCopiesHoldBackTheNextConversion() {
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{32});
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 4;
  spill.model.expertChunks = {1000};
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(4, 4, 24, 10000));
  spill.engine.submit(Spilling::prompt(5, 5, 48, 10000));
  spill.runUntil([&] { return spill.events.outputs.contains(4) && spill.events.outputs.contains(5); },
                 "the neighbours did not decode");
  spill.tier.writableFile = true;
  // The Spill lane's floor fits once an extent has left memory since it
  // first asked for it, or a chunk of the store went.
  std::optional<uint64_t> limit;
  spill.model.spilledReadsRoom = [&](uint64_t id) {
    if (id != 3)
      return true;
    if (!limit)
      limit = spill.storage.allocatedBytes() - 32 * 100;
    return spill.storage.allocatedBytes() <= *limit || !spill.model.releasedExpertChunks.empty();
  };
  spill.engine.submit(Spilling::prompt(3, 3, 20, 8));
  spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the Spill lane did not decode");
  std::unordered_map<uint64_t, KvMode> modes;
  for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
    modes[active.id] = active.kvMode;
  require(modes.at(5) == KvMode::Spill && modes.at(4) == KvMode::Resident &&
              spill.engine.snapshot().spill.converted == 1 && spill.model.releasedExpertChunks.empty(),
          "a lane converted, or the store went, while a converted lane's copies brought an extent back");
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0, "a request failed");
}

// A start refused at the engine's limit converts the Resident lane holding
// the most KV and waits for what its pages give back as their copies land.
// They land while a command runs and leave memory before the start tries
// again, which then fits: no other lane converts, and the store keeps its
// chunk.
void testARefusedStartTakesWhatItsConversionFreed() {
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
  // The tier writes every copy the conversion owes at once.
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 64;
  spill.model.expertChunks = {1000};
  // Two neighbours start Resident, as requests do while the SSD cache takes
  // no writes: 16 and 40 pages.
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(4, 4, 16, 10000));
  spill.engine.submit(Spilling::prompt(5, 5, 40, 10000));
  spill.runUntil([&] { return spill.events.outputs.contains(4) && spill.events.outputs.contains(5); },
                 "the neighbours did not decode");
  spill.tier.writableFile = true;
  // The start's state fits once an extent has left memory since it was
  // first refused, or a chunk of the store went.
  std::optional<uint64_t> limit;
  spill.model.beginGrowthBlocked = [&] {
    if (spill.model.lastBeginId != 3)
      return false;
    if (!limit)
      limit = spill.storage.allocatedBytes() - 16 * 100;
    return spill.storage.allocatedBytes() > *limit && spill.model.releasedExpertChunks.empty();
  };
  spill.engine.submit(Spilling::prompt(3, 3, 1, 8));
  spill.runUntil([&] { return spill.events.outputs.contains(3); }, "the start did not decode");
  std::unordered_map<uint64_t, KvMode> modes;
  for (const ActiveRequestSnapshot &active : spill.engine.snapshot().activeRequests)
    modes[active.id] = active.kvMode;
  require(modes.at(5) == KvMode::Spill && modes.at(4) == KvMode::Resident &&
              spill.engine.snapshot().spill.converted == 1 && spill.model.releasedExpertChunks.empty(),
          "a refused start climbed on before the pages its conversion freed left memory");
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0, "a request failed");
}

// A lane whose admission is not complete, its state's restore from the SSD
// cache in flight, holds the chain it restores, which only its admission's
// own conversion counts: a pass the host is short for converts no such lane,
// and the store gives its chunk back.
void testALaneBeingAdmittedDoesNotConvert() {
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
  // Every state goes straight to the SSD.
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.engine.submit(Spilling::prompt(1, 1, 40, 1));
  spill.runUntilIdle();
  // The follow-up starts Resident, as a request does while the SSD cache
  // takes no writes, and waits for its state's restore.
  std::vector<uint32_t> history(40 * KvCache::pageTokens + 1, 1);
  history.insert(history.end(), spill.events.outputs.at(1).begin(), spill.events.outputs.at(1).end());
  history.insert(history.end(), 64, 5);
  spill.tier.writableFile = false;
  const uint64_t reads = spill.model.diskReads;
  spill.engine.submit(request(2, history));
  spill.runUntil([&] { return spill.model.diskReads > reads; }, "the follow-up did not restore its state");
  spill.tier.writableFile = true;
  spill.model.expertChunks = {1000};
  static_cast<void>(spill.engine.reclaimMemory(
      {.targetBytes = 100'000, .upTo = ReclaimClass::Ordinary, .hostShort = true}));
  require(spill.engine.snapshot().spill.converted == 0 && spill.model.releasedExpertChunks.size() == 1,
          "a pass converted a lane whose admission was not complete");
  spill.model.restoreControl->ready = true;
  spill.runUntilIdle();
  require(spill.events.failedCount == 0, "the follow-up failed");
}

// A lone Spill lane's pages spill as its own copies land, which its
// write-through writes a few at a time, fewer than an extent holds. Refused
// its floor of the hot block cache once its first pages spilled, it waits
// for the copies it still owes rather than fail: written, they free its
// Full pages an extent at a time, and no lane grows into them. So it does
// when the memory governor's pass, between commands, collects the copies
// that landed, as the next ones start when it is refused again; while the
// copies being written are its dense history's, whose pages stay Full, as
// the owed ones that follow them free theirs; and at a wall that leaves no
// room for another Index extent, as the Full extents the spill empties fund
// it.
void testALoneSpillLaneWaitsForItsOwnCopies() {
  struct Case final {
    uint32_t writes;
    bool governorCollects;
    bool wall;
    const char *message;
  };
  for (const Case &each :
       {Case{4, false, false, "a lone Spill lane failed while its own copies were freeing its memory"},
        Case{4, true, false, "a lone Spill lane failed once the governor's pass collected its copies"},
        Case{1, false, false, "a lone Spill lane failed while its dense history's copy was written"},
        Case{4, false, true, "a lone Spill lane failed at a wall the Index runway could not refill at"}}) {
    // Extents of 16 pages. Every state goes straight to the SSD.
    Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
    spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
    spill.model.stateTier = std::make_shared<OffloadControl>();
    spill.model.stateTier->ready = true;
    spill.model.restoreControl->ready = true;
    // The tier writes a few of the lane's pages at a time, and lands none of
    // them before the lane decodes.
    spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = each.writes;
    bool land = false;
    const auto run = [&](const std::function<bool()> &done, const char *message) {
      for (uint32_t step = 0; step < 4000 && !done(); ++step) {
        static_cast<void>(spill.engine.tick(spill.now++));
        if (!land)
          continue;
        spill.tier.complete();
        if (each.governorCollects && !spill.engine.commandInFlight())
          static_cast<void>(spill.engine.reclaimMemory({.targetBytes = 1'000'000'000}));
      }
      require(done(), message);
    };
    // Its floor fits once two extents have left memory since it first asked
    // for it.
    std::optional<uint64_t> limit;
    spill.model.spilledReadsRoom = [&](uint64_t) {
      if (!limit)
        limit = spill.storage.allocatedBytes() - 2 * 16 * 100;
      return spill.storage.allocatedBytes() <= *limit;
    };
    spill.engine.submit(Spilling::prompt(1, 1, 60, 64));
    run([&] { return spill.events.outputs.contains(1); }, "the lane did not decode");
    require(!limit && !spill.cache.holdsIndexPages(1), "a page spilled before its copy landed");
    // At the wall, the memory left beside its Full extents and the Index
    // runway holds less than another Index extent.
    if (each.wall)
      spill.storage.budgetPages =
          static_cast<uint32_t>((spill.storage.allocatedBytes() + 16 * 10 / 2) / 100);
    land = true;
    run([&] { return idle(spill.engine); }, "the lane did not end");
    require(limit.has_value() && spill.events.failedCount == 0 &&
                spill.events.usage.at(1).second == 64,
            each.message);
  }
}

// A turn that restores a Spill chain without all its copies, as the retry of
// a turn the memory wall ended does, has no copy being written before its
// first command. Alone, refused its prefill history's staging there, it
// starts writing that chain and waits for those copies rather than fail.
void testALaneRefusedBeforeItsFirstCommandWritesItsChain() {
  // Extents of 16 pages. Every state goes straight to the SSD.
  Spilling spill(1024, true, 100'000, 0, Spilling::Extent{16});
  spill.model.deniedReservations = std::numeric_limits<uint32_t>::max();
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.model.restoreControl->ready = true;
  spill.tier.writeLimits[static_cast<size_t>(WriteClass::Spill)] = 4;
  // The first turn ends after its first token, its copies owed no more, and
  // the writes it started land.
  spill.engine.submit(Spilling::prompt(1, 1, 60, 1));
  spill.runUntilIdle();
  for (uint32_t step = 0; step < 5; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  require(spill.cache.spillSnapshot().writeThroughInFlight == 0,
          "the first turn's writes were still in flight");
  // The next turn's staging fits once an extent has left memory.
  const uint64_t limit = spill.storage.allocatedBytes() - 16 * 100;
  spill.model.refusedStaging = {2};
  spill.engine.submit(Spilling::prompt(2, 1, 60, 64));
  for (uint32_t step = 0; step < 4000 && !idle(spill.engine); ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
    if (spill.storage.allocatedBytes() <= limit)
      spill.model.refusedStaging.clear();
  }
  require(idle(spill.engine), "the requests did not end");
  require(spill.events.failedCount == 0 && spill.events.usage.at(2).second == 64,
          "a lone lane refused its staging failed while the chain it restored was written");
}

// A Spill lane's page spills once its own copy lands, whatever the pages
// before it: one whose write is still in flight holds back none past it.
void testAPageWaitingForItsCopyHoldsBackNoneAfterIt() {
  Spilling spill(1024);
  spill.engine.submit(Spilling::prompt(1, 1, 12, 64));
  // Every write lands but that of the lane's first page past its dense
  // history.
  std::optional<uint32_t> held;
  for (uint32_t step = 0; step < 2000; ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    const auto lane = spill.model.requests.find(1);
    if (!held && lane != spill.model.requests.end() && lane->second.resident &&
        spill.cache.pageTable(1).pages.size() > 1)
      held = spill.cache.pageTable(1).pages[1];
    for (size_t index = 0; index < spill.tier.writes.size(); ++index)
      if (spill.tier.transfers.at(index)->submitted && spill.tier.writes[index].page != held)
        spill.tier.transfers.at(index)->ready = true;
    if (spill.events.outputs.contains(1) && spill.cache.spillSnapshot().spilledPages >= 9)
      break;
  }
  require(held && spill.pool.form(*held) == kv::PageForm::Full &&
              spill.cache.spillSnapshot().spilledPages >= 9,
          "a page whose copy had not landed held back the pages after it");
  spill.engine.cancel(1);
  for (uint32_t step = 0; step < 2000 && !idle(spill.engine); ++step) {
    static_cast<void>(spill.engine.tick(spill.now++));
    spill.tier.complete();
  }
  require(idle(spill.engine), "the lane did not end");
}

// A lane's memory for its reads of spilled KV is reserved once every lane of
// the command has its KV: a lane that converts to Spill moves the landed
// pages of the blocks it shares with Spill lanes to the index form, those
// of a lane admitted before it in the command too. Here a Spill lane and a
// Resident one decode on the same prompt, whose copies the Spill lane
// wrote: those pages stay Full while the Resident lane holds them, so the
// Spill lane reads no Index page. Refused a page at the wall, the Resident
// lane, after it in the command, converts and its pages spill: both lanes
// read them through their floors of the hot block cache.
void testAConversionLeavesNoLaneWithoutItsFloor() {
  Spilling spill(1024);
  // The Spill lane writes its copies only once the Resident lane holds them.
  spill.allowSpillWrites(false);
  spill.engine.submit(Spilling::prompt(1, 1, 4, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(1); }, "the Spill lane did not decode");
  // The Resident lane starts while the SSD cache takes no writes.
  spill.tier.writableFile = false;
  spill.engine.submit(Spilling::prompt(2, 1, 4, 200));
  spill.runUntil([&] { return spill.events.outputs.contains(2); }, "the Resident lane did not decode");
  spill.tier.writableFile = true;
  spill.allowSpillWrites(true);
  // Lane 2's next command needs a page that lane 1's does not.
  const auto needsPage = [&](uint64_t id) {
    const uint64_t tokens = 4 * KvCache::pageTokens + 1 + spill.events.outputs[id].size() +
                            model::ExecutionLimits::targetVerifyRows;
    return tokens > uint64_t{spill.cache.pageTable(id).pages.size()} * KvCache::pageTokens;
  };
  spill.runUntil(
      [&] {
        return !spill.engine.commandInFlight() &&
               spill.cache.copiedBlocks(spill.cache.blockAt(1, 4 * KvCache::pageTokens)) == 4 &&
               needsPage(2) && !needsPage(1);
      },
      "lane 2 did not come to need a page that lane 1 does not");
  require(spill.engine.snapshot().spill.spill == 1 && spill.engine.snapshot().spill.resident == 1 &&
              !spill.cache.holdsIndexPages(1) &&
              spill.cache.pageTable(1).pages.front() == spill.cache.pageTable(2).pages.front(),
          "the lanes did not share the prompt, its pages Full beside their copies");
  // At the wall: no free Full page, and no growth.
  spill.storage.growthBlocked = true;
  const KvPageAcquisition held = spill.pool.acquirePages(spill.pool.freePageCount());
  static_cast<void>(spill.engine.tick(spill.now++));
  require(held.granted() && spill.engine.commandInFlight() &&
              spill.engine.snapshot().spill.converted == 1 && spill.cache.holdsIndexPages(1) &&
              spill.model.spilledReadFloors.contains(1) && spill.model.spilledReadFloors.contains(2),
          "a lane whose pages a later lane's conversion spilled read them without its floor");
  for (const uint32_t page : held.pages)
    spill.pool.releasePage(page, false);
  spill.storage.growthBlocked = false;
  spill.model.decodeFinishes = true;
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.failedCount == 0,
          "the lanes did not complete after the conversion");
}

// A Spill lane whose pages the SSD cache takes no more of keeps them Full.
// Alone at the memory wall, a decoding lane then ends with Length, as at
// the end of its context, and a prefilling one fails naming the SSD cache;
// either returns its commitment.
void testLoneSpillLaneEndsAtTheWall() {
  for (const bool decoding : {true, false}) {
    Spilling spill(64);
    // Two Full extents beside the Index runway: eight Full pages.
    spill.storage.budgetPages = 9;
    spill.allowSpillWrites(false);
    spill.engine.submit(Spilling::prompt(1, 1, decoding ? 4 : 10, 200));
    spill.runUntilIdle();
    if (decoding) {
      const uint32_t output =
          spill.events.usage.contains(1) ? spill.events.usage.at(1).second : 0;
      require(spill.events.failedCount == 0 && output > 0 &&
                  4 * KvCache::pageTokens + 1 + output <= 8 * KvCache::pageTokens &&
                  spill.engine.snapshot().spill.lengthEnds == 1 &&
                  spill.events.boundedIds == std::vector<uint64_t>{1},
              "a lone Spill lane at the wall did not end with Length at its bound");
    } else {
      require(spill.events.failures == std::vector<std::string>{"capacity_exhausted"} &&
                  spill.events.failureDetails.front().first.find("SSD cache") !=
                      std::string::npos,
              "a lone Spill prefill at the wall did not fail naming the SSD cache");
    }
    require(spill.committedSlots() == 0, "the lane kept its commitment");
  }
}

// The hot block cache grows toward its planned size between commands, an
// allocation that may be refused: refused, it takes no cache the model can
// rebuild, among which that cache's own slots past its floor are, nor a
// chunk of the expert store, and nothing while the host refuses growth. The
// lane decodes all the same.
void testHotCacheGrowthTakesNoModelCache() {
  for (const auto failure :
       {metal::AllocationFailure::EngineBudget, metal::AllocationFailure::HostPressure}) {
    Spilling spill(64);
    spill.model.cacheUnits = {100};
    spill.model.expertChunks = {1000};
    spill.model.refusedGrowth = failure;
    spill.engine.submit(Spilling::prompt(1, 1, 5, 20));
    spill.runUntilIdle();
    require(spill.model.growthAttempts > 0 && spill.model.cacheReclaims == 0 &&
                spill.model.cacheUnits.size() == 1 && spill.model.expertChunks.size() == 1 &&
                spill.events.completedCount == 1 && spill.events.failedCount == 0,
            "the hot block cache's growth took a cache the model can rebuild or the expert store");
  }
}

// A decoding lane's floor of the hot block cache that the host refuses takes
// no cache the model can rebuild either, as no allocation the host refuses
// does: the lane takes the floor as memory a request in service needs, and
// decodes.
void testHostRefusedHotFloorTakesNoModelCache() {
  Spilling spill(64);
  spill.model.cacheUnits = {100};
  spill.model.spilledReadsRefusal = metal::AllocationFailure::HostPressure;
  spill.model.spilledReadsRoom = [&](uint64_t) { return !spill.paused || spill.serving; };
  spill.paused = true;
  spill.engine.submit(Spilling::prompt(1, 1, 5, 20));
  spill.runUntilIdle();
  require(spill.events.completedCount == 1 && spill.events.failedCount == 0 &&
              spill.events.usage.at(1).second == 20,
          "the lane did not decode through the host's refusal of its hot floor");
  require(spill.model.cacheReclaims == 0 && spill.model.cacheUnits.size() == 1,
          "a hot floor the host refused took a cache the model can rebuild");
}

// KV that the SSD cache cannot read back fails only the lane that read it,
// as a retry recomputes it (503 kv_unreadable), before that lane's command
// publishes or emits anything, and without taking the state it restored for
// the failure: the read says nothing of it. The other lanes go on, and the
// block's copy is given up, so no later lookup matches it or what follows.
void testUnreadableKvEndsOnlyItsLane() {
  Spilling spill(64);
  // A first request leaves the state at 128 tokens that the failing lane
  // restores.
  spill.engine.submit(Spilling::prompt(30, 3, 4, 2));
  spill.runUntil([&] { return spill.events.outputs.contains(30); },
                 "the first request did not decode");
  const uint64_t unreadable = spill.cache.blockAt(30, 2 * KvCache::pageTokens);
  spill.runUntilIdle();
  const uint64_t invalidations = spill.cache.snapshot().stateCache.invalidations;
  // The failing lane's first step after its restore reads the block, and
  // cannot.
  spill.model.unreadableBlocks[3] = {unreadable};
  for (uint64_t id = 1; id <= 4; ++id)
    spill.engine.submit(Spilling::prompt(id, static_cast<uint32_t>(id), id == 3 ? 6 : 4, 4));
  spill.runUntilIdle();
  require(spill.events.failures == std::vector<std::string>{"kv_unreadable"} &&
              spill.events.failureDetails.front().second && spill.events.completedCount == 4 &&
              !spill.events.outputs.contains(3) &&
              std::ranges::all_of(std::array<uint64_t, 3>{1, 2, 4},
                                  [&](uint64_t id) {
                                    return spill.events.usage.at(id).second == 4;
                                  }),
          "unreadable KV failed other lanes, or its own lane emitted its step");
  require(spill.cache.snapshot().stateCache.invalidations == invalidations,
          "the state the failing lane restored was taken for the failure");
  const std::vector<uint32_t> prompt(6 * KvCache::pageTokens + 1, 3);
  require(spill.cache.lookup(prompt, {}).kvBoundary == KvCache::pageTokens &&
              spill.committedSlots() == 0,
          "a lookup still matched the block whose copy could not be read");
}

// A request's KV is projected to hold its prompt and at most
// kProjectedOutputTokens (4096) of output, whatever its output limit: a
// Spill start commits for that much.
void testProjectionClipsTheOutput() {
  Spilling spill(200);
  spill.engine.submit(Spilling::prompt(1, 1, 1, 10'000));
  spill.runUntil([&] { return spill.events.startIds.size() == 1; }, "the request did not start");
  require(spill.committedSlots() == (KvCache::pageTokens + 1 + kProjectedOutputTokens) /
                                        KvCache::pageTokens,
          "the projection did not clip the output limit");
  spill.engine.cancel(1);
  spill.runUntilIdle();
  require(spill.committedSlots() == 0, "the cancelled request kept its commitment");
}

// A Spill start commits once it holds its lane, before it restores its
// prefix: a restore that fails vacates the lane and returns the commitment,
// and the start runs later from the prefix that remains.
void testFailedRestoreReturnsTheCommitment() {
  Spilling spill(64);
  // The first request's state at 128 tokens goes to the disk tier: no cache
  // slot takes it.
  spill.model.deniedReservations = 1000;
  spill.model.stateTier = std::make_shared<OffloadControl>();
  spill.model.stateTier->ready = true;
  spill.engine.submit(Spilling::prompt(1, 1, 4, 1));
  spill.runUntilIdle();
  require(spill.engine.snapshot().diskStatePublications == 1,
          "fixture did not write the replay state to disk");
  spill.engine.submit(Spilling::prompt(2, 1, 6, 1));
  spill.runUntil([&] { return spill.model.diskReads == 1; }, "the start did not restore its state");
  require(spill.committedSlots() > 0, "the start did not commit before its restore");
  // The read fails; the start's next attempt waits for the host.
  bool hostRefuses = true;
  spill.model.beginAllocationFailure = metal::AllocationFailure::HostPressure;
  spill.model.beginGrowthBlocked = [&] { return hostRefuses; };
  spill.model.restoreControl->ready = true;
  spill.model.restoreControl->success = false;
  static_cast<void>(spill.engine.tick(spill.now++));
  require(spill.committedSlots() == 0 && spill.events.startIds.size() == 1,
          "the failed restore kept its commitment");
  hostRefuses = false;
  spill.runUntilIdle();
  require(spill.events.completedCount == 2 && spill.events.starts.back() == 0 &&
              spill.committedSlots() == 0,
          "the start did not run from the prefix that remained");
}

} // namespace

std::span<const test::TestCase> spillTests() {
  static constexpr test::TestCase kTests[] = {
      {"testUnwritableTierKeepsRequestsResident", testUnwritableTierKeepsRequestsResident},
      {"testAdmissionSpillsWhatMemoryCannotHold", testAdmissionSpillsWhatMemoryCannotHold},
      {"testSpillCommitmentCoversTheChainWithoutCopies", testSpillCommitmentCoversTheChainWithoutCopies},
      {"testConversionCommitsTheHistoryWithoutCopies", testConversionCommitsTheHistoryWithoutCopies},
      {"testConversionSpillsAHistoryWithCopies", testConversionSpillsAHistoryWithCopies},
      {"testResidentLaneConvertsBeforeTheExpertStore", testResidentLaneConvertsBeforeTheExpertStore},
      {"testAStartThatConvertsCountsNoConversion", testAStartThatConvertsCountsNoConversion},
      {"testShortQuotaBoundsWaitsOrRefusesASpillStart", testShortQuotaBoundsWaitsOrRefusesASpillStart},
      {"testAStartThatCannotSpillStartsResidentWithinTheResidentContext",
       testAStartThatCannotSpillStartsResidentWithinTheResidentContext},
      {"testWriteBudgetRefusalSaysWhenToRetry", testWriteBudgetRefusalSaysWhenToRetry},
      {"testOnlySpillingLanesRunPrefillGroups", testOnlySpillingLanesRunPrefillGroups},
      {"testSpilledFullExtentsLeaveOnceNoPromptGrows", testSpilledFullExtentsLeaveOnceNoPromptGrows},
      {"testSpillMovesThroughTheHostPause", testSpillMovesThroughTheHostPause},
      {"testSpillResumeCommitsAgain", testSpillResumeCommitsAgain},
      {"testSpillResumeOnAClosedTier", testSpillResumeOnAClosedTier},
      {"testSpillResumeBoundLeavesOutputPastItsAnchor", testSpillResumeBoundLeavesOutputPastItsAnchor},
      {"testSpillResumeInItsPromptHoldsNoAnchor", testSpillResumeInItsPromptHoldsNoAnchor},
      {"testSpillResumeTakesItsReplaysPagesAsItRuns", testSpillResumeTakesItsReplaysPagesAsItRuns},
      {"testAResumeThatConvertsTakesItsReplaysPagesAsItRuns", testAResumeThatConvertsTakesItsReplaysPagesAsItRuns},
      {"testDecodeLeavesOutALaneRefusedItsHotFloor", testDecodeLeavesOutALaneRefusedItsHotFloor},
      {"testARefusedDecodeTakesThePrefillLanesMemory", testARefusedDecodeTakesThePrefillLanesMemory},
      {"testRefusedModelMemoryTakesWhatGivesMemoryBack", testRefusedModelMemoryTakesWhatGivesMemoryBack},
      {"testALoneLaneWaitsForTheChainANeighbourLeft", testALoneLaneWaitsForTheChainANeighbourLeft},
      {"testAWaitingFloorKeepsTheMemoryOnItsWayBack", testAWaitingFloorKeepsTheMemoryOnItsWayBack},
      {"testAWaitThatNeverLandsHoldsLanesUpToItsLimit", testAWaitThatNeverLandsHoldsLanesUpToItsLimit},
      {"testAHeldSpillLaneLeavesTheFloorToTheWaiter", testAHeldSpillLaneLeavesTheFloorToTheWaiter},
      {"testAWaitingStartHoldsNoLane", testAWaitingStartHoldsNoLane},
      {"testAHeldLaneEndsItsCommandAtItsCapacity", testAHeldLaneEndsItsCommandAtItsCapacity},
      {"testAJoinerHeldBackLeavesACommandItsPasses", testAJoinerHeldBackLeavesACommandItsPasses},
      {"testAFloorConvertsAResidentNeighbourBeforeTheExpertStore",
       testAFloorConvertsAResidentNeighbourBeforeTheExpertStore},
      {"testAFloorConvertsTheLargestLaneThatGivesMemoryBack", testAFloorConvertsTheLargestLaneThatGivesMemoryBack},
      {"testOwedCopiesHoldBackTheNextConversion", testOwedCopiesHoldBackTheNextConversion},
      {"testARefusedStartTakesWhatItsConversionFreed", testARefusedStartTakesWhatItsConversionFreed},
      {"testALaneBeingAdmittedDoesNotConvert", testALaneBeingAdmittedDoesNotConvert},
      {"testALoneSpillLaneWaitsForItsOwnCopies", testALoneSpillLaneWaitsForItsOwnCopies},
      {"testALaneRefusedBeforeItsFirstCommandWritesItsChain", testALaneRefusedBeforeItsFirstCommandWritesItsChain},
      {"testAPageWaitingForItsCopyHoldsBackNoneAfterIt", testAPageWaitingForItsCopyHoldsBackNoneAfterIt},
      {"testAConversionLeavesNoLaneWithoutItsFloor", testAConversionLeavesNoLaneWithoutItsFloor},
      {"testLoneSpillLaneEndsAtTheWall", testLoneSpillLaneEndsAtTheWall},
      {"testHotCacheGrowthTakesNoModelCache", testHotCacheGrowthTakesNoModelCache},
      {"testHostRefusedHotFloorTakesNoModelCache", testHostRefusedHotFloorTakesNoModelCache},
      {"testUnreadableKvEndsOnlyItsLane", testUnreadableKvEndsOnlyItsLane},
      {"testProjectionClipsTheOutput", testProjectionClipsTheOutput},
      {"testFailedRestoreReturnsTheCommitment", testFailedRestoreReturnsTheCommitment},
  };
  return kTests;
}
