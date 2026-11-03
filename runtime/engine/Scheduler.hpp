#pragma once

#include "engine/CostModel.hpp"
#include "engine/Types.hpp"
#include "model/CommandCost.hpp"
#include "model/LatencyHistogram.hpp"
#include "model/Model.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace splash::engine {

enum class Phase : uint8_t {
  Queued,
  WaitingResources,
  WaitingPrefix,
  Prefill,
  Decode,
  WaitingMask,
  Completed,
  Cancelled,
  Failed,
};

// What the scheduler plans with: decode's share of contended prefill time,
// the time G a prefill unit may take while lanes of its priority or a higher
// one decode, and what one prefill command of the model may hold.
struct SchedulerConfig final {
  double decodeShare = 0.5;
  double decodeGapMilliseconds = 500.0;
  model::CommandLimits limits;
};

struct RequestSpec final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  // Tokens to prefill: the prompt, and after resumeFromResources the
  // replayed history.
  uint32_t prefillTokens = 0;
};

struct PrefillAdmission final {
  uint64_t requestId = 0;
  uint32_t cachedTokens = 0;
};

// A prefill lane as the engine knows it when the scheduler plans: the pages
// of its history only the SSD cache holds now, whether it may gain such pages
// between a command's passes (a Spill lane past its dense history and
// window), the most rows a state capture prepare cut this tick leaves it,
// whether it is a Spill lane, and whether its request ended between its
// command's passes (a cancel, its deadline), so that it runs to the
// command's last pass for no one.
struct LaneShape final {
  uint64_t requestId = 0;
  uint32_t indexPages = 0;
  bool gainsIndexPages = false;
  uint32_t rowCap = std::numeric_limits<uint32_t>::max();
  bool spill = false;
  bool ended = false;
};

// The command between its passes as planning reads it: its plan at its next
// pass, the ready lanes of its priority its first pass was planned without,
// the starts this admission pass held back for its end, and the wall time of
// its passes so far.
struct PassProgress final {
  BatchPlan plan;
  std::vector<uint64_t> leftOut;
  std::vector<uint64_t> heldStarts;
  double elapsedMilliseconds = 0.0;
};

// What a plan is made from, rebuilt before each: the lanes that cannot run
// in it (they wait for memory on its way back, or prepare refused or held
// them back this tick), the shapes of the resident prefill lanes and, for
// admission, of the candidates, the command between its passes, if any,
// and, for admission, the admitted requests that restore their prefix from
// the SSD cache before they run, each from the boundary it restores to.
struct PlanInputs final {
  std::span<const uint64_t> excluded;
  std::span<const LaneShape> shapes;
  const PassProgress *passes = nullptr;
  std::span<const PrefillAdmission> restoring;
};

// One rate a command estimate learned: the quantity of a command's work it
// multiplies ("units", "chunk_layers", "row_layers", "attention_pairs",
// "index_pairs", "weight_read_bytes", "spilled_kv_read_bytes" for prefill;
// "token" for decode), what it charges per unit of it, learned and seeded,
// and for a quantity the model estimates before a unit runs, the realized
// over the expected the price scales it by (1 for the others).
struct CostTermSnapshot final {
  std::string quantity;
  double millisecondsPerUnit = 0.0;
  double seedMillisecondsPerUnit = 0.0;
  double realizedPerExpected = 1.0;
};

// One kind of command's estimate (prefill or decode): the commands or passes
// that trained it and those whose time it clamped to within half and twice
// its price, its rates, and |predicted - measured| / measured over the last
// 256 that trained it.
struct CostEstimateSnapshot final {
  uint64_t samples = 0;
  uint64_t clamped = 0;
  std::vector<CostTermSnapshot> terms;
  double relativeErrorP50 = 0.0;
  double relativeErrorP95 = 0.0;
};

struct SchedulerSnapshot final {
  uint32_t queued = 0;
  uint32_t waitingResources = 0;
  uint32_t waitingPrefix = 0;
  uint32_t prefilling = 0;
  uint32_t decoding = 0;
  uint32_t waitingMask = 0;
  uint32_t terminal = 0;
  uint64_t prefillBatches = 0;
  // The prefill commands' rows, a command's with its first pass, as 1.5.0
  // counts them (the matrix's work); rows its lanes plan again after it was
  // abandoned between its passes count once.
  uint64_t prefillRows = 0;
  uint64_t decodeBatches = 0;
  std::array<uint64_t, model::ExecutionLimits::maximumBatchWidth>
      decodeBatchesByWidth{};
  // The decode gap target contended prefill work is sized by (G).
  double decodeGapTargetMilliseconds = 0.0;
  // The wall time of each prefill unit, a command or one pass of it, that ran
  // while a lane of its priority or a higher one decoded: what the target
  // bounds, its quantiles within a millisecond (model::FineLatencyHistogram).
  model::LatencyStats contendedPrefill;
  // Prefill units run; commands that ran in passes, and those a superseding
  // plan abandoned between passes; plans whose rows were halved because
  // passes could not bound a contended unit; and plans that left a lane out
  // that could have joined them because a command holds at most
  // model::CommandLimits::indexPageLanes lanes whose history may hold Index
  // pages.
  uint64_t prefillUnits = 0;
  uint64_t passCommands = 0;
  uint64_t abandonedPassCommands = 0;
  uint64_t boundCuts = 0;
  uint64_t indexPageConflicts = 0;
  CostEstimateSnapshot prefillEstimate;
  CostEstimateSnapshot decodeEstimate;
};

// One single-owner scheduling policy, one price for every prefill unit (a
// command or one pass of it): CostModel's, of the work the model expects of
// it (model::PrefillWorkSource).
//
// Prefill takes the ready lanes of the most urgent priority. Within it a lane
// that has waited behind later arrivals of its priority kAgingSlowdown times
// its own expected remaining cost is overdue and leads, to its prompt's end;
// otherwise the lane of the least expected remaining cost leads, ties in
// submission order. A leader takes a chunk, or a group of the model's command
// rows where the fixed part a group saves is worth a wait of G. While a lane
// of its priority or a higher one decodes (contended), or when it leads a
// group, each unit is bounded: the fewest passes whose every pass's price is
// within max(G, turnaround / kPassOverheadShare), or failing that, of no
// fewer passes, within twice its fixed part; a contended unit no passes
// bound halves its rows where fewer rows can bound it, never below the
// fewest rows the model's faster path takes
// (model::CommandLimits::minimumUnitRows) nor leaving its prompt fewer, past
// the bound if need be. A layer's attention pairs bound rows, never passes.
// A lane joins with the rows that spare it a command of its own when
// what it adds to the others is less than what it saves (summed
// completion). The command between its
// passes runs its later passes before any other prefill but a plan whose
// summed completion of the lanes concerned is shorter (supersedes). Decode
// takes the ready lanes of one priority and stage, constrained or not,
// without waiting for more, so it runs only the four real M8/M16/M24/M32
// shapes, between a command's passes too.
class Scheduler final {
public:
  Scheduler(SchedulerConfig config, const model::PrefillWorkSource &work);

  // The prefill warm-ups' samples and the SSD reads of loading the weights
  // seed the prefill price (CostModel::seed); the warm-ups never train it.
  void seedCost(std::span<const model::CommandSample> warmups, model::SsdReads load);
  void submit(RequestSpec request);
  void deferAdmission(uint64_t requestId);
  void waitForResources(uint64_t requestId);
  void waitForPrefix(uint64_t requestId);
  void resourcesReady(uint64_t requestId, uint32_t alreadyProcessed);
  void suspendForResources(uint64_t requestId);
  void resumeFromResources(uint64_t requestId, uint32_t alreadyProcessed,
                           uint32_t replayTokens);
  void maskReady(uint64_t requestId);
  void cancel(uint64_t requestId);
  // Ends a request outside a command's completion, as at its length.
  void finish(uint64_t requestId);
  void fail(uint64_t requestId);
  void remove(uint64_t requestId);

  [[nodiscard]] std::vector<uint64_t> admissionOrder() const;
  // The candidates their first command would serve and leave waiting for no
  // other prompt: running them to their prompt's end, leading them now or,
  // beside no Spill lane, next, or as they are overdue, beside the resident
  // prefill lanes (those of the command between its passes past its rows),
  // and led by none of the requests restoring in `in` once they land.
  // Admission starts only those, planning again without the others until it
  // serves all it plans, with the candidates' shapes in `in`.
  [[nodiscard]] std::vector<uint64_t>
  prefillAdmissionOrder(std::span<const PrefillAdmission> candidates, const PlanInputs &in) const;
  // The highest priority among requests that prefill or decode. A waiting
  // request of a lower priority cannot be in prefillAdmissionOrder's result
  // while it lasts: the plan takes one tier, the highest among resident
  // prefill lanes and candidates, and is dropped when a higher tier decodes.
  [[nodiscard]] std::optional<RequestPriority> highestRunnablePriority() const noexcept;
  // The next command, planned without the excluded lanes. While a command is
  // between its passes (in.passes), that is its next pass, decode, as its
  // priority, or that of the most urgent lane it holds back or start it left
  // waiting, and the decode share order them, or a plan that supersedes it.
  [[nodiscard]] std::optional<BatchPlan> next(const PlanInputs &in) const;
  // Whether `plan` is a prefill plan other than the next pass of the command
  // between its passes, which is abandoned before it commits.
  [[nodiscard]] bool abandonsPasses(const BatchPlan &plan, const PlanInputs &in) const;
  // Whether a start of `lanes` lanes of `priority` supersedes the command
  // between its passes: a higher priority, while the summed completion of
  // the lanes concerned is shorter abandoned (supersedes). The engine asks
  // it for a start of that priority (Engine::admit, Engine::preemptBelow).
  [[nodiscard]] bool yieldsTo(RequestPriority priority, uint32_t lanes, const PlanInputs &in) const;
  // The excluded lanes are those the plan was made without: blocked, not
  // passed over, so they lose nothing to it. A later pass must be the
  // command's next. For a first pass of several, the ready lanes of its
  // priority it leaves out (PassProgress::leftOut).
  std::vector<uint64_t> commit(const BatchPlan &plan, const PlanInputs &in);
  // A prefill unit trains the price with its sample's realized work against
  // the work planning expected of it, weighing 1 / its passes; a pass before
  // the command's last consumes no rows and ends no lane. Of its wall time,
  // the image encodes ahead of it stay out of the price. A decode command's
  // time and the tokens its lanes committed train the decode time per token.
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                const model::CommandSample &sample);
  // The command between its passes was abandoned: each of its lanes counts
  // the time its passes took as time it waited.
  void abandoned(const PassProgress &command);
  // The price, in milliseconds, of a prefill command or pass doing `work`,
  // the work a model expects (PrefillWorkSource): its read bytes count at
  // the share of them units realized.
  [[nodiscard]] double prefillMilliseconds(const model::CommandWork &work) const noexcept;
  // The price, in milliseconds, of a byte of weights a prefill read from the
  // SSD, as the expert store counts its reads (ExpertStore::readBytes): the
  // cost model's rate. prefillMilliseconds charges an expected byte that
  // rate times the share of expected bytes units read, so a difference of
  // its prices is what an expected byte costs, not a byte read.
  [[nodiscard]] double weightReadMillisecondsPerByte() const noexcept;
  // The wall time of a decode command per token it committed per lane, in
  // milliseconds: the mean over the decode commands completed, each weighing
  // kDecodeTokenDecay of the next; 0 before the first.
  [[nodiscard]] double decodeTokenMilliseconds() const noexcept;
  // The tokens a lane commits per decode command, the same decayed mean; 0
  // before the first.
  [[nodiscard]] double decodeTokensPerLaneCommand() const noexcept;
  // The most a bounded unit may take: G, or turnaround / kPassOverheadShare,
  // which a decode gap past it exceeds (Engine::DecodeGapCauses).
  [[nodiscard]] double bound() const;

  [[nodiscard]] Phase phase(uint64_t requestId) const;
  // The stage of its decode commands, which a suspension keeps.
  [[nodiscard]] DecodeStage decodeStage(uint64_t requestId) const;
  // Suspended by suspendForResources and not resumed since; a terminal
  // phase keeps it until remove(), so the engine can end the request's
  // continuation.
  [[nodiscard]] bool suspended(uint64_t requestId) const;
  // The request's place in submission order.
  [[nodiscard]] uint64_t submissionOrder(uint64_t requestId) const;
  [[nodiscard]] uint32_t promptProcessed(uint64_t requestId) const;
  [[nodiscard]] SchedulerSnapshot snapshot() const noexcept;

private:
  struct Request final {
    RequestSpec spec;
    Phase phase = Phase::Queued;
    uint32_t promptProcessed = 0;
    // Set by suspendForResources: the request comes back through
    // resumeFromResources, never through resourcesReady.
    bool suspended = false;
    DecodeStage decodeStage = DecodeStage::Regular;
    uint64_t order = 0;
    uint64_t lastDecodeDispatch = 0;
    // Until its prompt completes: the wall time of the units of its priority
    // that served later arrivals while it waited without rows, and of its
    // commands abandoned between their passes; and whether that reached
    // kAgingSlowdown times its expected remaining cost at a commit, after
    // which it stays overdue.
    double waitedMilliseconds = 0.0;
    bool overdue = false;
    // The prompt position its committed rows reach in prefillRows, never
    // short of promptProcessed, which rows planned again after their command
    // was abandoned between its passes do not pass; a resume's replay counts
    // again the rows it runs that ran before it, and none it runs that were
    // counted and never ran.
    uint32_t rowsCounted = 0;
  };

  // A prefill lane as planning weighs it.
  struct Lane final {
    const Request *request = nullptr;
    uint32_t processed = 0;
    LaneShape shape;
    double remainingMilliseconds = 0.0;
    bool overdue = false;
    [[nodiscard]] uint32_t remaining() const noexcept { return request->spec.prefillTokens - processed; }
  };
  // A plan and what planning it counted: rows halved for a contended unit
  // no passes bound, and joinable lanes left out for the Index-page lanes.
  struct Planned final {
    BatchPlan plan;
    uint32_t boundCuts = 0;
    uint32_t indexPageConflicts = 0;
  };
  // The passes a unit runs in and whether they bound every pass.
  struct UnitFit final {
    uint32_t passes = 1;
    bool within = true;
  };

  [[nodiscard]] Request &get(uint64_t requestId);
  [[nodiscard]] const Request &get(uint64_t requestId) const;
  [[nodiscard]] static bool terminal(Phase phase) noexcept;
  [[nodiscard]] static bool byPriorityThenOrder(const Request *a,
                                                const Request *b) noexcept;
  [[nodiscard]] static bool samePlan(const BatchPlan &a, const BatchPlan &b) noexcept;
  // Whether prefill takes lane `a` before lane `b`: the order planning leads
  // by and joins in.
  [[nodiscard]] static bool leadsBefore(const Lane &a, const Lane &b) noexcept;

  [[nodiscard]] LaneShape shapeOf(uint64_t requestId, const PlanInputs &in) const;
  [[nodiscard]] Lane lane(const Request &request, uint32_t processed, const PlanInputs &in) const;
  [[nodiscard]] std::vector<model::PrefillLane> prefillLanes(std::span<const BatchItem> items,
                                                             const PlanInputs &in) const;
  // The price of `lanes` as one command in one pass, and summed over the
  // passes [first, passes) of a command of `passes`, nothing for no lanes.
  [[nodiscard]] double commandMilliseconds(std::span<const model::PrefillLane> lanes) const;
  [[nodiscard]] double passesMilliseconds(std::span<const model::PrefillLane> lanes, uint32_t first,
                                          uint32_t passes) const;
  // Whether a lane's chunk saves enough fixed part in a group of the
  // model's command rows to be worth a wait of G.
  [[nodiscard]] bool groups(const Lane &lane) const;
  // The most rows a command of the lane alone holds: the model's command
  // rows where it groups, else a chunk.
  [[nodiscard]] uint32_t groupRows(const Lane &lane) const;
  // The rows of an uncontended command of the lane: a chunk, or a group's.
  [[nodiscard]] uint32_t ownRows(const Lane &lane) const;
  [[nodiscard]] double remainingMilliseconds(const Lane &lane) const;
  [[nodiscard]] bool contended(RequestPriority priority) const;
  [[nodiscard]] UnitFit passesFor(std::span<const model::PrefillLane> lanes, bool bounded) const;
  [[nodiscard]] bool withinPairs(std::span<const model::PrefillLane> lanes) const;
  // The resident prefill lanes a plan may serve: those it is not made
  // without and that did not end.
  [[nodiscard]] std::vector<Lane> readyLanes(const PlanInputs &in) const;
  [[nodiscard]] std::optional<Planned> planPrefill(std::vector<Lane> ready) const;
  [[nodiscard]] std::optional<BatchPlan> nextPrefill(const PlanInputs &in) const;
  [[nodiscard]] std::optional<BatchPlan> nextDecode(std::span<const uint64_t> excluded) const;
  // Whether a lane of the command between its passes that did not end is
  // overdue, or would be once abandoning it adds the time its passes took to
  // its wait (abandoned): such a lane leads the next plan, so a plan made
  // without that wait would not run, and its passes would be lost.
  [[nodiscard]] bool overdueOnceAbandoned(const PlanInputs &in) const;
  // Whether abandoning the command between its passes for `plan` shortens
  // the summed completion of the lanes concerned (gain), with no lane of the
  // command overdue once abandoned and none of the plan left out of its first
  // pass.
  [[nodiscard]] bool supersedes(const BatchPlan &plan, const PlanInputs &in) const;
  // What abandoning the command between its passes for a plan of `lanes`
  // (and `starts` more lanes not yet resident) of `priority` saves the
  // lanes of the command, of the plan, and the ready lanes of their
  // priorities that wait behind both, summed: keeping it, its lanes end
  // with its passes and the plan's new lanes run after them; abandoning it,
  // the plan runs first and the command's lanes it leaves out run their rows
  // again after it; one that ended loses nothing. A lane of a lower priority
  // than the plan's does not count the wait for the plan, which strict
  // priority gives it either way.
  // `planMilliseconds` and `joinersMilliseconds` price the plan and its new
  // lanes' own command, which a plan of no new lanes does not run.
  [[nodiscard]] double supersedeGain(std::span<const BatchItem> lanes, uint32_t starts, RequestPriority priority,
                                     double planMilliseconds, double joinersMilliseconds,
                                     const PlanInputs &in) const;
  // Debt is owed only to requests that decode or wait for a mask: once the
  // last one leaves, a later decoder starts without it.
  void dropStaleDecodeDebt() noexcept;
  // A prompt that completed leaves its aging behind.
  static void promptCompleted(Request &request) noexcept;

  SchedulerConfig config_;
  const model::PrefillWorkSource &work_;
  CostModel cost_;
  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  // The prefill plan next() made last and what planning it counted, which
  // its commit adds to the counters; the commit of any plan clears it.
  mutable std::optional<Planned> lastPlanned_;
  // The work planning expected of the prefill unit in flight, and whether it
  // was contended when it committed, as planning bounds units: what
  // contendedPrefill_ counts.
  model::CommandWork activeExpected_;
  bool activeContended_ = false;
  // The lanes the active command was planned without.
  std::vector<uint64_t> excluded_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  // The decode commands' wall time per token committed per lane and tokens
  // committed per lane, and their count, each decayed by kDecodeTokenDecay
  // per command after it; the relative errors of the latest 256.
  double decodeTokenTime_ = 0.0;
  double decodeLaneTokens_ = 0.0;
  double decodeTokenWeight_ = 0.0;
  std::array<double, CostModel::kErrorSamples> decodeErrors_{};
  size_t decodeErrorCount_ = 0;
  uint64_t decodeSamples_ = 0;
  double decodeShare_;
  // Decode time that prefill still owes the lanes that decoded beside it
  // (not those waiting for a mask): equal-priority decode runs until its
  // commands' wall time has worked it off.
  double decodeDebtMilliseconds_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  SchedulerSnapshot counters_;
  model::FineLatencyHistogram contendedPrefill_;
};

} // namespace splash::engine
