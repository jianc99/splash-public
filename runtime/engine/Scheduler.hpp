#pragma once

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

struct RequestSpec final {
  uint64_t id = 0;
  RequestPriority priority = RequestPriority::Normal;
  // Tokens to prefill: the prompt, and after resumeFromResources the
  // replayed history.
  uint32_t prefillTokens = 0;
  // The most rows of one of its prefill commands (Model::prefillGroupRows): a
  // prefill group's, or a chunk's; the engine may set it again once it knows
  // where the request's KV lives (setPrefillGroupRows).
  uint32_t prefillGroupRows = model::ExecutionLimits::prefillTokenBudget;
  // The most passes over the model's layers one of its prefill commands may
  // run in (Model::prefillPasses, BatchPlan::passes).
  uint32_t prefillPasses = 1;
};

struct PrefillAdmission final {
  uint64_t requestId = 0;
  uint32_t cachedTokens = 0;
};

// A prefill lane as the engine knows it when the scheduler plans: the pages
// of its history only the SSD cache holds now, whether it may gain such pages
// between a command's passes (a Spill lane past its dense history and
// window), and the most rows a state capture prepare cut this tick leaves it.
struct LaneShape final {
  uint64_t requestId = 0;
  uint32_t indexPages = 0;
  bool gainsIndexPages = false;
  uint32_t rowCap = std::numeric_limits<uint32_t>::max();
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
// admission, of the candidates, and the command between its passes, if any.
struct PlanInputs final {
  std::span<const uint64_t> excluded;
  std::span<const LaneShape> shapes;
  const PassProgress *passes = nullptr;
};

// One rate a command estimate learned: the quantity of a command's work it
// multiplies ("units", "chunk_layers", "row_layers", "attention_pairs",
// "index_pairs", "weight_read_bytes", "spilled_kv_read_bytes" for prefill;
// "token" for decode) and what it charges per unit of it, learned and seeded.
struct CostTermSnapshot final {
  std::string quantity;
  double millisecondsPerUnit = 0.0;
  double seedMillisecondsPerUnit = 0.0;
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
  uint64_t prefillRows = 0;
  uint64_t decodeBatches = 0;
  std::array<uint64_t, model::ExecutionLimits::maximumBatchWidth>
      decodeBatchesByWidth{};
  // Prefill commands of a lane that streams its history from the SSD cache
  // that ran while lanes of its priority or a higher one decoded, and the
  // gaps decoding lanes saw between their commands across such a command:
  // the command time from one of a lane's decode commands to its next.
  uint64_t contendedStreamingPrefillBatches = 0;
  model::LatencyStats decodeGap;
  // The decode gap target contended prefill work is sized by.
  double decodeGapTargetMilliseconds = 0.0;
  // The wall time of each prefill unit, a command or one pass of it, that ran
  // while a lane of its priority or a higher one decoded: what the target
  // bounds.
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

// One single-owner scheduling policy. Prefill batches the shortest remaining
// sequences first into an actual-row budget, with a bound k on how often
// later arrivals may overtake a lane, led by the oldest whose remaining
// prompt exceeds the shortest by at most floor((m-1)/k) of the shortest's m
// commands; a lane that streams its history from the SSD cache runs
// alone. A command that runs in passes over the model's layers runs its later
// passes before any other prefill, but for a command that supersedes it
// (abandonsPasses), as often as later arrivals may overtake a lane. Decode
// takes the ready lanes of one priority and stage, constrained or not, without
// waiting for more, so it runs only the four real M8/M16/M24/M32 shapes,
// between a command's passes too.
class Scheduler final {
public:
  // Decode time owed for each unit of time a prefill runs while requests of
  // equal or higher priority decode; zero alternates one command of each kind.
  // A prefill command's attention computes at most maximumAttentionPairs
  // query-key pairs (attentionPairs): its lanes' rows halve until they fit.
  explicit Scheduler(double decodeShare,
                     uint64_t maximumAttentionPairs = std::numeric_limits<uint64_t>::max()) noexcept
      : decodeShare_(decodeShare), maximumAttentionPairs_(maximumAttentionPairs) {}

  // The query-key pairs the attention of `rows` prompt rows from `position`
  // computes: each row attends to its history and itself, so the pairs, and
  // a command's time, grow with the depth of its rows.
  [[nodiscard]] static constexpr uint64_t attentionPairs(uint64_t position, uint32_t rows) noexcept {
    return rows * position + uint64_t{rows} * (rows + 1) / 2;
  }

  void submit(RequestSpec request);
  // A prefill command of `rows` rows took wallMilliseconds, of which its GPU
  // work waited hostMilliseconds on host steps. The first command seeds the
  // command estimate, a fixed part and a part per row: its waits on host
  // steps (a streamed target's expert reads) fixed, the rest per row. The
  // commands after it fit the estimate as the least-squares line of their
  // times by their rows, each weighing three quarters of the next and
  // counting at most twice and at least half the time the estimate gave it,
  // the fixed part shrunk toward their waits on host steps as much as one
  // command a chunk's rows from the others would pull it: commands of one
  // size keep their waits fixed and the rest per row, as the depth of a
  // prompt's rows lengthens them, and commands of different sizes split what
  // the waits alone miss on a streamed target, the GPU's work a command does
  // whatever its rows. Neither part is negative.
  void observePrefill(uint32_t rows, double wallMilliseconds, double hostMilliseconds = 0.0);
  void deferAdmission(uint64_t requestId);
  void waitForResources(uint64_t requestId);
  void waitForPrefix(uint64_t requestId);
  void resourcesReady(uint64_t requestId, uint32_t alreadyProcessed);
  void suspendForResources(uint64_t requestId);
  void resumeFromResources(uint64_t requestId, uint32_t alreadyProcessed,
                           uint32_t replayTokens);
  void maskReady(uint64_t requestId);
  // Whether the resident lane's prefill commands read its history from the
  // SSD cache (KvHistoryStream), which a command does for one lane alone and
  // whatever its rows; the engine says so before it plans.
  void setStreamsHistory(uint64_t requestId, bool streams);
  void setPrefillGroupRows(uint64_t requestId, uint32_t rows);
  void cancel(uint64_t requestId);
  // Ends a request outside a command's completion, as at its length.
  void finish(uint64_t requestId);
  void fail(uint64_t requestId);
  void remove(uint64_t requestId);

  [[nodiscard]] std::vector<uint64_t> admissionOrder() const;
  // Preview the dispatch row budget before allocating new lanes.
  [[nodiscard]] std::vector<uint64_t>
  prefillAdmissionOrder(std::span<const PrefillAdmission> candidates) const;
  // The highest priority among requests that prefill or decode. A waiting
  // request of a lower priority cannot be in prefillAdmissionOrder's result
  // while it lasts: the plan takes one tier, the highest among resident
  // prefill lanes and candidates, and is dropped when a higher tier decodes.
  [[nodiscard]] std::optional<RequestPriority> highestRunnablePriority() const noexcept;
  // The next command, planned without the excluded lanes: those that wait
  // for memory on its way back and cannot run before it lands. While a
  // command is between its passes, that is its next pass, decode, as its
  // priority, or that of the most urgent request it holds back, and the
  // decode share order them, or a command that supersedes it
  // (abandonsPasses).
  [[nodiscard]] std::optional<BatchPlan> next(std::span<const uint64_t> excluded) const;
  // Whether `plan` supersedes the command between its passes, which is
  // abandoned before it commits (abandonPasses): any prefill plan but its
  // next pass. next() makes one for a prompt of a higher priority, and for a
  // command of its lanes and lanes of its priority ready since, while the
  // passes the command's lanes lose by it are fewer than those the lanes
  // that supersede it save (supersedes).
  [[nodiscard]] bool abandonsPasses(const BatchPlan &plan) const;
  // Whether the command between its passes yields to the prompts of
  // `joined` lanes of a higher priority, as next() supersedes it with their
  // command: while it has run fewer passes, lane for lane, than those lanes
  // would wait for its end, and until a lane of it has seen
  // kMaximumOvertakes of its commands abandoned. False without one. The
  // engine asks it for a start of that priority (Engine::admit).
  [[nodiscard]] bool yieldsTo(RequestPriority priority, uint32_t joined) const;
  // The excluded lanes are those the plan was made without: blocked, not
  // passed over, so they lose nothing to it. A command in passes commits
  // them in turn, its first recording it.
  void commit(const BatchPlan &plan, std::span<const uint64_t> excluded);
  // Abandons the command between its passes, none of which is in flight:
  // nothing of its rows was committed, and its lanes, which count it
  // (Request::abandoned), run them again in a later command. A lane leaves
  // the command no other way before its last pass: while the command holds
  // it, it neither ends nor is suspended (Engine::abandonPasses).
  void abandonPasses();
  // A prefill command of one lane that streamed its history from the SSD
  // cache waited streamMilliseconds of its hostMilliseconds on the reads
  // (ModelBatchTicket); none for a command that streamed nothing. A pass
  // before a command's last consumes no rows and ends no lane (StepResult),
  // and the command's estimate learns from its passes' time together. Of
  // wallMilliseconds, imageEncodeMilliseconds encoded the command's images
  // ahead of it, which decode's share counts and the estimate leaves out. A
  // decode command's time and the tokens its lanes committed train the
  // decode time per token (decodeTokenMilliseconds).
  void complete(const BatchPlan &plan, std::span<const StepResult> results,
                double wallMilliseconds, bool representativePrefillTiming,
                double hostMilliseconds = 0.0,
                std::optional<double> streamMilliseconds = std::nullopt,
                double imageEncodeMilliseconds = 0.0);
  // The price, in milliseconds, of a prefill command or pass doing `work`:
  // the command estimate's for its rows (observePrefill).
  [[nodiscard]] double prefillMilliseconds(const model::CommandWork &work) const noexcept;
  // The wall time of a decode command per token it committed per lane, in
  // milliseconds: the mean over the decode commands completed, each weighing
  // kDecodeTokenDecay of the next; 0 before the first.
  [[nodiscard]] double decodeTokenMilliseconds() const noexcept;

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
    // Consecutive prefill commands that served a later arrival while this
    // lane received no rows. Reset whenever the lane is served.
    uint32_t overtaken = 0;
    // The commands in passes it was a lane of that were abandoned since its
    // rows last committed: from kMaximumOvertakes on, the command between
    // its passes it is a lane of runs to its end (supersedes).
    uint32_t abandoned = 0;
    // Its prefill commands stream its history (setStreamsHistory), and the
    // running estimate of how long one waits on the reads; none before the
    // first that read any. The command estimate leaves that wait out: it is
    // this lane's own cost, which grows with its history.
    bool streamsHistory = false;
    std::optional<double> streamMilliseconds;
    // While it decodes, where its gap since its last decode command began
    // (startDecodeGap): the command time and streaming prefill commands
    // completed then.
    double decodeGapStart = 0.0;
    uint64_t decodeGapStreams = 0;
  };

  // The command between its passes, from its first pass's commit to its
  // last's completion: its plan, at its next pass, the lanes ready then that
  // it was planned without, not excluded, and the time its passes took so
  // far, a sample of its rows unless one was not.
  struct Passes final {
    BatchPlan plan;
    std::vector<uint64_t> leftOut;
    double wallMilliseconds = 0.0;
    double hostMilliseconds = 0.0;
    bool representative = true;
  };

  struct PrefillRequestView final {
    const Request *request = nullptr;
    uint32_t promptProcessed = 0;
  };

  [[nodiscard]] Request &get(uint64_t requestId);
  [[nodiscard]] const Request &get(uint64_t requestId) const;
  [[nodiscard]] static bool terminal(Phase phase) noexcept;
  [[nodiscard]] static bool byPriorityThenOrder(const Request *a,
                                                const Request *b) noexcept;
  [[nodiscard]] std::optional<BatchPlan>
  nextPrefill(std::span<const uint64_t> excluded) const;
  // The bound every command that supersedes the command between its passes
  // keeps (supersedes, yieldsTo): no lane of it has seen kMaximumOvertakes
  // of its commands abandoned, and the passes its lanes lose, those they ran
  // and, `added`, those the joining rows add, are fewer than those the
  // `joined` lanes that supersede it save, what is left of its passes for
  // each.
  [[nodiscard]] bool givesWay(uint32_t joined, double added) const;
  // Whether `plan` replaces the command between its passes: a command of a
  // higher priority, or one of its priority that holds every lane of it with
  // the same rows and more, none of which its first pass was planned without
  // while ready, while the passes its lanes lose by it, together, are fewer
  // than those the lanes that supersede it save, and until a lane of it has
  // seen kMaximumOvertakes of its commands abandoned (Request::abandoned),
  // after which it runs to its end. The lanes that supersede it save what is left
  // of its passes. Superseded by a higher priority, its lanes lose the
  // passes they ran, which they run again after it. Planned again, its lanes
  // and those that join wait the planned command, where going on, its lanes
  // wait what is left of its passes and those that join that and a command
  // of their own, about as long as the planned one: a command in passes
  // costs about its fixed part (a streamed target's expert reads,
  // Model::prefillPasses), which theirs would pay again, and a lane joins
  // one with the rows that spare it a command of its own (planPrefill). So
  // its lanes lose the passes they ran and those the joining rows add, and
  // the waits sum to less planned again while that holds. Lanes that become
  // ready early in its passes, as requests that arrive during its first do,
  // so join it, and a higher priority's prompt supersedes it about while it
  // has fewer passes behind it than ahead; later, they wait for its end, as
  // do lanes it was planned without while ready (prepare refused them memory
  // the others fit in), which a command planned again would leave out too.
  // A lane it was planned without because it waited for memory on its way
  // back (an excluded lane) joins once its retry is ready, as an arrival
  // does.
  [[nodiscard]] bool supersedes(const BatchPlan &plan) const;
  [[nodiscard]] std::optional<BatchPlan>
  nextDecode(std::span<const uint64_t> excluded) const;
  [[nodiscard]] std::optional<BatchPlan>
  planPrefill(std::vector<PrefillRequestView> ready) const;
  [[nodiscard]] uint32_t
  prefillBudget(const PrefillRequestView &leader,
                std::span<const PrefillRequestView> ready) const;
  // The rows of a contended command of a lane that streams its history,
  // which reads all of it whatever its rows: the fewest, halving from its
  // prefill group's, whose time covers its estimated wait on the reads, so
  // that fewer would not end the command sooner, and no fewer than
  // `contendedRows`, those of another contended command, which it takes
  // before its first streamed command.
  [[nodiscard]] uint32_t streamingRows(const Request &leader, uint32_t contendedRows) const;
  // A prefill command's estimated time with `rows` rows (observePrefill).
  [[nodiscard]] double prefillMilliseconds(uint32_t rows) const noexcept;
  // The passes a command of `rows` rows the lane leads runs in: as many as
  // keep each within the contended slice by the command estimate, up to its
  // model's; one for a lane that streams its history, or before the first
  // estimate.
  [[nodiscard]] uint32_t passesFor(const Request &leader, uint32_t rows) const;
  // A lane's gap between decode commands starts when it begins to decode
  // and after each of its decode commands.
  void startDecodeGap(Request &request) noexcept;
  // Debt is owed only to requests that decode or wait for a mask: once the
  // last one leaves, a later decoder starts without it.
  void dropStaleDecodeDebt() noexcept;

  std::unordered_map<uint64_t, Request> requests_;
  std::optional<BatchPlan> active_;
  std::optional<Passes> passes_;
  // The lanes the active command was planned without.
  std::vector<uint64_t> excluded_;
  uint64_t order_ = 0;
  uint64_t decodeDispatchOrder_ = 0;
  // A prefill command's estimated cost: fixed, plus per row, and the
  // decayed sums of the later commands' rows, times and waits on host steps
  // it fits (observePrefill).
  double prefillFixedMilliseconds_ = 0.0;
  double prefillMillisecondsPerToken_ = 0.0;
  struct PrefillFit final {
    bool seeded = false;
    double weight = 0.0;
    double rows = 0.0;
    double rowsSquared = 0.0;
    double time = 0.0;
    double rowsTime = 0.0;
    double host = 0.0;
  };
  PrefillFit prefillFit_;
  // The decode commands' wall time per token committed per lane, and their
  // count, each decayed by kDecodeTokenDecay per command after it
  // (decodeTokenMilliseconds).
  double decodeTokenTime_ = 0.0;
  double decodeTokenWeight_ = 0.0;
  double decodeShare_;
  uint64_t maximumAttentionPairs_;
  // Decode time that prefill still owes the lanes that decoded beside it
  // (not those waiting for a mask): equal-priority decode runs until its
  // commands' wall time has worked it off.
  double decodeDebtMilliseconds_ = 0.0;
  std::optional<WorkKind> lastCommittedKind_;
  SchedulerSnapshot counters_;
  // The time of the commands completed so far, and the streaming prefill
  // commands among them, which lanes' decode gaps are measured on; the gaps
  // across a streaming command.
  double commandMilliseconds_ = 0.0;
  uint64_t streamingCommands_ = 0;
  model::LatencyHistogram decodeGaps_;
};

} // namespace splash::engine
