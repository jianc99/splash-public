#include "engine/Scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::engine {
namespace {

// A lane yields to later arrivals at most this many prefill commands in a
// row: the other lanes of a full batch, each finishing one short prompt. So
// often at most a command in passes it is a lane of is abandoned for later
// work, of its priority or a higher one, before its rows commit: as often as
// one planned again can widen.
constexpr uint32_t kMaximumOvertakes =
    model::ExecutionLimits::maximumBatchWidth - 1;

// The slice a contended prefill command aims for: while peers of its
// priority decode, the budget halves until its estimated cost fits, so they
// wait about this long between steps. A latency target, not a deadline: the
// first command has no measurement and the budget stops at
// kMinimumPrefillRows. A command with a fixed cost (a streamed target's
// expert reads, which any chunk pays in full) gets twice that cost instead,
// so its rows take at least as long as its fixed part and its peers wait
// that long, seconds rather than this slice; where its model runs a command
// in passes over its layers (Model::prefillPasses), its leader keeps its rows
// whatever the contention (prefillBudget), and each pass aims for this
// slice, contended or not, so that decode steps run between passes
// (passesFor). A lane that streams its history sizes its commands by its own
// reads instead (streamingRows).
constexpr double kContendedPrefillMilliseconds = 500.0;
// The fewest rows a contended prefill command is cut to, and the fewest whose
// time measures prefill throughput: below it a command's fixed costs dominate.
constexpr uint32_t kMinimumPrefillRows = 64;
// The weight of a command's measured time in a running estimate (the
// command estimate's fit, or a lane's wait on its history's reads): one
// unusually slow or fast command moves it a quarter of the way, and a
// lasting change carries it within a few commands.
constexpr double kPrefillTimeWeight = 0.25;
// What the commands' waits on host steps are worth to the command
// estimate's fixed part: one command a chunk's rows from the others
// (observePrefill).
constexpr double kPrefillHostSpread = double{model::ExecutionLimits::prefillTokenBudget} *
                                      model::ExecutionLimits::prefillTokenBudget;

// A command excludes at most the lanes of a full batch: a linear search.
bool listed(std::span<const uint64_t> ids, uint64_t id) noexcept {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// The rows of an uncontended command of a lane with `remaining` rows to
// prefill: up to a chunk, or for a lane whose commands may be prefill groups
// of up to `groupRows` rows, its remaining chunks split evenly over the
// fewest groups that hold them (a short last group would pay a whole pass
// over a target's streamed experts for a few rows).
uint32_t uncontendedRows(uint32_t remaining, uint32_t groupRows) noexcept {
  constexpr uint32_t chunk = model::ExecutionLimits::prefillTokenBudget;
  if (remaining <= chunk || groupRows <= chunk)
    return std::min(remaining, chunk);
  const uint32_t chunks = (remaining + chunk - 1) / chunk;
  const uint32_t groups = (chunks + groupRows / chunk - 1) / (groupRows / chunk);
  return std::min(remaining, (chunks + groups - 1) / groups * chunk);
}

// The fewest of a lane's `remaining` rows that spare it one of its
// uncontended commands, each of up to `groupRows` rows (uncontendedRows):
// all of them where one command ends its prompt.
uint32_t sparingRows(uint32_t remaining, uint32_t groupRows) noexcept {
  return (remaining - 1) % std::max(groupRows, model::ExecutionLimits::prefillTokenBudget) + 1;
}

} // namespace

void Scheduler::submit(RequestSpec request) {
  Request state;
  state.spec = std::move(request);
  state.order = ++order_;
  requests_.emplace(state.spec.id, std::move(state));
}

void Scheduler::resourcesReady(uint64_t id, uint32_t processed) {
  Request &request = get(id);
  if (request.phase != Phase::Queued &&
      request.phase != Phase::WaitingResources &&
      request.phase != Phase::WaitingPrefix) {
    throw std::logic_error("only queued work can be admitted");
  }
  if (processed > request.spec.prefillTokens) {
    throw std::invalid_argument("processed prompt exceeds request");
  }
  if (request.suspended) {
    throw std::logic_error("suspended request requires resumeFromResources");
  }
  request.promptProcessed = processed;
  request.decodeStage = DecodeStage::Regular;
  request.phase =
      processed == request.spec.prefillTokens ? Phase::Decode : Phase::Prefill;
  if (request.phase == Phase::Decode)
    startDecodeGap(request);
}

void Scheduler::suspendForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Prefill && request.phase != Phase::Decode) {
    throw std::logic_error("only runnable resident work can be suspended");
  }
  request.suspended = true;
  request.phase = Phase::WaitingResources;
  dropStaleDecodeDebt();
}

void Scheduler::resumeFromResources(uint64_t id, uint32_t processed,
                                    uint32_t replayTokens) {
  Request &request = get(id);
  if (request.phase != Phase::WaitingResources || !request.suspended) {
    throw std::logic_error("request is not suspended for resources");
  }
  if (processed >= replayTokens)
    throw std::invalid_argument("resource replay must leave an input token");
  request.spec.prefillTokens = replayTokens;
  request.promptProcessed = processed;
  request.phase = Phase::Prefill;
  request.suspended = false;
}

void Scheduler::deferAdmission(uint64_t id) {
  Request &request = get(id);
  if (request.suspended ||
      (request.phase != Phase::Queued &&
       request.phase != Phase::WaitingResources &&
       request.phase != Phase::WaitingPrefix))
    throw std::logic_error("only unstarted work can wait for scheduling");
  request.phase = Phase::Queued;
}

void Scheduler::waitForResources(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::Queued &&
      request.phase != Phase::WaitingResources &&
      request.phase != Phase::WaitingPrefix) {
    throw std::logic_error("resident request cannot wait before admission");
  }
  request.phase = Phase::WaitingResources;
}

void Scheduler::waitForPrefix(uint64_t id) {
  Request &request = get(id);
  if (request.suspended ||
      (request.phase != Phase::Queued &&
       request.phase != Phase::WaitingResources &&
       request.phase != Phase::WaitingPrefix)) {
    throw std::logic_error("only unstarted requests can wait for a prefix");
  }
  request.phase = Phase::WaitingPrefix;
}

void Scheduler::maskReady(uint64_t id) {
  Request &request = get(id);
  if (request.phase != Phase::WaitingMask ||
      !waitsForMask(request.decodeStage)) {
    throw std::logic_error("request is not waiting for a mask");
  }
  request.phase = Phase::Decode;
  startDecodeGap(request);
}

void Scheduler::setStreamsHistory(uint64_t id, bool streams) {
  get(id).streamsHistory = streams;
}

void Scheduler::setPrefillGroupRows(uint64_t id, uint32_t rows) {
  get(id).spec.prefillGroupRows = rows;
}

void Scheduler::cancel(uint64_t id) {
  Request &request = get(id);
  if (!terminal(request.phase))
    request.phase = Phase::Cancelled;
  dropStaleDecodeDebt();
}

void Scheduler::finish(uint64_t id) {
  Request &request = get(id);
  if (!terminal(request.phase))
    request.phase = Phase::Completed;
  dropStaleDecodeDebt();
}

void Scheduler::fail(uint64_t id) {
  Request &request = get(id);
  if (terminal(request.phase))
    return;
  request.phase = Phase::Failed;
  dropStaleDecodeDebt();
}

void Scheduler::remove(uint64_t id) {
  if (active_) {
    for (const BatchItem &item : active_->items) {
      if (item.requestId == id) {
        throw std::logic_error("cannot remove an active batch member");
      }
    }
  }
  auto found = requests_.find(id);
  if (found == requests_.end() || !terminal(found->second.phase)) {
    throw std::logic_error("only terminal requests can be removed");
  }
  requests_.erase(found);
}

std::vector<uint64_t> Scheduler::admissionOrder() const {
  std::vector<const Request *> ready;
  ready.reserve(requests_.size());
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Queued ||
        request.phase == Phase::WaitingResources ||
        request.phase == Phase::WaitingPrefix) {
      ready.push_back(&request);
    }
  }
  std::sort(ready.begin(), ready.end(), byPriorityThenOrder);
  std::vector<uint64_t> result;
  result.reserve(ready.size());
  for (const Request *request : ready)
    result.push_back(request->spec.id);
  return result;
}

std::vector<uint64_t> Scheduler::prefillAdmissionOrder(
    std::span<const PrefillAdmission> candidates) const {
  std::vector<PrefillRequestView> pending;
  pending.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    const Request &value = get(candidate.requestId);
    if (value.suspended ||
        (value.phase != Phase::Queued && value.phase != Phase::WaitingResources &&
         value.phase != Phase::WaitingPrefix) ||
        candidate.cachedTokens >= value.spec.prefillTokens)
      throw std::logic_error("invalid pending prefill admission");
    pending.push_back({&value, candidate.cachedTokens});
  }
  std::vector<PrefillRequestView> lanes;
  lanes.reserve(requests_.size());
  std::vector<uint64_t> result;
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Prefill)
      lanes.push_back({&request, request.promptProcessed});
  }
  // Admission starts only lanes the next command serves: it plans again
  // without the candidates a plan leaves out until it keeps all it plans, as
  // without them the plan can take another leader (planPrefill measures its
  // near-equal band from the shortest lane).
  std::optional<BatchPlan> plan;
  size_t planned = 0;
  do {
    planned = pending.size();
    std::vector<PrefillRequestView> ready = lanes;
    ready.insert(ready.end(), pending.begin(), pending.end());
    plan = planPrefill(std::move(ready));
    if (!plan)
      return result;
    std::erase_if(pending, [&](const PrefillRequestView &view) {
      return std::none_of(plan->items.begin(), plan->items.end(), [&](const BatchItem &item) {
        return item.requestId == view.request->spec.id;
      });
    });
  } while (pending.size() != planned);
  // Admission ranks candidates against every decoder, left out of a plan or
  // not.
  const auto decode = nextDecode({});
  if (decode && get(decode->items.front().requestId).spec.priority <
                    get(plan->items.front().requestId).spec.priority)
    return result;
  for (const auto &item : plan->items)
    if (get(item.requestId).phase != Phase::Prefill)
      result.push_back(item.requestId);
  return result;
}

std::optional<RequestPriority> Scheduler::highestRunnablePriority() const noexcept {
  std::optional<RequestPriority> result;
  for (const auto &[_, request] : requests_) {
    if ((request.phase == Phase::Prefill || request.phase == Phase::Decode) &&
        (!result || request.spec.priority < *result))
      result = request.spec.priority;
  }
  return result;
}

bool Scheduler::yieldsTo(RequestPriority priority, uint32_t joined) const {
  return passes_ && priority < get(passes_->plan.items.front().requestId).spec.priority &&
         givesWay(joined, 0.0);
}

bool Scheduler::givesWay(uint32_t joined, double added) const {
  const BatchPlan &command = passes_->plan;
  return std::ranges::none_of(command.items,
                              [&](const BatchItem &item) {
                                return get(item.requestId).abandoned >= kMaximumOvertakes;
                              }) &&
         command.width() * (command.pass + added) < joined * double(command.passes - command.pass);
}

bool Scheduler::abandonsPasses(const BatchPlan &plan) const {
  return passes_ && plan.kind == WorkKind::Prefill &&
         (plan.pass != passes_->plan.pass || plan.passes != passes_->plan.passes ||
          plan.items != passes_->plan.items);
}

std::optional<BatchPlan> Scheduler::next(std::span<const uint64_t> excluded) const {
  if (active_)
    return std::nullopt;
  auto decode = nextDecode(excluded);
  auto prefill = nextPrefill(excluded);
  if (!decode)
    return prefill;
  if (!prefill)
    return decode;

  const RequestPriority decodePriority =
      get(decode->items.front().requestId).spec.priority;
  RequestPriority prefillPriority =
      get(prefill->items.front().requestId).spec.priority;
  // A command between its passes runs at the priority of the most urgent
  // request it holds back, a prompt it does not yield to or a start it leaves
  // waiting for its end (Engine::preemptBelow), or that one would wait for
  // every decode above the command's own priority and not only its passes.
  if (passes_)
    for (const auto &[id, request] : requests_)
      if ((request.phase == Phase::Prefill && !listed(excluded, id)) ||
          request.phase == Phase::WaitingResources)
        prefillPriority = std::min(prefillPriority, request.spec.priority);
  if (decodePriority != prefillPriority) {
    return decodePriority < prefillPriority ? std::move(decode)
                                             : std::move(prefill);
  }

  // Prefill and decode use different Metal graphs and cannot share one
  // command. Honor request priority first. Then decode runs while
  // prefill owes it time, and otherwise the kinds alternate at command
  // boundaries so equal-priority work cannot starve.
  if (decodeDebtMilliseconds_ > 0.0)
    return decode;
  return lastCommittedKind_ == WorkKind::Decode ? std::move(prefill)
                                                 : std::move(decode);
}

std::optional<BatchPlan> Scheduler::nextPrefill(std::span<const uint64_t> excluded) const {
  std::vector<PrefillRequestView> ready;
  for (const auto &[id, request] : requests_) {
    if (request.phase == Phase::Prefill && !listed(excluded, id))
      ready.push_back({&request, request.promptProcessed});
  }
  std::optional<BatchPlan> plan = planPrefill(std::move(ready));
  // A command between its passes goes on before any other prefill, whose
  // rows would take the model's prefill memory its rows' work is in, but one
  // that supersedes it.
  if (!passes_ || (plan && supersedes(*plan)))
    return plan;
  for (const BatchItem &item : passes_->plan.items)
    if (listed(excluded, item.requestId))
      return std::nullopt;
  return passes_->plan;
}

bool Scheduler::supersedes(const BatchPlan &plan) const {
  const BatchPlan &command = passes_->plan;
  const auto joins = [&](const BatchItem &item) {
    return std::ranges::none_of(command.items,
                                [&](const BatchItem &lane) { return lane.requestId == item.requestId; });
  };
  const auto joined = static_cast<uint32_t>(std::ranges::count_if(plan.items, joins));
  const RequestPriority priority = get(plan.items.front().requestId).spec.priority;
  if (priority < get(command.items.front().requestId).spec.priority)
    return yieldsTo(priority, joined);
  const auto kept = [&](const BatchItem &lane) { return std::ranges::find(plan.items, lane) != plan.items.end(); };
  const auto leftOut = [&](const BatchItem &item) { return listed(passes_->leftOut, item.requestId); };
  if (!std::ranges::all_of(command.items, kept) || std::ranges::any_of(plan.items, leftOut))
    return false;
  // Planned again with lanes that join, each lane of the command loses the
  // passes it ran and what their rows add to its time, in passes of it.
  const auto rows = [](const BatchPlan &of) {
    uint32_t sum = 0;
    for (const BatchItem &item : of.items)
      sum += item.tokenCount;
    return sum;
  };
  const double time = prefillMilliseconds(rows(command));
  return givesWay(joined, command.passes * (prefillMilliseconds(rows(plan)) - time) / time);
}

std::optional<BatchPlan>
Scheduler::planPrefill(std::vector<PrefillRequestView> ready) const {
  if (ready.empty())
    return std::nullopt;
  // Order by the remaining prompt. After kMaximumOvertakes consecutive skips,
  // an older lane leads the next command to prevent starvation.
  const auto overdue = [](const PrefillRequestView &view) {
    return view.request->overtaken >= kMaximumOvertakes;
  };
  const auto remaining = [](const PrefillRequestView &view) {
    return uint64_t{view.request->spec.prefillTokens - view.promptProcessed};
  };
  std::sort(ready.begin(), ready.end(),
            [&](const PrefillRequestView &a, const PrefillRequestView &b) {
              if (a.request->spec.priority != b.request->spec.priority)
                return a.request->spec.priority < b.request->spec.priority;
              if (overdue(a) != overdue(b))
                return overdue(a);
              if (remaining(a) != remaining(b))
                return remaining(a) < remaining(b);
              return a.request->order < b.request->order;
            });
  // Unless the front lane is overdue, the oldest lane of its priority leads
  // whose remaining prompt exceeds the shortest by at most floor((m-1)/k)
  // commands of c rows, with c the row budget of the shortest's commands, m
  // their number and k = kMaximumOvertakes: as many as an earlier lane would
  // lead, overdue, before the shortest's last, were the shortest to lead. An
  // earlier lane it makes overdue comes in as a second long lane, which can
  // push a Resident lane to Spill. Between two lanes with serial prefill, the
  // earlier one leading does not raise the summed first-token latency where
  // those overdue commands would take c rows each. It raises it where one
  // comes before a last command of the shortest that fits a contended slice,
  // and so takes the slice, and where a lane older than the leader and
  // outside the band, overtaken by it as by the shortest, leads overdue
  // commands ahead of both. While the first lane to finish decodes, the
  // other's rows run at 1/(1 + decodeShare_) of their speed, and the earlier
  // lane leading leaves up to the tolerance more of them: the later first
  // token can come up to decodeShare_ times their time later. The rest stay
  // shortest first. A tolerance is not transitive, so it cannot go in the
  // comparator.
  if (!overdue(ready.front())) {
    const RequestPriority priority = ready.front().request->spec.priority;
    const uint64_t shortest = remaining(ready.front());
    const uint64_t rows = prefillBudget(ready.front(), ready);
    const uint64_t tolerance = (shortest - 1) / (rows * kMaximumOvertakes) * rows;
    auto oldest = ready.begin();
    for (auto view = ready.begin();
         view != ready.end() && view->request->spec.priority == priority; ++view)
      if (remaining(*view) <= shortest + tolerance &&
          view->request->order < oldest->request->order)
        oldest = view;
    std::rotate(ready.begin(), oldest, oldest + 1);
  }
  const PrefillRequestView &leader = ready.front();
  const RequestPriority selectedPriority = leader.request->spec.priority;

  BatchPlan plan;
  plan.kind = WorkKind::Prefill;
  uint32_t budget = prefillBudget(leader, ready);
  uint64_t pairs = 0;
  // The command's rows so far: a lane joins only while they lie within the
  // first chunk, and its group rows bound the command's
  // (ExecutionLimits::prefillTokenBudget).
  uint32_t planned = 0;
  for (const PrefillRequestView &view : ready) {
    if (planned > model::ExecutionLimits::prefillTokenBudget)
      break;
    budget = std::min(budget, view.request->spec.prefillGroupRows - planned);
    if (!budget || view.request->spec.priority != selectedPriority ||
        plan.width() == model::ExecutionLimits::maximumBatchWidth)
      break;
    // A lane that streams its history runs alone: nothing joins a command
    // it leads, and it joins none it does not lead; overtaken often enough,
    // it leads one.
    if (view.request->streamsHistory && &view != &leader)
      continue;
    uint32_t rows =
        std::min(view.request->spec.prefillTokens - view.promptProcessed, budget);
    // The command's attention stays within maximumAttentionPairs_: a lane's
    // rows halve until they fit, and one that does not fit at
    // kMinimumPrefillRows waits for a command of its own.
    const auto exceeds = [&](uint32_t candidate) {
      return pairs + attentionPairs(view.promptProcessed, candidate) > maximumAttentionPairs_;
    };
    // Past the rows of a leader whose command may run in passes, a lane
    // takes the fewest rows that spare it a command of its own, or none:
    // fewer would leave it as many, and more would only make the others
    // wait. It joins while the fixed part it saves outweighs what its rows
    // add for the lanes planned before it.
    if (&view != &leader && leader.request->spec.prefillPasses > 1) {
      rows = sparingRows(view.request->spec.prefillTokens - view.promptProcessed,
                         view.request->spec.prefillGroupRows);
      if (rows > budget || exceeds(rows) ||
          plan.width() * rows * prefillMillisecondsPerToken_ >= prefillFixedMilliseconds_)
        continue;
    }
    while (rows > kMinimumPrefillRows && exceeds(rows))
      rows /= 2;
    if (!plan.items.empty() && exceeds(rows))
      break;
    pairs += attentionPairs(view.promptProcessed, rows);
    plan.items.push_back(
        {view.request->spec.id, rows, view.promptProcessed});
    planned += rows;
    budget -= rows;
    if (leader.request->streamsHistory)
      break;
  }
  plan.passes = passesFor(*leader.request, planned);
  return plan;
}

uint32_t Scheduler::prefillBudget(
    const PrefillRequestView &leader,
    std::span<const PrefillRequestView> ready) const {
  const uint32_t leaderRemaining =
      leader.request->spec.prefillTokens - leader.promptProcessed;
  const uint32_t maximum = model::ExecutionLimits::prefillTokenBudget;
  // Passes bound a command's time, which fewer rows would not shorten: a
  // leader whose command may run in them keeps its rows whatever contends,
  // and at least a chunk's, of which the ready prompts of its priority take
  // past its own those that spare them a command of their own, which would
  // pay the command's fixed part (a streamed target's expert reads) again
  // (planPrefill).
  if (leader.request->spec.prefillPasses > 1 && !leader.request->streamsHistory)
    return std::max(maximum, uncontendedRows(leaderRemaining, leader.request->spec.prefillGroupRows));
  if (prefillMillisecondsPerToken_ <= 0.0 && prefillFixedMilliseconds_ <= 0.0)
    return maximum;
  uint32_t rows = maximum;
  const double slice = std::max(kContendedPrefillMilliseconds, 2.0 * prefillFixedMilliseconds_);
  while (rows > kMinimumPrefillRows &&
         prefillMilliseconds(rows) > slice)
    rows /= 2;
  const bool leaderFinishing = leaderRemaining <= rows;
  const bool streams = leader.request->streamsHistory;
  const bool contended = std::any_of(
      requests_.begin(), requests_.end(), [&](const auto &entry) {
        const Request &peer = entry.second;
        return (peer.phase == Phase::Decode || peer.phase == Phase::WaitingMask) &&
               peer.spec.priority <= leader.request->spec.priority;
      }) || std::any_of(ready.begin(), ready.end(), [&](
                           const PrefillRequestView &peer) {
        return peer.request->spec.id != leader.request->spec.id &&
               peer.request->spec.priority <= leader.request->spec.priority &&
               (streams || leaderFinishing ||
                peer.request->spec.prefillTokens - peer.promptProcessed <= rows);
      });
  // Bound commands for peers that decode or wait for a CPU mask, including
  // decoders left out of this command (their streams resume once their
  // memory lands), and for peers that can finish prefill within this slice,
  // or, beside a lane that streams its history and so runs alone, any ready
  // peer. The first sample and minimum matrix shape remain limits.
  if (contended)
    return streams ? streamingRows(*leader.request, rows) : rows;
  // A leader that finishes within the full budget ends the command at its
  // last row: prefill cost is linear above the slice, so shortest-first
  // sequential commands minimise first-token latency. Long prefills share a
  // command only when no lane finishes. A leader that runs prefill groups
  // takes up to a group; planPrefill gives what its rows leave to the next
  // lanes within ExecutionLimits::prefillTokenBudget's contract.
  return uncontendedRows(leaderRemaining, leader.request->spec.prefillGroupRows);
}

double Scheduler::prefillMilliseconds(uint32_t rows) const noexcept {
  return prefillFixedMilliseconds_ + rows * prefillMillisecondsPerToken_;
}

uint32_t Scheduler::passesFor(const Request &leader, uint32_t rows) const {
  if (leader.spec.prefillPasses <= 1 || leader.streamsHistory)
    return 1;
  return static_cast<uint32_t>(std::clamp(std::ceil(prefillMilliseconds(rows) / kContendedPrefillMilliseconds), 1.0,
                                          static_cast<double>(leader.spec.prefillPasses)));
}

uint32_t Scheduler::streamingRows(const Request &leader, uint32_t contendedRows) const {
  if (!leader.streamMilliseconds)
    return contendedRows;
  uint32_t rows = std::max(leader.spec.prefillGroupRows, contendedRows);
  while (rows / 2 >= contendedRows &&
         rows / 2 * prefillMillisecondsPerToken_ >= *leader.streamMilliseconds)
    rows /= 2;
  return rows;
}

std::optional<BatchPlan> Scheduler::nextDecode(std::span<const uint64_t> excluded) const {
  std::vector<const Request *> ready;
  for (const auto &[id, request] : requests_) {
    if (request.phase == Phase::Decode && !listed(excluded, id))
      ready.push_back(&request);
  }
  if (ready.empty())
    return std::nullopt;
  std::sort(ready.begin(), ready.end(), [](const Request *a, const Request *b) {
    if (a->spec.priority != b->spec.priority)
      return a->spec.priority < b->spec.priority;
    if (a->lastDecodeDispatch != b->lastDecodeDispatch)
      return a->lastDecodeDispatch < b->lastDecodeDispatch;
    return a->order < b->order;
  });
  const DecodeStage decodeStage = ready.front()->decodeStage;
  const RequestPriority selectedPriority = ready.front()->spec.priority;
  BatchPlan plan;
  plan.kind = WorkKind::Decode;
  plan.decodeStage = decodeStage;
  for (const Request *request : ready) {
    if (request->spec.priority != selectedPriority ||
        request->decodeStage != decodeStage)
      continue;
    plan.items.push_back({request->spec.id, 0, 0});
    if (plan.width() == model::ExecutionLimits::maximumBatchWidth)
      break;
  }
  return plan;
}

void Scheduler::commit(const BatchPlan &plan, std::span<const uint64_t> excluded) {
  if (active_ || plan.empty() ||
      plan.width() > model::ExecutionLimits::maximumBatchWidth) {
    throw std::logic_error("invalid scheduler commit");
  }
  for (const BatchItem &item : plan.items) {
    const Request &request = get(item.requestId);
    const Phase expected =
        plan.kind == WorkKind::Prefill ? Phase::Prefill : Phase::Decode;
    if (request.phase != expected ||
        (plan.kind == WorkKind::Prefill &&
         (!item.tokenCount || item.promptOffset != request.promptProcessed)) ||
        (plan.kind == WorkKind::Decode &&
         (item.tokenCount || item.promptOffset ||
          request.decodeStage != plan.decodeStage))) {
      throw std::logic_error("batch no longer matches scheduler state");
    }
  }
  if (abandonsPasses(plan) ||
      (!passes_ && (plan.pass || (plan.passes > 1 && plan.kind != WorkKind::Prefill))))
    throw std::logic_error("batch no longer matches scheduler state");
  if (plan.passes > 1 && !passes_) {
    passes_ = Passes{.plan = plan};
    for (const auto &[id, request] : requests_)
      if (request.phase == Phase::Prefill && !listed(excluded, id) &&
          std::ranges::none_of(plan.items, [&](const BatchItem &item) { return item.requestId == id; }))
        passes_->leftOut.push_back(id);
  }
  excluded_.assign(excluded.begin(), excluded.end());
  active_ = plan;
  lastCommittedKind_ = plan.kind;
  // A command counts, and passes others over, once: with its first pass.
  if (plan.kind == WorkKind::Prefill && !plan.pass) {
    ++counters_.prefillBatches;
    for (const BatchItem &item : plan.items)
      counters_.prefillRows += item.tokenCount;
    uint64_t youngestServed = 0;
    for (const BatchItem &item : plan.items)
      youngestServed = std::max(youngestServed, get(item.requestId).order);
    for (auto &[id, request] : requests_) {
      if (terminal(request.phase) || request.phase == Phase::Decode ||
          request.phase == Phase::WaitingMask || request.suspended ||
          listed(excluded, id))
        continue;
      const bool served = std::any_of(
          plan.items.begin(), plan.items.end(),
          [id](const BatchItem &item) { return item.requestId == id; });
      if (served)
        request.overtaken = 0;
      else if (request.order < youngestServed)
        ++request.overtaken;
    }
  } else if (plan.kind == WorkKind::Decode) {
    const uint64_t dispatchOrder = ++decodeDispatchOrder_;
    for (const BatchItem &item : plan.items)
      get(item.requestId).lastDecodeDispatch = dispatchOrder;
    ++counters_.decodeBatches;
    ++counters_.decodeBatchesByWidth[plan.width() - 1];
  }
}

void Scheduler::complete(const BatchPlan &plan,
                         std::span<const StepResult> results,
                         double wallMilliseconds,
                         bool representativePrefillTiming,
                         double hostMilliseconds,
                         std::optional<double> streamMilliseconds,
                         double imageEncodeMilliseconds) {
  if (!active_ || active_->kind != plan.kind ||
      active_->items.size() != plan.items.size() ||
      results.size() != plan.items.size()) {
    throw std::logic_error("completion does not match active batch");
  }
  if (streamMilliseconds && (plan.kind != WorkKind::Prefill || plan.items.size() != 1))
    throw std::logic_error("only a prefill command of one lane streams its history");
  const bool timed = std::isfinite(wallMilliseconds) && wallMilliseconds > 0.0;
  // The lane of a prefill command that streamed its history, alone in it. A
  // lane that streams may run commands that read nothing yet, before its
  // first pages spill: those say nothing of its reads.
  Request *const streaming =
      streamMilliseconds ? &get(plan.items.front().requestId) : nullptr;
  // Its wait on the reads, which its waits on host steps include; none
  // without a measurement.
  const std::optional<double> stream =
      streaming && std::isfinite(*streamMilliseconds) && std::isfinite(hostMilliseconds)
          ? std::optional(std::clamp(*streamMilliseconds, 0.0, std::max(hostMilliseconds, 0.0)))
          : std::nullopt;
  if (timed) {
    commandMilliseconds_ += wallMilliseconds;
    streamingCommands_ += streaming != nullptr;
  }
  for (size_t index = 0; index < results.size(); ++index) {
    const BatchItem &item = plan.items[index];
    const StepResult &result = results[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("completion request order changed");
    }
    Request &request = get(item.requestId);
    if (plan.kind == WorkKind::Prefill && plan.pass + 1 < plan.passes) {
      // A pass before the last consumes nothing and ends no lane: one that
      // ends meanwhile stays in the command to its last pass.
      if (result.consumedPromptTokens || result.finished)
        throw std::logic_error("prefill completion row count changed");
    } else if (plan.kind == WorkKind::Prefill) {
      if (result.consumedPromptTokens != item.tokenCount ||
          item.promptOffset != request.promptProcessed) {
        throw std::logic_error("prefill completion row count changed");
      }
      request.promptProcessed += result.consumedPromptTokens;
      request.abandoned = 0;
      // A stop token or a one-token budget is selected by prefill itself; a
      // constrained prompt waits for the mask its first token needs.
      if (result.finished) {
        request.phase = Phase::Completed;
      } else if (request.promptProcessed != request.spec.prefillTokens) {
        request.phase = Phase::Prefill;
      } else if (waitsForMask(result.nextDecodeStage)) {
        request.decodeStage = result.nextDecodeStage;
        request.phase = Phase::WaitingMask;
      } else {
        request.phase = Phase::Decode;
        startDecodeGap(request);
      }
    } else {
      // The gap since its last decode command, recorded when a streaming
      // prefill command ran within it.
      if (request.decodeGapStreams != streamingCommands_)
        decodeGaps_.add(
            static_cast<uint64_t>((commandMilliseconds_ - request.decodeGapStart) * 1e6));
      startDecodeGap(request);
      request.decodeStage = result.nextDecodeStage;
      request.phase = result.finished ? Phase::Completed : Phase::Decode;
    }
  }
  // A streaming command trains the command estimate as if it had not waited
  // on its history's reads, and its lane's own estimate with that wait; a
  // command trains it without the encodes of its images. A command in passes
  // trains it once, with its passes' time together.
  const double rowsWall =
      wallMilliseconds - std::clamp(imageEncodeMilliseconds, 0.0, std::max(wallMilliseconds, 0.0));
  double commandWall = rowsWall, commandHost = hostMilliseconds;
  bool representative = representativePrefillTiming;
  if (plan.kind == WorkKind::Prefill && plan.passes > 1) {
    commandWall = passes_->wallMilliseconds += rowsWall;
    commandHost = passes_->hostMilliseconds += hostMilliseconds;
    representative = passes_->representative = passes_->representative && representativePrefillTiming;
    if (plan.pass + 1 < plan.passes) {
      representative = false;
      ++passes_->plan.pass;
    } else {
      passes_.reset();
    }
  }
  if (representative && plan.kind == WorkKind::Prefill) {
    uint32_t rows = 0;
    for (const BatchItem &item : plan.items)
      rows += item.tokenCount;
    const double waited = stream.value_or(0.0);
    observePrefill(rows, commandWall - waited, commandHost - waited);
  }
  if (stream)
    streaming->streamMilliseconds =
        streaming->streamMilliseconds
            ? (1.0 - kPrefillTimeWeight) * *streaming->streamMilliseconds +
                  kPrefillTimeWeight * *stream
            : *stream;
  // While requests of the same or a higher priority decode, lanes this
  // prefill finished included, it owes decode a share of its time; decode
  // commands work the debt off with their own. A lane waiting for its mask
  // is owed nothing: it could not have decoded meanwhile, and the slice
  // already bounds the prefill it waits behind. Nor is a decoder this
  // command was planned without: it could not run either.
  if (timed) {
    if (plan.kind == WorkKind::Decode) {
      decodeDebtMilliseconds_ =
          std::max(0.0, decodeDebtMilliseconds_ - wallMilliseconds);
    } else if (streaming || decodeShare_ > 0.0) {
      const RequestPriority priority =
          get(plan.items.front().requestId).spec.priority;
      const bool contended = std::any_of(
          requests_.begin(), requests_.end(), [&](const auto &entry) {
            const auto &[id, peer] = entry;
            return peer.phase == Phase::Decode && peer.spec.priority <= priority &&
                   !listed(excluded_, id);
          });
      counters_.contendedStreamingPrefillBatches += contended && streaming;
      if (contended && decodeShare_ > 0.0)
        decodeDebtMilliseconds_ += decodeShare_ * wallMilliseconds;
    }
  }
  dropStaleDecodeDebt();
  excluded_.clear();
  active_.reset();
}

void Scheduler::abandonPasses() {
  if (!passes_ || (active_ && active_->passes > 1))
    throw std::logic_error("no command between its passes to abandon");
  for (const BatchItem &item : passes_->plan.items)
    ++get(item.requestId).abandoned;
  passes_.reset();
}

void Scheduler::startDecodeGap(Request &request) noexcept {
  request.decodeGapStart = commandMilliseconds_;
  request.decodeGapStreams = streamingCommands_;
}

void Scheduler::dropStaleDecodeDebt() noexcept {
  const bool decoding = std::any_of(
      requests_.begin(), requests_.end(), [](const auto &entry) {
        return entry.second.phase == Phase::Decode ||
               entry.second.phase == Phase::WaitingMask;
      });
  if (!decoding)
    decodeDebtMilliseconds_ = 0.0;
}

void Scheduler::observePrefill(uint32_t rows, double wallMilliseconds,
                               double hostMilliseconds) {
  // Tiny tails are dominated by fixed command costs, not prefill throughput.
  if (rows < kMinimumPrefillRows || !std::isfinite(wallMilliseconds) ||
      wallMilliseconds <= 0.0)
    return;
  PrefillFit &fit = prefillFit_;
  const auto waited = [&](double time) {
    return std::isfinite(hostMilliseconds) ? std::clamp(hostMilliseconds, 0.0, time) : 0.0;
  };
  // The first command seeds the estimate: its waits on host steps fixed,
  // the rest per row.
  if (!fit.seeded) {
    prefillFixedMilliseconds_ = waited(wallMilliseconds);
    prefillMillisecondsPerToken_ = (wallMilliseconds - prefillFixedMilliseconds_) / rows;
    fit.seeded = true;
    return;
  }
  // A command counts at most twice and at least half the time the estimate
  // gives it, so that a stall does not price the commands after it.
  const double expected = prefillMilliseconds(rows);
  const double time = std::clamp(wallMilliseconds, 0.5 * expected, 2.0 * expected);
  const double decay = 1.0 - kPrefillTimeWeight;
  fit.weight = decay * fit.weight + 1.0;
  fit.rows = decay * fit.rows + rows;
  fit.rowsSquared = decay * fit.rowsSquared + double(rows) * rows;
  fit.time = decay * fit.time + time;
  fit.rowsTime = decay * fit.rowsTime + rows * time;
  fit.host = decay * fit.host + waited(time);
  const double meanRows = fit.rows / fit.weight, meanTime = fit.time / fit.weight;
  const double spread = std::max(fit.rowsSquared / fit.weight - meanRows * meanRows, 0.0);
  const double covariance = fit.rowsTime / fit.weight - meanRows * meanTime;
  // The cost per row the commands' waits on host steps leave, were they the
  // fixed part.
  const double hostPerRow = (meanTime - fit.host / fit.weight) / meanRows;
  const double slope = (fit.weight * covariance + kPrefillHostSpread * hostPerRow) /
                       (fit.weight * spread + kPrefillHostSpread);
  prefillMillisecondsPerToken_ = std::clamp(slope, 0.0, meanTime / meanRows);
  prefillFixedMilliseconds_ = meanTime - prefillMillisecondsPerToken_ * meanRows;
}

Phase Scheduler::phase(uint64_t id) const { return get(id).phase; }

DecodeStage Scheduler::decodeStage(uint64_t id) const { return get(id).decodeStage; }

bool Scheduler::suspended(uint64_t id) const { return get(id).suspended; }

uint64_t Scheduler::submissionOrder(uint64_t id) const { return get(id).order; }

uint32_t Scheduler::promptProcessed(uint64_t id) const {
  return get(id).promptProcessed;
}

SchedulerSnapshot Scheduler::snapshot() const noexcept {
  SchedulerSnapshot result = counters_;
  result.decodeGap = decodeGaps_.snapshot();
  for (const auto &[_, request] : requests_) {
    switch (request.phase) {
    case Phase::Queued:
      ++result.queued;
      break;
    case Phase::WaitingResources:
      ++result.waitingResources;
      break;
    case Phase::WaitingPrefix:
      ++result.waitingPrefix;
      break;
    case Phase::Prefill:
      ++result.prefilling;
      break;
    case Phase::Decode:
      ++result.decoding;
      break;
    case Phase::WaitingMask:
      ++result.waitingMask;
      break;
    case Phase::Completed:
    case Phase::Cancelled:
    case Phase::Failed:
      ++result.terminal;
      break;
    }
  }
  return result;
}

Scheduler::Request &Scheduler::get(uint64_t id) {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

const Scheduler::Request &Scheduler::get(uint64_t id) const {
  auto found = requests_.find(id);
  if (found == requests_.end())
    throw std::out_of_range("unknown request");
  return found->second;
}

bool Scheduler::terminal(Phase phase) noexcept {
  return phase == Phase::Completed || phase == Phase::Cancelled ||
         phase == Phase::Failed;
}

bool Scheduler::byPriorityThenOrder(const Request *a,
                                    const Request *b) noexcept {
  if (a->spec.priority != b->spec.priority)
    return a->spec.priority < b->spec.priority;
  return a->order < b->order;
}

} // namespace splash::engine
