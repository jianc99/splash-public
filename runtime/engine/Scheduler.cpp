#include "engine/Scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace splash::engine {
namespace {

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
// The fewest rows halving cuts a lane's command to.
constexpr uint32_t kMinimumPrefillRows = 64;
// A lane waits behind later arrivals of its priority at most this many times
// its own expected remaining cost before it leads to its prompt's end: the
// other lanes of a full batch, each finishing a prompt no longer than its own
// remaining one.
constexpr double kAgingSlowdown = model::ExecutionLimits::maximumBatchWidth - 1;
// The weight a decode command's time per token keeps against the next
// command's in the decode time per token (decodeTokenMilliseconds).
constexpr double kDecodeTokenDecay = 0.9;

// A command excludes at most the lanes of a full batch: a linear search.
bool listed(std::span<const uint64_t> ids, uint64_t id) noexcept {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

bool holds(std::span<const BatchItem> items, uint64_t id) noexcept {
  return std::ranges::any_of(items, [id](const BatchItem &item) { return item.requestId == id; });
}

// The rows of an uncontended command of a lane with `remaining` rows to
// prefill: up to a chunk, or for a lane whose commands are groups of up to
// `groupRows` rows, its remaining chunks split evenly over the fewest groups
// that hold them (a short last group would pay a whole fixed part for a few
// rows).
uint32_t uncontendedRows(uint32_t remaining, uint32_t groupRows) noexcept {
  if (remaining <= kChunk || groupRows <= kChunk)
    return std::min(remaining, kChunk);
  const uint32_t chunks = (remaining + kChunk - 1) / kChunk;
  const uint32_t groups = (chunks + groupRows / kChunk - 1) / (groupRows / kChunk);
  return std::min(remaining, (chunks + groups - 1) / groups * kChunk);
}

// The lanes whose rows run past a chunk as they fill the chunks in order
// (model::commandViolation's count).
uint32_t lanesPastChunk(std::span<const model::PrefillLane> lanes) noexcept {
  uint64_t rows = 0;
  uint32_t past = 0;
  for (const model::PrefillLane &lane : lanes) {
    past += lane.rows && rows / kChunk != (rows + lane.rows - 1) / kChunk;
    rows += lane.rows;
  }
  return past;
}

uint64_t rowsOf(std::span<const model::PrefillLane> lanes) noexcept {
  uint64_t rows = 0;
  for (const model::PrefillLane &lane : lanes)
    rows += lane.rows;
  return rows;
}

} // namespace

Scheduler::Scheduler(SchedulerConfig config, const model::PrefillWorkSource &work)
    : config_(config), work_(work), decodeShare_(config.decodeShare) {
  if (!std::isfinite(config_.decodeGapMilliseconds) || config_.decodeGapMilliseconds <= 0.0)
    throw std::invalid_argument("the decode gap target must be positive and finite");
  if (!config_.limits.commandRows || !config_.limits.maximumPasses)
    throw std::invalid_argument("a prefill command must hold a row in at least one pass");
  counters_.decodeGapTargetMilliseconds = config_.decodeGapMilliseconds;
}

void Scheduler::seedCost(std::span<const model::CommandSample> warmups, model::SsdReads load) {
  cost_.seed(warmups, load);
}

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
    promptCompleted(request);
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
  if (active_ && holds(active_->items, id))
    throw std::logic_error("cannot remove an active batch member");
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

std::vector<uint64_t> Scheduler::prefillAdmissionOrder(std::span<const PrefillAdmission> candidates,
                                                       const PlanInputs &in) const {
  std::vector<Lane> pending;
  pending.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    const Request &value = get(candidate.requestId);
    if (value.suspended ||
        (value.phase != Phase::Queued && value.phase != Phase::WaitingResources &&
         value.phase != Phase::WaitingPrefix) ||
        candidate.cachedTokens >= value.spec.prefillTokens)
      throw std::logic_error("invalid pending prefill admission");
    pending.push_back(lane(value, candidate.cachedTokens, in));
  }
  std::vector<Lane> lanes;
  lanes.reserve(requests_.size());
  std::vector<uint64_t> result;
  for (const auto &[_, request] : requests_) {
    if (request.phase == Phase::Prefill)
      lanes.push_back(lane(request, request.promptProcessed, in));
  }
  // Admission starts only lanes the next command serves: it plans again
  // without the candidates a plan leaves out until it keeps all it plans, as
  // without them the plan can take another leader.
  std::optional<Planned> planned;
  size_t count = 0;
  do {
    count = pending.size();
    std::vector<Lane> ready = lanes;
    ready.insert(ready.end(), pending.begin(), pending.end());
    planned = planPrefill(std::move(ready));
    if (!planned)
      return result;
    std::erase_if(pending, [&](const Lane &view) { return !holds(planned->plan.items, view.request->spec.id); });
  } while (pending.size() != count);
  // Admission ranks candidates against every decoder, left out of a plan or
  // not.
  const auto decode = nextDecode({});
  if (decode && get(decode->items.front().requestId).spec.priority <
                    get(planned->plan.items.front().requestId).spec.priority)
    return result;
  for (const auto &item : planned->plan.items)
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

bool Scheduler::yieldsTo(RequestPriority priority, uint32_t lanes, const PlanInputs &in) const {
  if (!in.passes || !lanes)
    return false;
  const BatchPlan &command = in.passes->plan;
  if (priority >= get(command.items.front().requestId).spec.priority)
    return false;
  for (const BatchItem &item : command.items) {
    const Request &request = get(item.requestId);
    if (lane(request, request.promptProcessed, in).overdue)
      return false;
  }
  // The start runs first either way, so the plan's price drops out.
  return supersedeGain({}, lanes, priority, 0.0, 0.0, in) > 0.0;
}

bool Scheduler::abandonsPasses(const BatchPlan &plan, const PlanInputs &in) const {
  return in.passes && plan.kind == WorkKind::Prefill && !samePlan(plan, in.passes->plan);
}

std::optional<BatchPlan> Scheduler::next(const PlanInputs &in) const {
  if (active_)
    return std::nullopt;
  auto decode = nextDecode(in.excluded);
  auto prefill = nextPrefill(in);
  if (!decode)
    return prefill;
  if (!prefill)
    return decode;

  const RequestPriority decodePriority =
      get(decode->items.front().requestId).spec.priority;
  RequestPriority prefillPriority =
      get(prefill->items.front().requestId).spec.priority;
  // A command between its passes runs at the priority of the most urgent
  // lane it holds back, a prompt it does not yield to or a start it left
  // waiting for its end (Engine::preemptBelow), or that one would wait for
  // every decode above the command's own priority and not only its passes.
  if (in.passes)
    for (const auto &[id, request] : requests_)
      if ((request.phase == Phase::Prefill && !listed(in.excluded, id)) ||
          (request.phase == Phase::WaitingResources && listed(in.passes->heldStarts, id)))
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

std::optional<BatchPlan> Scheduler::nextPrefill(const PlanInputs &in) const {
  std::optional<Planned> planned = planPrefill(readyLanes(in));
  // Its commit counts what planning it counted.
  lastPlanned_ = planned;
  // A command between its passes goes on before any other prefill, whose
  // rows would take the model's prefill memory its rows' work is in, but one
  // that supersedes it.
  if (!in.passes || (planned && supersedes(planned->plan, in)))
    return planned ? std::optional(std::move(planned->plan)) : std::nullopt;
  for (const BatchItem &item : in.passes->plan.items)
    if (listed(in.excluded, item.requestId))
      return std::nullopt;
  return in.passes->plan;
}

bool Scheduler::supersedes(const BatchPlan &plan, const PlanInputs &in) const {
  const PassProgress &command = *in.passes;
  for (const BatchItem &item : command.plan.items) {
    const Request &request = get(item.requestId);
    if (lane(request, request.promptProcessed, in).overdue)
      return false;
  }
  if (std::ranges::any_of(plan.items, [&](const BatchItem &item) { return listed(command.leftOut, item.requestId); }))
    return false;
  std::vector<BatchItem> joiners;
  for (const BatchItem &item : plan.items)
    if (!holds(command.plan.items, item.requestId))
      joiners.push_back(item);
  return supersedeGain(plan.items, 0, get(plan.items.front().requestId).spec.priority,
                       passesMilliseconds(prefillLanes(plan.items, in), 0, plan.passes),
                       passesMilliseconds(prefillLanes(joiners, in), 0, plan.passes), in) > 0.0;
}

double Scheduler::supersedeGain(std::span<const BatchItem> lanes, uint32_t starts, RequestPriority priority,
                                double planMilliseconds, double joinersMilliseconds,
                                const PlanInputs &in) const {
  const PassProgress &command = *in.passes;
  const BatchPlan &kept = command.plan;
  const RequestPriority commandPriority = get(kept.items.front().requestId).spec.priority;
  // The command's measured passes and its predicted rest; what the lanes it
  // leaves out lose: their rows' whole command again, which is all of it
  // when the plan keeps none of its lanes.
  const double done = command.elapsedMilliseconds;
  const double rest = passesMilliseconds(prefillLanes(kept.items, in), kept.pass, kept.passes);
  std::vector<BatchItem> dropped;
  for (const BatchItem &item : kept.items)
    if (!holds(lanes, item.requestId))
      dropped.push_back(item);
  const bool keepsSome = dropped.size() != kept.items.size();
  const double redone =
      dropped.empty() ? 0.0
      : keepsSome     ? passesMilliseconds(prefillLanes(dropped, in), 0, kept.passes)
                      : done + rest;
  const auto lower = [&](RequestPriority of) { return of > priority; };
  double gain = 0.0;
  for (const BatchItem &item : dropped) {
    const double keep = rest;
    double abandon = planMilliseconds + redone;
    if (lower(get(item.requestId).spec.priority))
      abandon -= planMilliseconds;
    gain += keep - abandon;
  }
  for (const BatchItem &item : lanes) {
    const bool inCommand = holds(kept.items, item.requestId);
    gain += (inCommand ? rest : rest + joinersMilliseconds) - planMilliseconds;
  }
  gain += starts * (rest + joinersMilliseconds - planMilliseconds);
  // The other ready lanes of either priority wait behind both.
  for (const auto &[id, request] : requests_) {
    if ((request.spec.priority != commandPriority && request.spec.priority != priority) ||
        holds(kept.items, id) || holds(lanes, id))
      continue;
    if (!((request.phase == Phase::Prefill && !listed(in.excluded, id)) ||
          (request.phase == Phase::WaitingResources && listed(command.heldStarts, id))))
      continue;
    double keep = rest + joinersMilliseconds;
    double abandon = planMilliseconds + (!dropped.empty() && request.spec.priority >= commandPriority ? redone : 0.0);
    if (lower(request.spec.priority)) {
      keep -= joinersMilliseconds;
      abandon -= planMilliseconds;
    }
    gain += keep - abandon;
  }
  return gain;
}

LaneShape Scheduler::shapeOf(uint64_t id, const PlanInputs &in) const {
  const auto found =
      std::ranges::find_if(in.shapes, [id](const LaneShape &shape) { return shape.requestId == id; });
  return found == in.shapes.end() ? LaneShape{.requestId = id} : *found;
}

Scheduler::Lane Scheduler::lane(const Request &request, uint32_t processed, const PlanInputs &in) const {
  Lane result{.request = &request, .processed = processed, .shape = shapeOf(request.spec.id, in)};
  result.remainingMilliseconds = remainingMilliseconds(result);
  // A lane's wait is bounded by its own priced cost; before any price there
  // is none to bound it by.
  result.overdue = request.overdue || (result.remainingMilliseconds > 0.0 &&
                                       request.waitedMilliseconds >= kAgingSlowdown * result.remainingMilliseconds);
  return result;
}

std::vector<model::PrefillLane> Scheduler::prefillLanes(std::span<const BatchItem> items,
                                                        const PlanInputs &in) const {
  std::vector<model::PrefillLane> result;
  result.reserve(items.size());
  for (const BatchItem &item : items)
    result.push_back({item.promptOffset, item.tokenCount, shapeOf(item.requestId, in).indexPages});
  return result;
}

double Scheduler::commandMilliseconds(std::span<const model::PrefillLane> lanes) const {
  return cost_.price(work_.prefillWork(lanes, 0, 1));
}

double Scheduler::passesMilliseconds(std::span<const model::PrefillLane> lanes, uint32_t first,
                                     uint32_t passes) const {
  double total = 0.0;
  for (uint32_t pass = first; pass < passes; ++pass)
    total += cost_.price(work_.prefillWork(lanes, pass, passes));
  return total;
}

bool Scheduler::groups(const Lane &lane) const {
  const uint32_t chunks = config_.limits.commandRows / kChunk;
  if (chunks <= 1)
    return false;
  const model::PrefillLane probe{lane.processed, kChunk, lane.shape.indexPages};
  return (chunks - 1) * cost_.fixed(work_.prefillWork({&probe, 1}, 0, 1)) >= config_.decodeGapMilliseconds;
}

uint32_t Scheduler::ownRows(const Lane &lane) const {
  return std::min({uncontendedRows(lane.remaining(), groups(lane) ? config_.limits.commandRows : kChunk),
                   config_.limits.commandRows, lane.shape.rowCap});
}

double Scheduler::remainingMilliseconds(const Lane &lane) const {
  // Its remaining commands alone, each of its uncontended rows from where the
  // one before ends, at its history's Index pages now.
  const uint32_t groupRows = groups(lane) ? config_.limits.commandRows : kChunk;
  double total = 0.0;
  for (uint32_t position = lane.processed, remaining = lane.remaining(); remaining;) {
    const uint32_t rows = std::min(uncontendedRows(remaining, groupRows), config_.limits.commandRows);
    const model::PrefillLane command{position, rows, lane.shape.indexPages};
    total += commandMilliseconds({&command, 1});
    position += rows;
    remaining -= rows;
  }
  return total;
}

bool Scheduler::contended(RequestPriority priority) const {
  // Decoders this command is planned without count: they resume once their
  // memory lands.
  return std::ranges::any_of(requests_, [&](const auto &entry) {
    const Request &peer = entry.second;
    return (peer.phase == Phase::Decode || peer.phase == Phase::WaitingMask) && peer.spec.priority <= priority;
  });
}

double Scheduler::bound() const {
  return std::max(config_.decodeGapMilliseconds, cost_.turnaround() / kPassOverheadShare);
}

Scheduler::UnitFit Scheduler::passesFor(std::span<const model::PrefillLane> lanes, bool bounded) const {
  if (!bounded)
    return {};
  const uint32_t most = config_.limits.maximumPasses;
  const double limit = bound();
  // The fewest passes whose every pass is within the bound; or failing that
  // within twice its fixed part, so that no pass spends more than half its
  // time on what its rows do not need. A pass's price includes its
  // turnaround, so the passes' mean, and the heaviest, is at least the
  // command's over their count: fewer cannot fit.
  const auto fits = [&](uint32_t passes, bool relaxed) {
    for (uint32_t pass = 0; pass < passes; ++pass) {
      const model::CommandWork unit = work_.prefillWork(lanes, pass, passes);
      const double allowed = relaxed ? std::max(limit, 2.0 * cost_.fixed(unit)) : limit;
      if (cost_.price(unit) > allowed)
        return false;
    }
    return true;
  };
  const double whole = commandMilliseconds(lanes);
  const auto first = static_cast<uint32_t>(std::clamp(std::ceil(whole / limit), 1.0, double(most)));
  for (uint32_t passes = first; passes <= most; ++passes)
    if (fits(passes, false))
      return {passes, true};
  for (uint32_t passes = 1; passes <= most; ++passes)
    if (fits(passes, true))
      return {passes, true};
  return {most, false};
}

bool Scheduler::withinPairs(std::span<const model::PrefillLane> lanes) const {
  return work_.prefillWork(lanes, 0, 1).layerAttentionPairs <= config_.limits.attentionPairs;
}

std::vector<Scheduler::Lane> Scheduler::readyLanes(const PlanInputs &in) const {
  std::vector<Lane> ready;
  for (const auto &[id, request] : requests_)
    if (request.phase == Phase::Prefill && !listed(in.excluded, id))
      ready.push_back(lane(request, request.promptProcessed, in));
  return ready;
}

std::optional<Scheduler::Planned> Scheduler::planPrefill(std::vector<Lane> ready) const {
  if (ready.empty())
    return std::nullopt;
  // Priority first; then overdue lanes, oldest first; then the least expected
  // remaining cost, of equal cost (as before any price) the fewest remaining
  // rows, then submission order.
  std::sort(ready.begin(), ready.end(), [](const Lane &a, const Lane &b) {
    if (a.request->spec.priority != b.request->spec.priority)
      return a.request->spec.priority < b.request->spec.priority;
    if (a.overdue != b.overdue)
      return a.overdue;
    if (!a.overdue && a.remainingMilliseconds != b.remainingMilliseconds)
      return a.remainingMilliseconds < b.remainingMilliseconds;
    if (!a.overdue && a.remaining() != b.remaining())
      return a.remaining() < b.remaining();
    return a.request->order < b.request->order;
  });
  const Lane &leader = ready.front();
  const RequestPriority tier = leader.request->spec.priority;
  const bool contend = contended(tier);
  // A unit is bounded where someone waits for it: a lane of its priority or
  // a higher one decoding, or arrivals that may join a group at a pass.
  const bool bounded = contend || groups(leader);
  Planned result;
  std::vector<model::PrefillLane> lanes{{leader.processed, ownRows(leader), leader.shape.indexPages}};
  // A layer's attention pairs bound the rows, which passes never lower.
  while (lanes.front().rows > kMinimumPrefillRows && !withinPairs(lanes))
    lanes.front().rows /= 2;
  UnitFit fit = passesFor(lanes, bounded);
  // A contended unit no passes bound halves its rows where fewer rows can
  // bound it: a cost per chunk or per pass no rows shorten leaves its rows.
  // Halving is how fewer rows meet the bound; where even the fewest exceed it
  // in the most passes (each pass carries its layers' share of the chunk's
  // compute, which ChunkLayers prices and fixed() leaves out), halving meets
  // nothing and multiplies the commands: a 2.2 s dense chunk in four passes
  // would take 32 commands of about its time for its 2048 rows
  // (testAContendedUnitKeepsRowsNoFewerBound).
  const model::PrefillLane fewest{leader.processed, std::min(lanes.front().rows, kMinimumPrefillRows),
                                  leader.shape.indexPages};
  const bool halves = !fit.within && contend && passesFor({&fewest, 1}, bounded).within;
  while (halves && !fit.within && lanes.front().rows > kMinimumPrefillRows) {
    lanes.front().rows = std::max(lanes.front().rows / 2, kMinimumPrefillRows);
    ++result.boundCuts;
    fit = passesFor(lanes, bounded);
  }
  const auto indexPageLanes = [&](const std::vector<model::PrefillLane> &with, std::span<const Lane> of,
                                  uint32_t passes) {
    uint32_t count = 0;
    for (size_t index = 0; index < with.size(); ++index)
      count += with[index].indexPages || (passes > 1 && of[index].shape.gainsIndexPages);
    return count;
  };
  std::vector<Lane> members{leader};
  for (auto view = ready.begin() + 1; view != ready.end(); ++view) {
    if (view->request->spec.priority != tier || lanes.size() == model::ExecutionLimits::maximumBatchWidth)
      break;
    // The fewest rows that spare it one of its own commands.
    const uint32_t own = ownRows(*view);
    const uint32_t rows = std::min((view->remaining() - 1) % own + 1, view->shape.rowCap);
    std::vector<model::PrefillLane> with = lanes;
    with.push_back({view->processed, rows, view->shape.indexPages});
    if (rowsOf(with) > config_.limits.commandRows || lanesPastChunk(with) > config_.limits.lanesPastChunk ||
        !withinPairs(with))
      continue;
    // It joins unless what its rows add for the lanes planned, each, exceeds
    // what joining saves it: its own command, less that wait.
    const double added = commandMilliseconds(with) - commandMilliseconds(lanes);
    if (lanes.size() * added > commandMilliseconds({&with.back(), 1}) - added)
      continue;
    // A bound the plan meets, it meets with the lane.
    const UnitFit joined = passesFor(with, bounded);
    if (fit.within && !joined.within)
      continue;
    members.push_back(*view);
    if (indexPageLanes(with, members, joined.passes) > config_.limits.indexPageLanes) {
      members.pop_back();
      ++result.indexPageConflicts;
      continue;
    }
    lanes = std::move(with);
    fit = joined;
  }
  result.plan.kind = WorkKind::Prefill;
  for (size_t index = 0; index < lanes.size(); ++index)
    result.plan.items.push_back({members[index].request->spec.id, lanes[index].rows, lanes[index].position});
  result.plan.passes = fit.passes;
  return result;
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

std::vector<uint64_t> Scheduler::commit(const BatchPlan &plan, const PlanInputs &in) {
  if (active_ || plan.empty() ||
      plan.width() > model::ExecutionLimits::maximumBatchWidth) {
    throw std::logic_error("invalid scheduler commit");
  }
  const bool prefill = plan.kind == WorkKind::Prefill;
  for (const BatchItem &item : plan.items) {
    const Request &request = get(item.requestId);
    const Phase expected = prefill ? Phase::Prefill : Phase::Decode;
    if (request.phase != expected ||
        (prefill && (!item.tokenCount || item.promptOffset != request.promptProcessed)) ||
        (!prefill && (item.tokenCount || item.promptOffset || request.decodeStage != plan.decodeStage))) {
      throw std::logic_error("batch no longer matches scheduler state");
    }
  }
  // A later pass is the command's next; no other prefill commits while a
  // command is between its passes.
  if (in.passes ? prefill && !samePlan(plan, in.passes->plan)
                : plan.pass || (plan.passes > 1 && !prefill))
    throw std::logic_error("batch no longer matches scheduler state");
  std::vector<uint64_t> leftOut;
  const RequestPriority priority = get(plan.items.front().requestId).spec.priority;
  if (prefill && !plan.pass) {
    for (const auto &[id, request] : requests_)
      if (plan.passes > 1 && request.phase == Phase::Prefill && request.spec.priority == priority &&
          !listed(in.excluded, id) && !holds(plan.items, id))
        leftOut.push_back(id);
    counters_.passCommands += plan.passes > 1;
    ++counters_.prefillBatches;
    for (const BatchItem &item : plan.items)
      counters_.prefillRows += item.tokenCount;
    // What planning counted for this plan, as next() made it, and the lanes
    // now overdue, which stay so to their prompt's end.
    if (lastPlanned_ && samePlan(lastPlanned_->plan, plan)) {
      counters_.boundCuts += lastPlanned_->boundCuts;
      counters_.indexPageConflicts += lastPlanned_->indexPageConflicts;
    }
    for (const Lane &view : readyLanes(in))
      if (view.overdue)
        get(view.request->spec.id).overdue = true;
  } else if (!prefill) {
    const uint64_t dispatchOrder = ++decodeDispatchOrder_;
    for (const BatchItem &item : plan.items)
      get(item.requestId).lastDecodeDispatch = dispatchOrder;
    ++counters_.decodeBatches;
    ++counters_.decodeBatchesByWidth[plan.width() - 1];
  }
  lastPlanned_.reset();
  activeExpected_ = prefill ? work_.prefillWork(prefillLanes(plan.items, in), plan.pass, plan.passes)
                            : model::CommandWork{};
  excluded_.assign(in.excluded.begin(), in.excluded.end());
  active_ = plan;
  lastCommittedKind_ = plan.kind;
  return leftOut;
}

void Scheduler::complete(const BatchPlan &plan, std::span<const StepResult> results,
                         const model::CommandSample &sample) {
  if (!active_ || active_->kind != plan.kind ||
      active_->items.size() != plan.items.size() ||
      results.size() != plan.items.size()) {
    throw std::logic_error("completion does not match active batch");
  }
  const double wallMilliseconds = sample.wallMilliseconds;
  const bool timed = std::isfinite(wallMilliseconds) && wallMilliseconds > 0.0;
  const bool prefill = plan.kind == WorkKind::Prefill;
  for (size_t index = 0; index < results.size(); ++index) {
    const BatchItem &item = plan.items[index];
    const StepResult &result = results[index];
    if (result.requestId != item.requestId) {
      throw std::logic_error("completion request order changed");
    }
    Request &request = get(item.requestId);
    if (prefill && plan.pass + 1 < plan.passes) {
      // A pass before the last consumes nothing and ends no lane: one that
      // ends meanwhile stays in the command to its last pass.
      if (result.consumedPromptTokens || result.finished)
        throw std::logic_error("prefill completion row count changed");
    } else if (prefill) {
      if (result.consumedPromptTokens != item.tokenCount ||
          item.promptOffset != request.promptProcessed) {
        throw std::logic_error("prefill completion row count changed");
      }
      request.promptProcessed += result.consumedPromptTokens;
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
      }
      if (request.phase != Phase::Prefill)
        promptCompleted(request);
    } else {
      request.decodeStage = result.nextDecodeStage;
      request.phase = result.finished ? Phase::Completed : Phase::Decode;
    }
  }
  const RequestPriority priority = get(plan.items.front().requestId).spec.priority;
  // Whether lanes of its priority or a higher one decoded beside it, not
  // counting decoders it was planned without, which could not run either.
  const bool contendedCommand = prefill && std::ranges::any_of(requests_, [&](const auto &entry) {
                                  const auto &[id, peer] = entry;
                                  return peer.phase == Phase::Decode && peer.spec.priority <= priority &&
                                         !listed(excluded_, id);
                                });
  if (prefill) {
    ++counters_.prefillUnits;
    const double rowsMilliseconds =
        wallMilliseconds - std::clamp(sample.imageEncodeMilliseconds, 0.0, std::max(wallMilliseconds, 0.0));
    if (timed) {
      if (contendedCommand)
        contendedPrefill_.add(static_cast<uint64_t>(wallMilliseconds * 1e6));
      // The lanes of its priority submitted before a lane it served, which
      // waited without rows meanwhile, count its time as waited.
      uint64_t youngestServed = 0;
      for (const BatchItem &item : plan.items)
        youngestServed = std::max(youngestServed, get(item.requestId).order);
      for (auto &[id, request] : requests_)
        if (request.spec.priority == priority && request.order < youngestServed && !request.suspended &&
            (request.phase == Phase::Queued || request.phase == Phase::WaitingResources ||
             request.phase == Phase::WaitingPrefix || request.phase == Phase::Prefill) &&
            !listed(excluded_, id) && !holds(plan.items, id))
          request.waitedMilliseconds += rowsMilliseconds;
    }
    // A unit that reports no work, or whose completion added work, says
    // nothing of the rates.
    if (sample.representative && sample.work.rows)
      cost_.observe(sample.work, activeExpected_, rowsMilliseconds, 1.0 / plan.passes);
  }
  // While requests of the same or a higher priority decode, lanes this
  // prefill finished included, it owes decode a share of its time; decode
  // commands work the debt off with their own. A lane waiting for its mask
  // is owed nothing: it could not have decoded meanwhile, and the bound
  // already bounds the prefill it waits behind. Nor is a decoder this
  // command was planned without: it could not run either.
  if (timed) {
    if (!prefill) {
      decodeDebtMilliseconds_ = std::max(0.0, decodeDebtMilliseconds_ - wallMilliseconds);
      // Its time over the tokens a lane committed on average trains the
      // decode time per token; a command whose lanes all ended in flight
      // committed none and trains nothing.
      uint32_t committed = 0;
      for (const StepResult &result : results)
        committed += result.committedTokens;
      if (committed) {
        const double laneTokens = double(committed) / plan.width();
        if (decodeTokenWeight_ > 0.0)
          decodeErrors_[decodeErrorCount_++ % CostModel::kErrorSamples] =
              std::fabs(decodeTokenMilliseconds() * laneTokens - wallMilliseconds) / wallMilliseconds;
        ++decodeSamples_;
        decodeTokenTime_ = kDecodeTokenDecay * decodeTokenTime_ + wallMilliseconds / laneTokens;
        decodeLaneTokens_ = kDecodeTokenDecay * decodeLaneTokens_ + laneTokens;
        decodeTokenWeight_ = kDecodeTokenDecay * decodeTokenWeight_ + 1.0;
      }
    } else if (contendedCommand && decodeShare_ > 0.0) {
      decodeDebtMilliseconds_ += decodeShare_ * wallMilliseconds;
    }
  }
  dropStaleDecodeDebt();
  excluded_.clear();
  active_.reset();
}

void Scheduler::abandoned(const PassProgress &command) {
  // A command of other lanes may be in flight, never a pass of this one.
  if (active_ && active_->passes > 1)
    throw std::logic_error("a command in flight cannot be abandoned");
  for (const BatchItem &item : command.plan.items)
    get(item.requestId).waitedMilliseconds += command.elapsedMilliseconds;
  ++counters_.abandonedPassCommands;
}

void Scheduler::promptCompleted(Request &request) noexcept {
  request.waitedMilliseconds = 0.0;
  request.overdue = false;
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

double Scheduler::prefillMilliseconds(const model::CommandWork &work) const noexcept {
  return cost_.price(work);
}

double Scheduler::decodeTokenMilliseconds() const noexcept {
  return decodeTokenWeight_ > 0.0 ? decodeTokenTime_ / decodeTokenWeight_ : 0.0;
}

double Scheduler::decodeTokensPerLaneCommand() const noexcept {
  return decodeTokenWeight_ > 0.0 ? decodeLaneTokens_ / decodeTokenWeight_ : 0.0;
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
  result.contendedPrefill = contendedPrefill_.snapshot();
  const CostModelStats cost = cost_.stats();
  result.prefillEstimate = {.samples = cost.samples,
                            .clamped = cost.clamped,
                            .relativeErrorP50 = cost.relativeErrorP50,
                            .relativeErrorP95 = cost.relativeErrorP95};
  for (size_t term = 0; term < kCostTerms; ++term)
    result.prefillEstimate.terms.push_back({std::string(costTermName(static_cast<CostTerm>(term))),
                                            cost.rates[term], cost.seedRates[term], cost.realizedPerExpected[term]});
  result.decodeEstimate.samples = decodeSamples_;
  result.decodeEstimate.terms.push_back({"token", decodeTokenMilliseconds(), 0.0, 1.0});
  if (const size_t count = std::min<size_t>(decodeErrorCount_, CostModel::kErrorSamples)) {
    std::array<double, CostModel::kErrorSamples> sorted = decodeErrors_;
    std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(count));
    result.decodeEstimate.relativeErrorP50 = sorted[static_cast<size_t>(std::ceil(0.5 * double(count))) - 1];
    result.decodeEstimate.relativeErrorP95 = sorted[static_cast<size_t>(std::ceil(0.95 * double(count))) - 1];
  }
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

bool Scheduler::samePlan(const BatchPlan &a, const BatchPlan &b) noexcept {
  return a.kind == b.kind && a.pass == b.pass && a.passes == b.passes && a.items == b.items;
}

} // namespace splash::engine
