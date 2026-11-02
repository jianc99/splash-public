// The rules every command keeps (model::commandViolation), the attention
// pairs a command's rows compute (model::attentionPairs), the work a model
// reports by default (Model::prefillWork), the experts a load routes
// (model::expectedDistinctExperts) and the sample a ticket gives by default
// (ModelBatchTicket::sample).

#include "TestChecks.hpp"
#include "model/Model.hpp"
#include "model/WorkFormulas.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace splash;

namespace {

using splash::test::require;

constexpr uint32_t kChunk = model::ExecutionLimits::prefillTokenBudget;
// A streamed target whose K/V spills: four chunks' rows in up to 48
// passes, the rows of one lane past a chunk, the history of one lane
// streamed.
constexpr model::CommandLimits kLimits{
    .commandRows = 4 * kChunk, .maximumPasses = 48, .lanesPastChunk = 1, .indexPageLanes = 1};

struct Command final {
  BatchPlan plan;
  std::vector<ModelBatchItem> items;
};

// A command of a lane of each of `rows` rows (none for a decode), the
// history of the first `indexed` of them holding an Index page.
Command command(WorkKind kind, const std::vector<uint32_t> &rows, uint32_t indexed = 0) {
  auto chunk = std::make_shared<model::KvPageSources::Chunk>();
  chunk->pages[0].block = 1;
  chunk->indexPages = 1;
  const auto sources = std::make_shared<const model::KvPageSources>(
      std::vector<std::shared_ptr<const model::KvPageSources::Chunk>>{chunk}, 1);
  Command result;
  result.plan.kind = kind;
  for (uint32_t lane = 0; lane < rows.size(); ++lane) {
    result.plan.items.push_back({lane + 1, rows[lane], 0});
    ModelBatchItem &item = result.items.emplace_back();
    item.requestId = lane + 1;
    item.tokenCount = rows[lane];
    if (lane < indexed)
      item.pageSources = sources;
  }
  return result;
}

Command prefill(const std::vector<uint32_t> &rows, uint32_t indexed = 0) {
  return command(WorkKind::Prefill, rows, indexed);
}

Command inPasses(uint32_t pass, uint32_t passes) {
  Command result = prefill({kChunk});
  result.plan.pass = pass;
  result.plan.passes = passes;
  return result;
}

// The first rule the command breaks is the one whose message names `rule`.
void requireBreaks(const Command &command, std::string_view rule, std::string_view message,
                   const model::CommandLimits &limits = kLimits) {
  const std::optional<std::string_view> broken = model::commandViolation(limits, command.plan, command.items);
  require(broken && broken->find(rule) != std::string_view::npos, message);
}

void requireKeeps(const Command &command, std::string_view message,
                  const model::CommandLimits &limits = kLimits) {
  require(!model::commandViolation(limits, command.plan, command.items), message);
}

void testCommandsWithinTheLimitsBreakNoRule() {
  requireKeeps(command(WorkKind::Decode, {0, 0, 0, 0}), "a decode of four lanes broke a rule");
  requireKeeps(prefill({500, 500, 500, 548}), "four lanes within a chunk broke a rule");
  requireKeeps(prefill({1000, 4 * kChunk - 1000}, 1),
               "a command's rows with one lane past a chunk and one streaming broke a rule");
  requireKeeps(inPasses(47, 48), "the last of the most passes broke a rule");
}

void testAWideCommandBreaksTheWidth() {
  requireBreaks(command(WorkKind::Decode, {0, 0, 0, 0, 0}), "batch width", "a decode of five lanes broke no rule");
  requireBreaks(prefill({1, 1, 1, 1, 1}), "batch width", "a prefill of five lanes broke no rule");
}

void testPassesTheModelDoesNotRunBreakThePassRule() {
  requireBreaks(inPasses(0, 0), "passes", "a prefill of no passes broke no rule");
  requireBreaks(inPasses(1, 1), "passes", "a pass past a prefill's passes broke no rule");
  requireBreaks(inPasses(0, 49), "passes", "more passes than the model runs broke no rule");
  requireBreaks(inPasses(0, 2), "passes", "passes where the model runs a command whole broke no rule",
                {.commandRows = kChunk, .maximumPasses = 1});
}

void testRowsPastTheCommandsBreakTheRowRule() {
  requireBreaks(prefill({kChunk, kChunk, 2 * kChunk + 1}), "rows exceed", "a row past a command's broke no rule");
  requireKeeps(prefill({kChunk, kChunk, 2 * kChunk}), "a command's rows broke a rule");
  requireBreaks(prefill({kChunk + 1}), "rows exceed", "a row past a chunk broke no rule where a command holds one",
                {.commandRows = kChunk, .maximumPasses = 1});
}

void testTwoLanesPastAChunkBreakTheChunkRule() {
  requireBreaks(prefill({3000, 2000}), "run past", "two lanes past a chunk broke no rule");
  requireBreaks(prefill({kChunk - 1, 2, kChunk}), "run past",
                "a lane of two rows across a chunk's end and one past the next broke no rule");
}

// A model without GDN state lets every lane's rows run past a chunk.
void testLanesPastAChunkAreTheModels() {
  const model::CommandLimits stateless{.commandRows = 4 * kChunk,
                                       .maximumPasses = 48,
                                       .lanesPastChunk = model::ExecutionLimits::maximumBatchWidth,
                                       .indexPageLanes = 1};
  requireKeeps(prefill({3000, 2000}), "two lanes past a chunk broke a rule where the model runs them", stateless);
  requireKeeps(prefill({kChunk + 1, kChunk, kChunk}),
               "three lanes past a chunk broke a rule where the model runs them", stateless);
}

void testALaneEndingAtAChunkDoesNotRunPastIt() {
  requireKeeps(prefill({kChunk, 3000}), "a lane ending at a chunk's end ran past it");
  requireKeeps(prefill({1000, kChunk - 1000, kChunk + 1}), "lanes ending at a chunk's end ran past it");
  requireKeeps(prefill({kChunk, kChunk, kChunk, kChunk}), "lanes of a chunk each ran past one");
}

void testStreamingLanesPastTheModelsBreakTheIndexRule() {
  requireBreaks(prefill({100, 100}, 2), "streams", "a prefill of two lanes over Index pages broke no rule");
  requireBreaks(prefill({100}, 1), "streams", "a streaming lane broke no rule where the model streams none",
                {.commandRows = 4 * kChunk, .maximumPasses = 48, .indexPageLanes = 0});
  // Each decode lane reads its spilled K/V through the hot block cache.
  requireKeeps(command(WorkKind::Decode, {0, 0, 0, 0}, 4), "a decode of four lanes over Index pages broke a rule");
}

void testTheFirstRuleBrokenIsNamed() {
  requireBreaks(prefill({kChunk, kChunk, kChunk, kChunk, kChunk}), "batch width",
                "a wide command of too many rows did not break the width first");
  Command rows = prefill({3000, 6000});
  rows.plan.passes = 49;
  requireBreaks(rows, "passes", "too many passes of too many rows did not break the pass rule first");
  requireBreaks(prefill({3000, 6000}), "rows exceed",
                "too many rows in two lanes past a chunk did not break the row rule first");
  requireBreaks(prefill({3000, 2000}, 2), "run past",
                "two streaming lanes past a chunk did not break the chunk rule first");
}

// The pairs of `rows` rows from `position`, row by row: the row with k keys,
// its own included, attends to all of them, to the last windowTokens, or,
// block-sparse, to all while its complete blocks hold at most
// denseHistoryTokens, and past that to selectedTokens and its keys after
// its last complete block.
uint64_t pairsByRow(const model::AttentionReach &reach, uint64_t position, uint32_t rows) {
  uint64_t pairs = 0;
  for (uint64_t keys = position + 1; keys <= position + rows; ++keys) {
    if (reach.windowTokens)
      pairs += std::min<uint64_t>(keys, reach.windowTokens);
    else if (reach.denseHistoryTokens && keys / reach.blockTokens * reach.blockTokens > reach.denseHistoryTokens)
      pairs += reach.selectedTokens + keys % reach.blockTokens;
    else
      pairs += keys;
  }
  return pairs;
}

constexpr std::array<uint64_t, 9> kPositions{0, 1, 127, 128, 2047, 2048, 2111, 2112, 64000};
constexpr std::array<uint32_t, 6> kRows{1, 63, 64, 129, 500, kChunk};

void testDenseAttentionPairsAreEveryRowsKeys() {
  for (const uint64_t position : kPositions)
    for (const uint32_t rows : kRows)
      require(model::denseAttentionPairs(position, rows) == rows * position + uint64_t{rows} * (rows + 1) / 2 &&
                  model::denseAttentionPairs(position, rows) == pairsByRow({}, position, rows),
              "dense attention pairs are not every row's keys");
}

void testAWindowedLayerSaturatesAtItsWindow() {
  const model::AttentionReach window{.windowTokens = 128};
  require(model::attentionPairs(window, 0, 128) == 128 * 129 / 2,
          "rows within the window did not attend every key");
  require(model::attentionPairs(window, 1000, 10) == 10 * 128 &&
              model::attentionPairs(window, 100000, 10) == 10 * 128,
          "rows past the window did not attend the window alone");
  for (const uint64_t position : kPositions)
    for (const uint32_t rows : kRows)
      require(model::attentionPairs(window, position, rows) == pairsByRow(window, position, rows),
              "windowed attention pairs are not each row's window");
}

void testABlockSparseLayerSaturatesAtTheSelectionAndTail() {
  const model::AttentionReach sparse{.denseHistoryTokens = 2048, .selectedTokens = 512, .blockTokens = 64};
  require(model::attentionPairs(sparse, 0, kChunk) == model::denseAttentionPairs(0, kChunk),
          "rows within the dense history did not attend every key");
  // A block of rows past it attends the selection and, row by row, 1 to 63
  // keys after the last complete block, and then none, at any depth.
  require(model::attentionPairs(sparse, 64000, 64) == 64 * 512 + 63 * 64 / 2 &&
              model::attentionPairs(sparse, 640000, 64) == 64 * 512 + 63 * 64 / 2,
          "rows past the dense history did not attend the selection and their tail");
  for (const uint64_t position : kPositions)
    for (const uint32_t rows : kRows)
      require(model::attentionPairs(sparse, position, rows) == pairsByRow(sparse, position, rows),
              "block-sparse attention pairs are not each row's selection");
}

// A model of the defaults alone (Model::prefillWork, Model::commandLimits)
// whose commands run in up to its 48 layers' passes.
class DefaultModel final : public model::Model {
public:
  uint32_t prefillPasses() const noexcept override { return 48; }
  StateAdmission begin(const ModelRequest &) override { return {}; }
  void suspend(uint64_t) override {}
  StateAdmission resume(const ModelRequest &) override { return {}; }
  std::unique_ptr<StateRestore> beginRestore(uint64_t, uint32_t, std::shared_ptr<const CompositeState>, bool,
                                             std::function<void()>) override {
    return {};
  }
  void setDraftContextPlan(uint64_t, DraftContextPlan) override {}
  std::unique_ptr<ModelBatchTicket> submit(const BatchPlan &, std::span<const ModelBatchItem>,
                                           std::function<void()>) override {
    return {};
  }
  std::shared_ptr<StateReservation> reserveState(uint64_t, uint32_t, metal::AllocationFailure &) override {
    return {};
  }
  std::shared_ptr<const CompositeState> snapshot(uint64_t) override { return {}; }
  uint64_t snapshotBytes() const noexcept override { return 0; }
  uint64_t reclaimIdleState(bool, model::IdleMemory) noexcept override { return 0; }
  std::optional<std::string> provideMask(uint64_t, std::span<const uint32_t>) override { return {}; }
  void end(uint64_t) override {}
};

// A target of 48 layers, every fourth attention, and a native window of
// 262144 tokens (Qwen3.5-122B-A10B): a pass of 19 of a chunk at YaRN depth
// past the window that runs an attention layer computes in it the pairs of
// every row of the chunk, more than one layer may compute, as the whole
// command does. Passes do not lower a layer's pairs; half the rows do.
void testPassesDoNotLowerALayersAttentionPairs() {
  DefaultModel target;
  constexpr uint32_t position = 300000;
  const uint64_t layerBound = model::denseAttentionPairs(262144 - kChunk, kChunk);
  const std::array<model::PrefillLane, 1> chunk{{{.position = position, .rows = kChunk}}};
  const uint64_t whole = target.prefillWork(chunk, 0, 1).layerAttentionPairs;
  require(whole == model::denseAttentionPairs(position, kChunk) && whole > layerBound,
          "a chunk at YaRN depth did not compute its dense pairs in a layer");
  for (uint32_t pass = 0; pass < 19; ++pass) {
    requireKeeps(inPasses(pass, 19), "a pass of 19 broke a rule of a 48-layer target", target.commandLimits());
    // Layers 3, 7, ..., 47 attend: end / 4 of them lie below layer end.
    const model::LayerRange layers = model::passLayers(pass, 19, 48);
    if (layers.end / 4 > layers.first / 4)
      require(target.prefillWork(chunk, pass, 19).layerAttentionPairs == whole,
              "a pass of 19 lowered the attention pairs of the attention layer it runs");
  }
  const std::array<model::PrefillLane, 1> half{{{.position = position, .rows = kChunk / 2}}};
  require(target.prefillWork(half, 0, 1).layerAttentionPairs <= layerBound,
          "half a chunk at YaRN depth computed more pairs than one layer may");
}

void testALoadRoutesFromTopKToEveryExpert() {
  require(model::expectedDistinctExperts(0, 8, 256) == 0.0 && model::expectedDistinctExperts(64, 8, 0) == 0.0,
          "a load of no rows or experts routed some");
  require(std::abs(model::expectedDistinctExperts(1, 8, 256) - 8.0) < 1e-9, "one row routed other than its top k");
  require(std::abs(model::expectedDistinctExperts(2, 8, 256) - (16.0 - 8.0 * 8.0 / 256)) < 1e-9,
          "two rows routed other than their top k less the experts they share");
  require(model::expectedDistinctExperts(kChunk, 8, 256) > 255.99 &&
              model::expectedDistinctExperts(kChunk, 8, 256) <= 256.0,
          "a chunk did not route nearly every expert");
}

// A ticket that reports only the times every ticket reports.
class TimedTicket final : public ModelBatchTicket {
public:
  bool ready() const noexcept override { return true; }
  std::vector<ModelStepResult> wait() override { return {}; }
  double wallMilliseconds() const noexcept override { return 12.5; }
  double hostMilliseconds() const noexcept override { return 5.0; }
  double imageEncodeMilliseconds() const noexcept override { return 2.0; }
  bool prefillTimingIsRepresentative() const noexcept override { return false; }
};

void testTheDefaultSampleIsTheTicketsTimes() {
  const model::CommandSample sample = TimedTicket().sample();
  require(sample.work == model::CommandWork{} && sample.wallMilliseconds == 12.5 &&
              sample.spilledKvReadMilliseconds == 0.0 && sample.imageEncodeMilliseconds == 2.0 &&
              !sample.representative,
          "the default sample is not the ticket's times with no work");
}

} // namespace

int main() {
  try {
    testCommandsWithinTheLimitsBreakNoRule();
    testAWideCommandBreaksTheWidth();
    testPassesTheModelDoesNotRunBreakThePassRule();
    testRowsPastTheCommandsBreakTheRowRule();
    testTwoLanesPastAChunkBreakTheChunkRule();
    testLanesPastAChunkAreTheModels();
    testALaneEndingAtAChunkDoesNotRunPastIt();
    testStreamingLanesPastTheModelsBreakTheIndexRule();
    testTheFirstRuleBrokenIsNamed();
    testDenseAttentionPairsAreEveryRowsKeys();
    testAWindowedLayerSaturatesAtItsWindow();
    testABlockSparseLayerSaturatesAtTheSelectionAndTail();
    testPassesDoNotLowerALayersAttentionPairs();
    testALoadRoutesFromTopKToEveryExpert();
    testTheDefaultSampleIsTheTicketsTimes();
    std::cout << "model boundary tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "model boundary tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
