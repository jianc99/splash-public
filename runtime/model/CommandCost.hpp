#pragma once

// What a prefill command may hold (CommandLimits), what a command or pass does in the model's units
// (CommandWork: expected from PrefillWorkSource::prefillWork, realized in CommandSample) and what loading read
// from the SSD (SsdReads). Data only: the formulas behind the counts are the model's own
// (model/WorkFormulas.hpp), and no value is a time the model guesses.

#include <cstdint>
#include <limits>
#include <span>

namespace splash::model {

struct LayerRange final {
  uint32_t first = 0, end = 0;
  bool operator==(const LayerRange &) const = default;
};
// The layers pass `pass` of `passes` runs: an even share of them, as Runtime::Impl::passLayers splits them.
[[nodiscard]] constexpr LayerRange passLayers(uint32_t pass, uint32_t passes, uint32_t layers) noexcept {
  return {pass * layers / passes, (pass + 1) * layers / passes};
}

// The capabilities of one prefill command that differ between models, fixed when the model is built.
// The constant ones stay in ExecutionLimits (maximumBatchWidth, prefillTokenBudget).
struct CommandLimits final {
  uint32_t commandRows = 0;      // rows of one command, all lanes: prefillTokenBudget * RuntimeGeometry::prefillChunks
  uint32_t maximumPasses = 1;    // the target's layers; 1 with the Neural Engine FFN split
                                 // (QwenTarget::requirePrefill)
  uint32_t lanesPastChunk = 1;   // lanes whose rows may run past a chunk as the lanes fill the chunks in order: 1
                                 // with GDN state, whose prefill group has one scratch cell for the state between a
                                 // lane's parts (ExecutionLimits::prefillTokenBudget), else maximumBatchWidth
  uint32_t indexPageLanes = 0;   // lanes of one prefill command whose history may hold Index pages while it runs:
                                 // 1 with the readers (KvHistoryStream), 0 for K/V that does not spill
  // CommandWork::layerAttentionPairs one attention layer of a command, or of one pass of it, may compute:
  // denseAttentionPairs(nativeContextTokens - prefillTokenBudget, prefillTokenBudget), the pairs of a full chunk
  // at the end of the native window, the bound Bootstrap sets in EngineConfig::maximumPrefillAttentionPairs. A layer
  // whose reach is bounded saturates below it. Passes do not lower a layer's pairs, as a pass does not split a layer
  // and the GPU's wait limit binds each layer; fewer rows do. A planning bound; the runtime does not refuse it.
  uint64_t attentionPairs = std::numeric_limits<uint64_t>::max();
  bool operator==(const CommandLimits &) const = default;
};

// One lane of a prefill command as the engine would run it.
struct PrefillLane final {
  uint32_t position = 0;    // rows already in its KV (BatchItem::promptOffset)
  uint32_t rows = 0;        // rows this command adds
  uint32_t indexPages = 0;  // pages before the command's end holding only their index when it runs
  bool operator==(const PrefillLane &) const = default;
};

// What a command or one pass of it does. Counts cover the layers the pass runs.
struct CommandWork final {
  uint32_t lanes = 0;
  uint32_t rows = 0;                // prefill: prompt rows; decode: verify rows of all lanes
  uint32_t chunks = 0;              // prefill chunks the rows fill in order; 0 for decode
  uint32_t layers = 0;              // target layers run
  uint64_t attentionPairs = 0;      // query-key pairs over the attention layers run, each within its reach
  uint64_t layerAttentionPairs = 0; // the most of them one attention layer run computes (CommandLimits::attentionPairs)
  uint64_t indexPairs = 0;          // rows * complete blocks a block index scores, over the layers run; 0 without one
  uint64_t weightReadBytes = 0;     // weights read from the SSD on demand (streamed experts, PLE rows)
  uint64_t spilledKvReadBytes = 0;  // spilled K/V read from the SSD (a streaming lane's history; hot-block misses)
  bool operator==(const CommandWork &) const = default;
};

// One completed command or pass (ModelBatchTicket::sample, valid once wait() returned).
struct CommandSample final {
  CommandWork work;                       // realized: bytes as the store's and the readers' counters moved
  double wallMilliseconds = 0.0;
  double spilledKvReadMilliseconds = 0.0; // decode: host-step work of the hot block cache's blocking reads during
                                          // the command (KvBlockCacheStats::work delta); prefill 0
  double imageEncodeMilliseconds = 0.0;   // the encode command ahead of a prefill (Runtime::prefillAsync)
  bool representative = true;             // false when the completion added work (the Neural Engine FFN's rerun)
};

// The bytes of a stretch of SSD reads and the time they took (Model::loadReads).
struct SsdReads final {
  uint64_t bytes = 0;
  double milliseconds = 0.0;
};

// What planning asks a model. Engine thread, no command in flight; no allocation beyond the result.
class PrefillWorkSource {
public:
  virtual ~PrefillWorkSource() = default;
  // Expected work of pass `pass` of `passes` of a prefill command whose lanes fill its chunks in order, with
  // the residency now.
  [[nodiscard]] virtual CommandWork prefillWork(std::span<const PrefillLane> lanes, uint32_t pass,
                                                uint32_t passes) const = 0;
};

} // namespace splash::model
