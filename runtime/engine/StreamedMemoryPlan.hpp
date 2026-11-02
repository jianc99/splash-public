#pragma once

// The startup planner of the parts of the weights that can stream from the
// SSD and of a spill-capable target's floors (RuntimeResources::create): it
// sizes the stores the model loads with, before the engine's plan
// (EngineMemoryPlan) is made from what loaded.

#include "engine/MemoryPlan.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace splash::engine {

// What the planner sizes the streamed stores and a spill-capable target's
// floors by, before the model loads (RuntimeResources::create), and once it
// loaded with the arenas its geometry takes.
struct StreamedMemoryInputs final {
  uint64_t hardBudgetBytes = 0;
  // Everything the plan holds besides the streamed parts, the arenas, the
  // lanes' state and KV: the weights that load, the reserves, the state
  // staging and the Neural Engine split.
  uint64_t fixedBytes = 0;
  // Of fixedBytes, the reserve the host's budget holds already when it is
  // sampled: the process's runtime overhead
  // (model::kRuntimeOverheadReserveBytes), which the spare memory counts.
  // The pipeline reserve's pipelines and encoder scratch are built at
  // warmup, after the sample, so the host room charges them.
  uint64_t reserveBytes = 0;
  // The arenas of the model's geometry (model::plannedRuntimeMemory) by what
  // streams, [experts][table]: a target whose routed experts stream runs its
  // prefill commands as groups, and one whose table streams stages its rows,
  // so streaming only grows them. None before the model loads.
  std::array<std::array<uint64_t, 2>, 2> arenaBytes{};
  // The arenas of the geometry of a runtime whose K/V spills, by whether the
  // table streams (model::StreamedParts::kv): it runs prefill groups whether
  // the experts stream or not, and holds where its verifies' listed blocks
  // are, so they are at least arenaBytes'. A plan that spills takes them.
  std::array<uint64_t, 2> spillArenaBytes{};
  uint64_t laneStateBytes = 0;
  kv::Layout kvLayout;
  // The model's window (servedContextTokens), the context --max-context
  // asks for, zero for auto, and the one a relaunch keeps
  // (--relaunch-context), zero for a first start.
  uint32_t windowTokens = 0;
  uint32_t requestedContextTokens = 0;
  uint32_t relaunchContextTokens = 0;
  StreamedPart experts;
  StreamedPart table;
  // The row cache an automatic table takes at most
  // (model::StreamableWeights::tableShareBytes).
  uint64_t tableShareBytes = 0;
  // What the host leaves the engine as the governor counts it when the
  // stores are sized (hostBudgetBytes): an automatic part's store starts
  // with at most what it leaves beside what an engine without requests
  // holds once warm, reserveBytes aside.
  uint64_t hostBudgetBytes = 0;
  // --sparse-kv-memory: the hot block cache's target, at least its floor;
  // none keeps it at its floor.
  std::optional<uint64_t> hotCacheBytes;
  // The block-sparse attention a target declares, none for a dense one.
  std::optional<model::SparseKvShape> sparseKv;
  // The SSD cache: whether it takes writes, its slots and the room active
  // KV may take of its volume (model::DiskBudget::activeLimitBytes).
  bool tierWritable = false;
  uint64_t kvSlotBytes = 0;
  uint64_t stateSlotBytes = 0;
  uint64_t activeLimitBytes = 0;
  // The pages of a prefill group's rows (model::kPrefillGroupChunks chunks).
  uint32_t groupPages = 0;
};

// The stores the plan streams the parts through, zero for a part that loads
// whole or a model without it, and a spill-capable target's plan. The
// routed experts' store takes expertBytes, its cap, and gives back what
// memory needs down to its floor, expertFloorBytes: --expert-memory's size,
// or zero for the store's own (model::ExpertStoreOptions::floorBytes). An
// automatic store starts with expertStartBytes, what the host's budget
// leaves it, zero for its cap, and takes the rest back once memory allows
// (model::ExpertStoreOptions::startBytes). hostHeldBackBytes is what the
// host's budget held back of an automatic table's share, whose row cache
// does not grow.
struct StreamedMemoryPlan final {
  uint64_t expertBytes = 0;
  uint64_t expertFloorBytes = 0;
  uint64_t expertStartBytes = 0;
  uint64_t tableBytes = 0;
  uint64_t hostHeldBackBytes = 0;
  KvSpillPlan spill{};
};

struct StreamedMemoryResult final {
  std::optional<StreamedMemoryPlan> plan;
  // Why nothing was planned, and when the floors exceed the budget, their
  // bytes.
  std::string error;
  uint64_t requiredBytes = 0;
};

// Plans the stores of the parts that stream and a spill-capable target's
// floors from one budget (docs/server.md, Memory and context). The floors
// come first: the fixed bytes and the arenas, a lane's state and the KV
// runway (minimumRequiredBytes), the stores at their smallest (the routed
// experts' at --expert-memory's size where it is set) and the spill floors
// at the context spilling serves. Then the routed experts load whole where
// that leaves KV its allowance, and otherwise stream through all that the
// allowance and the PLE table's share leave; the table loads whole where the
// rest holds it, and otherwise streams through its share, tableShareBytes at
// most. The allowance is the KV of a context in the fewest pages whose whole
// extents hold it, as one request's capacity rounds them (kvCapacityPagesFor):
// a spill-capable target's first 64K tokens; a dense target's --max-context,
// or none, so that its experts load wherever main loads them, and the first
// 64K tokens once they stream. Last, the host's budget: an engine without
// requests holds what loads, the floors but a Spill request's and the
// stores, the reserve the budget holds already aside, and the automatic
// stores, the table's share first, take above their charge only what the
// budget leaves of what the engine's limit gives them: the table's row cache
// that much, what the budget holds back of it the plan's all the same
// (hostHeldBackBytes), and the experts' store starts with it, its cap the
// limit's (expertStartBytes), so that what loads whole, the floors and the
// context follow the engine's limit alone. The arenas are those of the
// parts that stream: where the plan with the arenas of fewer parts streams
// more, it plans again with those. An explicit size of the PLE table
// replaces its part; one of the routed experts is charged among the floors,
// and they load whole or stream as automatic experts do, a store that
// streams them taking all that the rest leaves them, so at least that size,
// with that size as its floor, and starting as an automatic store does.
// A size below the part's smallest store, a hot cache target beyond what
// the floors leave, and floors beyond the budget are refused.
[[nodiscard]] StreamedMemoryResult planStreamedMemory(const StreamedMemoryInputs &inputs);

} // namespace splash::engine
