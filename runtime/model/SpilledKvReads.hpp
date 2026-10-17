#pragma once

// How a block-sparse target's attention reads K/V whose pages hold only
// their index (kv::PageForm::Index), with two calls in its family code: the
// runtime builds one per command, over the hot block cache (decode) and the
// history stream (prefill) and the command's page sources. A command none of
// whose lanes holds an Index page gets the null one, which changes nothing:
// its graph is the one it encodes without spilling.

#include "metal/CommandGraph.hpp"
#include "model/KvHistoryStream.hpp"
#include "ops/PagedAttention.hpp"

#include <cstdint>

namespace splash::model {

class SpilledKvReads final {
public:
  // The null one.
  SpilledKvReads() = default;

  // Some lane of the command holds an Index page.
  [[nodiscard]] bool active() const noexcept;
  // After a layer's block lists, before its attention: the lists with the
  // locations of their entries' blocks (KvBlockCache::addResolve), for the
  // located split. Only an active one, for lists.
  [[nodiscard]] ops::AttentionBlocks verify(metal::CommandGraph &graph, uint32_t layer,
                                            const ops::AttentionBlocks &blocks);
  // After a prefill chunk's store: the K/V its attention reads, staged when
  // the history of `sequence` holds Index pages (KvHistoryStream), else
  // `pool`. The chunk's pages are [chunkFirstPage, chunkFirstPage +
  // chunkPages).
  [[nodiscard]] PrefillKv prefill(metal::CommandGraph &graph, uint32_t layer, uint32_t sequence,
                                  uint32_t chunkFirstPage, uint32_t chunkPages, const PrefillKv &pool);
  // After a layer's attention: the hand-off of a stream with one buffer to
  // the next layer's reads.
  void layerEnd(metal::CommandGraph &graph, uint32_t layer);
};

} // namespace splash::model
