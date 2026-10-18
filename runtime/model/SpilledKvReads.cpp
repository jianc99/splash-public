#include "model/SpilledKvReads.hpp"

#include "model/KvBlockCache.hpp"

#include <stdexcept>
#include <utility>

namespace splash::model {

SpilledKvReads::SpilledKvReads(KvBlockCache &cache, std::vector<std::shared_ptr<const KvPageSources>> lanes,
                               metal::MetalBuffer locations)
    : cache_(&cache), sources_(std::move(lanes)), locations_(std::move(locations)) {
  // A lane whose pages all hold their K/V reads each of its blocks through
  // its page table.
  static const KvPageSources resident;
  for (const std::shared_ptr<const KvPageSources> &sources : sources_)
    lanes_.push_back(sources ? sources.get() : &resident);
}

SpilledKvReads::SpilledKvReads(KvHistoryStream &stream, uint32_t sequence,
                               std::shared_ptr<const KvPageSources> sources, uint32_t pages,
                               metal::MetalBuffer poolTable, std::span<const SplashKvLayer> poolLayers)
    : stream_(&stream), sources_{std::move(sources)}, sequence_(sequence), pages_(pages),
      poolTable_(std::move(poolTable)), poolLayers_(poolLayers) {
  if (!sources_.front()) throw std::invalid_argument("a streamed prefill has no page sources");
}

SpilledKvReads::~SpilledKvReads() { end(); }

bool SpilledKvReads::active() const noexcept { return cache_ || stream_; }

ops::AttentionBlocks SpilledKvReads::verify(metal::CommandGraph &graph, uint32_t layer,
                                            const ops::AttentionBlocks &blocks) {
  if (!cache_) throw std::logic_error("only a decode command's reads of spilled K/V resolve blocks");
  if ((uint64_t{blocks.list} + lanes_.size()) * blocks.stride * sizeof(uint32_t) > locations_.sizeBytes())
    throw std::logic_error("a verify's block lists are longer than their locations");
  running_ = true;
  return cache_->addResolve(graph, layer, lanes_, blocks, locations_, stats_);
}

// The sequence's first chunk of the command begins the stream, and its first
// chunk of each layer stages the layer.
PrefillKv SpilledKvReads::prefill(metal::CommandGraph &graph, uint32_t layer, uint32_t sequence,
                                  uint32_t chunkFirstPage, uint32_t chunkPages, const PrefillKv &pool) {
  if (!stream_ || sequence != sequence_) return pool;
  if (!running_) {
    stream_->begin(*sources_.front(), pages_, poolTable_, poolLayers_);
    running_ = true;
  }
  if (stagedLayer_ != layer) {
    staged_ = stream_->addLayer(graph, layer);
    stagedLayer_ = layer;
  }
  stream_->addChunk(graph, layer, chunkFirstPage, chunkPages);
  if (chunkFirstPage + chunkPages == pages_) attendedLayer_ = layer;
  return staged_;
}

void SpilledKvReads::layerEnd(metal::CommandGraph &graph, uint32_t layer) {
  if (attendedLayer_ != layer) return;
  stream_->addLayerEnd(graph, layer);
  attendedLayer_.reset();
}

void SpilledKvReads::end() {
  if (!running_) return;
  running_ = false;
  if (cache_) {
    cache_->mergeStats(std::exchange(stats_, {}));
    const std::vector<KvReadFailure> failed = cache_->takeFailures();
    failures_.insert(failures_.end(), failed.begin(), failed.end());
    return;
  }
  const KvStreamResult result = stream_->end();
  for (const KvReadFailure &failure : result.failures) failures_.push_back({sequence_, failure.block});
  streamMilliseconds_ += result.streamMilliseconds;
  stagedLayer_.reset();
  attendedLayer_.reset();
}

} // namespace splash::model
