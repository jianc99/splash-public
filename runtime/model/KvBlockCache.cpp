#include "model/KvBlockCache.hpp"

#include <stdexcept>

namespace splash::model {

// Step 0 stubs, which M5 (spilled-KV readers) replaces: no cache is built.

namespace {

[[noreturn]] void notBuilt() { throw std::logic_error("the hot block cache is not built yet (M5)"); }

} // namespace

KvBlockCache::KvBlockCache(metal::MetalBackend &, metal::AllocationAdmission, const kv::Layout &, uint32_t,
                           KvBlockCacheOptions) {
  notBuilt();
}

KvBlockCache::~KvBlockCache() = default;

ModelMemoryAdmission KvBlockCache::reserveLane(uint32_t) { notBuilt(); }

void KvBlockCache::releaseLane(uint32_t) noexcept {}

ModelMemoryAdmission KvBlockCache::growTowardTarget() { notBuilt(); }

uint64_t KvBlockCache::releaseAboveFloor() { notBuilt(); }

ops::AttentionBlocks KvBlockCache::addResolve(metal::CommandGraph &, uint32_t, std::span<const KvPageSources *const>,
                                              const ops::AttentionBlocks &, const metal::MetalBuffer &,
                                              KvReadStats &) {
  notBuilt();
}

std::vector<KvReadFailure> KvBlockCache::takeFailures() { notBuilt(); }

void KvBlockCache::mergeStats(const KvReadStats &) { notBuilt(); }

KvBlockCacheStats KvBlockCache::stats() const { notBuilt(); }

} // namespace splash::model
