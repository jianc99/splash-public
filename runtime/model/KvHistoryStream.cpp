#include "model/KvHistoryStream.hpp"

#include <stdexcept>

namespace splash::model {

// Step 0 stubs, which M5 (spilled-KV readers) replaces: no stream is built.

namespace {

[[noreturn]] void notBuilt() { throw std::logic_error("the history stream is not built yet (M5)"); }

} // namespace

KvHistoryStream::KvHistoryStream(metal::MetalBackend &, metal::AllocationAdmission, const kv::Layout &, uint32_t,
                                 std::shared_ptr<SsdReaders>, kv::SlotFormat, uint32_t) {
  notBuilt();
}

KvHistoryStream::~KvHistoryStream() = default;

ModelMemoryAdmission KvHistoryStream::reserve(uint32_t) { notBuilt(); }

uint64_t KvHistoryStream::releaseIdle() { notBuilt(); }

void KvHistoryStream::begin(const KvPageSources &, uint32_t, const metal::MetalBuffer &,
                            std::span<const SplashKvLayer>) {
  notBuilt();
}

PrefillKv KvHistoryStream::addLayer(metal::CommandGraph &, uint32_t) { notBuilt(); }

void KvHistoryStream::addChunk(metal::CommandGraph &, uint32_t, uint32_t, uint32_t) { notBuilt(); }

void KvHistoryStream::addLayerEnd(metal::CommandGraph &, uint32_t) { notBuilt(); }

KvStreamResult KvHistoryStream::end() { notBuilt(); }

KvHistoryStreamStats KvHistoryStream::stats() const { notBuilt(); }

} // namespace splash::model
