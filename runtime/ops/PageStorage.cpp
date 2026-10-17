#include "ops/PageStorage.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace splash::kv {

PageStorage::PageStorage(metal::MetalBackend &backend,
    metal::AllocationAdmission admitAllocation, Layout layout,
    uint32_t pageCount, uint32_t extentPages)
    : backend_(backend), admitAllocation_(std::move(admitAllocation)),
      layout_(layout), pageCount_(pageCount), extentPages_(extentPages) {
    if (!admitAllocation_) {
        throw std::invalid_argument(
            "KV page storage requires allocation admission");
    }
    if (!layout_.valid()) {
        throw std::invalid_argument("KV page storage layout is invalid");
    }
    // Page entries carry the index in an extent in their low bits, and
    // SplashKvLayer places a layer's region with a 32-bit offset, so an
    // extent stays below 4 GiB.
    if (!extentPages_ || extentPages_ % layout_.extentAlignmentPages() ||
        extentPages_ > SPLASH_KV_PAGE_INDEX_MASK ||
        extentBytes() > std::numeric_limits<uint32_t>::max() ||
        extentBytes() > backend_.capabilities().maxBufferLengthBytes) {
        throw std::invalid_argument("KV extent geometry is invalid");
    }
    if (!pageCount_ || pageCount_ % extentPages_) {
        throw std::invalid_argument(
            "KV page pool is not a whole number of extents");
    }
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
        layers_.push_back(splash_kv_layer(extentPages_, data, scale, layer));
    extents_.resize(pageCount_ / extentPages_);
    indexes_.resize(extents_.size());
    extentAddresses_.resize(extents_.size());
    indexAddresses_.resize(extents_.size());
    allocated_.resize(extents_.size());
    spilledPages_.resize(extents_.size());
    spilled_.resize(pageCount_);
    pins_.resize(pageCount_);
}

metal::MetalBuffer PageStorage::allocateBuffer(uint64_t bytes, const std::string &name) const {
    metal::MetalBuffer allocated = backend_.allocateBuffer(
        bytes, metal::BufferStorage::Shared, name, metal::BufferReach::Addressed);
    if (allocated.gpuAddress() & SPLASH_KV_PAGE_INDEX_MASK) {
        throw std::logic_error(
            "KV extent address leaves no room for the page index");
    }
    return allocated;
}

size_t PageStorage::extentIndex(uint32_t page) const {
    if (page >= pageCount_) throw std::out_of_range("invalid KV page id");
    return page / extentPages_;
}

uint64_t PageStorage::allocatedExtentBytes() const noexcept {
    return static_cast<uint64_t>(std::count(allocated_.begin(), allocated_.end(), true)) * extentBytes();
}

uint64_t PageStorage::actualAllocatedBytes() const noexcept {
    uint64_t bytes = 0;
    for (size_t extent = 0; extent < extents_.size(); ++extent)
        if (allocated_[extent]) bytes += indexBytes() + (extents_[extent] ? kvBytes() : 0);
    return bytes;
}

bool PageStorage::isAllocated(uint32_t page) const {
    return allocated_[extentIndex(page)];
}

metal::AllocationResult PageStorage::allocateExtent(uint32_t extent) {
    if (allocated_.at(extent)) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is already allocated");
    }
    const std::string name = "kv-extent-" + std::to_string(extent * extentPages_);
    return admitAllocation_(extentBytes(), [&] {
        metal::MetalBuffer kv = allocateBuffer(kvBytes(), name);
        metal::MetalBuffer index = indexBytes() ? allocateBuffer(indexBytes(), name + "-index") : metal::MetalBuffer{};
        extentAddresses_[extent] = kv.gpuAddress();
        indexAddresses_[extent] = index ? index.gpuAddress() : 0;
        extents_[extent] = std::move(kv);
        indexes_[extent] = std::move(index);
        allocated_[extent] = true;
    });
}

void PageStorage::releaseExtent(uint32_t extent) {
    if (!allocated_.at(extent)) {
        throw std::logic_error(
            "KV extent " + std::to_string(extent) + " is not allocated");
    }
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot release a KV extent while a command is in flight");
    }
    extents_[extent] = {};
    indexes_[extent] = {};
    extentAddresses_[extent] = indexAddresses_[extent] = 0;
    allocated_[extent] = false;
    const uint32_t first = extent * extentPages_;
    for (uint32_t page = first; page < first + extentPages_; ++page) spilled_[page] = false;
    spilledPages_[extent] = 0;
}

void PageStorage::spill(uint32_t page) {
    const size_t extent = extentIndex(page);
    if (!allocated_[extent])
        throw std::logic_error("cannot spill a KV page of an extent that is not allocated");
    if (!spilled_[page]) {
        spilled_[page] = true;
        ++spilledPages_[extent];
    }
}

bool PageStorage::spilled(uint32_t page) const {
    (void)extentIndex(page);
    return spilled_[page];
}

bool PageStorage::kvResident(uint32_t page) const { return static_cast<bool>(extents_[extentIndex(page)]); }

metal::AllocationResult PageStorage::unspill(uint32_t page) {
    const size_t extent = extentIndex(page);
    if (!allocated_[extent])
        throw std::logic_error("cannot restore a KV page of an extent that is not allocated");
    if (!extents_[extent]) {
        const metal::AllocationResult granted = admitAllocation_(kvBytes(), [&] {
            metal::MetalBuffer kv = allocateBuffer(kvBytes(), "kv-extent-" + std::to_string(extent * extentPages_));
            extentAddresses_[extent] = kv.gpuAddress();
            extents_[extent] = std::move(kv);
        });
        if (!granted) return granted;
        ++kvRevision_;
    }
    if (spilled_[page]) {
        spilled_[page] = false;
        --spilledPages_[extent];
    }
    return {};
}

uint64_t PageStorage::releaseSpilledKv() {
    if (backend_.commandInFlight()) return 0;
    uint64_t released = 0;
    for (size_t extent = 0; extent < extents_.size(); ++extent) {
        if (!extents_[extent] || spilledPages_[extent] != extentPages_) continue;
        const uint32_t first = static_cast<uint32_t>(extent) * extentPages_;
        if (std::any_of(pins_.begin() + first, pins_.begin() + first + extentPages_,
                        [](uint32_t pins) { return pins != 0; }))
            continue;
        extents_[extent] = {};
        extentAddresses_[extent] = 0;
        released += kvBytes();
    }
    if (released) ++kvRevision_;
    return released;
}

void PageStorage::pin(uint32_t page) { ++pins_.at(page); }

void PageStorage::unpin(uint32_t page) {
    if (!pins_.at(page)) throw std::logic_error("KV page is not pinned");
    --pins_[page];
}

// A page of an extent whose K/V buffer went gets it back, and its new holder
// writes its K/V.
metal::AllocationResult PageStorage::prepareAcquired(uint32_t page) { return unspill(page); }

void PageStorage::copyPages(std::span<const PageCopy> copies) {
    if (backend_.commandInFlight()) {
        throw std::logic_error(
            "cannot copy KV pages while a command is in flight");
    }
    for (const PageCopy &copy : copies) {
        if (!isAllocated(copy.from) || !isAllocated(copy.to)) {
            throw std::logic_error(
                "cannot copy a KV page of an extent that is not allocated");
        }
    }
    for (const PageCopy &copy : copies) {
        const auto source = spans(copy.from);
        const auto destination = spans(copy.to);
        for (size_t tensor = 0; tensor < source.size(); ++tensor) {
            std::memcpy(destination[tensor].data(), source[tensor].data(),
                        source[tensor].size());
        }
    }
}

void PageStorage::clearPages(std::span<const uint32_t> pages) {
    for (const uint32_t page : pages) {
        if (!isAllocated(page)) {
            throw std::logic_error(
                "cannot clear a KV page of an extent that is not allocated");
        }
    }
    for (const uint32_t page : pages) {
        for (const std::span<std::byte> tensor : spans(page))
            std::memset(tensor.data(), 0, tensor.size());
    }
}

uint32_t PageStorage::indexOffset(uint32_t layer) const {
    if (!layout_.indexDimension || layer >= layout_.attentionLayers)
        throw std::out_of_range("KV layer has no QSA index region");
    // A region per layer of the index buffer.
    return static_cast<uint32_t>(uint64_t{extentPages_} * layer * layout_.indexBytesPerLayerPage());
}

SplashKvPage PageStorage::entry(uint32_t page) const {
    const size_t extent = extentIndex(page);
    if (!allocated_[extent]) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               " is in an extent that is not allocated");
    }
    const uint64_t address = extentAddresses_[extent];
    return address ? splash_kv_page_entry(address, page - static_cast<uint32_t>(extent) * extentPages_)
                   : SplashKvPage(0);
}

void PageStorage::writeEntries(std::span<const uint32_t> pages,
                               uint32_t first,
                               const metal::MetalBuffer &table) const {
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    if (!entries || table.sizeBytes() / sizeof(SplashKvPage) < pages.size()) {
        throw std::logic_error(
            "KV page table is not CPU-visible or too small for its entries");
    }
    if (first > pages.size()) {
        throw std::invalid_argument(
            "KV page table entries start past the end of its pages");
    }
    for (size_t index = first; index < pages.size(); ++index)
        entries[index] = entry(pages[index]);
}

void PageStorage::writeIndexEntries(std::span<const uint32_t> pages, uint32_t first,
                                    const metal::MetalBuffer &table) const {
    auto *entries = static_cast<SplashKvPage *>(table.contents());
    if (!layout_.indexDimension)
        throw std::logic_error("KV pages hold no QSA index");
    if (!entries || table.sizeBytes() / sizeof(SplashKvPage) < pages.size())
        throw std::logic_error("KV index table is not CPU-visible or too small for its entries");
    if (first > pages.size())
        throw std::invalid_argument("KV index table entries start past the end of its pages");
    for (size_t index = first; index < pages.size(); ++index) {
        const size_t extent = extentIndex(pages[index]);
        if (!allocated_[extent])
            throw std::logic_error("KV page " + std::to_string(pages[index]) +
                                   " is in an extent that is not allocated");
        entries[index] = splash_kv_page_entry(indexAddresses_[extent],
                                              pages[index] - static_cast<uint32_t>(extent) * extentPages_);
    }
}

std::vector<std::span<std::byte>> PageStorage::spans(uint32_t page) const {
    const size_t at = extentIndex(page);
    auto *extent = static_cast<std::byte *>(extents_[at].contents());
    if (!extent) {
        throw std::logic_error("KV page " + std::to_string(page) +
                               (allocated_[at] ? " gave its K/V back to its sparse KV store"
                                               : " is in an extent that is not allocated"));
    }
    auto *indexes = static_cast<std::byte *>(indexes_[at].contents());
    const auto data = static_cast<uint32_t>(layout_.dataBytesPerLayerPage());
    const auto scale = static_cast<uint32_t>(layout_.scaleBytesPerLayerPage());
    const uint32_t index = page % extentPages_;
    std::vector<std::span<std::byte>> result;
    result.reserve(size_t{layout_.attentionLayers} * ((scale ? 4 : 2) + (layout_.indexDimension ? 1 : 0)));
    for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer) {
        for (uint32_t tensor = SPLASH_KV_KEYS; tensor <= SPLASH_KV_VALUE_SCALES;
             ++tensor) {
            if (const uint32_t bytes = splash_kv_page_bytes(data, scale, tensor)) {
                result.emplace_back(extent + splash_kv_offset(extentPages_, data,
                                                              scale, layer,
                                                              tensor, index),
                                    bytes);
            }
        }
    }
    if (layout_.indexDimension) {
        const uint64_t bytes = layout_.indexBytesPerLayerPage();
        for (uint32_t layer = 0; layer < layout_.attentionLayers; ++layer)
            result.emplace_back(indexes + indexOffset(layer) + index * bytes, bytes);
    }
    return result;
}

}  // namespace splash::kv
