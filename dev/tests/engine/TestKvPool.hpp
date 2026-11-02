#pragma once

#include "FakeMemory.hpp"
#include "engine/KvPool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace splash::test {

// Extents of extentPages pages, none allocated until the pool allocates
// them (its runway first), as PageStorage, of bytesPerPage bytes a Full page,
// indexBytesPerPage an Index page and slimBytesPerPage a Slim one; without
// index bytes no page is Index, and without slim bytes none is Slim. Growth
// is refused with allocationFailure (EngineBudget, the governor's cause,
// unless a test names another) past the bytes of budgetPages Full pages,
// while growthBlocked, or when growthAllowed(extent) says no; a
// budget-limited pool needs more page ids than its budget, as production's
// has. With `memory`, growth is allocated through it under `admission`
// instead (held as "kv"), and a release gives the bytes back. A release or
// a copy throws std::logic_error while commandInFlight()
// says a command is in flight, as PageStorage's does. Every page holds one
// value, which a test sets, a copy carries (between forms too, as what both
// pages hold) and a clear zeroes, so a test can follow the content of pages
// the pool moves. An extent is active from its allocation, as PageStorage's,
// until the pool says otherwise (setExtentActive), which counts each change
// and, as a misuse, a mark of an unallocated extent or one the extent already
// has. Releasing an active extent throws std::logic_error, changing nothing:
// the pool releases only extents none of whose pages a request holds.
class TestKvStorage final : public kv::ExtentStorage {
public:
    TestKvStorage(uint32_t pages, uint64_t bytesPerPage, uint32_t extentPages,
                  uint64_t indexBytesPerPage = 0, uint64_t slimBytesPerPage = 0)
        : content(pages), pageCount_(pages), bytesPerPage_(bytesPerPage),
          indexBytesPerPage_(indexBytesPerPage), slimBytesPerPage_(slimBytesPerPage),
          extentPages_(extentPages) {
        if (!pages || !bytesPerPage || !extentPages || pages % extentPages) {
            throw std::invalid_argument("invalid test KV extent storage");
        }
        forms_.resize(pages / extentPages);
        active_.resize(pages / extentPages);
    }

    uint32_t pageCount() const noexcept override { return pageCount_; }
    uint64_t bytesPerPage(kv::PageForm form) const noexcept override {
        switch (form) {
        case kv::PageForm::Full: return bytesPerPage_;
        case kv::PageForm::Index: return indexBytesPerPage_;
        case kv::PageForm::Slim: return slimBytesPerPage_;
        }
        return 0;
    }
    uint32_t extentPages() const noexcept override { return extentPages_; }
    metal::AllocationResult allocateExtent(uint32_t extent, kv::PageForm form) override {
        ++allocationAttempts;
        if (forms_.at(extent))
            throw std::logic_error("test KV extent is already allocated");
        if (form == kv::PageForm::Index && !indexBytesPerPage_)
            throw std::invalid_argument("test KV pages without an index are all Full");
        if (form == kv::PageForm::Slim && !slimBytesPerPage_)
            throw std::invalid_argument("test KV pages without a window keep every layer");
        if (memory) {
            if (const metal::AllocationResult result =
                    memory->allocate(admission, "kv", extentPages_ * bytesPerPage(form));
                !result)
                return result;
        } else if (growthBlocked || (growthAllowed && !growthAllowed(extent)) ||
                   allocatedBytes() + extentPages_ * bytesPerPage(form) > budgetPages * bytesPerPage_) {
            return allocationFailure;
        }
        forms_[extent] = form;
        active_[extent] = true;
        return {};
    }
    void releaseExtent(uint32_t extent) override {
        if (!forms_.at(extent))
            throw std::logic_error("test KV extent is not allocated");
        if (active_[extent])
            throw std::logic_error("the pool released a KV extent a request holds a page of");
        if (commandInFlight && commandInFlight()) {
            throw std::logic_error(
                "cannot release a KV extent while a command is in flight");
        }
        if (memory)
            memory->release("kv", extentPages_ * bytesPerPage(*forms_[extent]));
        forms_[extent].reset();
        const auto start = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(releaseTime);
        const double milliseconds = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - start)
                                        .count();
        longestRelease = std::max(longestRelease, milliseconds);
        totalRelease += milliseconds;
        ++releasedExtents;
        if (released) released();
    }
    void copyPages(std::span<const kv::PageCopy> pages) override {
        if (commandInFlight && commandInFlight()) {
            throw std::logic_error(
                "cannot copy KV pages while a command is in flight");
        }
        for (const kv::PageCopy &copy : pages) {
            content.at(copy.to) = content.at(copy.from);
            copies.push_back(copy);
        }
    }
    void setExtentActive(uint32_t extent, bool active) noexcept override {
        if (extent >= forms_.size() || !forms_[extent] || active_[extent] == active) {
            ++activityMisuses;
            return;
        }
        active_[extent] = active;
        ++activityChanges;
    }
    void clearPages(std::span<const uint32_t> pages) override {
        for (uint32_t page : pages) {
            if (!forms_.at(page / extentPages_)) {
                throw std::logic_error(
                    "cannot clear a KV page of an extent that is not allocated");
            }
        }
        for (uint32_t page : pages) {
            content.at(page) = 0;
            cleared.push_back(page);
        }
    }

    [[nodiscard]] bool allocated(uint32_t extent) const {
        return forms_.at(extent).has_value();
    }
    // The form of the page's extent, which must be allocated.
    [[nodiscard]] kv::PageForm form(uint32_t page) const {
        return forms_.at(page / extentPages_).value();
    }
    [[nodiscard]] uint32_t allocatedPages() const noexcept {
        uint32_t extents = 0;
        for (const auto &form : forms_) extents += form.has_value();
        return extents * extentPages_;
    }
    [[nodiscard]] uint64_t allocatedBytes() const noexcept {
        uint64_t bytes = 0;
        for (const auto &form : forms_)
            if (form) bytes += extentPages_ * bytesPerPage(*form);
        return bytes;
    }
    // Allocates its extents through `fakeMemory` under `growth` from now on;
    // returns itself, for the pool built on it.
    TestKvStorage &through(FakeMemory &fakeMemory, metal::AllocationAdmission growth) {
        memory = &fakeMemory;
        admission = std::move(growth);
        return *this;
    }
    // Whether the allocated extent is active, as the pool last said.
    [[nodiscard]] bool active(uint32_t extent) const {
        if (!forms_.at(extent))
            throw std::logic_error("test KV extent is not allocated");
        return active_[extent];
    }
    // Whether exactly the allocated extents that hold a page a request holds
    // are active, and the pool told only changes.
    [[nodiscard]] bool activityFollows(const engine::KvPool &pool) const {
        for (uint32_t extent = 0; extent < forms_.size(); ++extent) {
            if (!forms_[extent]) continue;
            bool held = false;
            for (uint32_t page = extent * extentPages_; page < (extent + 1) * extentPages_; ++page)
                held = held || pool.pageActive(page);
            if (active_[extent] != held) return false;
        }
        return activityMisuses == 0;
    }

    FakeMemory *memory = nullptr;
    metal::AllocationAdmission admission;
    uint32_t budgetPages = std::numeric_limits<uint32_t>::max();
    bool growthBlocked = false;
    std::function<bool(uint32_t extent)> growthAllowed;
    metal::AllocationFailure allocationFailure = metal::AllocationFailure::EngineBudget;
    std::function<bool()> commandInFlight;
    // After each release: a budget shared with states returns its bytes.
    std::function<void()> released;
    // How long releasing one extent takes.
    std::chrono::milliseconds releaseTime{0};
    uint32_t allocationAttempts = 0;
    uint32_t releasedExtents = 0;
    // The changes setExtentActive made, and the calls it took as misuses.
    uint32_t activityChanges = 0;
    uint32_t activityMisuses = 0;
    // What the releases took as measured here, in milliseconds: the longest,
    // and all of them together.
    double longestRelease = 0.0;
    double totalRelease = 0.0;
    std::vector<uint64_t> content;
    std::vector<kv::PageCopy> copies;
    std::vector<uint32_t> cleared;

private:
    uint32_t pageCount_ = 0;
    uint64_t bytesPerPage_ = 0;
    uint64_t indexBytesPerPage_ = 0;
    uint64_t slimBytesPerPage_ = 0;
    uint32_t extentPages_ = 0;
    // The form of each allocated extent.
    std::vector<std::optional<kv::PageForm>> forms_;
    // Whether each allocated extent is active.
    std::vector<bool> active_;
};

}  // namespace splash::test
