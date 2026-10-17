#include "engine/MemoryGovernor.hpp"
#include "metal/abi/PagedAttention.h"
#include "model/SparseKvStore.hpp"
#include "ops/PageStorage.hpp"
#include "tests/engine/TestChecks.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace splash;
using namespace splash::engine;

namespace {

using splash::test::rejects;
using splash::test::require;

constexpr uint32_t kPageTokens = kv::kPageTokens;
constexpr uint32_t kBlockTokens = SPLASH_ATTENTION_BLOCK_TOKENS;
constexpr uint32_t kPageBlocks = kPageTokens / kBlockTokens;

// Two attention layers of two heads of 256 values, values token-major as a
// sparse layout keeps them.
kv::Layout sparseLayout(kv::Format format) {
  kv::Layout layout{2, 2, 256, format};
  layout.valuesTokenMajor = true;
  return layout;
}

// Fills every byte of every tensor of `page` from a pattern of the page and
// the byte's place.
void fill(const kv::PageStorage &pages, uint32_t page) {
  uint32_t tensor = 0;
  for (const std::span<std::byte> span : pages.spans(page)) {
    for (size_t at = 0; at < span.size(); ++at)
      span[at] = std::byte(static_cast<uint8_t>(page * 131 + tensor * 29 + at * 7 + at / 251));
    ++tensor;
  }
}

// Whether block `place` of every tensor of `layer` of the cache page
// `cached` holds block `block` of `layer` of pool page `page`.
bool sameBlock(const kv::PageStorage &pool, uint32_t page, uint32_t block,
               const std::vector<std::span<std::byte>> &cached, uint32_t place, uint32_t layer) {
  const kv::Layout layout = pool.layout();
  const uint32_t tensors = layout.format == kv::Format::Int8 ? 4 : 2;
  const std::vector<std::span<std::byte>> source = pool.spans(page);
  for (uint32_t tensor = layer * tensors; tensor < (layer + 1) * tensors; ++tensor) {
    const uint64_t perHead = source[tensor].size() / layout.kvHeads, perBlock = perHead / kPageBlocks;
    for (uint32_t head = 0; head < layout.kvHeads; ++head)
      if (std::memcmp(source[tensor].data() + head * perHead + block % kPageBlocks * perBlock,
                      cached[tensor].data() + head * perHead + place * perBlock, perBlock))
        return false;
  }
  return true;
}

void checkFormat(metal::MetalBackend &backend, MemoryGovernor &governor, kv::Format format) {
  const kv::Layout layout = sparseLayout(format);
  // Extents of as many pages in either format, so that a lane's cache holds
  // fewer blocks a layer than the request has in its file.
  constexpr uint32_t extentPages = 256;
  require(extentPages % layout.extentAlignmentPages() == 0, "the test's extents are not whole alignment units");
  kv::PageStorage pool(backend, governor.allocationAdmission(), layout, 2 * extentPages, extentPages);
  require(pool.allocateExtent(0) && pool.allocateExtent(1), "the pool did not allocate its extents");
  // A request of 2 * extentPages - 1 pages, the last one open.
  std::vector<uint32_t> table(2 * extentPages - 1);
  std::iota(table.rbegin(), table.rend(), 1u);
  for (const uint32_t page : table) fill(pool, page);
  const uint32_t slots = extentPages * kPageBlocks;
  const auto options = [&](uint32_t selected) {
    return model::SparseKvStoreOptions{.memoryBytes = pool.extentBytes() + pool.extentBytes() / 2,
                                       .lanes = 2,
                                       .maximumContextTokens = 2 * extentPages * kPageTokens,
                                       .selectedBlocks = selected,
                                       .directory = std::filesystem::temp_directory_path()};
  };
  rejects([&] { model::SparseKvStore(backend, governor.allocationAdmission(), pool, options(slots + 1)); },
          "--sparse-kv-memory holds fewer blocks", "a cache smaller than a step's selection was accepted");
  model::SparseKvStore store(backend, governor.allocationAdmission(), pool, options(slots));
  require(store.laneCacheBytes() == pool.extentBytes() && !store.cacheBytes(),
          "the cache is not one whole extent a lane, or was allocated before a request used a lane");

  const uint64_t committed = uint64_t{table.size() - 1} * kPageTokens + 5;
  store.prepare(1, 7, committed, table);
  require(store.stats().writtenPages == table.size() - 1, "the committed pages did not go to the file");
  require(store.cacheBytes() == pool.extentBytes(), "the request's lane did not allocate its cache alone");
  const uint32_t inFile = static_cast<uint32_t>(table.size() - 1) * kPageBlocks;
  const auto resolve = [&](uint32_t layer, std::span<const uint32_t> blocks) {
    std::vector<uint32_t> lists(slots + 8), locations(lists.size(), 0);
    for (size_t entry = 0; entry < blocks.size(); ++entry)
      lists[entry] = blocks[entry] << SPLASH_ATTENTION_BLOCK_ROW_BITS | 1u;
    const std::array<uint32_t, 1> lanes{1};
    const std::array<uint32_t, 1> counts{static_cast<uint32_t>(blocks.size())};
    store.resolve(layer, lanes, lists.data(), counts.data(), static_cast<uint32_t>(lists.size()), locations.data());
    locations.resize(blocks.size());
    return locations;
  };
  // Blocks of the file land in the lane's slots with their pages' content;
  // the open page's are read from the request's page.
  const std::vector<uint32_t> blocks{0, 3, 9, inFile - 1, inFile, inFile + 3};
  const std::vector<uint32_t> first = resolve(1, blocks);
  for (size_t entry = 0; entry < blocks.size(); ++entry) {
    if (blocks[entry] >= inFile) {
      require(first[entry] == SPLASH_ATTENTION_PAGED, "a block of the open page was not left to the page table");
      continue;
    }
    const uint32_t location = first[entry];
    require(location / slots == 1, "a block went to another lane's slots");
    require(sameBlock(pool, table[blocks[entry] / kPageBlocks], blocks[entry],
                      store.cacheSpans(location >> SPLASH_ATTENTION_LOCATION_BLOCK_BITS), location % kPageBlocks, 1),
            "a cached block does not hold its page's keys, values and scales");
  }
  require(store.stats().reads == 4 && store.stats().hits == 0, "the first resolve did not read each block once");
  // Again: every block is cached where it was.
  require(resolve(1, blocks) == first && store.stats().reads == 4 && store.stats().hits == 4,
          "a cached block was read again or moved");
  // The other layer keeps its own slots.
  (void)resolve(0, std::span(blocks).first(2));
  require(store.stats().reads == 6, "another layer's resolve found the first layer's blocks");

  // A step of every slot's worth of other blocks evicts the four, oldest
  // first; reading them again reads them from the file.
  std::vector<uint32_t> many;
  for (uint32_t block = 10; many.size() < slots; ++block)
    if (block != inFile - 1) many.push_back(block);
  (void)resolve(1, many);
  const uint64_t before = store.stats().reads;
  (void)resolve(1, std::span(blocks).first(4));
  require(store.stats().reads == before + 4, "blocks evicted by a full step were not read again");
  rejects([&] {
    std::vector<uint32_t> over(slots + 1);
    std::iota(over.begin(), over.end(), 0u);
    (void)resolve(1, over);
  }, "smaller than its selection", "a step selecting more blocks than the cache holds was accepted");

  // Another request on the lane starts over: nothing of the last is in its
  // file, so its blocks come from its pages until it commits them.
  store.prepare(1, 8, kPageTokens - 1, table);
  require(resolve(1, std::span(blocks).first(2)) ==
              std::vector<uint32_t>(2, SPLASH_ATTENTION_PAGED),
          "a new request found the last one's blocks");
  std::cout << "sparse KV store: format=" << kv::formatName(format) << " PASS\n";
}

// A lane whose cache the budget refuses reads every block from its pages.
void checkRefused(metal::MetalBackend &backend, MemoryGovernor &governor) {
  const kv::Layout layout = sparseLayout(kv::Format::Int8);
  const uint32_t extentPages = layout.extentAlignmentPages();
  kv::PageStorage pool(backend, governor.allocationAdmission(), layout, extentPages, extentPages);
  require(static_cast<bool>(pool.allocateExtent(0)), "the pool did not allocate its extent");
  std::vector<uint32_t> table(4);
  std::iota(table.begin(), table.end(), 0u);
  const metal::AllocationAdmission refuse = [](uint64_t, const std::function<void()> &) {
    return metal::AllocationResult(metal::AllocationFailure::EngineBudget);
  };
  model::SparseKvStore store(backend, refuse, pool,
                             {.memoryBytes = pool.extentBytes(), .lanes = 1, .maximumContextTokens = 4096,
                              .selectedBlocks = 16, .directory = std::filesystem::temp_directory_path()});
  store.prepare(0, 1, 4 * kPageTokens, table);
  std::vector<uint32_t> lists(16), locations(16, 0);
  for (uint32_t entry = 0; entry < 8; ++entry) lists[entry] = entry << SPLASH_ATTENTION_BLOCK_ROW_BITS;
  const std::array<uint32_t, 1> lane{0}, counts{8};
  store.resolve(0, lane, lists.data(), counts.data(), 16, locations.data());
  require(!store.cacheBytes() && !store.stats().writtenPages &&
              std::all_of(locations.begin(), locations.begin() + 8,
                          [](uint32_t location) { return location == SPLASH_ATTENTION_PAGED; }),
          "a lane without a cache read or wrote anything of its own");
  std::cout << "sparse KV store without memory: PASS\n";
}

void run(const std::string &metallib) {
  metal::MetalBackend backend(metallib);
  MemoryGovernor governor(backend, backend.capabilities().recommendedMaxWorkingSetBytes, 1, 1, queryHostMemory, 0);
  kv::Layout dense{2, 2, 256};
  kv::PageStorage densePool(backend, governor.allocationAdmission(), dense, dense.extentAlignmentPages(),
                            dense.extentAlignmentPages());
  rejects([&] {
    model::SparseKvStore(backend, governor.allocationAdmission(), densePool,
                         {.memoryBytes = densePool.extentBytes(), .lanes = 1, .maximumContextTokens = 4096,
                          .selectedBlocks = 1, .directory = std::filesystem::temp_directory_path()});
  }, "token-major values", "a store over dimension-major values was accepted");
  checkFormat(backend, governor, kv::Format::Int8);
  checkFormat(backend, governor, kv::Format::BFloat16);
  checkRefused(backend, governor);
  std::cout << "sparse KV store tests passed\n";
}

} // namespace

int main(int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "usage: sparse_kv_store_test METALLIB\n";
    return EXIT_FAILURE;
  }
  try {
    run(argv[1]);
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "sparse KV store test failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
