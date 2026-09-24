#pragma once

// Layout constants of the weight files: preparation writes them, the weight
// store reads them. Preparation code takes them from this header, so the
// preparation identity does not follow the reader's API.

#include <cstdint>
#include <string_view>

namespace splash::model {

inline constexpr uint32_t kQ4GroupElements = 64;
inline constexpr uint64_t kBFloat16Bytes = 2;

inline constexpr uint64_t kWeightFileAlignment = 16 * 1024;
inline constexpr uint32_t kQ4StorageN = 256;

// The prepared GGUF target image and the packed vision tower.
inline constexpr std::string_view kGgufImageMagic = "MDGG0001";
inline constexpr std::string_view kVisionMagic = "MDFV0001";

} // namespace splash::model
