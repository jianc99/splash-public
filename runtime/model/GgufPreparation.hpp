#pragma once

#include "metal/MetalBackend.hpp"
#include "model/GgufImage.hpp"
#include "model/PreparedWeights.hpp"

namespace splash::model {

// At most two 256 x 8192 staging buffers, plus bounded hashing/copy space.
// No tensor, layer or expert count can increase this bound.

[[nodiscard]] std::string ggufImageKey(const std::string &sourceDigest,
                                      const gguf::Image &image);
void prepareGgufImage(metal::MetalBackend &backend, int source, int destination,
                      const gguf::Image &image, const PreparationCheck &check = {});

} // namespace splash::model
