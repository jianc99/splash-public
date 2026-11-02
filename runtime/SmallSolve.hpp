#pragma once

// x with h x = g for a small positive definite h, by Gaussian elimination
// with partial pivoting: the Newton steps of VerifyBudget's acceptance fit
// and the engine's CostModel's ridge fit.

#include <array>
#include <cmath>
#include <cstddef>
#include <utility>

namespace splash {

template <size_t N>
[[nodiscard]] std::array<double, N> solve(std::array<std::array<double, N>, N> h, std::array<double, N> g) noexcept {
  for (size_t column = 0; column < N; ++column) {
    size_t pivot = column;
    for (size_t row = column + 1; row < N; ++row)
      if (std::fabs(h[row][column]) > std::fabs(h[pivot][column]))
        pivot = row;
    std::swap(h[column], h[pivot]);
    std::swap(g[column], g[pivot]);
    for (size_t row = column + 1; row < N; ++row) {
      const double factor = h[row][column] / h[column][column];
      for (size_t k = column; k < N; ++k)
        h[row][k] -= factor * h[column][k];
      g[row] -= factor * g[column];
    }
  }
  std::array<double, N> x{};
  for (size_t row = N; row-- > 0;) {
    double sum = g[row];
    for (size_t k = row + 1; k < N; ++k)
      sum -= h[row][k] * x[k];
    x[row] = sum / h[row][row];
  }
  return x;
}

} // namespace splash
