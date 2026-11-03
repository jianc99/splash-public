#pragma once

// What a prefill command or one pass of it costs, learned from the units
// that ran. A unit's price is a rate per unit of each of seven quantities of
// its work (model::CommandWork) summed; the rates are a decayed ridge fit of
// the units' measured times on their realized work, toward a seed of weak
// priors, no rate below zero.

#include "model/CommandCost.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace splash::engine {

enum class CostTerm : uint8_t {
  Units,              // one per command or pass: its turnaround
  ChunkLayers,        // chunks its rows fill times layers run
  RowLayers,          // rows times layers run
  AttentionPairs,     // query-key pairs over the attention layers run
  IndexPairs,         // rows times complete blocks a block index scores
  WeightReadBytes,    // weights read from the SSD on demand
  SpilledKvReadBytes, // spilled K/V read from the SSD
};
inline constexpr size_t kCostTerms = 7;
using CostVector = std::array<double, kCostTerms>;

// The share of a command's time its passes may spend on their turnaround:
// a unit may take up to turnaround / kPassOverheadShare before it is split
// further, so passes add about a tenth to a command (Scheduler).
inline constexpr double kPassOverheadShare = 0.1;

[[nodiscard]] constexpr CostVector costTerms(const model::CommandWork &work) noexcept {
  return {1.0,
          double(work.chunks) * work.layers,
          double(work.rows) * work.layers,
          double(work.attentionPairs),
          double(work.indexPairs),
          double(work.weightReadBytes),
          double(work.spilledKvReadBytes)};
}
// The term's quantity as /status names it.
[[nodiscard]] std::string_view costTermName(CostTerm term) noexcept;

struct CostModelStats final {
  CostVector rates{}, seedRates{};
  // Of the terms the model estimates before a unit runs (WeightReadBytes,
  // SpilledKvReadBytes), the realized bytes over the expected ones the
  // prices scale them by; 1 for the others.
  CostVector realizedPerExpected{};
  uint64_t samples = 0, clamped = 0;
  // |price - measured| / measured of the latest 256 units, the price of the
  // work planning expected, before each unit trained the fit.
  double relativeErrorP50 = 0.0, relativeErrorP95 = 0.0;
};

class CostModel final {
public:
  // What each unit's sums keep against the next command's: a command decays
  // them once, whatever its passes. 0.9 is DESIGN §4.3's value, not yet
  // derived: amend/S.md A10 chooses it by replaying recorded commands
  // (cost_model_test testDecayFitsRecordedCommands over V-tools' fixtures
  // prefill_commands_*.json, which do not exist yet).
  static constexpr double kDecay = 0.9;
  static constexpr size_t kErrorSamples = 256;
  // The second prefill warm-up's rows, beside the first's chunk: the second
  // size the seed puts its line through, as short as an agent's turn (300
  // rows in the recorded M6 commands) and an eighth of the chunk; on a
  // streamed target it costs about one command's fixed part at startup, ~2 s
  // (accepted, DESIGN §0 O4).
  static constexpr uint32_t kSeedRows = 256;

  CostModel() noexcept;

  // The startup measurements: the prefill warm-ups' samples (their work and
  // wall time) and the SSD reads of loading the weights. Both read terms
  // take the load's time per byte; the warm-ups, one chunk over every layer
  // in one pass each, less their reads at that rate, give the line through
  // them of RowLayers and, at no rows, ChunkLayers (Units for a model that
  // reports no layers); the other terms, Units among them, start at zero.
  // Each term's seed weighs as many units as measured it, at least one;
  // ChunkLayers' weighs on it and the warm-ups' Units together, whose
  // turnaround the line's constant holds. A warm-up that reports no work
  // measures nothing.
  void seed(std::span<const model::CommandSample> warmups, model::SsdReads load);
  // A unit's price, in milliseconds, for the work planning expects of it: the
  // model-estimated read bytes scaled by their realized share (stats), never
  // below zero.
  [[nodiscard]] double price(const model::CommandWork &expected) const noexcept;
  // The rate of Units: what every command or pass costs whatever its work.
  [[nodiscard]] double turnaround() const noexcept;
  // The part of a unit's price fewer rows do not shorten: its Units and
  // read terms.
  [[nodiscard]] double fixed(const model::CommandWork &expected) const noexcept;
  // A unit that ran: what it did and what planning expected of it, its
  // measured time (wall less image encodes), weighing `weight` of a command
  // (1 / its passes). The time counts within half and twice the price of its
  // realized work.
  void observe(const model::CommandWork &realized, const model::CommandWork &expected, double milliseconds,
               double weight);
  [[nodiscard]] CostModelStats stats() const noexcept;

private:
  [[nodiscard]] double rawPrice(const CostVector &terms) const noexcept;
  [[nodiscard]] double scale(CostTerm term) const noexcept;
  void solve();

  // Decayed sums of w·x̂·x̂ᵀ and w·x̂·t over the terms normalized by `unit_`,
  // each term's first nonzero quantity; a term without one keeps its seed
  // and stays out of the system.
  std::array<CostVector, kCostTerms> normal_{};
  CostVector moment_{}, unit_{}, seedRates_{}, precision_{}, rates_{};
  // The warm-ups' normalized Units per normalized ChunkLayers of theirs,
  // which ChunkLayers' seed weighs on with it (solve); 0 where the seed's
  // constant is Units'.
  double seedUnits_ = 0.0;
  // Decayed realized and expected bytes of the two estimated terms.
  double realizedWeight_ = 0.0, expectedWeight_ = 0.0;
  double realizedSpilled_ = 0.0, expectedSpilled_ = 0.0;
  std::array<double, kErrorSamples> errors_{};
  size_t errorCount_ = 0;
  uint64_t samples_ = 0, clamped_ = 0;
};

} // namespace splash::engine
