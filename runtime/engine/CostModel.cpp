#include "engine/CostModel.hpp"
#include "SmallSolve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

constexpr size_t index(CostTerm term) noexcept { return static_cast<size_t>(term); }

constexpr size_t kUnits = index(CostTerm::Units);
constexpr size_t kChunkLayers = index(CostTerm::ChunkLayers);
constexpr size_t kRowLayers = index(CostTerm::RowLayers);
constexpr size_t kWeightReads = index(CostTerm::WeightReadBytes);
constexpr size_t kSpilledReads = index(CostTerm::SpilledKvReadBytes);

bool timed(double milliseconds) noexcept { return std::isfinite(milliseconds) && milliseconds > 0.0; }

} // namespace

std::string_view costTermName(CostTerm term) noexcept {
  switch (term) {
  case CostTerm::Units:
    return "units";
  case CostTerm::ChunkLayers:
    return "chunk_layers";
  case CostTerm::RowLayers:
    return "row_layers";
  case CostTerm::AttentionPairs:
    return "attention_pairs";
  case CostTerm::IndexPairs:
    return "index_pairs";
  case CostTerm::WeightReadBytes:
    return "weight_read_bytes";
  case CostTerm::SpilledKvReadBytes:
    return "spilled_kv_read_bytes";
  }
  return "";
}

CostModel::CostModel(double decodeGapMilliseconds, double decay) noexcept
    : resolution_(decodeGapMilliseconds * kPassOverheadShare), decay_(decay) {
  precision_.fill(1.0);
  unit_[kUnits] = 1.0;
}

void CostModel::seed(std::span<const model::CommandSample> warmups, model::SsdReads load) {
  // Both SSD terms read past the page cache, as loading does (F_NOCACHE): one
  // measurement, the load's, prices a byte of either.
  seedRates_ = {};
  precision_.fill(1.0);
  seedUnits_ = 0.0;
  const double perByte = load.bytes && timed(load.milliseconds) ? load.milliseconds / double(load.bytes) : 0.0;
  seedRates_[kWeightReads] = seedRates_[kSpilledReads] = perByte;
  // Each warm-up's time less its reads at that rate, against its rows'
  // layers: the line through them, kept from going below zero. The warm-ups
  // run a chunk over every layer in one pass, so what the line leaves at no
  // rows is a cost per chunk-layer as much as per pass: it seeds ChunkLayers,
  // which passes divide as they divide the layers, and Units, the turnaround
  // only commands of other passes measure, starts at zero. Seeded as Units it
  // would price every pass what a chunk's command costs at no rows (1633 ms
  // on the M6 profile, TimingProfile.cpp) and bound units at ten times that,
  // so no contended unit would run in passes before samples of several
  // passes; at zero the bound is G, the base's slice, until such samples
  // price the turnaround (testTheBoundFollowsTheTurnaround). The constant
  // holds the warm-ups' one turnaround too, so ChunkLayers' prior is the
  // constant less the turnaround (solve). Units of one pass over one chunk,
  // as a dense target runs every uncontended command, never tell the
  // turnaround from the chunk's cost and leave it at zero whatever they cost
  // (testWholeCommandsLeaveTheTurnaroundAtZero). A model that reports no
  // layers leaves the constant nowhere but Units.
  std::vector<std::pair<double, double>> points;
  double chunkLayers = std::numeric_limits<double>::infinity();
  for (const model::CommandSample &warmup : warmups) {
    const double milliseconds = warmup.wallMilliseconds - warmup.imageEncodeMilliseconds;
    if (!warmup.work.rows || !timed(milliseconds))
      continue;
    const CostVector terms = costTerms(warmup.work);
    for (size_t term = 0; term < kCostTerms; ++term)
      if (!unit_[term] && terms[term])
        unit_[term] = std::fabs(terms[term]);
    points.emplace_back(terms[kRowLayers],
                        milliseconds - perByte * (terms[kWeightReads] + terms[kSpilledReads]));
    chunkLayers = std::min(chunkLayers, terms[kChunkLayers]);
  }
  if (!points.empty()) {
    double meanX = 0.0, meanR = 0.0;
    for (const auto &[x, r] : points) {
      meanX += x / double(points.size());
      meanR += r / double(points.size());
    }
    double spread = 0.0, covariance = 0.0, squares = 0.0, products = 0.0;
    for (const auto &[x, r] : points) {
      spread += (x - meanX) * (x - meanX);
      covariance += (x - meanX) * (r - meanR);
      squares += x * x;
      products += x * r;
    }
    double slope = spread > 0.0 ? covariance / spread : 0.0;
    double intercept = meanR - slope * meanX;
    if (slope < 0.0) {
      slope = 0.0;
      intercept = meanR;
    }
    if (intercept < 0.0) {
      intercept = 0.0;
      slope = squares > 0.0 ? std::max(products / squares, 0.0) : 0.0;
    }
    const size_t constant = chunkLayers > 0.0 ? kChunkLayers : kUnits;
    seedRates_[constant] = constant == kChunkLayers ? intercept / chunkLayers : intercept;
    precision_[constant] = double(points.size());
    // Units' unit is 1.
    if (constant == kChunkLayers)
      seedUnits_ = unit_[kChunkLayers] / chunkLayers;
    seedRates_[kRowLayers] = slope;
    if (spread > 0.0)
      precision_[kRowLayers] = double(points.size());
  }
  normal_ = {};
  moment_ = {};
  sums_ = {};
  timeSquares_ = commands_ = times_ = 0.0;
  rates_ = seedRates_;
}

double CostModel::scale(CostTerm term) const noexcept {
  const bool weights = term == CostTerm::WeightReadBytes;
  const double expected = weights ? expectedWeight_ : expectedSpilled_;
  return expected > 0.0 ? (weights ? realizedWeight_ : realizedSpilled_) / expected : 1.0;
}

double CostModel::rawPrice(const CostVector &terms) const noexcept {
  double milliseconds = 0.0;
  for (size_t term = 0; term < kCostTerms; ++term)
    milliseconds += rates_[term] * terms[term];
  return std::max(milliseconds, 0.0);
}

double CostModel::price(const model::CommandWork &expected) const noexcept {
  CostVector terms = costTerms(expected);
  terms[kWeightReads] *= scale(CostTerm::WeightReadBytes);
  terms[kSpilledReads] *= scale(CostTerm::SpilledKvReadBytes);
  return rawPrice(terms);
}

double CostModel::turnaround() const noexcept { return rates_[kUnits]; }

double CostModel::weightReadRate() const noexcept { return rates_[kWeightReads]; }

double CostModel::fixed(const model::CommandWork &expected) const noexcept {
  return rates_[kUnits] +
         rates_[kWeightReads] * scale(CostTerm::WeightReadBytes) * double(expected.weightReadBytes) +
         rates_[kSpilledReads] * scale(CostTerm::SpilledKvReadBytes) * double(expected.spilledKvReadBytes);
}

void CostModel::observe(const model::CommandWork &realized, const model::CommandWork &expected, double milliseconds,
                        double weight) {
  if (!timed(milliseconds) || !(weight > 0.0))
    return;
  errors_[errorCount_++ % kErrorSamples] = std::fabs(price(expected) - milliseconds) / milliseconds;
  // A command's passes decay the sums once together, and their bytes sum to
  // the command's.
  const double decay = std::pow(decay_, weight);
  realizedWeight_ = decay * realizedWeight_ + double(realized.weightReadBytes);
  expectedWeight_ = decay * expectedWeight_ + double(expected.weightReadBytes);
  realizedSpilled_ = decay * realizedSpilled_ + double(realized.spilledKvReadBytes);
  expectedSpilled_ = decay * expectedSpilled_ + double(expected.spilledKvReadBytes);
  const CostVector terms = costTerms(realized);
  // A stall, or a unit far faster than its work, moves the fit no more than
  // a unit of half or twice its price.
  const double price = rawPrice(terms);
  double time = milliseconds;
  if (price > 0.0)
    time = std::clamp(milliseconds, 0.5 * price, 2.0 * price);
  clamped_ += time != milliseconds;
  // A pass is `weight` of its command: at the command's scale, its work and
  // time over that share, it weighs that share, so the passes of a command
  // tell the fit what the whole command would of the work they divide, and
  // their turnarounds, one each, what only passes measure.
  CostVector normalized{};
  for (size_t term = 0; term < kCostTerms; ++term) {
    if (!unit_[term] && terms[term])
      unit_[term] = std::fabs(terms[term]);
    normalized[term] = unit_[term] ? terms[term] / unit_[term] : 0.0;
  }
  for (size_t row = 0; row < kCostTerms; ++row) {
    for (size_t column = 0; column < kCostTerms; ++column)
      normal_[row][column] = decay * normal_[row][column] + normalized[row] * normalized[column] / weight;
    moment_[row] = decay * moment_[row] + normalized[row] * time / weight;
    sums_[row] = decay * sums_[row] + normalized[row];
  }
  timeSquares_ = decay * timeSquares_ + time * time / weight;
  commands_ = decay * commands_ + weight;
  times_ = decay * times_ + time;
  ++samples_;
  solve();
}

void CostModel::age(uint32_t commands) {
  const double decay = std::pow(decay_, commands);
  for (double *sum :
       {&realizedWeight_, &expectedWeight_, &realizedSpilled_, &expectedSpilled_, &timeSquares_, &commands_, &times_})
    *sum *= decay;
  for (CostVector &row : normal_)
    for (double &sum : row)
      sum *= decay;
  for (CostVector *sums : {&moment_, &sums_})
    for (double &sum : *sums)
      sum *= decay;
  solve();
}

void CostModel::solve() {
  // The turnaround u first, from what the units say of it beside every
  // measured term whatever their rates (identifiedTurnaround). Then
  // (N + K) β = m − N·u + K·ŝ over the other terms with a unit, N and m the
  // normal matrix and the moments, and again without the most negative
  // rate, held at zero, until none is negative. K is diag κ, ChunkLayers'
  // prior κ (β_c + s u − ŝ_c)², s = seedUnits_: the warm-ups' line at no
  // rows holds their turnaround.
  std::array<bool, kCostTerms> free{};
  for (size_t term = 0; term < kCostTerms; ++term)
    free[term] = term != kUnits && unit_[term] > 0.0;
  const double turnaround = identifiedTurnaround(free);
  CostVector beta{};
  for (size_t round = 0; round < kCostTerms; ++round) {
    std::array<CostVector, kCostTerms> system{};
    CostVector target{};
    for (size_t row = 0; row < kCostTerms; ++row) {
      if (!free[row]) {
        system[row][row] = 1.0;
        continue;
      }
      for (size_t column = 0; column < kCostTerms; ++column)
        if (free[column])
          system[row][column] = normal_[row][column];
      system[row][row] += precision_[row];
      target[row] = moment_[row] - normal_[row][kUnits] * turnaround + precision_[row] * seedRates_[row] * unit_[row];
    }
    if (free[kChunkLayers])
      target[kChunkLayers] -= precision_[kChunkLayers] * seedUnits_ * turnaround;
    beta = splash::solve<kCostTerms>(system, target);
    size_t negative = kCostTerms;
    for (size_t term = 0; term < kCostTerms; ++term)
      if (free[term] && beta[term] < 0.0 && (negative == kCostTerms || beta[term] < beta[negative]))
        negative = term;
    if (negative == kCostTerms)
      break;
    free[negative] = false;
  }
  for (size_t term = 0; term < kCostTerms; ++term) {
    if (free[term])
      rates_[term] = beta[term] / unit_[term];
    else
      rates_[term] = unit_[term] > 0.0 ? 0.0 : seedRates_[term];
  }
  // Units' unit is 1.
  rates_[kUnits] = turnaround;
}

double CostModel::identifiedTurnaround(const std::array<bool, kCostTerms> &measured) const noexcept {
  // Eliminates from N, m and the times' sum of squares, bordered by a
  // constant per command (its sums commands_, sums_ and times_), that
  // constant's column and then the measured terms', each that anything is
  // left of: Units' entries then hold S and e, what neither a constant per
  // command nor any combination of the work explains of Units' sum of
  // squares and of its products with the times, and the times' entry what
  // those leave of them. A command's Units are its passes, so S is the
  // spread of the pass counts beyond their mean and the work: units all in
  // one pass count, whatever their work, leave none.
  constexpr size_t kCommand = kCostTerms, kBordered = kCostTerms + 1;
  std::array<std::array<double, kBordered>, kBordered> rest{};
  std::array<double, kBordered> moment{};
  for (size_t row = 0; row < kCostTerms; ++row) {
    std::copy(normal_[row].begin(), normal_[row].end(), rest[row].begin());
    rest[row][kCommand] = rest[kCommand][row] = sums_[row];
    moment[row] = moment_[row];
  }
  rest[kCommand][kCommand] = commands_;
  moment[kCommand] = times_;
  double times = timeSquares_;
  size_t fitted = 0;
  std::array<bool, kBordered> left{};
  std::copy(measured.begin(), measured.end(), left.begin());
  left[kCommand] = true;
  for (size_t step = 0; step < kBordered; ++step) {
    const size_t pivot = step ? step - 1 : kCommand;
    if (!left[pivot] || !(rest[pivot][pivot] > 0.0))
      continue;
    left[pivot] = false;
    ++fitted;
    times -= moment[pivot] * moment[pivot] / rest[pivot][pivot];
    for (size_t row = 0; row < kBordered; ++row) {
      if (!left[row] && row != kUnits)
        continue;
      const double factor = rest[row][pivot] / rest[pivot][pivot];
      for (size_t column = 0; column < kBordered; ++column)
        if (left[column] || column == kUnits)
          rest[row][column] -= factor * rest[pivot][column];
      moment[row] -= factor * moment[pivot];
    }
  }
  // u = (e + κ ŝ) / (S + κ) beside its seed's weight κ: what other pass
  // counts over the same work say of it, units of one pass count nothing
  // (owner 11-03 00:10). It counts once σ sqrt(n / (S + κ)) is within
  // G * kPassOverheadShare, what moves the bound by G; until then it keeps
  // its seed and the bound stays G. σ² is the units' scatter about the work
  // and u per command (a command's passes weigh one in all) over the
  // commands less the rates fitted, the constant among them, and n the
  // commands: the standard error of one command's worth of the evidence,
  // S / n, not of all n commands'. What a linear price misses of the units,
  // reads hidden behind compute, is no noise that more commands average
  // away. Units in the passes their prices set, whose pass counts follow
  // their work, leave it in what their pass counts say beyond the work
  // however many run, and a standard error of all of them took it for u as
  // the decayed sums grew (kDecay 0.99 keeps about a hundred commands, 0.9
  // ten): 24 and 47 ms where the profile's is 10 after 8 and 23 commands,
  // after the M6 startup as planned
  // (testThePlannedStartupLeavesTheTurnaroundWithinTwiceItsTruth) and among
  // 80 prompts beside a decode (testVariedPromptsLeaveTheBoundAtG). Units of
  // other passes over the same work pin it down
  // (testOnlyOtherPassesOfTheWorkTellTheTurnaround); units of one pass count
  // and varied work, whose constant per command holds as much of what the
  // price misses as of u, do not
  // (testAMisfitOfOnePassCountDoesNotTellTheTurnaround). Fitted whatever its
  // error, u took that misfit, 25-115 ms among those 80 prompts and 62-72 ms
  // after that startup, and fitted beside the other terms 301 ms after the
  // M6 smoke's startup, 8192 rows in 12 and then 6 even passes beside
  // compute rates seeded at zero, where the hardware's was about 10 (M6-diag
  // B (2), testStartupUnitsOfOneShapeLeaveTheTurnaroundAtItsSeed). Every
  // measured term stays in S whether or not its rate is held at zero: a
  // sign held on a work term identifies no turnaround.
  const double units = rest[kUnits][kUnits], products = moment[kUnits];
  const double seed = seedRates_[kUnits], weight = precision_[kUnits];
  const double estimate = (products + weight * seed) / (units + weight);
  const double freedom = commands_ - double(fitted) - 1.0;
  if (!(freedom > 0.0))
    return seed;
  const double scatter =
      std::max(times - 2.0 * estimate * products + estimate * estimate * units, 0.0) / freedom;
  return scatter * commands_ <= resolution_ * resolution_ * (units + weight) ? std::max(estimate, 0.0) : seed;
}

CostModelStats CostModel::stats() const noexcept {
  CostModelStats result{.rates = rates_, .seedRates = seedRates_, .samples = samples_, .clamped = clamped_};
  result.realizedPerExpected.fill(1.0);
  result.realizedPerExpected[kWeightReads] = scale(CostTerm::WeightReadBytes);
  result.realizedPerExpected[kSpilledReads] = scale(CostTerm::SpilledKvReadBytes);
  const size_t count = std::min(errorCount_, kErrorSamples);
  if (count) {
    std::array<double, kErrorSamples> sorted = errors_;
    std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(count));
    const auto quantile = [&](double share) {
      return sorted[static_cast<size_t>(std::ceil(share * double(count))) - 1];
    };
    result.relativeErrorP50 = quantile(0.5);
    result.relativeErrorP95 = quantile(0.95);
  }
  return result;
}

} // namespace splash::engine
