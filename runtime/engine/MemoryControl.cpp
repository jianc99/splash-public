#include "engine/MemoryControl.hpp"

#include "StderrLine.hpp"

#include <optional>
#include <ostream>
#include <sstream>
#include <utility>

namespace splash::engine {

namespace {

uint64_t mebibytes(uint64_t bytes) noexcept { return (bytes + (1ULL << 19)) >> 20; }

// The funding since the last line that reported it: the starts funded, on
// what was released since the last start reported, and the lanes whose KV
// moved to the SSD.
void describeFunding(std::ostream &out, const SpillFundingSnapshot &now, const SpillFundingSnapshot &before) {
  const char *separator = "";
  if (const uint64_t starts = now.starts - before.starts) {
    const SpillFundingSnapshot::Released released{now.released.idle - before.released.idle,
                                                  now.released.cache - before.released.cache,
                                                  now.released.streamed - before.released.streamed};
    if (starts == 1)
      out << "request " << now.latestStart << " started";
    else
      out << starts << " requests started, the latest request " << now.latestStart << ',';
    out << " on " << mebibytes(released.total()) << " MiB Splash released (";
    const char *kind = "";
    for (const auto &[name, bytes] : {std::pair{"streamed", released.streamed}, std::pair{"cache", released.cache},
                                      std::pair{"idle", released.idle}}) {
      if (!bytes)
        continue;
      out << kind << name << ' ' << mebibytes(bytes);
      kind = ", ";
    }
    out << ')';
    separator = "; ";
  }
  if (const uint64_t conversions = now.conversions - before.conversions) {
    out << separator;
    if (conversions == 1)
      out << "request " << now.latestDonor << "'s KV moves to the SSD";
    else
      out << "the KV of " << conversions << " requests moves to the SSD, the latest request "
          << now.latestDonor << "'s";
  }
}

} // namespace

std::string MemoryStatusReporter::update(const ResourceWaitSnapshot &wait,
                                         bool hostGrowthAllowed) {
  const unsigned state = (!hostGrowthAllowed ? 1u : 0u) |
                         (wait.memory ? 2u : 0u) |
                         (wait.suspended ? 4u : 0u) |
                         (wait.draining ? 8u : 0u) |
                         (wait.heldBehindRefusal ? 16u : 0u);
  const bool funded = wait.funding.starts != funding_.starts || wait.funding.conversions != funding_.conversions;
  if (state == state_ && !funded)
    return {};
  std::ostringstream out;
  if (state != state_) {
    state_ = state;
    if (!state) {
      out << "Memory: growth available; resource wait cleared";
    } else {
      out << "Memory: growth " << (hostGrowthAllowed ? "available" : "paused")
          << "; waiting=" << wait.memory << "; suspended=" << wait.suspended;
      if (wait.heldBehindRefusal)
        out << "; held=" << wait.heldBehindRefusal;
      if (wait.draining)
        out << "; waiting for resident requests to finish";
    }
  } else {
    out << "Memory: growth " << (hostGrowthAllowed ? "available" : "paused");
  }
  if (funded) {
    out << "; ";
    describeFunding(out, wait.funding, funding_);
    // What was released stays to report with the next start.
    const SpillFundingSnapshot::Released released =
        wait.funding.starts != funding_.starts ? wait.funding.released : funding_.released;
    funding_ = wait.funding;
    funding_.released = released;
  }
  return out.str();
}

bool MemoryControl::run(MemoryPressure pressure) {
  loop_.releaseIdleWeights();
  governor_.setPressure(pressure);
  const double now = loop_.monotonicMilliseconds();
  // Requests held back by a refusal wait for memory too, the refused one
  // included while a pass defers it.
  const ResourceWaitSnapshot wait = loop_.resourceWaitSnapshot();
  const bool requestWaiting = wait.memory || wait.suspended || wait.heldBehindRefusal;
  governor_.setRequestWaiting(requestWaiting);
  static_cast<void>(backend_.refreshMemoryStats());
  const MemoryGovernorSnapshot memory = governor_.evaluate();
  const std::string diagnostic = reporter_.update(wait, memory.hostGrowthAllowed);
  if (!diagnostic.empty())
    logLine(diagnostic);
  const std::optional<MemoryReclaimDirective> directive =
      policy_.update(memory, now, requestWaiting);
  if (!directive)
    return false;
  const MemoryReclaimResult reclaim = loop_.reclaimMemory(*directive);
  policy_.reclaimed(*directive, reclaim);
  governor_.reclaimed(reclaim.outcome);
  static_cast<void>(backend_.refreshMemoryStats());
  // What transfers held back continues at the next command-free point.
  return reclaim.outcome == ReclaimOutcome::Pending;
}

} // namespace splash::engine
