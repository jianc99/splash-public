#include "engine/MemoryGovernor.hpp"

#include "engine/Engine.hpp"
#include "engine/MemoryPlan.hpp"

#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace splash::engine {

namespace {

// What compressing the anonymous pages frees: the share the compressor does
// not keep at its ratio of held to own pages, at most half of them.
uint64_t compressionSavings(const HostMemoryPages &pages) noexcept {
  if (!pages.compressor || pages.compressed / 2 >= pages.compressor) return pages.anonymous / 2;
  if (pages.compressed <= pages.compressor) return 0;
  return pages.anonymous -
         static_cast<uint64_t>(static_cast<unsigned __int128>(pages.anonymous) * pages.compressor / pages.compressed);
}

// The host's spare memory above its spare reserve, beside `reservedBytes`
// being allocated: the room growth no request in service needs takes from.
uint64_t spareHeadroom(const HostMemory &host, uint64_t spareReserveBytes, uint64_t reservedBytes) noexcept {
  if (host.spare <= spareReserveBytes)
    return 0;
  const uint64_t spareAfterReserve = host.spare - spareReserveBytes;
  return reservedBytes < spareAfterReserve ? spareAfterReserve - reservedBytes : 0;
}

} // namespace

uint64_t hostBudgetBytes(const HostMemory &host, uint64_t spareReserveBytes, uint64_t engineBytes) noexcept {
  const uint64_t headroom = spareHeadroom(host, spareReserveBytes, 0);
  const uint64_t room = headroom > kHostWarningMarginBytes ? headroom - kHostWarningMarginBytes : 0;
  return room <= std::numeric_limits<uint64_t>::max() - engineBytes ? engineBytes + room
                                                                     : std::numeric_limits<uint64_t>::max();
}

HostMemory estimateHostMemory(const HostMemoryPages &statistics, uint64_t pageSize,
                              bool compression) noexcept {
  // free_count includes the speculative pages, which external_page_count
  // also counts; wired file pages are in neither.
  if (!pageSize || statistics.speculative > statistics.free) return {};
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t pages = statistics.free - statistics.speculative;
  for (uint64_t reclaimable : {statistics.fileBacked, statistics.purgeable}) {
    if (reclaimable > maximum - pages) return {};
    pages += reclaimable;
  }
  const uint64_t credit = compression ? compressionSavings(statistics) : 0;
  if (credit > maximum - pages || pages + credit > maximum / pageSize) return {};
  return {(pages + credit) * pageSize, (pages + credit / 2) * pageSize};
}

std::optional<HostMemory> queryHostMemory() noexcept {
  mach_port_t host = mach_host_self();
  vm_size_t pageSize = 0;
  vm_statistics64_data_t statistics{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t pageResult = host_page_size(host, &pageSize);
  kern_return_t statisticsResult = pageResult == KERN_SUCCESS
      ? host_statistics64(host, HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&statistics),
                          &count)
      : pageResult;
  mach_port_deallocate(mach_task_self(), host);
  if (statisticsResult != KERN_SUCCESS || !pageSize) return std::nullopt;

  return estimateHostMemory(
      {.free = statistics.free_count,
       .speculative = statistics.speculative_count,
       .fileBacked = statistics.external_page_count,
       .purgeable = statistics.purgeable_count,
       .anonymous = statistics.internal_page_count,
       .compressor = statistics.compressor_page_count,
       .compressed = statistics.total_uncompressed_pages_in_compressor},
      pageSize, querySystemMemoryPressure().value_or(MemoryPressure::Critical) != MemoryPressure::Critical);
}

std::optional<MemoryPressure> querySystemMemoryPressure() noexcept {
  uint32_t level = 0;
  size_t size = sizeof(level);
  if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &size,
                   nullptr, 0) != 0 || size != sizeof(level))
    return std::nullopt;
  switch (level) {
  case DISPATCH_MEMORYPRESSURE_NORMAL: return MemoryPressure::Normal;
  case DISPATCH_MEMORYPRESSURE_WARN: return MemoryPressure::Warning;
  case DISPATCH_MEMORYPRESSURE_CRITICAL: return MemoryPressure::Critical;
  default: return std::nullopt;
  }
}

MemoryGovernor::Reservation::Reservation(MemoryGovernor *owner, uint64_t bytes)
    : owner_(owner), bytes_(bytes) {}

MemoryGovernor::Reservation::~Reservation() { release(); }

MemoryGovernor::Reservation::Reservation(Reservation &&other) noexcept
    : owner_(other.owner_), bytes_(other.bytes_) {
  other.owner_ = nullptr;
  other.bytes_ = 0;
}

void MemoryGovernor::Reservation::release() noexcept {
  if (owner_ && bytes_)
    owner_->release(bytes_);
  owner_ = nullptr;
  bytes_ = 0;
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, uint64_t limitBytes,
                               uint64_t hostReserveBytes, uint64_t hostSpareReserveBytes,
                               HostMemoryProvider hostMemory, uint64_t untrackedReserveBytes, uint64_t hostHeldBytes)
    : backend_(backend), limitBytes_(limitBytes), hostReserveBytes_(hostReserveBytes),
      hostSpareReserveBytes_(hostSpareReserveBytes), hostMemory_(std::move(hostMemory)),
      untrackedReserveBytes_(untrackedReserveBytes), hostHeldBytes_(hostHeldBytes) {
  if (!limitBytes_) {
    throw std::invalid_argument("memory governor limit must be positive");
  }
  if (!hostReserveBytes_) {
    throw std::invalid_argument("host memory reserve must be positive");
  }
  if (!hostMemory_) {
    throw std::invalid_argument("host memory provider must be present");
  }
  if (chargedBytes(true) > limitBytes_) {
    throw metal::MetalAllocationError(
        "existing Metal allocations exceed memory governor limit",
        metal::AllocationFailure::EngineBudget);
  }
  const std::optional<HostMemory> host = hostMemory_();
  if (!host || host->available <= hostReserveBytes_) {
    throw metal::MetalAllocationError(
        "host available memory does not satisfy the system reserve",
        metal::AllocationFailure::HostPressure);
  }
}

MemoryGovernor::MemoryGovernor(metal::MetalBackend &backend, const EngineMemoryBreakdown &budget,
                               HostMemoryProvider hostMemory)
    : MemoryGovernor(backend, budget.hardBudgetBytes,
                     EngineMemoryPolicy::hostAvailableReserveBytes(budget.physicalMemoryBytes),
                     EngineMemoryPolicy::hostSpareReserveBytes(budget.physicalMemoryBytes), std::move(hostMemory),
                     budget.pipelineReserveBytes + budget.runtimeOverheadReserveBytes, budget.streamedTableBytes) {}

uint64_t
MemoryGovernor::chargedBytes(bool refreshDevice) const noexcept {
  metal::MetalMemoryStats memory = refreshDevice
      ? backend_.refreshMemoryStats()
      : backend_.memoryStats();
  // The backend's buffers are charged in full, the rest of the device's
  // footprint only where it exceeds the untracked reserve.
  const uint64_t limit = std::numeric_limits<uint64_t>::max();
  const uint64_t accounted = untrackedReserveBytes_ <= limit - memory.allocatedBytes
      ? memory.allocatedBytes + untrackedReserveBytes_
      : limit;
  const uint64_t device = std::max(accounted, memory.deviceCurrentAllocatedBytes);
  return hostHeldBytes_ <= limit - device ? device + hostHeldBytes_ : limit;
}

std::optional<HostMemory> MemoryGovernor::sampleHost() const noexcept {
  try {
    return hostMemory_();
  } catch (...) {
    return std::nullopt;
  }
}

uint64_t MemoryGovernor::hostHeadroomBytes(const std::optional<HostMemory> &host,
                                           uint64_t reservedBytes) const noexcept {
  return host ? spareHeadroom(*host, hostSpareReserveBytes_, reservedBytes) : 0;
}

std::optional<MemoryGovernor::Reservation>
MemoryGovernor::tryReserve(uint64_t bytes, bool optional, metal::AllocationFailure &failure) {
  if (!bytes) {
    throw std::invalid_argument("memory reservation must be positive");
  }
  uint64_t observed = chargedBytes(true);
  bool overflows =
      reservedBytes_ > std::numeric_limits<uint64_t>::max() - bytes;
  uint64_t requested =
      overflows ? std::numeric_limits<uint64_t>::max() : reservedBytes_ + bytes;
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  sampleHostRule(host);
  bool engineFits = !overflows && observed <= limitBytes_ &&
                    requested <= limitBytes_ - observed;
  // Inside a claim's scope: the room claims ranked above it hold is not its,
  // and the host admits it by its class.
  if (scope_) {
    const ClaimState &claim = ledger_.claim(scope_);
    const bool fits = engineFits && ledger_.heldBytes(scope_, growing_) <= limitBytes_ - observed - requested;
    const bool hostAdmits = hostRule_.admits(claim.spec.cls, bytes, claim.hostExempt, scopeCredit_);
    if (!fits || !hostAdmits) {
      failure = hostAdmits ? metal::AllocationFailure::EngineBudget : metal::AllocationFailure::HostPressure;
      if (deniedReservations_ != std::numeric_limits<uint64_t>::max())
        ++deniedReservations_;
      ledger_.refused(scope_, {.bytes = bytes, .hostRefused = !hostAdmits});
      return std::nullopt;
    }
    ledger_.setHostExempt(scope_, false);
    reservedBytes_ = requested;
    return Reservation(this, bytes);
  }
  const MemoryPressure pressure = effectivePressure(host_);
  // Growth leaves the warning margin free above the host's spare reserve,
  // stops while macOS warns and, while a request waits for memory, waits for
  // the recovery margin once the host has run short, unless a request in
  // service needs it (setServing). Optional growth leaves the recovery
  // margin free, whatever the mark, but takes from the host only the bytes
  // beyond those the engine returned to the system for it (setGrowthCredit):
  // with none beyond them it leaves the host's footprint where it was, and
  // neither a margin nor macOS's warning holds it back on a measured host.
  const bool serving = serving_ && !optional;
  const uint64_t hostRoom = hostHeadroomBytes(host, 0);
  const uint64_t margin = optional ? kHostRecoveryMarginBytes : kHostWarningMarginBytes;
  const uint64_t credited = optional ? std::min(bytes, growthCreditBytes_) : 0;
  const uint64_t hostRequested = requested - credited;
  const bool covered = host && credited == bytes;
  const bool hostRoomFits =
      covered || (host && hostRequested <= hostRoom && hostRoom - hostRequested >= margin);
  // A request that only the host headroom refuses waits for host memory
  // while the idle headroom may still clear the margin. Hold host pressure:
  // once a control pass finds the request waiting, the paced reclaim frees
  // toward the recovery margin for it. Optional growth that the host
  // refuses is done without, and asks no reclaim.
  if (engineFits && !serving && !optional && !hostRoomFits)
    host_.constrained = true;
  const bool hostRefuses =
      pressure == MemoryPressure::Critical ||
      (!serving && ((systemPressure_ == MemoryPressure::Warning && !covered) || !hostRoomFits ||
                    growthHeld(host_)));
  if (hostRefuses || !engineFits) {
    // The host's refusal lifts with its pressure, the limit's only once
    // memory is freed: a refusal they share is the host's.
    failure = hostRefuses ? metal::AllocationFailure::HostPressure
                          : metal::AllocationFailure::EngineBudget;
    if (deniedReservations_ != std::numeric_limits<uint64_t>::max()) {
      ++deniedReservations_;
    }
    return std::nullopt;
  }
  reservedBytes_ = requested;
  return Reservation(this, bytes);
}

metal::AllocationAdmission MemoryGovernor::allocationAdmission() noexcept {
  return admission(false);
}

metal::AllocationAdmission MemoryGovernor::growthAdmission() noexcept {
  return admission(true);
}

metal::AllocationAdmission MemoryGovernor::admission(bool optional) noexcept {
  return [this, optional](uint64_t bytes, const std::function<void()> &allocate)
             -> metal::AllocationResult {
    metal::AllocationFailure failure;
    auto reservation = tryReserve(bytes, optional, failure);
    if (!reservation)
      return failure;
    try {
      allocate();
    } catch (const metal::MetalAllocationError &error) {
      // Host headroom is an estimate; the driver can still deny the allocation.
      return error.failure();
    }
    return {};
  };
}

bool MemoryGovernor::setServing(bool serving) noexcept {
  return std::exchange(serving_, serving);
}

uint64_t MemoryGovernor::setGrowthCredit(uint64_t bytes) noexcept {
  return std::exchange(growthCreditBytes_, bytes);
}

void MemoryGovernor::setRequestWaiting(bool waiting) noexcept {
  requestWaiting_ = waiting;
}

void MemoryGovernor::setPressure(MemoryPressure pressure) noexcept {
  systemPressure_ = pressure;
}

void MemoryGovernor::reclaimed(ReclaimOutcome outcome) noexcept {
  if (outcome == ReclaimOutcome::Untargeted)
    return;
  host_.reclaimExhausted = host_.constrained && outcome == ReclaimOutcome::Exhausted;
}

MemoryGovernorSnapshot MemoryGovernor::snapshot() const noexcept {
  const std::optional<HostMemory> host = sampleHost();
  return snapshot(host, hostState(host));
}

MemoryGovernorSnapshot MemoryGovernor::evaluate() noexcept {
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(host);
  return snapshot(host, host_);
}

MemoryGovernorSnapshot
MemoryGovernor::snapshot(const std::optional<HostMemory> &hostMemory,
                         const HostState &host) const noexcept {
  uint64_t observed = chargedBytes();
  uint64_t used = observed;
  if (reservedBytes_ <= std::numeric_limits<uint64_t>::max() - used) {
    used += reservedBytes_;
  } else {
    used = std::numeric_limits<uint64_t>::max();
  }
  uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, reservedBytes_);
  const MemoryPressure pressure = effectivePressure(host);
  bool hostGrowthAllowed = systemPressure_ == MemoryPressure::Normal &&
      !growthHeld(host) && hostHeadroom >= kHostWarningMarginBytes;
  MemoryGovernorSnapshot snapshot{
      limitBytes_,
      observed,
      used < limitBytes_ ? limitBytes_ - used : 0,
      pressure,
      deniedReservations_,
      hostMemory.has_value(),
      hostMemory ? hostMemory->available : 0,
      hostReserveBytes_,
      hostMemory ? hostMemory->spare : 0,
      hostSpareReserveBytes_,
      hostHeadroom,
      systemPressure_,
      hostGrowthAllowed,
  };
  ledger_.fill(snapshot, observed);
  return snapshot;
}

MemoryGovernor::HostState
MemoryGovernor::hostState(const std::optional<HostMemory> &hostMemory) const noexcept {
  const uint64_t hostHeadroom = hostHeadroomBytes(hostMemory, reservedBytes_);
  HostState next;
  next.constrained = !hostMemory ||
      hostHeadroom < kHostWarningMarginBytes ||
      (host_.constrained && hostHeadroom < kHostRecoveryMarginBytes);
  next.reclaimExhausted = host_.reclaimExhausted && next.constrained;
  return next;
}

MemoryPressure
MemoryGovernor::effectivePressure(const HostState &host) const noexcept {
  // Availability and the system's warning control growth and paced reclaim.
  // Only the system's critical signal warrants dropping every evictable cache
  // entry.
  if (systemPressure_ == MemoryPressure::Critical)
    return MemoryPressure::Critical;
  if (host.constrained || systemPressure_ == MemoryPressure::Warning)
    return MemoryPressure::Warning;
  return MemoryPressure::Normal;
}

void MemoryGovernor::release(uint64_t bytes) noexcept {
  if (bytes > reservedBytes_) {
    reservedBytes_ = 0;
    return;
  }
  reservedBytes_ -= bytes;
}

MemoryGovernor::Scope::Scope(MemoryGovernor &governor, ClaimId claim, std::optional<AccountId> growing,
                             uint64_t hostCreditBytes) noexcept
    : governor_(governor), previousClaim_(std::exchange(governor.scope_, claim)),
      previousGrowing_(std::exchange(governor.growing_, growing)),
      previousCredit_(std::exchange(governor.scopeCredit_, hostCreditBytes)) {}

MemoryGovernor::Scope::~Scope() {
  governor_.scope_ = previousClaim_;
  governor_.growing_ = previousGrowing_;
  governor_.scopeCredit_ = previousCredit_;
}

void MemoryGovernor::HostRule::sample(const std::optional<HostMemory> &host, uint64_t spareReserve, uint64_t inFlight,
                                      MemoryPressure system, uint64_t hysteresisBytes, bool reclaimShort) noexcept {
  measured_ = host.has_value();
  headroom_ = host ? spareHeadroom(*host, spareReserve, inFlight) : 0;
  system_ = system;
  if (!measured_ || headroom_ < kHostMarginBytes)
    constrained_ = true;
  else if (reclaimShort || headroom_ - kHostMarginBytes >= hysteresisBytes)
    constrained_ = false;
}

bool MemoryGovernor::HostRule::admits(ClaimClass cls, uint64_t bytes, bool hostExempt,
                                      uint64_t creditBytes) const noexcept {
  if (system_ == MemoryPressure::Critical)
    return false;
  if (cls == ClaimClass::Floor || (cls == ClaimClass::InService && hostExempt) ||
      (cls == ClaimClass::Growth && bytes && bytes <= creditBytes))
    return true;
  const bool keepsMargin = measured_ && headroom_ >= bytes && headroom_ - bytes >= kHostMarginBytes;
  if (cls == ClaimClass::InService)
    return keepsMargin && system_ == MemoryPressure::Normal;
  return keepsMargin && system_ == MemoryPressure::Normal && !constrained_;
}

uint64_t MemoryGovernor::HostRule::deficitBytes(uint64_t refusedWaitingBytes, uint64_t returningBytes) const noexcept {
  // Missing telemetry holds elective work back but is no evidence that
  // memory must be given back.
  if (!measured_)
    return 0;
  const uint64_t wanted = refusedWaitingBytes <= std::numeric_limits<uint64_t>::max() - kHostMarginBytes
                              ? kHostMarginBytes + refusedWaitingBytes
                              : std::numeric_limits<uint64_t>::max();
  const uint64_t lacking = wanted > headroom_ ? wanted - headroom_ : 0;
  return lacking > returningBytes ? lacking - returningBytes : 0;
}

void MemoryGovernor::sampleHostRule(const std::optional<HostMemory> &host) noexcept {
  hostRule_.sample(host, hostSpareReserveBytes_, reservedBytes_, systemPressure_, growthUnitBytes_, hostPassShort_);
  // A waiver lasts the band it ended: the next band starts unwaived.
  if (!hostRule_.constrained())
    hostPassShort_ = false;
}

uint64_t MemoryGovernor::roomBytes() const noexcept {
  const uint64_t used = chargedBytes();
  return used <= limitBytes_ && reservedBytes_ <= limitBytes_ - used ? limitBytes_ - used - reservedBytes_ : 0;
}

Need MemoryGovernor::need(ClaimId claim, const std::optional<Need> &inPlace) const {
  Need need = ledger_.claim(claim).need;
  if (inPlace) {
    need.pages = inPlace->pages;
    need.form = inPlace->form;
    need.stateBuffers = inPlace->stateBuffers;
  }
  return need;
}

FreeResult MemoryGovernor::free(ClaimId claim, const Need &need) {
  FreeResult result = ledger_.free(claim, need, roomBytes(), hostRule_.level() == MemoryPressure::Critical, false);
  const ClaimClass cls = ledger_.claim(claim).spec.cls;
  if (need.hostRefused && result.outcome == FreeOutcome::Short &&
      (cls == ClaimClass::Floor || cls == ClaimClass::InService)) {
    ledger_.setHostExempt(claim, true);
    result.outcome = FreeOutcome::Met;
  }
  return result;
}

uint64_t MemoryGovernor::available(const ClaimSpec &spec, uint64_t enough) const {
  return ledger_.available(spec, enough, roomBytes(), hostRule_.level() == MemoryPressure::Critical, false);
}

bool MemoryGovernor::admits(ClaimClass cls) const noexcept { return hostRule_.admits(cls, 0, false, 0); }

bool MemoryGovernor::copyDemand() const noexcept { return ledger_.copyDemand(); }

bool MemoryGovernor::claimWaiting() const noexcept { return ledger_.claimWaiting(); }

// The host's claim of what the pass finds, up to `target`, or of what lands:
// free() gives back the units that cost nothing and are sized only when
// taken (the idle state buffers) first, whatever it finds besides. A pass
// that freed nothing now or later is Short: it found nothing to take.
FreeResult MemoryGovernor::hostPass(uint64_t target, bool critical, bool deficit) {
  const ClaimSpec spec{.cls = ClaimClass::Host,
                       .reach = critical ? ReclaimClass::InUse : ReclaimClass::Ordinary,
                       .holdSeconds = kValueHorizonSeconds};
  const uint64_t found = ledger_.available(spec, target, 0, critical, deficit);
  const uint64_t landing = ledger_.returning().landingBytes;
  const ClaimId claim = ledger_.open(spec);
  FreeResult result = ledger_.free(claim, {.bytes = std::min(target, std::max(found, landing))}, 0, critical, deficit);
  ledger_.close(claim);
  if (result.outcome == FreeOutcome::Met && !result.nowBytes && !result.laterBytes)
    result.outcome = FreeOutcome::Short;
  return result;
}

GrowthHost MemoryGovernor::growthHost(bool displaces) {
  return {.admits = [this](uint64_t bytes) { return hostRule_.admits(ClaimClass::Growth, bytes, false, 0); },
          .displaces = displaces,
          .grow = [this](ClaimId claim, AccountId account, model::ElasticPool &pool, uint64_t credit) {
            const Scope scope(*this, claim, account, credit);
            return pool.grow(allocationAdmission());
          }};
}

bool MemoryGovernor::controlPass(double nowMilliseconds) {
  const std::optional<HostMemory> host = sampleHost();
  host_ = hostState(host);
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(host);
  std::optional<FreeResult> pass;
  if (hostRule_.level() == MemoryPressure::Critical) {
    ledger_.dropToFloors();
    pass = hostPass(std::numeric_limits<uint64_t>::max(), true, true);
  } else {
    const uint64_t deficit =
        hostRule_.deficitBytes(ledger_.hostRefusedWaitingBytes(), ledger_.returning().landingBytes);
    if (nowMilliseconds >= nextHostPassMilliseconds_) {
      if (hostRule_.level() == MemoryPressure::Warning)
        pass = hostPass(kHostMarginBytes, false, deficit != 0);
      else if (hostRule_.constrained() && deficit)
        pass = hostPass(std::min(kHostMarginBytes, deficit), false, true);
    }
    ledger_.grow(roomBytes(), growthHost(true));
  }
  if (pass) {
    // Only a pass while the host is short can waive the band.
    if (hostRule_.constrained())
      hostPassShort_ = pass->outcome == FreeOutcome::Short;
    if (pass->nowBytes || pass->laterBytes)
      nextHostPassMilliseconds_ = nowMilliseconds + kHostSettleMilliseconds;
  }
  return pass && pass->outcome == FreeOutcome::Pending;
}

void MemoryGovernor::growToTargets(double) {
  growthUnitBytes_ = ledger_.largestGrowthUnit();
  sampleHostRule(sampleHost());
  ledger_.grow(roomBytes(), growthHost(false));
}

std::optional<MemoryReclaimDirective> MemoryPressurePolicy::update(
    const MemoryGovernorSnapshot &snapshot, double nowMilliseconds,
    bool requestWaiting) noexcept {
  if (snapshot.pressure == MemoryPressure::Normal) {
    nextReclaimMilliseconds_ = 0.0;
    return std::nullopt;
  }
  if (snapshot.pressure == MemoryPressure::Critical)
    return MemoryReclaimDirective{.critical = true, .hostShort = true};
  if (nowMilliseconds < nextReclaimMilliseconds_)
    return continued_.value_or(MemoryReclaimDirective{});
  continued_.reset();

  // Missing telemetry pauses allocation, but is not evidence that live
  // cache must be discarded. Empty extents can still be returned.
  if (!snapshot.hostMeasurementValid &&
      snapshot.systemPressure == MemoryPressure::Normal)
    return MemoryReclaimDirective{};

  // A pass a request waits for frees toward the recovery margin, where growth
  // reopens (hostState). One that nothing waits for only restores the warning
  // margin that growth leaves free: freeing on to the recovery margin would
  // discard cache nothing needs, on every pass while the host's idle headroom
  // stays below it. Such a pass also keeps the resume point, whose loss costs
  // the next request a replay of its whole prompt; a waiting request outranks
  // it. Nor does such a pass take what is in use: the states of the requests
  // in service and the KV they restore through. The replay point it would
  // take is the next resume point, and what it would free goes back to those
  // requests' growth, which neither the margin nor macOS's warning holds
  // back. macOS's warning asks caches to give memory back, whatever the
  // headroom: while it lasts, every pass returns as much as a pass may,
  // still short of what is in use unless a request waits, and keeps the
  // resume point even for a waiting request, whose growth the warning
  // refuses as well, so that what the pass frees would not reach it. Only a
  // measured deficit has the pass take the expert store above its floor: the
  // warning alone asks caches, and the experts a smaller store evicts are
  // read from the SSD again at every step that routes them.
  const bool warning = snapshot.systemPressure == MemoryPressure::Warning;
  const uint64_t margin = requestWaiting ? kHostRecoveryMarginBytes : kHostWarningMarginBytes;
  const uint64_t deficit =
      snapshot.hostHeadroomBytes < margin ? margin - snapshot.hostHeadroomBytes : 0;
  const bool hostShort = snapshot.hostMeasurementValid && deficit != 0;
  const uint64_t desired = warning ? kHostWarningMarginBytes : deficit;
  // The host samples every 500 ms. Allow counters to settle after a pass
  // with a target, but keep responding if another application continues
  // consuming memory. A pass with none leaves nothing to settle: a request
  // that starts waiting gets its pass at once.
  if (desired)
    nextReclaimMilliseconds_ = nowMilliseconds + 1000.0;
  return MemoryReclaimDirective{
      .targetBytes = std::min(desired, kHostWarningMarginBytes),
      .keepResumePoint = !requestWaiting || warning,
      .upTo = requestWaiting ? ReclaimClass::InUse : ReclaimClass::Ordinary,
      .hostShort = hostShort};
}

void MemoryPressurePolicy::reclaimed(const MemoryReclaimDirective &directive,
                                     const MemoryReclaimResult &result) noexcept {
  continued_.reset();
  // Every critical pass evicts everything again by itself.
  if (result.outcome != ReclaimOutcome::Pending || directive.critical)
    return;
  continued_ = directive;
  continued_->targetBytes -= std::min(result.releasedBytes, directive.targetBytes);
}

void connectToGovernor(EngineConfig &config, MemoryGovernor &governor) {
  config.governor = &governor;
  config.growthPaused = [&governor] {
    return !governor.snapshot().hostGrowthAllowed;
  };
  config.serving = [&governor](bool serving) { return governor.setServing(serving); };
}

} // namespace splash::engine
