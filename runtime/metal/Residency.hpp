#pragma once

#include "AwakeClock.hpp"
#include "MetalBackend.hpp"

#import <Metal/Metal.h>
#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace splash::metal {

// Holds every buffer of a backend in one residency set, wired between the
// commands of its command queue. One request holds them only about two
// seconds, so a heartbeat requests residency every 500 ms. After keepAlive
// without a command, never when it is infinite, it ends residency, which Metal
// applies at its next GPU operation on any queue: one dispatch of the kick
// kernel, whose pipeline the backend builds with its library, on a queue of
// its own, as the runtime keeps exactly one command in flight on the command
// queue. Destruction ends residency without GPU work: a backend being torn
// down must not start any. The backend's set is used only on the heartbeat's
// serial queue.
//
// The command queue holds a set of its own, of the buffers kernels reach only
// through their GPU addresses (BufferReach::Addressed) while their owners say
// commands reach them (MetalBackend::setQueueResident): Metal makes it
// resident for every command buffer, as it does each buffer a dispatch binds,
// so each command buffer pays for all of it. Holding every buffer there would
// make each command buffer wait for all of them first: with memory nearly
// full, 1-2% of command buffers waited 100-170 ms before they started (M5 Max
// with 36 GB), so a command of many buffers waited at nearly every command.
// Changes to that set are staged (stage()) and made as a command begins
// (commitQueue()), while none of the queue's commands runs; a buffer that
// goes leaves it at once (unqueue()). The heartbeat never touches that set,
// so it is used under a mutex of its own on the threads that change it, and a
// command begins without waiting behind a residency request.
class Residency final {
public:
  // A one-thread kernel that writes one word of its buffer 0
  // (kernels/shared/residency.metal).
  static constexpr std::string_view kKickPipeline = "residency_kick";

  Residency(id<MTLDevice> device, id<MTLCommandQueue> commands,
            id<MTLComputePipelineState> kick, double keepAliveSeconds)
      : commands_(commands), kick_(kick), keepAlive_(keepAliveSeconds) {
    set_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                           error:nil];
    addressed_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new]
                                                 error:nil];
    kickQueue_ = [device newCommandQueue];
    kickTarget_ = [device newBufferWithLength:sizeof(uint32_t)
                                      options:MTLResourceStorageModePrivate];
    if (!set_ || !addressed_ || !kickQueue_ || !kickTarget_)
      throw MetalBackendError("unable to create the Metal residency set");
    [commands_ addResidencySet:addressed_];
    queue_ = dispatch_queue_create(
        "splash.metal.residency",
        dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM));
    heartbeat_ = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue_);
    dispatch_source_set_event_handler(heartbeat_, ^{ beat(); });
    dispatch_activate(heartbeat_);
  }

  ~Residency() {
    dispatch_source_cancel(heartbeat_);
    // Runs after any beat or request already queued: they use this object.
    // Metal applies the end at the process's next GPU operation, or frees
    // the set at exit.
    dispatch_sync(queue_, ^{
      [commands_ removeResidencySet:addressed_];
      std::lock_guard lock(mutex_);
      if (held_) [set_ endResidency];
    });
  }

  Residency(const Residency &) = delete;
  Residency &operator=(const Residency &) = delete;

  // Adds the buffer to the set and restarts the keep-alive: a held set wires
  // it at the commit, and a lapsed one is requested again off the caller's
  // thread (use()).
  void add(id<MTLBuffer> buffer) {
    dispatch_sync(queue_, ^{
      [set_ addAllocation:buffer];
      [set_ commit];
    });
    use();
  }

  void remove(id<MTLBuffer> buffer) {
    dispatch_sync(queue_, ^{
      [set_ removeAllocation:buffer];
      [set_ commit];
    });
  }

  // Stages whether the queue's set holds an Addressed buffer, as its owner
  // says commands reach it or not (MetalBackend::setQueueResident): the next
  // command's commitQueue() makes it so.
  void stage(id<MTLBuffer> buffer, bool resident) {
    std::lock_guard lock(queueMutex_);
    staged_[(__bridge const void *)buffer] = {buffer, resident};
  }

  // Makes the staged changes of the queue's set as a command begins, with one
  // commit when any changes it: none of the queue's commands runs then, so a
  // removal takes residency from none, and the command finds resident every
  // buffer staged in before it. A buffer staged in and out again since the
  // last command leaves the set as it was. The staged changes are a map of
  // their own, not the set's uncommitted ones: unqueue() commits the set at
  // once, which must not make a removal staged for the next command.
  void commitQueue() noexcept {
    std::lock_guard lock(queueMutex_);
    bool changed = false;
    for (const auto &[key, change] : staged_) {
      if (static_cast<bool>([addressed_ containsAllocation:change.buffer]) == change.resident)
        continue;
      if (change.resident)
        [addressed_ addAllocation:change.buffer];
      else
        [addressed_ removeAllocation:change.buffer];
      changed = true;
    }
    if (changed) [addressed_ commit];
    staged_.clear();
  }

  // Takes an Addressed buffer that goes out of the queue's set at once, with
  // whatever is staged for it: the sets retain the buffer, whose memory goes
  // with its last view, which no command still reaches.
  void unqueue(id<MTLBuffer> buffer) noexcept {
    std::lock_guard lock(queueMutex_);
    staged_.erase((__bridge const void *)buffer);
    @autoreleasepool {
      if ([addressed_ containsAllocation:buffer]) {
        [addressed_ removeAllocation:buffer];
        [addressed_ commit];
      }
    }
  }

  // Marks a command. A lapsed set is requested again at once, off the
  // caller's thread, and the heartbeat resumes.
  void use() {
    {
      std::lock_guard lock(mutex_);
      lastUse_ = AwakeClock::now();
      if (held_) return;
      held_ = true;
    }
    dispatch_async(queue_, ^{
      [set_ requestResidency];
      dispatch_source_set_timer(heartbeat_, dispatch_time(DISPATCH_TIME_NOW, kBeat),
                                kBeat, kBeat / 10);
    });
  }

private:
  static constexpr uint64_t kBeat = 500 * NSEC_PER_MSEC;

  // A staged change of the queue's set: the buffer joins it or leaves it.
  struct QueueChange final {
    __strong id<MTLBuffer> buffer = nil;
    bool resident = false;
  };

  void beat() {
    bool lapsed;
    {
      std::lock_guard lock(mutex_);
      lapsed = AwakeClock::now() - lastUse_ >= keepAlive_;
      held_ = !lapsed;
    }
    if (!lapsed) {
      [set_ requestResidency];
      return;
    }
    dispatch_source_set_timer(heartbeat_, DISPATCH_TIME_FOREVER, 0, 0);
    [set_ endResidency];
    kickResidencyEnd();
  }

  void kickResidencyEnd() {
    id<MTLCommandBuffer> command = [kickQueue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:kick_];
    [encoder setBuffer:kickTarget_ offset:0 atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command commit];
  }

  __strong id<MTLCommandQueue> commands_ = nil;
  __strong id<MTLComputePipelineState> kick_ = nil;
  __strong id<MTLResidencySet> set_ = nil;
  // The command queue's: the Addressed buffers commands reach.
  __strong id<MTLResidencySet> addressed_ = nil;
  __strong id<MTLCommandQueue> kickQueue_ = nil;
  __strong id<MTLBuffer> kickTarget_ = nil;
  __strong dispatch_queue_t queue_ = nil;
  __strong dispatch_source_t heartbeat_ = nil;
  const std::chrono::duration<double> keepAlive_;
  std::mutex mutex_;
  AwakeClock::time_point lastUse_;
  bool held_ = false;
  // Guards addressed_ and staged_.
  std::mutex queueMutex_;
  // The staged changes of the queue's set, by buffer; each retains its buffer
  // until it is made or the buffer goes.
  std::unordered_map<const void *, QueueChange> staged_;
};

} // namespace splash::metal
