#pragma once

// Scratch diagnostics for the CPU/GPU handoff of host steps (not for
// merging): counters the stress test reads, and lines appended to
// /tmp/splash-handoff-diag.log.

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace splash::handoff {

// Steps whose event the runner raised and re-read, and re-reads below the
// value it set (the event went back down).
inline std::atomic<uint64_t> checks{0}, regressions{0};

__attribute__((format(printf, 1, 2))) inline void log(const char *format, ...) {
  static std::mutex mutex;
  static FILE *file = std::fopen("/tmp/splash-handoff-diag.log", "a");
  if (!file) return;
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  const std::lock_guard lock(mutex);
  std::fprintf(file, "%lld.%03lld ", static_cast<long long>(ms / 1000), static_cast<long long>(ms % 1000));
  va_list args;
  va_start(args, format);
  std::vfprintf(file, format, args);
  va_end(args);
  std::fputc('\n', file);
  std::fflush(file);
}

} // namespace splash::handoff
