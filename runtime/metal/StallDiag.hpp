#pragma once

// Scratch diagnostics for the 5-s host-step stall: lines appended to
// /tmp/splash-stall-diag.log, wall-clock stamped, with the thread. Not for
// merging.

#include <pthread.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace splash::diag {

// What the host step on this thread is (set by its work), for the runner's
// lines.
inline thread_local const char *stepKind = "unlabeled";

inline double nowSeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

__attribute__((format(printf, 1, 2))) inline void log(const char *format, ...) {
  static std::mutex mutex;
  static FILE *file = std::fopen("/tmp/splash-stall-diag.log", "a");
  if (!file) return;
  const auto ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  uint64_t thread = 0;
  pthread_threadid_np(nullptr, &thread);
  const std::lock_guard lock(mutex);
  std::fprintf(file, "%lld.%03lld t%llu ", static_cast<long long>(ms / 1000), static_cast<long long>(ms % 1000),
               static_cast<unsigned long long>(thread));
  va_list args;
  va_start(args, format);
  std::vfprintf(file, format, args);
  va_end(args);
  std::fputc('\n', file);
  std::fflush(file);
}

} // namespace splash::diag
