#pragma once

#include <unistd.h>

#include <cerrno>
#include <ctime>
#include <sstream>
#include <string>
#include <string_view>

namespace splash::engine {

// The server and this runtime write to the same stderr. Each line goes out
// in one write, newline included, so that lines written at once stay whole.
inline void writeStderrLine(std::string_view text) noexcept {
  try {
    std::string line(text);
    line += '\n';
    for (std::string_view rest = line; !rest.empty();) {
      const ssize_t written = ::write(STDERR_FILENO, rest.data(), rest.size());
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        return;
      rest.remove_prefix(static_cast<size_t>(written));
    }
  } catch (...) {
    // Diagnostics must not affect startup or serving.
  }
}

template <typename... Parts>
void logKernelStartup(const Parts &...parts) noexcept {
  try {
    std::ostringstream text;
    (text << ... << parts);
    const std::string message = text.str();
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    char timestamp[9] = "--:--:--";
    if (localtime_r(&now, &local))
      std::strftime(timestamp, sizeof(timestamp), "%H:%M:%S", &local);
    std::ostringstream line;
    line << timestamp << ' ';
    // Native stderr is inherited by serve. Keep each optional startup notice
    // bounded and on one line, including messages from caught exceptions.
    for (unsigned char character : std::string_view(message).substr(0, 768))
      line << (character < 32 || character == 127 ? ' ' : char(character));
    if (message.size() > 768) line << "...";
    writeStderrLine(line.str());
  } catch (...) {
    // Optional diagnostics must not affect startup or serving.
  }
}

} // namespace splash::engine
