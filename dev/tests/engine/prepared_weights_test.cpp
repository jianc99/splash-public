#include "model/PreparedWeights.hpp"

#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splash::model;

namespace {
void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F run, const char *message) {
  bool rejected = false;
  try { run(); } catch (const std::exception &) { rejected = true; }
  require(rejected, message);
}
std::string key(uint8_t value) { return weightDigest(std::span(&value, 1)); }
}

int main() {
  char path[] = "/tmp/splash-prepared-weights-XXXXXX";
  const char *directory = mkdtemp(path);
  if (!directory) return 1;
  const std::filesystem::path root(directory);
  setenv("SPLASH_WEIGHT_CACHE", root.c_str(), 1);
  try {
    PreparedWeights store(root);
    std::vector<uint8_t> bytes(128 * 1024);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 37);
    int builds = 0;
    const auto write = [&](int fd) { ++builds; writeWeightBytes(fd, 0, bytes); };
    const auto cached = store.prepare(key(1), bytes.size(), write);
    require(builds == 1 && std::filesystem::file_size(cached) == bytes.size(), "cold preparation");
    static_cast<void>(store.prepare(key(1), bytes.size(), write));
    require(builds == 1, "warm preparation rebuilt weights");
    require(weightDigest(bytes) == WeightSource(cached).digest(), "cached content differs");
    struct stat info{};
    require(stat(cached.c_str(), &info) == 0 && !(info.st_mode & 0222), "cache is writable");
    // Same-size corruption must not pass a metadata-only check.
    require(chmod(cached.c_str(), 0600) == 0, "chmod fixture");
    int fd = open(cached.c_str(), O_WRONLY);
    const uint8_t bad = 9;
    writeWeightBytes(fd, 0, std::span(&bad, 1));
    close(fd);
    static_cast<void>(store.prepare(key(1), bytes.size(), write));
    require(builds == 2 && WeightSource(cached).digest() == weightDigest(bytes), "corruption not repaired");
    // Interrupted and ENOSPC writes do not publish anything and can be retried.
    rejects([&] { static_cast<void>(store.prepare(key(2), bytes.size(), [&](int output) {
      writeWeightBytes(output, 0, std::span(bytes).first(64));
      throw std::system_error(ENOSPC, std::generic_category());
    })); }, "failed write accepted");
    require(!std::filesystem::exists(root / key(2)), "partial file published");
    static_cast<void>(store.prepare(key(2), bytes.size(), write));
    rejects([&] { static_cast<void>(store.prepare(key(3), bytes.size(), write, [] { throw std::runtime_error("pressure"); })); }, "pressure ignored");
    require(!std::filesystem::exists(root / key(3)), "pressure rejection published weights");
    rejects([&] { static_cast<void>(store.prepare("../outside", bytes.size(), write)); }, "unsafe cache key accepted");
    const pid_t crash = fork();
    if (crash == 0) {
      static_cast<void>(store.prepare(key(4), bytes.size(), [&](int output) {
        writeWeightBytes(output, 0, std::span(bytes).first(64));
        _exit(7);
      }));
      _exit(8);
    }
    require(crash > 0, "fork failed");
    int status = 0;
    waitpid(crash, &status, 0);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 7, "crash fixture failed");
    require(!std::filesystem::exists(root / key(4)), "crash published partial weights");
    static_cast<void>(store.prepare(key(4), bytes.size(), write));
    require(!std::filesystem::exists(root / (key(4) + ".partial")), "abandoned staging not cleaned");
    // Two processes requesting the same identity must run its writer only once.
    const auto counter = root / "builds";
    const auto competing = [&](int output) {
      const int count = open(counter.c_str(), O_CREAT | O_WRONLY | O_APPEND, 0600);
      require(count >= 0 && ::write(count, "x", 1) == 1, "write build counter");
      close(count);
      writeWeightBytes(output, 0, bytes);
    };
    const pid_t child = fork();
    if (child == 0) {
      try { static_cast<void>(store.prepare(key(5), bytes.size(), competing)); _exit(0); }
      catch (...) { _exit(1); }
    }
    require(child > 0, "fork competitor");
    static_cast<void>(store.prepare(key(5), bytes.size(), competing));
    waitpid(child, &status, 0);
    require(WIFEXITED(status) && WEXITSTATUS(status) == 0 && std::filesystem::file_size(counter) == 1,
            "concurrent cache miss rebuilt or corrupted weights");
    WeightSource source(cached);
    require(chmod(cached.c_str(), 0600) == 0, "chmod source");
    fd = open(cached.c_str(), O_WRONLY);
    writeWeightBytes(fd, 5, std::span(&bad, 1)); close(fd);
    rejects([&] { source.checkUnchanged(); }, "changed source was accepted");
    // An open descriptor surviving a rename must not hide path replacement.
    WeightSource beforeReplacement(cached);
    std::filesystem::rename(cached, root / "old-source");
    std::filesystem::copy_file(root / "old-source", cached);
    rejects([&] { beforeReplacement.checkUnchanged(); }, "replaced source was accepted");
    // Damaged memoization proofs must be recomputed, not trusted.
    for (const auto &entry : std::filesystem::directory_iterator(root / "verified")) {
      const int proof = open(entry.path().c_str(), O_WRONLY | O_TRUNC);
      require(proof >= 0, "open proof fixture");
      writeWeightBytes(proof, 0, std::span(&bad, 1));
      close(proof);
    }
    static_cast<void>(store.prepare(key(2), bytes.size(), write));
    require(builds == 4, "damaged proof forced an unnecessary rebuild");
    std::filesystem::remove_all(root);
    std::cout << "prepared weights: content, reuse, corruption, interruption, pressure and concurrency PASS\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    std::filesystem::remove_all(root);
    return 1;
  }
}
