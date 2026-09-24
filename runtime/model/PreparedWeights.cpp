#include "model/PreparedWeights.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <system_error>
#include <vector>

namespace splash::model {
namespace {

[[noreturn]] void fail(const char *operation) {
  throw std::system_error(errno, std::generic_category(), operation);
}

class Descriptor final {
public:
  explicit Descriptor(int fd) : fd_(fd) { if (fd < 0) fail("open prepared weights"); }
  ~Descriptor() { close(fd_); }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  operator int() const noexcept { return fd_; }
private:
  int fd_;
};

class PreparationLock final {
public:
  PreparationLock(const std::filesystem::path &root, const PreparationCheck &check)
      : file_(open((root / "prepare.lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600)) {
    while (flock(file_, LOCK_EX | LOCK_NB) < 0) {
      if (errno != EINTR && errno != EWOULDBLOCK) fail("lock weight preparation");
      if (check) check();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
private:
  Descriptor file_;
};

std::string hex(const unsigned char *digest) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result;
  for (size_t i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    result += digits[digest[i] >> 4];
    result += digits[digest[i] & 15];
  }
  return result;
}

bool unchanged(const struct stat &a, const struct stat &b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
         a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec &&
         a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
         a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec &&
         a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec;
}

std::filesystem::path defaultCacheRoot() {
  if (const char *path = std::getenv("SPLASH_WEIGHT_CACHE"); path && *path) return path;
  const char *home = std::getenv("HOME");
  if (!home || !*home) throw std::runtime_error("cannot locate prepared weight cache");
  return std::filesystem::path(home) / "Library/Caches/Splash/weights";
}

std::string verificationKey(const struct stat &state) {
  std::ostringstream identity;
  identity << "splash-verified-file-v1 " << state.st_dev << ' ' << state.st_ino << ' ' << state.st_size << ' '
           << state.st_mtimespec.tv_sec << ' ' << state.st_mtimespec.tv_nsec << ' '
           << state.st_ctimespec.tv_sec << ' ' << state.st_ctimespec.tv_nsec << ' '
           << state.st_birthtimespec.tv_sec << ' ' << state.st_birthtimespec.tv_nsec;
  const auto value = identity.str();
  return weightDigest({reinterpret_cast<const uint8_t *>(value.data()), value.size()});
}

std::string verificationRecord(std::string_view key, std::string_view digest) {
  const std::string value = std::string(key) + std::string(digest);
  return std::string(digest) + weightDigest({reinterpret_cast<const uint8_t *>(value.data()), value.size()});
}

void rememberDigest(const std::filesystem::path &root, const struct stat &state, const std::string &digest) {
  const auto directory = root / "verified";
  std::filesystem::create_directories(directory);
  const auto key = verificationKey(state);
  std::string temporary = (directory / ".pending-XXXXXX").string();
  Descriptor file(mkstemp(temporary.data()));
  try {
    const auto record = verificationRecord(key, digest);
    writeWeightBytes(file, 0, {reinterpret_cast<const uint8_t *>(record.data()), record.size()});
    if (fsync(file)) fail("flush weight verification");
    std::filesystem::rename(temporary, directory / key);
  } catch (...) {
    unlink(temporary.c_str());
    throw;
  }
}

// The content hash is computed on first use. A proof is reusable only for the
// same inode, length, birth time, mtime and ctime. Replacing or writing even a
// same-size file invalidates it. No upstream file or extended attribute is
// modified. This avoids rereading two entire models at every warm startup.
std::string verifiedDigest(int fd, const std::filesystem::path &root, const PreparationCheck &check) {
  if (check) check();
  struct stat before{}, after{};
  if (fstat(fd, &before)) fail("stat verified weights");
  if (!S_ISREG(before.st_mode) || before.st_size < 0) throw std::runtime_error("weights must be a regular file");
  const auto key = verificationKey(before);
  const int existing = open((root / "verified" / key).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  std::string digest;
  if (existing >= 0) {
    Descriptor proof(existing);
    struct stat info{};
    if (!fstat(proof, &info) && S_ISREG(info.st_mode) && info.st_size == 128) {
      std::array<uint8_t, 128> bytes;
      readWeightBytes(proof, 0, bytes);
      const std::string record(bytes.begin(), bytes.end());
      const auto value = record.substr(0, 64);
      if (value.find_first_not_of("0123456789abcdef") == value.npos && verificationRecord(key, value) == record)
        digest = value;
    }
  } else if (errno != ENOENT) fail("open weight verification");
  const bool missing = digest.empty();
  if (missing) digest = weightFileDigest(fd, check);
  if (fstat(fd, &after)) fail("stat verified weights after read");
  if (!unchanged(before, after)) throw std::runtime_error("weight file changed during verification");
  if (missing) rememberDigest(root, before, digest);
  return digest;
}

bool complete(const std::filesystem::path &directory, uint64_t bytes,
              const PreparationCheck &check) {
  const int input = open((directory / "weights").c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (input < 0) {
    if (errno == ENOENT) return false;
    fail("open cached weights");
  }
  Descriptor file(input);
  struct stat state{};
  if (fstat(file, &state)) fail("stat cached weights");
  if (!S_ISREG(state.st_mode) || state.st_size < 0 || uint64_t(state.st_size) != bytes)
    return false;
  const int manifest = open((directory / "sha256").c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (manifest < 0) {
    if (errno == ENOENT) return false;
    fail("open cached weight digest");
  }
  Descriptor hashFile(manifest);
  if (fstat(hashFile, &state)) fail("stat cached weight digest");
  if (!S_ISREG(state.st_mode) || state.st_size != 64) return false;
  std::array<uint8_t, 64> digest{};
  readWeightBytes(hashFile, 0, digest);
  return verifiedDigest(file, directory.parent_path(), check) == std::string(digest.begin(), digest.end());
}

} // namespace

void readWeightBytes(int fd, uint64_t offset, std::span<uint8_t> bytes) {
  if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
      offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
    throw std::overflow_error("weight read offset overflow");
  while (!bytes.empty()) {
    const ssize_t count = pread(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fail("read weight bytes");
    if (!count) throw std::runtime_error("truncated weight file");
    offset += count;
    bytes = bytes.subspan(count);
  }
}

void writeWeightBytes(int fd, uint64_t offset, std::span<const uint8_t> bytes) {
  if (bytes.size() > uint64_t(std::numeric_limits<off_t>::max()) ||
      offset > uint64_t(std::numeric_limits<off_t>::max()) - bytes.size())
    throw std::overflow_error("weight write offset overflow");
  while (!bytes.empty()) {
    const ssize_t count = pwrite(fd, bytes.data(), bytes.size(), static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) fail("write prepared weights");
    if (!count) throw std::runtime_error("zero-length prepared weight write");
    offset += count;
    bytes = bytes.subspan(count);
  }
}

std::string weightDigest(std::span<const uint8_t> bytes) {
  if (bytes.size() > std::numeric_limits<CC_LONG>::max())
    throw std::overflow_error("weight identity is too large");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest);
  return hex(digest);
}

std::string weightFileDigest(int fd, const PreparationCheck &check) {
  struct stat before{}, after{};
  if (fstat(fd, &before)) fail("stat source weights");
  if (!S_ISREG(before.st_mode) || before.st_size < 0)
    throw std::runtime_error("weights must be a regular file");
  if (check) check();
  std::vector<uint8_t> buffer(1024 * 1024);
  CC_SHA256_CTX context;
  CC_SHA256_Init(&context);
  for (uint64_t at = 0; at < uint64_t(before.st_size); at += buffer.size()) {
    if (check) check();
    const auto part = std::span(buffer).first(std::min<uint64_t>(buffer.size(), before.st_size - at));
    readWeightBytes(fd, at, part);
    CC_SHA256_Update(&context, part.data(), static_cast<CC_LONG>(part.size()));
  }
  if (fstat(fd, &after)) fail("stat source weights after read");
  if (!unchanged(before, after)) throw std::runtime_error("weight file changed while reading");
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &context);
  return hex(digest);
}

struct WeightSource::Impl {
  std::filesystem::path path;
  Descriptor file;
  struct stat state{};
  std::string digest;
  Impl(const std::filesystem::path &path, const PreparationCheck &check)
      : path(path), file(open(path.c_str(), O_RDONLY | O_CLOEXEC)) {
    if (fstat(file, &state)) fail("stat weight source");
    digest = verifiedDigest(file, defaultCacheRoot(), check);
  }
};

WeightSource::WeightSource(const std::filesystem::path &path, const PreparationCheck &check)
    : impl_(std::make_unique<Impl>(path, check)) { checkUnchanged(); }
WeightSource::~WeightSource() = default;
const std::filesystem::path &WeightSource::path() const noexcept { return impl_->path; }
int WeightSource::descriptor() const noexcept { return impl_->file; }
const std::string &WeightSource::digest() const noexcept { return impl_->digest; }
void WeightSource::checkUnchanged() const {
  struct stat current{};
  if (fstat(impl_->file, &current)) fail("stat weight source");
  struct stat named{};
  if (stat(impl_->path.c_str(), &named)) fail("stat weight source path");
  if (!unchanged(impl_->state, current) || !unchanged(impl_->state, named))
    throw std::runtime_error("source weights changed during preparation; retry with an immutable source");
}

PreparedWeights::PreparedWeights(std::filesystem::path root) : root_(std::move(root)) {
  if (root_.empty()) root_ = defaultCacheRoot();
}

void requireWeightDiskSpace(uint64_t available, uint64_t required) {
  if (required && (available < kWeightCacheDiskReserve || required > available - kWeightCacheDiskReserve))
    throw std::runtime_error("not enough disk space to prepare weights: need " +
        std::to_string(required) + " bytes plus a 2 GiB free-space reserve");
}

void PreparedWeights::requireSpace(std::span<const PreparedWeight> weights,
                                   const PreparationCheck &check) const {
  const auto missingBytes = [&] {
    uint64_t missing = 0;
    for (const auto &weight : weights) {
      if (weight.key.size() != 64 || weight.key.find_first_not_of("0123456789abcdef") != weight.key.npos || !weight.bytes)
        throw std::invalid_argument("invalid prepared weight identity or size");
      if (check) check();
      if (complete(root_ / weight.key, weight.bytes, check)) continue;
      if (weight.bytes > std::numeric_limits<uint64_t>::max() - missing)
        throw std::overflow_error("prepared model size overflow");
      missing += weight.bytes;
    }
    return missing;
  };
  if (!missingBytes()) return;
  std::filesystem::create_directories(root_);
  PreparationLock lock(root_, check);
  // Reclaim abandoned writes before budgeting a retry. Live converters hold
  // the lock; complete generations are retained and excluded from the budget.
  for (const auto &weight : weights) std::filesystem::remove_all(root_ / (weight.key + ".partial"));
  requireWeightDiskSpace(std::filesystem::space(root_).available, missingBytes());
}

std::filesystem::path PreparedWeights::prepare(
    const PreparedWeight &weight, const std::function<void(int)> &write,
    const PreparationCheck &check, const PreparationCheck &prepareCheck) const {
  const auto &key = weight.key;
  const auto bytes = weight.bytes;
  if (key.size() != 64 || key.find_first_not_of("0123456789abcdef") != key.npos ||
      !bytes || bytes > uint64_t(std::numeric_limits<off_t>::max()))
    throw std::invalid_argument("invalid prepared weight identity or size");
  if (check) check();
  const auto destination = root_ / key;
  // Immutable hits need neither conversion admission nor the converter lock.
  if (complete(destination, bytes, check)) return destination / "weights";
  std::filesystem::create_directories(root_);
  // One converter per user cache: concurrent cold loads cannot multiply the
  // bounded conversion workspace. OS locks are released on crashes.
  PreparationLock lock(root_, check);
  if (complete(destination, bytes, check)) return destination / "weights";
  if (check) check();
  if (prepareCheck) prepareCheck();
  // This name belongs only to this key under the converter lock. An abandoned
  // staging directory is never a cache hit and is safe to replace.
  const auto staging = root_ / (std::string(key) + ".partial");
  std::filesystem::remove_all(staging);
  requireWeightDiskSpace(std::filesystem::space(root_).available, bytes);
  std::filesystem::create_directory(staging);
  const auto started = std::chrono::steady_clock::now();
  if (!weight.name.empty()) std::clog << "Preparing target weights: " << weight.name << std::endl;
  try {
    Descriptor file(open((staging / "weights").c_str(), O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC, 0600));
    // Do not let dirty filesystem pages grow into a hidden model-sized buffer.
    if (fcntl(file, F_NOCACHE, 1)) fail("set preparation uncached I/O");
    fstore_t allocation{};
    allocation.fst_flags = F_ALLOCATEALL;
    allocation.fst_posmode = F_PEOFPOSMODE;
    allocation.fst_length = static_cast<off_t>(bytes);
    if (fcntl(file, F_PREALLOCATE, &allocation)) fail("reserve prepared weight disk space");
    if (ftruncate(file, static_cast<off_t>(bytes))) fail("size prepared weights");
    write(file);
    if (check) check();
    struct stat state{};
    if (fstat(file, &state)) fail("stat prepared weights");
    if (state.st_size < 0 || uint64_t(state.st_size) != bytes)
      throw std::runtime_error("prepared weight size changed");
    const std::string digest = weightFileDigest(file, check);
    if (fchmod(file, 0400) || fsync(file)) fail("flush prepared weights");
    if (fstat(file, &state)) fail("stat completed weights");
    rememberDigest(root_, state, digest);
    Descriptor manifest(open((staging / "sha256").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0400));
    writeWeightBytes(manifest, 0, {reinterpret_cast<const uint8_t *>(digest.data()), digest.size()});
    if (fsync(manifest)) fail("flush prepared weight digest");
    if (!weight.source.empty()) {
      Descriptor origin(open((staging / "source").c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0400));
      const auto description = weight.source + "\n" + weight.name + "\n";
      writeWeightBytes(origin, 0, {reinterpret_cast<const uint8_t *>(description.data()), description.size()});
      if (fsync(origin)) fail("flush prepared weight source");
    }
    // Invalid cached generations may be replaced; existing read-only mappings
    // retain their inode. No valid generation is rewritten in place.
    std::filesystem::remove_all(destination);
    std::filesystem::rename(staging, destination);
    Descriptor directory(open(root_.c_str(), O_RDONLY | O_CLOEXEC));
    if (fsync(directory)) fail("flush prepared weight directory");
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove_all(staging, ignored);
    throw;
  }
  if (!weight.name.empty()) {
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::clog << "Prepared " << weight.name << " in " << seconds << " s" << std::endl;
  }
  return destination / "weights";
}

} // namespace splash::model
