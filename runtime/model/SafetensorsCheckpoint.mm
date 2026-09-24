#include "model/SafetensorsCheckpoint.hpp"
#include "model/WeightStore.hpp"

#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sys/stat.h>

namespace splash::model {
namespace {

uint64_t number(id value) {
  if (![value isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID())
    throw WeightStoreError("safetensors metadata requires an integer");
  const double real = [value doubleValue];
  // File offsets and shapes above 2^53 are neither usable on this backend nor
  // exactly represented by all JSON readers.
  if (!std::isfinite(real) || real < 0 || real > 9007199254740991.0 || std::floor(real) != real)
    throw WeightStoreError("invalid safetensors integer");
  return [value unsignedLongLongValue];
}

NSDictionary *object(NSData *data) {
  if (!data) throw WeightStoreError("unreadable safetensors JSON");
  id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
  if (![parsed isKindOfClass:[NSDictionary class]]) throw WeightStoreError("invalid safetensors JSON object");
  return parsed;
}

uint32_t elementBytes(const std::string &type) {
  if (type == "BF16" || type == "F16" || type == "I16" || type == "U16") return 2;
  if (type == "F32" || type == "I32" || type == "U32") return 4;
  if (type == "F64" || type == "I64" || type == "U64") return 8;
  if (type == "BOOL" || type == "I8" || type == "U8") return 1;
  throw WeightStoreError("unsupported safetensors dtype: " + type);
}

} // namespace

struct SafetensorsCheckpoint::Impl {
  std::vector<std::unique_ptr<WeightSource>> files;
  std::map<std::string, SourceTensor, std::less<>> tensors;
  NSDictionary *quantization = nil;
  NSDictionary *textConfig = nil;
};

SafetensorsCheckpoint::SafetensorsCheckpoint(const std::filesystem::path &directory, const PreparationCheck &check)
    : impl_(std::make_unique<Impl>()) {
  @autoreleasepool {
    if (check) check();
    const auto configPath = directory / "config.json";
    if (std::filesystem::file_size(configPath) > 1024 * 1024)
      throw WeightStoreError("source configuration exceeds metadata bound");
    NSData *configuration = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:configPath.c_str()]];
    if (!configuration) throw WeightStoreError("cannot read " + configPath.string());
    NSDictionary *config = object(configuration);
    id quantization = config[@"quantization"] ?: config[@"quantization_config"];
    if (quantization && ![quantization isKindOfClass:[NSDictionary class]])
      throw WeightStoreError("invalid source quantization configuration");
    impl_->quantization = quantization;
    impl_->textConfig = config[@"text_config"] ?: config;
    if (![impl_->textConfig isKindOfClass:[NSDictionary class]])
      throw WeightStoreError("source has no text model configuration");
    std::vector<std::filesystem::path> paths;
    for (const auto &entry : std::filesystem::directory_iterator(directory))
      if (entry.path().extension() == ".safetensors") {
        if (paths.size() == 1024) throw WeightStoreError("too many source shards");
        paths.push_back(entry.path());
      }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) throw WeightStoreError("source contains no safetensors weights");
    uint64_t metadataBytes = 0;
    for (const auto &path : paths) {
      @autoreleasepool {
        if (check) check();
        auto source = std::make_unique<WeightSource>(path, check);
        struct stat status{};
        if (fstat(source->descriptor(), &status) || status.st_size < 8)
          throw WeightStoreError("truncated safetensors file");
        uint64_t headerBytes = 0;
        readWeightBytes(source->descriptor(), 0, {reinterpret_cast<uint8_t *>(&headerBytes), 8});
        if (!headerBytes || headerBytes > 1024 * 1024 || headerBytes > uint64_t(status.st_size) - 8 ||
            (metadataBytes += headerBytes) > 4 * 1024 * 1024)
          throw WeightStoreError("safetensors header exceeds metadata or file bounds");
        NSMutableData *header = [NSMutableData dataWithLength:headerBytes];
        readWeightBytes(source->descriptor(), 8, {static_cast<uint8_t *>(header.mutableBytes), static_cast<size_t>(headerBytes)});
        NSDictionary *index = object(header);
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (id key in index) {
          if (![key isKindOfClass:[NSString class]] || [key lengthOfBytesUsingEncoding:NSUTF8StringEncoding] > 1024)
            throw WeightStoreError("invalid tensor name");
          if ([key isEqualToString:@"__metadata__"]) continue;
          NSDictionary *record = index[key];
          if (![record isKindOfClass:[NSDictionary class]] || ![record[@"dtype"] isKindOfClass:[NSString class]] ||
              ![record[@"shape"] isKindOfClass:[NSArray class]] || ![record[@"data_offsets"] isKindOfClass:[NSArray class]] ||
              [record[@"data_offsets"] count] != 2)
            throw WeightStoreError("invalid safetensors record");
          if ([record[@"shape"] count] > 8 || impl_->tensors.size() == 16384)
            throw WeightStoreError("source tensor metadata exceeds bounds");
          SourceTensor tensor;
          tensor.file = source.get();
          tensor.dtype = [record[@"dtype"] UTF8String];
          uint64_t bytes = elementBytes(tensor.dtype);
          for (id dimension in record[@"shape"]) {
            const uint64_t size = number(dimension);
            tensor.shape.push_back(size);
            bytes = checkedWeightMultiply(bytes, size, "source tensor size");
          }
          const uint64_t begin = number(record[@"data_offsets"][0]);
          const uint64_t end = number(record[@"data_offsets"][1]);
          if (end < begin || end - begin != bytes || end > uint64_t(status.st_size) - 8 - headerBytes)
            throw WeightStoreError("safetensors data range is invalid");
          tensor.offset = 8 + headerBytes + begin;
          tensor.bytes = bytes;
          tensor.dataOffset = 8 + headerBytes;
          const std::string name = [key UTF8String];
          if (!impl_->tensors.emplace(name, std::move(tensor)).second)
            throw WeightStoreError("duplicate source tensor: " + name);
          if (bytes) ranges.emplace_back(begin, end);
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i)
          if (ranges[i].first < ranges[i - 1].second) throw WeightStoreError("overlapping source tensors");
        source->checkUnchanged();
        impl_->files.push_back(std::move(source));
      }
    }
  }
}
SafetensorsCheckpoint::~SafetensorsCheckpoint() = default;

const SourceTensor *SafetensorsCheckpoint::find(std::string_view name) const noexcept {
  const auto found = impl_->tensors.find(name);
  return found == impl_->tensors.end() ? nullptr : &found->second;
}
const SourceTensor &SafetensorsCheckpoint::require(std::string_view name) const {
  const SourceTensor *tensor = find(name);
  if (!tensor) throw WeightStoreError("missing source tensor: " + std::string(name));
  return *tensor;
}
void SafetensorsCheckpoint::requireQuantization(std::string_view projection, uint32_t bits) const {
  @autoreleasepool {
    NSString *key = [[NSString alloc] initWithBytes:projection.data() length:projection.size() encoding:NSUTF8StringEncoding];
    NSDictionary *entry = impl_->quantization[key] ?: impl_->quantization;
    id mode = [entry isKindOfClass:[NSDictionary class]] ? entry[@"mode"] ?: impl_->quantization[@"mode"] : nil;
    if (![entry isKindOfClass:[NSDictionary class]] || number(entry[@"bits"] ?: impl_->quantization[@"bits"]) != bits ||
        number(entry[@"group_size"] ?: impl_->quantization[@"group_size"]) != 64 ||
        (mode && ![mode isEqual:@"affine"]))
      throw WeightStoreError("unsupported affine quantization for " + std::string(projection));
  }
}
namespace {
id configValue(NSDictionary *config, std::string_view key) {
  id value = config;
  while (!key.empty()) {
    const auto dot = key.find('.');
    const auto part = key.substr(0, dot);
    if (![value isKindOfClass:[NSDictionary class]]) return nil;
    NSString *name = [[NSString alloc] initWithBytes:part.data() length:part.size() encoding:NSUTF8StringEncoding];
    value = value[name];
    if (dot == key.npos) break;
    key.remove_prefix(dot + 1);
  }
  return value;
}
}
void SafetensorsCheckpoint::requireConfigNumber(std::string_view key, double expected) const {
  @autoreleasepool {
    id value = configValue(impl_->textConfig, key);
    if (![value isKindOfClass:[NSNumber class]] || [value doubleValue] != expected)
      throw WeightStoreError("source model configuration does not match: " + std::string(key));
  }
}
void SafetensorsCheckpoint::requireConfigString(std::string_view key, std::string_view expected) const {
  @autoreleasepool {
    id value = configValue(impl_->textConfig, key);
    if (![value isKindOfClass:[NSString class]] || std::string_view([value UTF8String]) != expected)
      throw WeightStoreError("source model configuration does not match: " + std::string(key));
  }
}
void SafetensorsCheckpoint::requireLayerTypes(uint32_t layers, uint32_t fullAttentionPeriod) const {
  @autoreleasepool {
    id types = impl_->textConfig[@"layer_types"];
    if (![types isKindOfClass:[NSArray class]] || [types count] != layers)
      throw WeightStoreError("source layer schedule does not match");
    for (uint32_t layer = 0; layer < layers; ++layer) {
      NSString *expected = (layer + 1) % fullAttentionPeriod ? @"linear_attention" : @"full_attention";
      if (![types[layer] isEqual:expected]) throw WeightStoreError("source layer schedule does not match");
    }
  }
}
void SafetensorsCheckpoint::checkUnchanged() const { for (const auto &source : impl_->files) source->checkUnchanged(); }
void SourceTensor::read(uint64_t at, std::span<uint8_t> destination) const {
  if (at > bytes || destination.size() > bytes - at) throw WeightStoreError("source tensor read is out of bounds");
  readWeightBytes(file->descriptor(), offset + at, destination);
}
void SourceTensor::identify(WeightIdentity &identity) const {
  identity.input(*file, dataOffset, offset, bytes, dtype, shape);
}

} // namespace splash::model
