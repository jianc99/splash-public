#pragma once

// Method replacements through which tests observe and steer what Metal
// objects do, and the recorder of command queues' residency sets built on
// one. Objective-C++ only.

#include "TestChecks.hpp"

#import <Metal/Metal.h>
#import <objc/runtime.h>

#include <vector>

namespace splash::test {

// Replaces a method of the class of `object`, every object's of that class,
// while it lives.
class MethodReplacement final {
public:
    MethodReplacement(id object, SEL selector, IMP replacement)
        : method_(class_getInstanceMethod(object_getClass(object), selector)) {
        require(method_ != nullptr, "replacement method is missing");
        original = method_setImplementation(method_, replacement);
    }
    ~MethodReplacement() { method_setImplementation(method_, original); }
    MethodReplacement(const MethodReplacement &) = delete;
    MethodReplacement &operator=(const MethodReplacement &) = delete;

    IMP original = nullptr;

private:
    Method method_ = nullptr;
};

// Records the residency sets added to command queues while it lives: Metal
// offers no way to read a queue's sets back. Arm it before the backend it
// observes.
class QueueSets final {
public:
    QueueSets() {
        original = replacement_.original;
        added.clear();
    }

    // Whether a set added holds the buffer labeled `label`.
    [[nodiscard]] bool holds(NSString *label) const {
        for (id<MTLResidencySet> set : added)
            for (id<MTLAllocation> allocation in set.allAllocations)
                if ([((id<MTLBuffer>)allocation).label isEqualToString:label]) return true;
        return false;
    }

    static inline std::vector<id<MTLResidencySet>> added;

private:
    static inline IMP original = nullptr;

    static void recordAdd(id queue, SEL selector, id<MTLResidencySet> set) {
        added.push_back(set);
        reinterpret_cast<void (*)(id, SEL, id)>(original)(queue, selector, set);
    }

    id<MTLCommandQueue> queue_ = [MTLCreateSystemDefaultDevice() newCommandQueue];
    MethodReplacement replacement_{queue_, @selector(addResidencySet:), reinterpret_cast<IMP>(recordAdd)};
};

} // namespace splash::test
