#pragma once

// Internal to src/cuda. The CUDA side of a gpud::Buffer handle; the
// storage operations (Device::alloc/write/read) live in Buffer.cpp.

#include <gpud/Device.h>

namespace gpud::cuda {

struct BufferImpl final : ::gpud::Buffer::Impl {
    // TODO(impl): CUdeviceptr from cuMemAlloc. Freed by ~BufferImpl,
    // which the handle-lifetime rule guarantees runs before the Device
    // (and its context) dies.
};

} // namespace gpud::cuda
