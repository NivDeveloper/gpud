#pragma once

// Internal to src/metal. The Metal side of a gpud::Buffer handle;
// the storage operations (Device::alloc/write/read) live in Buffer.cpp
// (→ Buffer.mm when real).

#include <gpud/Device.h>

namespace gpud::metal {

struct BufferImpl final : ::gpud::Buffer::Impl {
    // TODO(impl): id<MTLBuffer> (shared or managed storage mode).
    // Released by ~BufferImpl, which the handle-lifetime rule
    // guarantees runs before the Device dies.
};

} // namespace gpud::metal
