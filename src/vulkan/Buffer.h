#pragma once

// Internal to src/vulkan. The Vulkan side of a gpud::Buffer handle;
// the storage operations (Device::alloc/write/read) live in Buffer.cpp.

#include <gpud/Device.h>

namespace gpud::vulkan {

struct BufferImpl final : ::gpud::Buffer::Impl {
    // TODO(impl): VkBuffer + its memory (VkDeviceMemory or a VMA
    // allocation) + the buffer device address handed to kernels in the
    // push-constant blob. Released by ~BufferImpl, which the
    // handle-lifetime rule guarantees runs before the Device dies.
};

} // namespace gpud::vulkan
