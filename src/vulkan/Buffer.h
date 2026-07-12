#pragma once

// Internal to src/vulkan. The Vulkan side of a gpud::Buffer handle;
// the storage operations (Device::alloc/write/read) live in Buffer.cpp.

#include <gpud/Device.h>

#include <volk.h>

namespace gpud::vulkan {

// Host-visible, host-coherent, persistently mapped storage buffer with
// a device address (the BDA push-constant ABI). Simple and — on unified
// memory (Apple Silicon, iGPUs) — fast; staging is a post-v1 concern.
struct BufferImpl final : ::gpud::Buffer::Impl {
    VkDevice device{};   // non-owning, for destruction
    VkBuffer buffer{};
    VkDeviceMemory memory{};
    void *mapped{};
    VkDeviceAddress address{};

    ~BufferImpl() override {
        // Runs before the Device dies (handle-lifetime rule). Freeing
        // the memory unmaps it implicitly.
        if (buffer) vkDestroyBuffer(device, buffer, nullptr);
        if (memory) vkFreeMemory(device, memory, nullptr);
    }
};

inline BufferImpl &impl_of(const ::gpud::Buffer &b) {
    return *static_cast<BufferImpl *>(b.impl());
}

} // namespace gpud::vulkan
