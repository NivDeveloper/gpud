#pragma once

// Internal to src/vulkan. The Vulkan side of a gpud::Buffer handle;
// the storage operations (Device::alloc/write/read) live in Buffer.cpp.

#include <gpud/Device.h>

#include <volk.h>

#include <vk_mem_alloc.h>

#include <cstdint>

namespace gpud::vulkan {

class Device;   // ~BufferImpl defers its teardown to the owning Device

// Host-visible, host-coherent, persistently mapped storage buffer with
// a device address (the BDA push-constant ABI). Simple and — on unified
// memory (Apple Silicon, iGPUs) — fast; staging is a post-v1 concern.
// The device objects OUTLIVE the handle: a dead Buffer's VkBuffer goes
// back to the Device's pool and the next same-sized alloc() gets it —
// which is why a fresh buffer's contents are unspecified, and why a
// free-running producer never creates or destroys anything.
struct BufferImpl final : ::gpud::Buffer::Impl {
    Device *owner{};          // non-owning; the Device that allocated this
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void *mapped{};
    VkDeviceAddress address{};
    std::size_t size = 0;     // the VkBuffer's size: the pool's key

    // Ticket of the most recent run() that referenced this buffer; 0 =
    // never used by a dispatch. This is the "has the work touching this
    // memory completed?" key — for read()/write() hazards and for how
    // long the memory must outlive the handle.
    std::uint64_t last_use = 0;

    ~BufferImpl() override;   // Buffer.cpp
};

inline BufferImpl &impl_of(const ::gpud::Buffer &b) {
    return *static_cast<BufferImpl *>(b.impl());
}

} // namespace gpud::vulkan
