#include "Buffer.h"
#include "Device.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

namespace gpud::vulkan {

BufferImpl::~BufferImpl() {
    // The handle dies here, but the memory must not while queued work
    // can still reference it — hand it to the Device, which releases it
    // once last_use has completed (or immediately, if it already has).
    // Both destroy calls tolerate a null handle, so a half-built impl
    // from a throwing alloc() is fine.
    if (!owner || (!buffer && !memory)) return;
    owner->defer_release(last_use,
                         [dev = device, buf = buffer, mem = memory] {
                             vkDestroyBuffer(dev, buf, nullptr);
                             vkFreeMemory(dev, mem, nullptr);
                         });
}

Buffer Device::alloc(std::size_t bytes) {
    reclaim();   // hand back whatever the GPU has finished with

    auto impl = std::make_unique<BufferImpl>();
    impl->owner = this;
    impl->device = s.device;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes ? bytes : 4;   // Vulkan forbids zero-size buffers
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(s.device, &bci, nullptr, &impl->buffer),
          "vkCreateBuffer");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(s.device, impl->buffer, &req);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    std::uint32_t type = s.memory.memoryTypeCount;
    for (std::uint32_t i = 0; i < s.memory.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (s.memory.memoryTypes[i].propertyFlags & want) == want) {
            type = i;
            break;
        }
    if (type == s.memory.memoryTypeCount)
        throw std::runtime_error("gpud/vulkan: no host-visible memory type");

    VkMemoryAllocateFlagsInfo mfi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    mfi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &mfi;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    check(vkAllocateMemory(s.device, &mai, nullptr, &impl->memory),
          "vkAllocateMemory");
    check(vkBindBufferMemory(s.device, impl->buffer, impl->memory, 0),
          "vkBindBufferMemory");
    check(vkMapMemory(s.device, impl->memory, 0, VK_WHOLE_SIZE, 0,
                      &impl->mapped),
          "vkMapMemory");
    std::memset(impl->mapped, 0, bytes);   // deterministic (mock parity)

    VkBufferDeviceAddressInfo bai{
        VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    bai.buffer = impl->buffer;
    impl->address = vkGetBufferDeviceAddress(s.device, &bai);

    return Buffer(std::move(impl), bytes);
}

void Device::write(Buffer &dst, const void *src, std::size_t bytes) {
    // Overwriting storage a queued dispatch still reads is a hazard the
    // caller cannot see, so an operation on a buffer with queued work
    // synchronizes first (contract note 1).
    assert(bytes <= dst.bytes());
    BufferImpl &b = impl_of(dst);
    wait(b.last_use);
    std::memcpy(b.mapped, src, bytes);
}

void Device::read(const Buffer &src, void *dst, std::size_t bytes) {
    // The designated sync point: wait out the last dispatch that touched
    // this buffer, then copy. The memory is host-coherent, so once that
    // work is done a plain memcpy is complete and correct.
    assert(bytes <= src.bytes());
    BufferImpl &b = impl_of(src);
    wait(b.last_use);
    std::memcpy(dst, b.mapped, bytes);
}

} // namespace gpud::vulkan
