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
    // Only then can the allocator hand the same block to a later
    // alloc(), which is why a fresh buffer's contents are unspecified.
    if (!owner || !buffer) return;
    owner->defer_release(last_use,
                         [vma = allocator, buf = buffer, alloc = allocation] {
                             vmaDestroyBuffer(vma, buf, alloc);
                         });
}

Buffer Device::alloc(std::size_t bytes) {
    reclaim();   // hand back whatever the GPU has finished with

    auto impl = std::make_unique<BufferImpl>();
    impl->owner = this;
    impl->allocator = s.allocator;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes ? bytes : 4;   // Vulkan forbids zero-size buffers
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // MAPPED keeps the suballocation mapped for as long as it lives, so
    // the pointer below stays good and write()/read() are plain memcpys.
    // HOST_VISIBLE|HOST_COHERENT are *required*, not preferred: left to
    // itself USAGE_AUTO may pick host-cached non-coherent memory, which
    // would need an explicit flush/invalidate around every copy.
    VmaAllocationCreateInfo aci{};
    aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VmaAllocationInfo info{};
    VkResult r = vmaCreateBuffer(s.allocator, &bci, &aci, &impl->buffer,
                                 &impl->allocation, &info);
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
        // Queued work may be the only thing holding memory that is
        // already destined for release; finish it and try once more
        // before declaring the pool exhausted.
        flush();
        reclaim();
        r = vmaCreateBuffer(s.allocator, &bci, &aci, &impl->buffer,
                            &impl->allocation, &info);
    }
    check(r, "vmaCreateBuffer");

    // Already offset to this suballocation — no arithmetic of our own.
    // Deliberately not zeroed: a fresh buffer's contents are unspecified
    // (contract note 4), which is what makes recycling free.
    impl->mapped = info.pMappedData;

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
    wait(Ticket{b.last_use});
    std::memcpy(b.mapped, src, bytes);
}

void Device::read(const Buffer &src, void *dst, std::size_t bytes) {
    // The designated sync point: wait out the last dispatch that touched
    // this buffer, then copy. The memory is host-coherent, so once that
    // work is done a plain memcpy is complete and correct.
    assert(bytes <= src.bytes());
    BufferImpl &b = impl_of(src);
    wait(Ticket{b.last_use});
    std::memcpy(dst, b.mapped, bytes);
}

} // namespace gpud::vulkan
