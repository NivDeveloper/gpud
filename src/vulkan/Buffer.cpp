#include "Buffer.h"
#include "Device.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>

namespace gpud::vulkan {

std::uint64_t native_buffer(::gpud::Buffer &b) {
    // dynamic_cast rejects a foreign backend's Buffer (and an empty
    // handle) with 0 rather than a misread pointer. Offset is always 0:
    // each Buffer owns a whole VkBuffer, VMA binds beneath it.
    auto *impl = dynamic_cast<BufferImpl *>(b.impl());
    return impl ? reinterpret_cast<std::uintptr_t>(impl->buffer) : 0;
}


BufferImpl::~BufferImpl() {
    // The handle dies; the device objects go back to the pool, tagged
    // with the last dispatch that read them so alloc() hands them out
    // only once it has completed. Nothing is destroyed here — see
    // Device::recycle for why that is the whole point.
    if (!owner || !buffer) return;
    owner->recycle({buffer, allocation, mapped, address, size, last_use});
}

Buffer Device::alloc(std::size_t bytes) {
    reclaim();   // hand back whatever the GPU has finished with

    auto impl = std::make_unique<BufferImpl>();
    impl->owner = this;
    impl->size = bytes ? bytes : 4;   // Vulkan forbids zero-size buffers

    // The pool first: a same-sized buffer whose last reader has finished
    // is exactly what a producer allocating per step wants back.
    if (Idle idle; take_idle(impl->size, idle)) {
        impl->buffer = idle.buffer;
        impl->allocation = idle.allocation;
        impl->mapped = idle.mapped;
        impl->address = idle.address;
        return Buffer(std::move(impl), bytes);
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = impl->size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    // An adopted device may share buffers with the app's queue
    // families (AdoptDesc::share_families): CONCURRENT forgoes the
    // ownership-transfer barriers this library has no seam to emit.
    // Ignored fields under EXCLUSIVE, per spec.
    bci.sharingMode = s.concurrent_families.size() >= 2
                          ? VK_SHARING_MODE_CONCURRENT
                          : VK_SHARING_MODE_EXCLUSIVE;
    bci.queueFamilyIndexCount = std::uint32_t(s.concurrent_families.size());
    bci.pQueueFamilyIndices = s.concurrent_families.data();

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
        // Idle pool entries of OTHER sizes may be the only thing holding
        // memory: finish everything, free them all (the device is idle
        // after a flush, so directly), and try once more before
        // declaring the pool exhausted.
        flush();
        drop_idle();
        r = vmaCreateBuffer(s.allocator, &bci, &aci, &impl->buffer,
                            &impl->allocation, &info);
    }
    check(r, "vmaCreateBuffer");
    {
        std::lock_guard lock(m_);
        live_bytes_ += impl->size;
    }

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

// ── the pool ─────────────────────────────────────────────────────────

void Device::recycle(Idle idle) {
    // Why a pool rather than a free: a driver may capture EVERY live
    // buffer's residency when it encodes a batch (Metal does), so a
    // batch encoded while this buffer existed may reference its memory
    // long after the buffer's own last dispatch — destroying it then
    // loses the device ("object destroyed while still required by the
    // command buffer"). A buffer that is never destroyed under load has
    // no such moment, and a producer allocating per step gets its own
    // buffers back with no create, no destroy and no residency churn.
    std::lock_guard lock(m_);
    live_bytes_ -= idle.size;
    idle_bytes_ += idle.size;
    pool_[idle.size].push_back(idle);
    trim_locked();
}

bool Device::take_idle(std::size_t size, Idle &out) {
    std::lock_guard lock(m_);
    auto it = pool_.find(size);
    if (it == pool_.end()) return false;
    // Any entry whose last reader has completed; completed_cache_ was
    // refreshed by the reclaim alloc() just ran.
    auto &v = it->second;
    for (std::size_t i = v.size(); i-- > 0;) {
        if (v[i].last_use > completed_cache_) continue;
        out = v[i];
        v[i] = v.back();
        v.pop_back();
        idle_bytes_ -= size;
        live_bytes_ += size;
        return true;
    }
    return false;
}

// Idle memory beyond what the live set justifies is freed, two-phase:
// the VkBuffer once its last reader completed, the memory once every
// batch submitted by then has too (that residency capture again). The
// margin keeps a steady producer from oscillating between freeing and
// recreating.
void Device::trim_locked() {
    constexpr std::size_t kSlack = std::size_t(64) << 20;
    while (idle_bytes_ > 2 * live_bytes_ + kSlack) {
        auto it = pool_.begin();
        while (it != pool_.end() && it->second.empty()) ++it;
        if (it == pool_.end()) return;
        Idle victim = it->second.back();
        it->second.pop_back();
        idle_bytes_ -= victim.size;
        defer_locked({victim.last_use,
                      [dev = s.device, buf = victim.buffer] {
                          vkDestroyBuffer(dev, buf, nullptr);
                      },
                      [vma = s.allocator, alloc = victim.allocation] {
                          vmaFreeMemory(vma, alloc);
                      }});
    }
}

// Everything idle, destroyed outright — only where the device is known
// idle: after a flush, or at teardown.
void Device::drop_idle() {
    std::lock_guard lock(m_);
    for (auto &[size, v] : pool_)
        for (const Idle &i : v) vmaDestroyBuffer(s.allocator, i.buffer, i.allocation);
    pool_.clear();
    idle_bytes_ = 0;
}

} // namespace gpud::vulkan
