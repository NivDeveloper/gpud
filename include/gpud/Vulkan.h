#pragma once

// Vulkan backend factory. try_open creates everything itself and no
// SDK type appears in its declaration. ADOPTION cannot be spelled
// without naming the app's Vulkan handles, so this header is the
// SECOND deliberate exception to the no-SDK-names rule (Sdl.h is the
// first): dispatchable handles are forward-declared exactly as
// vulkan_core.h declares them — including vulkan.h alongside this
// header is fine (identical typedefs on 64-bit ABIs, the only ones
// supported) — and non-dispatchable handles cross this boundary as
// std::uint64_t, sidestepping their 32-bit conditional typedef.

#include "Device.h"

#include <cstddef>
#include <cstdint>
#include <memory>

typedef struct VkInstance_T *VkInstance;
typedef struct VkPhysicalDevice_T *VkPhysicalDevice;
typedef struct VkDevice_T *VkDevice;
typedef struct VkQueue_T *VkQueue;
typedef void (*PFN_vkVoidFunction)(void);
typedef PFN_vkVoidFunction (*PFN_vkGetInstanceProcAddr)(VkInstance instance,
                                                        const char *pName);

namespace gpud::vulkan {

// nullptr if the backend can't come up (no driver, no device).
// dialect() of the returned device will be "slang-vulkan"; the kernel
// compiler resolves lazily at first compile().
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

// ADOPT an externally created device: the app owns instance, device
// and queues; gpud computes on the ONE queue given here and its
// teardown idles only that queue — it never destroys what it was
// handed. The device must outlive the returned Device and must have
// been created with Vulkan >= 1.2, bufferDeviceAddress and
// timelineSemaphore enabled (the two features try_open demands).
//
// The loader seam: get_instance_proc_addr seeds this backend's
// process-global function tables, so gpud dispatches through the same
// loader as the app. The documented limit stays: one vulkan-backend
// Device per process at a time.
struct AdoptDesc {
    VkInstance instance = nullptr;
    VkPhysicalDevice physical = nullptr;
    VkDevice device = nullptr;
    VkQueue queue = nullptr;      // gpud's one queue — the doctrine holds
    std::uint32_t queue_family = 0;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr = nullptr;

    // Queue families (other than queue_family) whose queues will ALSO
    // touch gpud buffers — a renderer's graphics family. Non-empty:
    // buffers are created VK_SHARING_MODE_CONCURRENT over the union,
    // so no ownership-transfer barriers are ever owed.
    const std::uint32_t *share_families = nullptr;
    std::size_t share_family_count = 0;

    // Set when the SAME VkQueue is shared with the app (a one-queue
    // driver): these bracket every queue access gpud makes. Null when
    // the queue is gpud's alone.
    void (*queue_lock)(void *) = nullptr;
    void (*queue_unlock)(void *) = nullptr;
    void *queue_user = nullptr;
};

std::unique_ptr<::gpud::Device> try_open_on(const AdoptDesc &,
                                            const Options & = {});

// The visualizer seam, export side: a buffer's VkBuffer (offset is
// always 0, size Buffer::bytes()) and the device timeline's
// VkSemaphore, both as uint64_t handle values; 0 for an empty handle
// or a foreign Device. The timeline's signaled value IS
// Ticket::value — that identity is API, and with submit() it is what
// lets a renderer wait GPU-side for the compute that produced a
// buffer. The other half of the bargain is the consumer's: last_use
// tracks only THIS Device's dispatches, so work the consumer records
// against a native buffer must complete before that Buffer is
// destroyed or its storage rewritten (a frame loop bounds this with
// frames-in-flight).
std::uint64_t native_buffer(::gpud::Buffer &);
std::uint64_t native_timeline(::gpud::Device &);

} // namespace gpud::vulkan
