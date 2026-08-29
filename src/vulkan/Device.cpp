// try_open (all runtime probing lives here), run, and teardown.
//
// macOS note: there is deliberately nothing MoltenVK-specific in this
// backend. The two portability touches below (instance enumeration
// flag, portability_subset device extension) are the generic Khronos
// pattern for any portability driver, driven by runtime extension
// queries — no platform #ifdefs.

#include "Buffer.h"
#include "Device.h"
#include "Kernel.h"

#include <gpud/Vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpud::vulkan {
namespace {

void log(const char *msg) {
    static const bool on = std::getenv("GPUD_LOG") != nullptr;
    if (on) std::fprintf(stderr, "gpud/vulkan: try_open: %s\n", msg);
}

// std::system's shell may not have /usr/local/bin or /opt/homebrew/bin
// on PATH — probe the common install locations (vklib-proven).
std::string find_slangc() {
    for (const char *c :
         {"/usr/local/bin/slangc", "/opt/homebrew/bin/slangc", "slangc"}) {
#ifdef _WIN32
        const std::string probe = std::string(c) + " -h > NUL 2>&1";
#else
        const std::string probe = std::string(c) + " -h > /dev/null 2>&1";
#endif
        if (std::system(probe.c_str()) == 0) return c;
    }
    return {};
}

bool has_extension(const std::vector<VkExtensionProperties> &exts,
                   const char *name) {
    for (const auto &e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

} // namespace

std::unique_ptr<::gpud::Device> try_open(const Options &opts) {
    Device::State s;

    s.max_queued = opts.max_queued < 1 ? 1 : opts.max_queued;
    s.slangc = find_slangc();
    if (s.slangc.empty()) return log("no slangc"), nullptr;

    if (volkInitialize() != VK_SUCCESS)
        return log("no Vulkan loader"), nullptr;

    // Instance. Enable portability enumeration iff the loader offers it
    // (required to see portability drivers like MoltenVK; absent = no-op).
    std::uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> inst_ext_props(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, inst_ext_props.data());

    std::vector<const char *> inst_exts;
    VkInstanceCreateFlags flags = 0;
    if (has_extension(inst_ext_props, "VK_KHR_portability_enumeration")) {
        inst_exts.push_back("VK_KHR_portability_enumeration");
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "gpud";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.flags = flags;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = std::uint32_t(inst_exts.size());
    ici.ppEnabledExtensionNames = inst_exts.data();
    if (vkCreateInstance(&ici, nullptr, &s.instance) != VK_SUCCESS)
        return log("vkCreateInstance failed"), nullptr;
    volkLoadInstance(s.instance);

    const auto fail = [&](const char *why) {
        log(why);
        if (s.device) vkDestroyDevice(s.device, nullptr);
        vkDestroyInstance(s.instance, nullptr);
        return nullptr;
    };

    // Physical device: Options::device_index, or -1 = first discrete
    // GPU, else the first device.
    n = 0;
    vkEnumeratePhysicalDevices(s.instance, &n, nullptr);
    if (n == 0) return fail("no devices");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(s.instance, &n, devs.data());
    if (opts.device_index >= 0) {
        if (std::uint32_t(opts.device_index) >= n)
            return fail("device_index out of range");
        s.phys = devs[std::uint32_t(opts.device_index)];
    } else {
        s.phys = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                s.phys = d;
                break;
            }
        }
    }

    // The BDA push-constant ABI needs Vulkan 1.2 + bufferDeviceAddress;
    // no descriptor-set fallback path — nullptr instead. timelineSemaphore
    // backs the device timeline; it is mandatory in 1.2, so the probe is
    // belt-and-braces for portability drivers.
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(s.phys, &props);
    if (props.apiVersion < VK_API_VERSION_1_2)
        return fail("device below Vulkan 1.2");
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    vkGetPhysicalDeviceFeatures2(s.phys, &f2);
    if (!f12.bufferDeviceAddress) return fail("no bufferDeviceAddress");
    if (!f12.timelineSemaphore) return fail("no timelineSemaphore");

    n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> families(n);
    vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &n, families.data());
    s.queue_family = n;
    for (std::uint32_t i = 0; i < n; ++i)
        if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            s.queue_family = i;
            break;
        }
    if (s.queue_family == n) return fail("no compute queue");

    // Logical device. The spec requires enabling portability_subset
    // whenever the device advertises it.
    n = 0;
    vkEnumerateDeviceExtensionProperties(s.phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> dev_ext_props(n);
    vkEnumerateDeviceExtensionProperties(s.phys, nullptr, &n,
                                         dev_ext_props.data());
    std::vector<const char *> dev_exts;
    if (has_extension(dev_ext_props, "VK_KHR_portability_subset"))
        dev_exts.push_back("VK_KHR_portability_subset");

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = s.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features enable12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    enable12.bufferDeviceAddress = VK_TRUE;
    enable12.timelineSemaphore = VK_TRUE;   // core in 1.2, still opt-in
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &enable12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = std::uint32_t(dev_exts.size());
    dci.ppEnabledExtensionNames = dev_exts.data();
    if (vkCreateDevice(s.phys, &dci, nullptr, &s.device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");
    volkLoadDevice(s.device);
    vkGetDeviceQueue(s.device, s.queue_family, 0, &s.queue);

    // The command pool batches are recorded into (allocated lazily and
    // recycled — see begin_batch_locked) plus the timeline semaphore
    // every submission signals. RESET_COMMAND_BUFFER_BIT is what lets a
    // completed buffer be re-recorded by vkBeginCommandBuffer alone.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = s.queue_family;
    if (vkCreateCommandPool(s.device, &pci, nullptr, &s.pool) != VK_SUCCESS)
        return fail("vkCreateCommandPool failed");

    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;   // ticket 0 = "nothing has ever been queued"
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &stci;
    if (vkCreateSemaphore(s.device, &sci, nullptr, &s.timeline) != VK_SUCCESS) {
        vkDestroyCommandPool(s.device, s.pool, nullptr);
        return fail("vkCreateSemaphore failed");
    }

    // VMA suballocates from large blocks, which is what keeps alloc()
    // off vkAllocateMemory/vkMapMemory. BUFFER_DEVICE_ADDRESS is
    // load-bearing rather than stylistic: it is what makes VMA tag the
    // backing allocation with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
    // without which a buffer created for BDA cannot be bound. Only the
    // two getters are supplied — with dynamic functions VMA imports the
    // rest through them.
    VmaVulkanFunctions vf{};
    vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    aci.physicalDevice = s.phys;
    aci.device = s.device;
    aci.instance = s.instance;

    // Options::pool_budget_bytes, applied to every heap: the allocator
    // stops growing there and alloc() reports the pool exhausted rather
    // than the process quietly ballooning. Must outlive the call below.
    std::vector<VkDeviceSize> heap_limits;
    if (opts.pool_budget_bytes != 0) {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(s.phys, &mp);
        heap_limits.assign(mp.memoryHeapCount,
                           VkDeviceSize(opts.pool_budget_bytes));
        aci.pHeapSizeLimit = heap_limits.data();
    }
    // Matches the instance above: claiming a higher version would let VMA
    // reach for entry points this device never loaded.
    aci.vulkanApiVersion = VK_API_VERSION_1_2;
    aci.pVulkanFunctions = &vf;
    if (vmaCreateAllocator(&aci, &s.allocator) != VK_SUCCESS) {
        vkDestroySemaphore(s.device, s.timeline, nullptr);
        vkDestroyCommandPool(s.device, s.pool, nullptr);
        return fail("vmaCreateAllocator failed");
    }

    return std::make_unique<Device>(s);
}

Device::~Device() {
    // Nothing else may touch a dying Device (handles don't outlive it,
    // and the thread-safe corner is for live devices), so no lock here.
    vkDeviceWaitIdle(s.device);
    // Unconditional: the device is idle, so every deferred release is
    // safe to run regardless of what its ticket says. A ticket-checked
    // reclaim would strand entries sitting behind a ticket the timeline
    // never reached — an open batch that was never submitted has some.
    reclaim_locked(true);
    clear_kernels();   // KernelImpls hold pipelines — before the VkDevice
    // After the drain, whose release closures call vmaDestroyBuffer.
    // (It asserts if any allocation is still live, which is a free check
    // that the drain above really emptied out.)
    vmaDestroyAllocator(s.allocator);
    vkDestroySemaphore(s.device, s.timeline, nullptr);
    // Frees every command buffer allocated from it, batch_ included.
    vkDestroyCommandPool(s.device, s.pool, nullptr);
    vkDestroyDevice(s.device, nullptr);
    vkDestroyInstance(s.instance, nullptr);
}

std::uint64_t Device::completed() const {
    // Lock-free by design: vkGetSemaphoreCounterValue has no externally
    // synchronized parameter, so polling while another thread submits is
    // allowed.
    std::uint64_t value = 0;
    check(vkGetSemaphoreCounterValue(s.device, s.timeline, &value),
          "vkGetSemaphoreCounterValue");
    return value;
}

void Device::wait(std::uint64_t ticket) {
    std::unique_lock lock(m_);
    wait_locked(lock, ticket);
}

void Device::wait_locked(std::unique_lock<std::mutex> &lock,
                         std::uint64_t ticket) {
    // Clamp rather than hang: a ticket beyond the last one issued would
    // block on a value nothing will ever signal.
    const std::uint64_t issued = submitted_.load(std::memory_order_relaxed);
    if (ticket > issued) ticket = issued;
    if (ticket == 0) return;

    // The work may still be sitting in the open batch — waiting for it
    // without submitting it first would deadlock against ourselves.
    if (ticket > last_submitted_) submit_batch_locked();

    if (completed_cache_ < ticket) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &s.timeline;
        wi.pValues = &ticket;
        // Unlocked across the block: this is the only long wait, and
        // holding m_ through it would stall every other thread's poll.
        lock.unlock();
        const VkResult r = vkWaitSemaphores(s.device, &wi, UINT64_MAX);
        lock.lock();
        check(r, "vkWaitSemaphores");
    }
    reclaim_locked(false);
}

void Device::defer_release(std::uint64_t ticket, std::function<void()> release) {
    std::lock_guard lock(m_);
    if (ticket <= completed_cache_ || ticket <= completed()) {
        release();
        return;
    }
    deferred_.push_back({ticket, std::move(release)});
}

void Device::reclaim() {
    std::lock_guard lock(m_);
    reclaim_locked(false);
}

void Device::reclaim_locked(bool force) {
    if (!force) completed_cache_ = completed();
    const std::uint64_t done = completed_cache_;

    // Tickets only increase, so both queues are ordered: the first entry
    // that isn't ready means none behind it are either.
    //
    // A release closure must only destroy resources. Anything that
    // re-entered defer_release() — destroying a Buffer, say — would
    // deadlock on m_, which is deliberately not recursive.
    while (!deferred_.empty() && (force || deferred_.front().ticket <= done)) {
        deferred_.front().release();
        deferred_.pop_front();
    }
    while (!pending_.empty() && (force || pending_.front().ticket <= done)) {
        free_.push_back(pending_.front().cmd);
        pending_.pop_front();
    }
}

VkCommandBuffer Device::begin_batch_locked() {
    if (batch_) return batch_;

    if (free_.empty()) {
        VkCommandBufferAllocateInfo cai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = s.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cmd{};
        check(vkAllocateCommandBuffers(s.device, &cai, &cmd),
              "vkAllocateCommandBuffers");
        free_.push_back(cmd);
    }

    // Begin before popping, so a failure leaves the buffer on the free
    // list rather than losing it.
    VkCommandBuffer cmd = free_.back();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    free_.pop_back();
    batch_ = cmd;
    return cmd;
}

void Device::submit_batch_locked() {
    // Nothing recorded since the last submit: a timeline signal must be
    // strictly greater than the current value, so submitting an empty
    // batch would be invalid. Any partially-recorded batch stays open
    // for the next run() to continue into.
    const std::uint64_t ticket = submitted_.load(std::memory_order_relaxed);
    if (ticket == last_submitted_) return;

    // Make this batch's writes available to the host. read() waits on
    // the semaphore, and the memory model's route from the device domain
    // to the host domain is a dependency with HOST in the destination
    // stage — the same shape whether the wait is on a fence or a
    // timeline. One per batch, and free on unified memory.
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(batch_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0,
                         nullptr);

    VkResult r = vkEndCommandBuffer(batch_);
    if (r == VK_SUCCESS) {
        VkTimelineSemaphoreSubmitInfo tssi{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tssi.signalSemaphoreValueCount = 1;   // must match signalSemaphoreCount
        tssi.pSignalSemaphoreValues = &ticket;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &tssi;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &batch_;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &s.timeline;
        r = vkQueueSubmit(s.queue, 1, &si, VK_NULL_HANDLE);
    }

    pending_.push_back({ticket, batch_});
    batch_ = VK_NULL_HANDLE;
    last_submitted_ = ticket;

    if (r != VK_SUCCESS) {
        // The tickets were handed out when the dispatches were recorded,
        // so the timeline owes this value — but a failed submit means the
        // GPU will never signal it, and every later wait(), teardown
        // included, would block forever. Signal it from the host and let
        // the throw be what tells the caller the work did not happen.
        VkSemaphoreSignalInfo ssi{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        ssi.semaphore = s.timeline;
        ssi.value = ticket;
        vkSignalSemaphore(s.device, &ssi);
        check(r, "vkQueueSubmit");
    }
}

void Device::throttle_locked(std::unique_lock<std::mutex> &lock) {
    // Keep queued-but-unfinished work bounded, which bounds recorded
    // commands, in-flight submissions and deferred memory at once. Costs
    // one stall per max_queued_ dispatches. completed_cache_ was just
    // refreshed by reclaim_locked, so the common path polls nothing.
    const std::uint64_t issued = submitted_.load(std::memory_order_relaxed);
    if (issued - completed_cache_ < s.max_queued) return;
    wait_locked(lock, issued - s.max_queued + 1);
}

void Device::run(const Kernel &kernel, std::size_t groups,
                 std::span<const std::byte> scalars,
                 std::span<Buffer *const> buffers) {
    const auto *k = static_cast<const KernelImpl *>(kernel.impl());

    // Push data = the scalar blob, then each buffer's device address at
    // the next 8-aligned offset — matching the dialect's PC struct
    // layout (scalars first, then pointer members).
    std::byte push[128] = {};
    const std::size_t addr_off = (scalars.size() + 7) & ~std::size_t{7};
    if (addr_off + 8 * buffers.size() > sizeof push)
        throw std::runtime_error("gpud/vulkan: push data exceeds 128 bytes");
    if (!scalars.empty())
        std::memcpy(push, scalars.data(), scalars.size());
    for (std::size_t i = 0; i < buffers.size(); ++i)
        std::memcpy(push + addr_off + 8 * i, &impl_of(*buffers[i]).address, 8);

    // Record only — no submit, no wait. Work accumulates in the open
    // batch until something has to observe it: read(), a write() into a
    // buffer with queued work, an explicit wait()/flush(), the throttle
    // below, or teardown.
    std::unique_lock lock(m_);
    reclaim_locked(false);
    throttle_locked(lock);

    VkCommandBuffer cmd = begin_batch_locked();

    // Order this dispatch against every dispatch already queued. A
    // pipeline barrier's first synchronization scope covers all commands
    // earlier in *submission order on the queue*, so this single barrier
    // covers the previous dispatch whether it sits in this batch or in
    // one already submitted — which holds because a gpud Device owns
    // exactly one queue. The access masks cover read- and
    // write-after-write; write-after-read needs only the execution
    // dependency the stage masks already give.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    vkCmdPushConstants(cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof push, push);
    vkCmdDispatch(cmd, std::uint32_t(groups), 1, 1);

    // Claim the ticket only now that the dispatch is fully recorded. The
    // lock has not been released since begin_batch_locked(), so no other
    // thread can have submitted a batch promising work that isn't there.
    const std::uint64_t ticket = submitted_.load(std::memory_order_relaxed) + 1;
    submitted_.store(ticket, std::memory_order_release);
    for (Buffer *b : buffers) impl_of(*b).last_use = ticket;
}

} // namespace gpud::vulkan
