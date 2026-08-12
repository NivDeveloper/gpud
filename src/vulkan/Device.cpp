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
        const std::string probe = std::string(c) + " -h > /dev/null 2>&1";
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
    vkGetPhysicalDeviceMemoryProperties(s.phys, &s.memory);

    // One reusable command buffer — run() submits and blocks (v1) — plus
    // the timeline semaphore every submission signals.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = s.queue_family;
    if (vkCreateCommandPool(s.device, &pci, nullptr, &s.pool) != VK_SUCCESS)
        return fail("vkCreateCommandPool failed");
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = s.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(s.device, &cai, &s.cmd);

    VkSemaphoreTypeCreateInfo stci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;   // ticket 0 = "nothing has ever been queued"
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &stci;
    if (vkCreateSemaphore(s.device, &sci, nullptr, &s.timeline) != VK_SUCCESS) {
        vkDestroyCommandPool(s.device, s.pool, nullptr);
        return fail("vkCreateSemaphore failed");
    }

    return std::make_unique<Device>(s);
}

Device::~Device() {
    vkDeviceWaitIdle(s.device);
    // Unconditional: the device is idle, so every deferred release is
    // safe to run regardless of what its ticket says. A ticket-checked
    // drain here would strand entries behind any ticket the timeline
    // never reached.
    drain(true);
    clear_kernels();   // KernelImpls hold pipelines — before the VkDevice
    vkDestroySemaphore(s.device, s.timeline, nullptr);
    vkDestroyCommandPool(s.device, s.pool, nullptr);
    vkDestroyDevice(s.device, nullptr);
    vkDestroyInstance(s.instance, nullptr);
}

std::uint64_t Device::completed() const {
    std::uint64_t value = 0;
    check(vkGetSemaphoreCounterValue(s.device, s.timeline, &value),
          "vkGetSemaphoreCounterValue");
    return value;
}

void Device::wait(std::uint64_t ticket) {
    // Clamp rather than hang: a ticket beyond the last one issued would
    // block on a value nothing will ever signal.
    if (ticket > submitted_) ticket = submitted_;
    if (ticket != 0 && completed() < ticket) {
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &s.timeline;
        wi.pValues = &ticket;
        check(vkWaitSemaphores(s.device, &wi, UINT64_MAX), "vkWaitSemaphores");
    }
    drain(false);
}

void Device::defer_release(std::uint64_t ticket, std::function<void()> release) {
    if (ticket <= completed()) {
        release();
        return;
    }
    deferred_.push_back({ticket, std::move(release)});
}

void Device::drain(bool force) {
    // Tickets only ever increase, so the queue is ordered and the first
    // entry that isn't ready means none behind it are either.
    const std::uint64_t done = force ? 0 : completed();
    while (!deferred_.empty() &&
           (force || deferred_.front().ticket <= done)) {
        deferred_.front().release();
        deferred_.pop_front();
    }
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

    // Record + submit + wait. The per-run wait is what makes the ordering
    // contract trivially hold; batching with sync-at-read() is the
    // planned optimization and only touches this file.
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(s.cmd, &bi), "vkBeginCommandBuffer");
    vkCmdBindPipeline(s.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    vkCmdPushConstants(s.cmd, k->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof push, push);
    vkCmdDispatch(s.cmd, std::uint32_t(groups), 1, 1);
    check(vkEndCommandBuffer(s.cmd), "vkEndCommandBuffer");

    // This run's ticket. It is claimed only once the submit succeeded:
    // a throw before that point leaves the timeline consistent, so a
    // later wait() can never block on a value nothing will signal.
    const std::uint64_t ticket = submitted_ + 1;
    VkTimelineSemaphoreSubmitInfo tssi{
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    tssi.signalSemaphoreValueCount = 1;   // must match signalSemaphoreCount
    tssi.pSignalSemaphoreValues = &ticket;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &tssi;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &s.timeline;
    check(vkQueueSubmit(s.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
    submitted_ = ticket;
    for (Buffer *b : buffers) impl_of(*b).last_use = ticket;

    wait(ticket);
}

} // namespace gpud::vulkan
