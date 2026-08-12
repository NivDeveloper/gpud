#pragma once

// Internal to src/vulkan — inside the include firewall. Never installed,
// never included from include/gpud/*.
//
// volk resolves every Vulkan function at runtime (no loader is linked);
// a machine without a driver fails in try_open, not at build or load
// time. Function pointers are process-global (volkLoadDevice), which
// assumes all coexisting vulkan Devices go through the same driver —
// fine until someone runs two different ICDs in one process.

#include <gpud/Device.h>

#include <volk.h>

#include <cstdint>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>

namespace gpud::vulkan {

// Error contract: failures after a successful try_open throw.
inline void check(VkResult r, const char *what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string("gpud/vulkan: ") + what +
                                 " failed (VkResult " + std::to_string(r) +
                                 ")");
}

class Device final : public ::gpud::Device {
  public:
    // Everything try_open() brought up, owned (and torn down, in
    // reverse) by the Device.
    struct State {
        VkInstance instance{};
        VkPhysicalDevice phys{};
        VkDevice device{};
        VkQueue queue{};
        std::uint32_t queue_family{};
        VkCommandPool pool{};
        VkCommandBuffer cmd{};   // one reusable buffer: run() is blocking (v1)
        VkSemaphore timeline{};  // the device timeline; run() signals ticket N
        VkPhysicalDeviceMemoryProperties memory{};
        std::string slangc;      // resolved compiler path
    };

    explicit Device(const State &state) : s(state) {}
    ~Device() override;

    std::string_view dialect() const override { return "slang-vulkan"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    void run(const Kernel &kernel, std::size_t groups,
             std::span<const std::byte> scalars,
             std::span<Buffer *const> buffers) override;

    // The device timeline, backed by a timeline VkSemaphore (core 1.2).
    std::uint64_t submitted() const override { return submitted_; }
    std::uint64_t completed() const override;
    void wait(std::uint64_t ticket) override;

    // Deferred release. A Buffer handle may die while queued work still
    // reads its memory, so BufferImpl hands its teardown here instead of
    // doing it inline: `release` runs once `ticket` has completed, or
    // right now if it already has.
    void defer_release(std::uint64_t ticket, std::function<void()> release);

  protected:
    // Kernel.cpp
    Kernel do_compile(std::string_view source) override;

  private:
    // Run the deferred releases whose ticket has completed, oldest
    // first, stopping at the first that hasn't. `force` skips the ticket
    // test entirely — teardown only, once the device is idle, where
    // tickets from an unsubmitted batch would never be signalled and
    // would strand every entry behind them.
    void drain(bool force);

    State s;
    std::uint64_t submitted_ = 0;

    struct Pending {
        std::uint64_t ticket;
        std::function<void()> release;
    };
    std::deque<Pending> deferred_;
};

} // namespace gpud::vulkan
