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

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

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
        VkSemaphore timeline{};  // the device timeline; each batch signals
                                 // the ticket of its last dispatch
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
    // These three are the thread-safe corner of the interface: they may
    // be called while another thread is inside run()/read()/write().
    // Everything else stays externally synchronized.
    std::uint64_t submitted() const override {
        return submitted_.load(std::memory_order_acquire);
    }
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
    // ── everything below is guarded by m_ ────────────────────────────
    // The `_locked` suffix means "caller already holds m_". The two that
    // take the lock by reference release it across the semaphore wait,
    // which is also why the public wait() cannot be reused internally:
    // m_ is not recursive, and run()'s throttle already holds it.

    // Reclaim what the GPU has finished with — deferred releases run,
    // command buffers go back on the free list — oldest first, stopping
    // at the first ticket that isn't done. `force` skips the ticket test
    // entirely; teardown only, once the device is idle, where an
    // unsubmitted batch's tickets would never be signalled and would
    // strand every entry behind them.
    void reclaim_locked(bool force);
    void reclaim();   // the same, taking the lock

    // Begin (or return) the batch currently being recorded into.
    VkCommandBuffer begin_batch_locked();

    // End and submit the open batch, signalling the ticket of its last
    // dispatch. A no-op when nothing has been recorded since the last
    // submit: a timeline signal must strictly increase.
    void submit_batch_locked();

    void wait_locked(std::unique_lock<std::mutex> &lock, std::uint64_t ticket);

    // Bound the queued work: at most max_queued_ tickets outstanding.
    void throttle_locked(std::unique_lock<std::mutex> &lock);

    State s;
    std::mutex m_;

    // Last ticket handed out. Atomic so submitted() — and through it
    // flush() — can be read without the lock.
    std::atomic<std::uint64_t> submitted_{0};

    std::uint64_t last_submitted_ = 0;   // highest ticket actually submitted
    std::uint64_t completed_cache_ = 0;  // last polled completed(), to keep
                                         // the common path off the driver
    std::uint32_t max_queued_ = 64;      // Options::max_queued (commit 4)

    VkCommandBuffer batch_{};            // open for recording, or null
    std::vector<VkCommandBuffer> free_;  // recorded, completed, reusable
    struct InFlight {
        std::uint64_t ticket;
        VkCommandBuffer cmd;
    };
    std::deque<InFlight> pending_;       // submitted, awaiting completion

    struct Deferred {
        std::uint64_t ticket;
        std::function<void()> release;
    };
    std::deque<Deferred> deferred_;
};

} // namespace gpud::vulkan
