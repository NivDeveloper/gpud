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

#include <vk_mem_alloc.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpud::vulkan {

// Error contract: failures after a successful try_open throw. A lost
// device gets its own sentence — the one VkResult a user can act on.
std::string device_lost_sentence(const char *what);

inline void check(VkResult r, const char *what) {
    if (r == VK_SUCCESS) return;
    if (r == VK_ERROR_DEVICE_LOST)
        throw std::runtime_error(device_lost_sentence(what));
    throw std::runtime_error(std::string("gpud/vulkan: ") + what +
                             " failed (VkResult " + std::to_string(r) + ")");
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
        VmaAllocator allocator{};
        std::uint32_t max_queued{};   // Options::max_queued, clamped to >= 1
        std::uint32_t batch{};        // Options::batch, clamped to [1, max_queued]
        std::uint64_t wait_ns{};      // Options::wait_ms (GPUD_WAIT_MS wins);
                                      // 0 = every host wait is unbounded
        // Options::profile (GPUD_PROFILE wins), granted only where the
        // queue family timestamps: the pool holds a begin/end pair per
        // batch slot, ns_per_tick is timestampPeriod.
        bool profile = false;
        VkQueryPool timestamps{};
        std::uint32_t timestamp_slots = 0;
        float ns_per_tick = 1.0f;
        std::string slangc;      // resolved at FIRST compile; empty until
        bool owned = true;       // try_open created device+instance;
                                 // try_open_on adopts and never destroys
        // Size >= 2 means buffers are created CONCURRENT over these
        // families (compute first, then the app's share_families).
        std::vector<std::uint32_t> concurrent_families;
        // Bracket every queue access when the VkQueue itself is shared
        // with the app (one-queue drivers). Null = the queue is ours.
        void (*queue_lock)(void *) = nullptr;
        void (*queue_unlock)(void *) = nullptr;
        void *queue_user = nullptr;
    };

    explicit Device(const State &state) : s(state) {}
    ~Device() override;

    std::string_view dialect() const override { return "slang-vulkan"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    Ticket run(const Kernel &kernel, std::size_t groups,
               std::span<const std::byte> scalars,
               std::span<Buffer *const> buffers) override;

    // The device timeline, backed by a timeline VkSemaphore (core 1.2).
    // These three are the thread-safe corner of the interface: they may
    // be called while another thread is inside run()/read()/write().
    // Everything else stays externally synchronized.
    Ticket submitted() const override {
        return {submitted_.load(std::memory_order_acquire)};
    }
    Ticket completed() const override;
    void wait(Ticket ticket) override;
    void submit() override;
    std::size_t take_timings(std::span<BatchTiming> out) override;

    // The pool: a dead Buffer's device objects, waiting to be handed out
    // again by alloc() once their last reader has completed. Nothing is
    // destroyed under load — Buffer.cpp says why.
    struct Idle {
        VkBuffer buffer;
        VmaAllocation allocation;
        void *mapped;
        VkDeviceAddress address;
        std::size_t size;
        std::uint64_t last_use;
    };
    void recycle(Idle);   // from ~BufferImpl

    // Deferred release, in two phases — how the pool is TRIMMED: `release`
    // (the VkBuffer) runs once `ticket` has completed; `then` (the memory)
    // once everything submitted at that moment has completed, because a
    // batch encoded while the buffer existed may hold its memory resident
    // until it finishes. Either runs right away when its ticket has.
    void defer_release(std::uint64_t ticket, std::function<void()> release,
                       std::function<void()> then = {});

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

    // What a host wait past the bound says: the ticket, where the
    // timeline stood, and the two things it can mean.
    std::string hung_sentence(std::uint64_t ticket) const;

    // Bound the queued work: at most max_queued_ tickets outstanding.
    void throttle_locked(std::unique_lock<std::mutex> &lock);

    void q_lock() { if (s.queue_lock) s.queue_lock(s.queue_user); }
    void q_unlock() { if (s.queue_unlock) s.queue_unlock(s.queue_user); }

    friend std::uint64_t native_timeline(::gpud::Device &);

    State s;
    std::mutex m_;

    // Last ticket handed out. Atomic so submitted() — and through it
    // flush() — can be read without the lock.
    std::atomic<std::uint64_t> submitted_{0};

    std::uint64_t last_submitted_ = 0;   // highest ticket actually submitted
    std::uint64_t completed_cache_ = 0;  // last polled completed(), to keep
                                         // the common path off the driver

    VkCommandBuffer batch_{};            // open for recording, or null
    std::uint32_t batch_count_ = 0;      // dispatches recorded into batch_
    std::vector<VkCommandBuffer> free_;  // recorded, completed, reusable
    struct InFlight {
        std::uint64_t ticket;
        VkCommandBuffer cmd;
        // Profiling: the batch's first ticket, its dispatch count and
        // the timestamp slot its pair sits in (or UINT32_MAX).
        std::uint64_t first;
        std::uint32_t dispatches;
        std::uint32_t slot;
    };
    std::deque<InFlight> pending_;       // submitted, awaiting completion
    std::uint64_t batch_first_ = 0;      // first ticket of the open batch
    std::uint32_t batch_slot_ = 0;       // its timestamp slot, when profiling
    std::uint32_t next_slot_ = 0;        // ring over timestamp_slots
    std::deque<BatchTiming> timings_;    // completed, awaiting take_timings
    // Completed batches whose stamps the driver has not yet marked
    // available (MoltenVK publishes them from the command buffer's
    // completion handler, after the semaphore); retried each reclaim.
    std::deque<InFlight> unread_;
    bool collect_timing_locked(const InFlight &);
    // The pending half of reclaim: completed batches' stamps are read
    // (or queued unread) and their command buffers freed.
    void collect_pending_locked(std::uint64_t done, bool force);

    struct Deferred {
        std::uint64_t ticket;
        std::function<void()> release;
        std::function<void()> then;   // re-deferred to last_submitted_
    };
    std::deque<Deferred> deferred_;   // ordered by ticket

    // Run a due entry: its release, then its follow-up — now if nothing
    // submitted is outstanding, else queued behind the last submission.
    void run_deferred_locked(Deferred &d, bool force);
    void push_deferred_locked(Deferred d);
    void defer_locked(Deferred d);   // run now if due, else push

    // Buffer.cpp — the pool
    std::unordered_map<std::size_t, std::vector<Idle>> pool_;
    std::size_t idle_bytes_ = 0;
    std::size_t live_bytes_ = 0;
    bool take_idle(std::size_t size, Idle &out);
    void trim_locked();
    void drop_idle();
};

} // namespace gpud::vulkan
