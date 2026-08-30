#pragma once

// Internal to src/sdl — inside the include firewall. Never installed,
// never included from include/gpud/*. SDL types appear here and in
// siblings — these headers are only included by src/sdl sources — but
// nowhere reachable from the public headers.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace gpud::sdl {

class Device final : public ::gpud::Device {
  public:
    // Everything try_open() brings up; the Device owns it from here.
    // slangc stays empty until the first do_compile resolves it — a
    // device without a shader compiler opens fine and fails at the
    // operation that needs one.
    struct State {
        SDL_GPUDevice *dev = nullptr;
        std::string slangc;
        std::uint32_t max_queued = 64; // Options::max_queued, clamped >= 1
    };
    explicit Device(const State &s) : s_(s) {}
    ~Device() override;

    std::string_view dialect() const override { return "slang-slot"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    Ticket run(const Kernel &kernel, std::size_t groups,
               std::span<const std::byte> scalars,
               std::span<Buffer *const> buffers) override;

    // The thread-safe corner: submitted() is a lock-free load, and
    // completed() keeps the contract's "poll and never block" by
    // answering from the atomic mirror when the lock is busy.
    Ticket submitted() const override {
        return {submitted_.load(std::memory_order_acquire)};
    }
    Ticket completed() const override;
    void wait(Ticket ticket) override;

    // The visualizer seam behind gpud::sdl::native_device().
    SDL_GPUDevice *native() const { return s_.dev; }

  protected:
    // Kernel.cpp
    Kernel do_compile(std::string_view source) override;

  private:
    struct InFlight {
        std::uint64_t ticket;
        SDL_GPUFence *fence;
    };

    // The `_locked` suffix means "caller already holds m_" (the vulkan
    // backend's rule). wait_locked drops the lock across the fence
    // wait; the throttle must call IT, never the public wait(), which
    // would re-lock a non-recursive mutex.
    void reclaim_locked(bool force) const;
    void wait_locked(std::unique_lock<std::mutex> &lock,
                     std::uint64_t ticket);
    void throttle_locked(std::unique_lock<std::mutex> &lock);

    State s_;
    mutable std::mutex m_;
    mutable std::condition_variable cv_;
    // MUST stay atomic: submitted() and, through it, flush() read it
    // with no lock. Written only under m_.
    std::atomic<std::uint64_t> submitted_{0};
    // The mirror completed() answers from; advanced only under m_, and
    // only forward.
    mutable std::atomic<std::uint64_t> completed_{0};
    mutable std::deque<InFlight> pending_; // submitted, fence not yet retired
};

} // namespace gpud::sdl
