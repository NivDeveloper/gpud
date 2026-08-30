// try_open (device bring-up), the async run(), the fence ring behind
// the ticket timeline, and the native-handle accessors.
//
// run() submits and returns; the fence rides a ring the reclaimer
// drains. Waits happen only where the host observes: read(), a write()
// into storage queued work may still touch, wait()/flush(), the
// max_queued throttle, and teardown. What forced this: beside a
// presenting renderer on the shared device a fence wait costs a
// display refresh (~15 ms measured, vs 0.16 ms uncontended), and a
// blocking run() paid it once per dispatch.

#include "Device.h"

#include "Buffer.h"
#include "Kernel.h"

#include <gpud/Sdl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpud::sdl {
namespace {

void log(const char *msg) {
    static const bool on = std::getenv("GPUD_LOG") != nullptr;
    if (on) std::fprintf(stderr, "gpud/sdl: try_open: %s\n", msg);
}

} // namespace

std::unique_ptr<::gpud::Device> try_open(const Options &opts) {
    Device::State s;
    s.max_queued = opts.max_queued < 1 ? 1 : opts.max_queued;

    // SDL has no device-selection hook on SDL_CreateGPUDevice; the
    // driver picks. -1 and 0 both mean "the obvious one".
    if (opts.device_index > 0)
        return log("device_index selection is not available under SDL"),
               nullptr;

    // Ref-counted, paired with SDL_QuitSubSystem in ~Device — SDL's
    // process init stays inside the Device's lifetime, which is the
    // RAII reading of "no library init/shutdown".
    if (!SDL_InitSubSystem(SDL_INIT_VIDEO))
        return log(SDL_GetError()), nullptr;

    // SDL dlopens the Vulkan loader by bare name, which misses
    // /usr/local/lib on macOS — point the hint at a known loader when
    // unset (an SDL_VULKAN_LIBRARY env var outranks this and wins).
    if (!SDL_GetHint(SDL_HINT_VULKAN_LIBRARY))
        for (const char *p : {"/usr/local/lib/libvulkan.1.dylib",
                              "/opt/homebrew/lib/libvulkan.1.dylib"})
            if (std::filesystem::exists(p)) {
                SDL_SetHint(SDL_HINT_VULKAN_LIBRARY, p);
                break;
            }

    // SPIR-V only in v1: SDL selects a driver that accepts it (its
    // Vulkan driver); the native-Metal path is MSL, a later dialect
    // decision, not a device flag.
    s.dev = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV, false, nullptr);
    if (!s.dev) {
        log(SDL_GetError());
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        return nullptr;
    }
    return std::make_unique<Device>(s);
}


Device::~Device() {
    // Idle first, then force-reclaim: after the idle every ring fence
    // is signalled, so releasing without querying is safe, and the
    // ring must empty before the device dies.
    SDL_WaitForGPUIdle(s_.dev);
    {
        std::unique_lock lock(m_);
        reclaim_locked(true);
    }
    clear_kernels(); // KernelImpls hold pipelines — before the device
    SDL_DestroyGPUDevice(s_.dev);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

Ticket Device::run(const Kernel &kernel, std::size_t groups,
                   std::span<const std::byte> scalars,
                   std::span<Buffer *const> buffers) {
    const KernelImpl &k = impl_of(kernel);
    if (buffers.empty() || buffers.size() != 1 + k.n_readonly)
        throw std::runtime_error(
            "gpud/sdl: kernel binds 1 output + " +
            std::to_string(k.n_readonly) + " inputs but run() was given " +
            std::to_string(buffers.size()) + " buffers");

    std::unique_lock lock(m_);
    reclaim_locked(false);
    throttle_locked(lock);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(s_.dev);
    if (!cmd)
        throw std::runtime_error(std::string("gpud/sdl: run: ") +
                                 SDL_GetError());
    if (k.has_uniform && !scalars.empty())
        SDL_PushGPUComputeUniformData(cmd, 0, scalars.data(),
                                      Uint32(scalars.size()));

    SDL_GPUStorageBufferReadWriteBinding out{};
    out.buffer = impl_of(*buffers[0]).buf;
    SDL_GPUComputePass *pass = SDL_BeginGPUComputePass(cmd, nullptr, 0, &out, 1);
    SDL_BindGPUComputePipeline(pass, k.pipeline);
    if (k.n_readonly > 0) {
        std::vector<SDL_GPUBuffer *> ro;
        ro.reserve(k.n_readonly);
        for (std::size_t i = 1; i < buffers.size(); ++i)
            ro.push_back(impl_of(*buffers[i]).buf);
        SDL_BindGPUComputeStorageBuffers(pass, 0, ro.data(), k.n_readonly);
    }
    SDL_DispatchGPUCompute(pass, Uint32(groups), 1, 1);
    SDL_EndGPUComputePass(pass);

    // The ticket is claimed only AFTER a successful submit: an SDL
    // fence bakes in no value (unlike a recorded timeline signal), so
    // a failed submit throws with nothing promised and there is no
    // never-signalled ticket to host-signal.
    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence)
        throw std::runtime_error(std::string("gpud/sdl: submit: ") +
                                 SDL_GetError());
    const std::uint64_t ticket =
        submitted_.load(std::memory_order_relaxed) + 1;
    pending_.push_back({ticket, fence});
    submitted_.store(ticket, std::memory_order_release);
    return {ticket};
}

// Retire what the GPU has finished with, oldest first: fences signal
// in submission order on the one queue, so the first not-done means
// none behind it are either. `force` skips the query — teardown only,
// after the device idled, where every fence is already signalled.
void Device::reclaim_locked(bool force) const {
    bool advanced = false;
    while (!pending_.empty() &&
           (force || SDL_QueryGPUFence(s_.dev, pending_.front().fence))) {
        SDL_ReleaseGPUFence(s_.dev, pending_.front().fence);
        completed_.store(pending_.front().ticket, std::memory_order_release);
        pending_.pop_front();
        advanced = true;
    }
    if (advanced) cv_.notify_all();
}

Ticket Device::completed() const {
    // "Poll and never block" (the contract): when another thread holds
    // the lock, answer with the last published value rather than wait
    // behind its record+submit.
    if (std::unique_lock lock(m_, std::try_to_lock); lock)
        reclaim_locked(false);
    return {completed_.load(std::memory_order_acquire)};
}

void Device::wait(Ticket ticket) {
    std::unique_lock lock(m_);
    wait_locked(lock, ticket.value);
}

// A fence ring is not waiter-independent: a fence can be waited only
// by whoever holds it, while the contract lets two threads wait()
// concurrently. So a waiter whose tickets were already popped by
// ANOTHER in-flight waiter loops on the condition variable until that
// waiter relocks and publishes — returning early there was the bug
// this shape exists to prevent.
void Device::wait_locked(std::unique_lock<std::mutex> &lock,
                         std::uint64_t ticket) {
    // Clamp rather than hang: a ticket beyond the last one issued
    // would wait on a fence that does not exist.
    const std::uint64_t issued = submitted_.load(std::memory_order_relaxed);
    if (ticket > issued) ticket = issued;
    if (ticket == 0) return;

    while (completed_.load(std::memory_order_relaxed) < ticket) {
        if (pending_.empty() || pending_.front().ticket > ticket) {
            cv_.wait(lock);
            continue;
        }

        // Take ownership of our prefix under the lock, wait outside it.
        std::vector<InFlight> mine;
        while (!pending_.empty() && pending_.front().ticket <= ticket) {
            mine.push_back(pending_.front());
            pending_.pop_front();
        }
        std::vector<SDL_GPUFence *> fences;
        fences.reserve(mine.size());
        for (const InFlight &f : mine) fences.push_back(f.fence);
        lock.unlock();
        const bool ok = SDL_WaitForGPUFences(s_.dev, true, fences.data(),
                                             Uint32(fences.size()));
        lock.lock();
        for (const InFlight &f : mine) SDL_ReleaseGPUFence(s_.dev, f.fence);

        // Publish before any throw: the fences are gone, so nothing
        // could ever complete these tickets again, and a stuck
        // completed_ would strand every other waiter on the cv.
        if (completed_.load(std::memory_order_relaxed) < mine.back().ticket)
            completed_.store(mine.back().ticket, std::memory_order_release);
        cv_.notify_all();
        if (!ok)
            throw std::runtime_error(std::string("gpud/sdl: wait: ") +
                                     SDL_GetError());
    }
}

// Bound the queued work: at most max_queued tickets outstanding, which
// bounds ring fences and, through SDL's deferred frees, zombie memory.
// Costs one stall per max_queued dispatches.
void Device::throttle_locked(std::unique_lock<std::mutex> &lock) {
    const std::uint64_t issued = submitted_.load(std::memory_order_relaxed);
    if (issued - completed_.load(std::memory_order_relaxed) < s_.max_queued)
        return;
    wait_locked(lock, issued - s_.max_queued + 1);
}

SDL_GPUDevice *native_device(::gpud::Device &dev) {
    auto *d = dynamic_cast<Device *>(&dev);
    return d ? d->native() : nullptr;
}

SDL_GPUBuffer *native_buffer(::gpud::Buffer &buf) {
    return buf ? impl_of(buf).buf : nullptr;
}

} // namespace gpud::sdl
