// try_open (device bring-up), run (one blocking dispatch), and the
// native-handle accessors.
//
// v1 is FULLY BLOCKING: every run() submits and waits before returning,
// which the ordering contract explicitly permits. The base-class
// timeline defaults (submitted/completed pinned at 0) are then the
// truth; batching on fences is the known later optimization.

#include "Device.h"

#include "Buffer.h"
#include "Kernel.h"

#include <gpud/Sdl.h>

#include <cstdio>
#include <cstdlib>
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
    // Blocking run() leaves nothing in flight, but a driver may hold
    // internal work; idle before releasing anything.
    SDL_WaitForGPUIdle(s_.dev);
    clear_kernels(); // KernelImpls hold pipelines — before the device
    SDL_DestroyGPUDevice(s_.dev);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

void Device::run(const Kernel &kernel, std::size_t groups,
                 std::span<const std::byte> scalars,
                 std::span<Buffer *const> buffers) {
    const KernelImpl &k = impl_of(kernel);
    if (buffers.empty() || buffers.size() != 1 + k.n_readonly)
        throw std::runtime_error(
            "gpud/sdl: kernel binds 1 output + " +
            std::to_string(k.n_readonly) + " inputs but run() was given " +
            std::to_string(buffers.size()) + " buffers");

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

    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence)
        throw std::runtime_error(std::string("gpud/sdl: submit: ") +
                                 SDL_GetError());
    SDL_WaitForGPUFences(s_.dev, true, &fence, 1);
    SDL_ReleaseGPUFence(s_.dev, fence);
}

SDL_GPUDevice *native_device(::gpud::Device &dev) {
    auto *d = dynamic_cast<Device *>(&dev);
    return d ? d->native() : nullptr;
}

SDL_GPUBuffer *native_buffer(::gpud::Buffer &buf) { return impl_of(buf).buf; }

} // namespace gpud::sdl
