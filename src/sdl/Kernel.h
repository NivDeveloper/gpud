#pragma once

// Internal to src/sdl — see Device.h.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

namespace gpud::sdl {

// The pipeline plus the two facts run() needs that SDL made us declare
// at creation: how many read-only slots the kernel binds, and whether
// a uniform block exists to push the scalar blob into.
struct KernelImpl final : ::gpud::Kernel::Impl {
    SDL_GPUDevice *dev = nullptr; // non-owning; ~Device clear_kernels()s first
    SDL_GPUComputePipeline *pipeline = nullptr;
    Uint32 n_readonly = 0;
    bool has_uniform = false;
    ~KernelImpl() override {
        if (dev && pipeline) SDL_ReleaseGPUComputePipeline(dev, pipeline);
    }
};

inline KernelImpl &impl_of(const ::gpud::Kernel &k) {
    return *static_cast<KernelImpl *>(k.impl());
}

} // namespace gpud::sdl
