#pragma once

// Internal to src/sdl — see Device.h.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

namespace gpud::sdl {

class Device;

// A blocking backend: every run() has fully completed before it
// returns, so the "memory stays alive while work is queued" half of
// the lifetime contract is vacuous and the destructor releases
// immediately.
struct BufferImpl final : ::gpud::Buffer::Impl {
    SDL_GPUDevice *dev = nullptr; // non-owning; the Device outlives us
    SDL_GPUBuffer *buf = nullptr;
    ~BufferImpl() override {
        if (dev && buf) SDL_ReleaseGPUBuffer(dev, buf);
    }
};

inline BufferImpl &impl_of(const ::gpud::Buffer &b) {
    return *static_cast<BufferImpl *>(b.impl());
}

} // namespace gpud::sdl
