#pragma once

// Internal to src/sdl — see Device.h.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

namespace gpud::sdl {

class Device;

// A blocking backend: every run() has fully completed before it
// returns, so the "memory stays alive while work is queued" half of
// the lifetime contract is vacuous for gpud's OWN work and the
// destructor releases immediately. It says nothing about a renderer's
// command buffer on the shared device: SDL defers the real free past
// work already SUBMITTED that references the buffer, and a handle
// bound into a command buffer not yet submitted is the consumer's to
// keep alive (BufferSource's thread rule).
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
