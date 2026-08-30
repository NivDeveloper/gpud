#pragma once

// Internal to src/sdl — see Device.h.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

namespace gpud::sdl {

class Device;

// The handle may die while gpud's own dispatches still reference the
// memory: every such dispatch was SUBMITTED before run() returned, and
// SDL_ReleaseGPUBuffer defers the real free until submitted work that
// references the buffer completes — the async ring needs no lifetime
// bookkeeping of its own. What SDL's deferral does NOT cover is a
// renderer's command buffer recorded but not yet submitted on the
// shared device; that handle is the consumer's to keep alive
// (BufferSource's thread rule).
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
