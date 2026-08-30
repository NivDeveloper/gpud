#pragma once

// SDL_GPU backend factory. Declaration only — no SDL types are DEFINED
// here; the implementation links SDL3 PRIVATE inside src/sdl/. The two
// forward declarations below are the ONE deliberate exception to the
// no-SDK-names rule: this backend exists so a renderer can share the
// device, and a seam nobody can name is not a seam.

#include "Device.h"

#include <memory>

struct SDL_GPUDevice;
struct SDL_GPUBuffer;

namespace gpud::sdl {

// nullptr if the backend can't come up (no SDL GPU driver supporting
// SPIR-V). slangc is resolved lazily at the first compile() and a
// missing one fails there, never here — a consumer that only shares
// the device needs no shader toolchain. dialect() of the returned
// device will be "slang-slot": slot-bound Slang — numbered
// [[vk::binding]] resources, scalars in a uniform block — for
// consumers whose code generators emit that dialect.
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

// The visualizer seam: the SAME SDL_GPUDevice the compute runs on (a
// renderer claims a window on it and shares the queue), and a buffer's
// native object (bind it as a graphics storage read — allocations
// carry that usage). Non-owning; valid only while the Device / Buffer
// lives; the arguments must have come from THIS backend — a foreign
// Device yields nullptr, a foreign Buffer is undefined like any
// cross-device handle misuse. An EMPTY Buffer yields nullptr.
//
// Threads: two rules, not one. The Device's own calls are externally
// synchronized among THEMSELVES (Device.h). What this seam hands out
// are SDL objects, and SDL_GPU's own rules govern them — command
// buffers are per-thread, acquire/submit/create/release are
// thread-safe — which is why a renderer may record on one thread
// while run() blocks on another. What no rule orders is a buffer
// re-parked by a producer while a renderer still holds its native
// handle; see BufferSource.
SDL_GPUDevice *native_device(::gpud::Device &);
SDL_GPUBuffer *native_buffer(::gpud::Buffer &);

} // namespace gpud::sdl
