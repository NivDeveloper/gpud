// BufferImpl plus alloc/write/read — the simplest correct strategy:
// one transfer buffer per call, fence-waited, released. Pooling is a
// later optimization; blocking run() means no last_use bookkeeping.

#include "Buffer.h"

#include "Device.h"

#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace gpud::sdl {
namespace {

// The shared shape of write/read: map + copy pass + submit + wait.
SDL_GPUTransferBuffer *make_transfer(SDL_GPUDevice *dev,
                                     SDL_GPUTransferBufferUsage usage,
                                     std::size_t bytes) {
    SDL_GPUTransferBufferCreateInfo ci{};
    ci.usage = usage;
    ci.size = Uint32(bytes);
    SDL_GPUTransferBuffer *tb = SDL_CreateGPUTransferBuffer(dev, &ci);
    if (!tb)
        throw std::runtime_error(std::string("gpud/sdl: transfer alloc: ") +
                                 SDL_GetError());
    return tb;
}

void submit_and_wait(SDL_GPUDevice *dev, SDL_GPUCommandBuffer *cmd) {
    SDL_GPUFence *fence = SDL_SubmitGPUCommandBufferAndAcquireFence(cmd);
    if (!fence)
        throw std::runtime_error(std::string("gpud/sdl: submit: ") +
                                 SDL_GetError());
    SDL_WaitForGPUFences(dev, true, &fence, 1);
    SDL_ReleaseGPUFence(dev, fence);
}

} // namespace

Buffer Device::alloc(std::size_t bytes) {
    if (bytes > std::numeric_limits<Uint32>::max())
        throw std::runtime_error("gpud/sdl: alloc: " + std::to_string(bytes) +
                                 " bytes exceeds SDL's 32-bit buffer size");
    SDL_GPUBufferCreateInfo ci{};
    // GRAPHICS_STORAGE_READ is the zero-copy render seam — granted on
    // every allocation so native_buffer() handles are always bindable.
    ci.usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ |
               SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE |
               SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ;
    ci.size = Uint32(bytes ? bytes : 4); // SDL forbids zero-size buffers
    SDL_GPUBuffer *b = SDL_CreateGPUBuffer(s_.dev, &ci);
    if (!b)
        throw std::runtime_error(std::string("gpud/sdl: alloc: ") +
                                 SDL_GetError());
    auto impl = std::make_unique<BufferImpl>();
    impl->dev = s_.dev;
    impl->buf = b;
    return Buffer(std::move(impl), bytes);
}

void Device::write(Buffer &dst, const void *src, std::size_t bytes) {
    assert(bytes <= dst.bytes());
    if (bytes == 0) return;
    SDL_GPUTransferBuffer *tb =
        make_transfer(s_.dev, SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, bytes);
    void *map = SDL_MapGPUTransferBuffer(s_.dev, tb, false);
    std::memcpy(map, src, bytes);
    SDL_UnmapGPUTransferBuffer(s_.dev, tb);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(s_.dev);
    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUTransferBufferLocation loc{tb, 0};
    const SDL_GPUBufferRegion reg{impl_of(dst).buf, 0, Uint32(bytes)};
    SDL_UploadToGPUBuffer(cp, &loc, &reg, false);
    SDL_EndGPUCopyPass(cp);
    submit_and_wait(s_.dev, cmd);
    SDL_ReleaseGPUTransferBuffer(s_.dev, tb);
}

void Device::read(const Buffer &src, void *dst, std::size_t bytes) {
    assert(bytes <= src.bytes());
    if (bytes == 0) return;
    SDL_GPUTransferBuffer *tb =
        make_transfer(s_.dev, SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD, bytes);

    SDL_GPUCommandBuffer *cmd = SDL_AcquireGPUCommandBuffer(s_.dev);
    SDL_GPUCopyPass *cp = SDL_BeginGPUCopyPass(cmd);
    const SDL_GPUBufferRegion reg{impl_of(src).buf, 0, Uint32(bytes)};
    const SDL_GPUTransferBufferLocation loc{tb, 0};
    SDL_DownloadFromGPUBuffer(cp, &reg, &loc);
    SDL_EndGPUCopyPass(cp);
    submit_and_wait(s_.dev, cmd);

    const void *map = SDL_MapGPUTransferBuffer(s_.dev, tb, false);
    std::memcpy(dst, map, bytes);
    SDL_UnmapGPUTransferBuffer(s_.dev, tb);
    SDL_ReleaseGPUTransferBuffer(s_.dev, tb);
}

} // namespace gpud::sdl
