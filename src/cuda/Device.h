#pragma once

// Internal to src/cuda — inside the include firewall. Never installed,
// never included from include/gpud/*. Toolkit headers (<cuda.h>,
// <nvrtc.h>, …) may appear here and in siblings, but nowhere reachable
// from the public headers.
//
// STATUS: scaffolding — every method is an unimplemented stub and
// try_open() returns nullptr, so none of this can execute yet.

#include <gpud/Device.h>

#include <cstdio>
#include <cstdlib>

namespace gpud::cuda {

// Scaffolding marker: every stub body calls this. Delete it once the
// last stub is implemented.
[[noreturn]] inline void unimplemented(const char *what) {
    std::fprintf(stderr, "gpud/cuda: %s is not implemented\n", what);
    std::abort();
}

class Device final : public ::gpud::Device {
  public:
    // TODO(impl): construct from the state try_open() brings up
    // (CUdevice per Options, retained primary context, stream); the
    // destructor releases them in reverse order in Device.cpp.
    Device() = default;

    std::string_view dialect() const override { return "cuda"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    Ticket run(const Kernel &kernel, std::size_t groups,
               std::span<const std::byte> scalars,
               std::span<Buffer *const> buffers) override;

  protected:
    // Kernel.cpp
    Kernel do_compile(std::string_view source) override;

  private:
    // TODO(impl): CUdevice, primary CUcontext, CUstream — one stream
    // gives call-order execution for free; read() syncs the stream
    // (ordering contract).
};

} // namespace gpud::cuda
