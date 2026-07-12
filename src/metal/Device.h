#pragma once

// Internal to src/metal — inside the include firewall. Never installed,
// never included from include/gpud/*. Once the sources become
// Objective-C++ (.mm), framework types (id<MTLDevice>, …) may appear
// here and in siblings — these headers are only included by src/metal
// sources — but nowhere reachable from the public headers.
//
// STATUS: scaffolding — every method is an unimplemented stub and
// try_open() returns nullptr, so none of this can execute yet.

#include <gpud/Device.h>

#include <cstdio>
#include <cstdlib>

namespace gpud::metal {

// Scaffolding marker: every stub body calls this. Delete it once the
// last stub is implemented.
[[noreturn]] inline void unimplemented(const char *what) {
    std::fprintf(stderr, "gpud/metal: %s is not implemented\n", what);
    std::abort();
}

class Device final : public ::gpud::Device {
  public:
    // TODO(impl): construct from the state try_open() brings up
    // (MTLDevice picked per Options, command queue); ARC/RAII handles
    // teardown — quirks stay in Device.mm.
    Device() = default;

    std::string_view dialect() const override { return "metal"; }

    // Buffer.cpp (→ Buffer.mm)
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp (→ Device.mm)
    void run(const Kernel &kernel, std::size_t groups,
             std::span<const std::byte> scalars,
             std::span<Buffer *const> buffers) override;

  protected:
    // Kernel.cpp (→ Kernel.mm)
    Kernel do_compile(std::string_view source) override;

  private:
    // TODO(impl): id<MTLDevice>, id<MTLCommandQueue>, plus pending
    // command-buffer batch state (the ordering contract allows
    // batching runs and syncing only at read()).
};

} // namespace gpud::metal
