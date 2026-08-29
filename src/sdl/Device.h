#pragma once

// Internal to src/sdl — inside the include firewall. Never installed,
// never included from include/gpud/*. SDL types appear here and in
// siblings — these headers are only included by src/sdl sources — but
// nowhere reachable from the public headers.

#include <gpud/Device.h>

#include <SDL3/SDL.h>

#include <string>

namespace gpud::sdl {

class Device final : public ::gpud::Device {
  public:
    // Everything try_open() brings up; the Device owns it from here.
    // slangc stays empty until the first do_compile resolves it — a
    // device without a shader compiler opens fine and fails at the
    // operation that needs one.
    struct State {
        SDL_GPUDevice *dev = nullptr;
        std::string slangc;
        bool owned = true; // adopted devices are never destroyed here
    };
    explicit Device(const State &s) : s_(s) {}
    ~Device() override;

    std::string_view dialect() const override { return "slang-slot"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    void run(const Kernel &kernel, std::size_t groups,
             std::span<const std::byte> scalars,
             std::span<Buffer *const> buffers) override;

    // The visualizer seam behind gpud::sdl::native_device().
    SDL_GPUDevice *native() const { return s_.dev; }

  protected:
    // Kernel.cpp
    Kernel do_compile(std::string_view source) override;

  private:
    State s_;
};

} // namespace gpud::sdl
