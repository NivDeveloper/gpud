#pragma once

// Internal to src/vulkan — inside the include firewall. Never
// installed, never included from include/gpud/*. SDK headers
// (<vulkan/vulkan.h>, slang, …) may appear here and in siblings, but
// nowhere reachable from the public headers.
//
// STATUS: scaffolding — every method is an unimplemented stub and
// try_open() returns nullptr, so none of this can execute yet.

#include <gpud/Device.h>

#include <cstdio>
#include <cstdlib>

namespace gpud::vulkan {

// Scaffolding marker: every stub body calls this. Delete it once the
// last stub is implemented.
[[noreturn]] inline void unimplemented(const char *what) {
    std::fprintf(stderr, "gpud/vulkan: %s is not implemented\n", what);
    std::abort();
}

class Device final : public ::gpud::Device {
  public:
    // TODO(impl): construct from the state try_open() brings up
    // (instance, physical + logical device picked per Options, compute
    // queue, command pool); the destructor tears down in reverse order
    // — teardown ordering and driver quirks stay in Device.cpp.
    Device() = default;

    std::string_view dialect() const override { return "slang-vulkan"; }

    // Buffer.cpp
    Buffer alloc(std::size_t bytes) override;
    void write(Buffer &dst, const void *src, std::size_t bytes) override;
    void read(const Buffer &src, void *dst, std::size_t bytes) override;

    // Device.cpp
    void run(const Kernel &kernel, std::size_t groups,
             std::span<const std::byte> scalars,
             std::span<Buffer *const> buffers) override;

  protected:
    // Kernel.cpp
    Kernel do_compile(std::string_view source) override;

  private:
    // TODO(impl): VkInstance / VkPhysicalDevice / VkDevice / VkQueue /
    // command pool, plus pending-submission batch state (the ordering
    // contract allows batching runs and syncing only at read()).
};

} // namespace gpud::vulkan
