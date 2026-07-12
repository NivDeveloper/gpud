// Vulkan backend — scaffolding only; nothing is implemented yet.
//
// This TU (and its siblings in src/vulkan/) is the include firewall:
// SDK headers (<vulkan/vulkan.h>, slang, …) may be included HERE and
// must never be reachable from public headers. What lands here when
// the backend is implemented (see CLAUDE.md "Implementing a backend"):
//
//   - struct BufferImpl : Buffer::Impl — VkBuffer + memory allocation
//   - struct KernelImpl : Kernel::Impl — VkPipeline (+ layout)
//   - class Device final : ::gpud::Device — instance/device/queue and
//     command machinery; dialect() == "slang-vulkan"; do_compile()
//     invokes slang -> SPIR-V -> pipeline; run() appends buffer device
//     addresses to the push-constant blob after `scalars`; read() is
//     the sync point (ordering contract: batch submissions freely,
//     sync at read)
//   - teardown ordering and driver workarounds stay private to this dir

#include <gpud/Vulkan.h>

namespace gpud::vulkan {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    return nullptr;   // not implemented yet — callers/auto-selection fail over
}

} // namespace gpud::vulkan
