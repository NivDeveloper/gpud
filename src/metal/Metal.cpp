// Metal backend — scaffolding only; nothing is implemented yet.
//
// This TU (and its siblings in src/metal/) is the include firewall:
// framework headers (<Metal/Metal.h>, …) may be included HERE and must
// never be reachable from public headers. The real implementation will
// be Objective-C++ — rename to Metal.mm and enable_language(OBJCXX) in
// this directory's CMakeLists.txt. What lands here (see CLAUDE.md
// "Implementing a backend"):
//
//   - struct BufferImpl : Buffer::Impl — id<MTLBuffer>
//   - struct KernelImpl : Kernel::Impl — id<MTLComputePipelineState>
//   - class Device final : ::gpud::Device — MTLDevice + command queue;
//     dialect() == "metal"; do_compile() builds a pipeline from source
//     (slang -target metal or MSL); run() passes `scalars` via
//     setBytes and binds buffers positionally via setBuffer(k);
//     read() is the sync point (waitUntilCompleted)

#include <gpud/Metal.h>

namespace gpud::metal {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    return nullptr;   // not implemented yet — callers/auto-selection fail over
}

} // namespace gpud::metal
