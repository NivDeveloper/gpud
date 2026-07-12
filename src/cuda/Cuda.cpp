// CUDA backend — scaffolding only; nothing is implemented yet.
//
// This TU (and its siblings in src/cuda/) is the include firewall:
// toolkit headers (<cuda.h>, <nvrtc.h>, …) may be included HERE and
// must never be reachable from public headers. What lands here when
// the backend is implemented (see CLAUDE.md "Implementing a backend"):
//
//   - struct BufferImpl : Buffer::Impl — CUdeviceptr
//   - struct KernelImpl : Kernel::Impl — CUmodule + CUfunction
//   - class Device final : ::gpud::Device — context/stream; dialect()
//     == "cuda"; do_compile() via NVRTC (or slang -> PTX); run()
//     passes `scalars` and the device pointers as kernel parameters,
//     buffers positional; read() is the sync point (stream sync)

#include <gpud/Cuda.h>

namespace gpud::cuda {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    return nullptr;   // not implemented yet — callers/auto-selection fail over
}

} // namespace gpud::cuda
