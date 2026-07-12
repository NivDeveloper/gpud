#include "Kernel.h"
#include "Device.h"

namespace gpud::cuda {

Kernel Device::do_compile(std::string_view) {
    // TODO(impl): source in the "cuda" dialect → PTX via NVRTC (or
    // slang -target ptx upstream) → cuModuleLoadData →
    // cuModuleGetFunction; return
    // Kernel(std::make_unique<KernelImpl>(…)). Called once per
    // distinct source — memoization already happened in the base
    // class.
    unimplemented("Device::do_compile");
}

} // namespace gpud::cuda
