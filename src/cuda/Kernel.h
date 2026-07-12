#pragma once

// Internal to src/cuda. The CUDA side of a gpud::Kernel handle;
// compilation (Device::do_compile) lives in Kernel.cpp.

#include <gpud/Device.h>

namespace gpud::cuda {

struct KernelImpl final : ::gpud::Kernel::Impl {
    // TODO(impl): CUmodule + CUfunction (module unloaded by
    // ~KernelImpl, before the Device's context dies).
};

} // namespace gpud::cuda
