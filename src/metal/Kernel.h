#pragma once

// Internal to src/metal. The Metal side of a gpud::Kernel handle;
// compilation (Device::do_compile) lives in Kernel.cpp (→ Kernel.mm
// when real).

#include <gpud/Device.h>

namespace gpud::metal {

struct KernelImpl final : ::gpud::Kernel::Impl {
    // TODO(impl): id<MTLComputePipelineState> (and, if kept around,
    // the id<MTLLibrary> / id<MTLFunction> it came from).
};

} // namespace gpud::metal
