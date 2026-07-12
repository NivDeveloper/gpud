#include "Kernel.h"
#include "Device.h"

namespace gpud::metal {

Kernel Device::do_compile(std::string_view) {
    // TODO(impl): source in the "metal" dialect (MSL, or slang
    // -target metal upstream) → MTLLibrary → MTLFunction → compute
    // pipeline state; return Kernel(std::make_unique<KernelImpl>(…)).
    // Called once per distinct source — memoization already happened
    // in the base class.
    unimplemented("Device::do_compile");
}

} // namespace gpud::metal
