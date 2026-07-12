#include "Kernel.h"
#include "Device.h"

namespace gpud::vulkan {

Kernel Device::do_compile(std::string_view) {
    // TODO(impl): slang source (the "slang-vulkan" dialect) → SPIR-V →
    // shader module → compute pipeline; return
    // Kernel(std::make_unique<KernelImpl>(…)). Called once per distinct
    // source — memoization already happened in the base class.
    unimplemented("Device::do_compile");
}

} // namespace gpud::vulkan
