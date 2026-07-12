#pragma once

// Internal to src/vulkan. The Vulkan side of a gpud::Kernel handle;
// compilation (Device::do_compile) lives in Kernel.cpp.

#include <gpud/Device.h>

namespace gpud::vulkan {

struct KernelImpl final : ::gpud::Kernel::Impl {
    // TODO(impl): VkShaderModule, VkPipelineLayout (one push-constant
    // range: scalars then buffer device addresses), VkPipeline.
};

} // namespace gpud::vulkan
