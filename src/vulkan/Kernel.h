#pragma once

// Internal to src/vulkan. The Vulkan side of a gpud::Kernel handle;
// compilation (Device::do_compile) lives in Kernel.cpp.

#include <gpud/Device.h>

#include <volk.h>

namespace gpud::vulkan {

struct KernelImpl final : ::gpud::Kernel::Impl {
    VkDevice device{};   // non-owning, for destruction
    VkPipelineLayout layout{};
    VkPipeline pipeline{};

    ~KernelImpl() override {
        // Cached impls live in the base-class kernel map; ~Device calls
        // clear_kernels() before destroying the VkDevice.
        if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
    }
};

} // namespace gpud::vulkan
