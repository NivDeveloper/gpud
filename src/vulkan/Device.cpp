// try_open (device bring-up), run (dispatch), and — once implemented —
// destruction/teardown ordering and driver workarounds.

#include "Device.h"

#include <gpud/Vulkan.h>

namespace gpud::vulkan {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    // TODO(impl): create instance, enumerate physical devices, pick by
    // Options::device_index (-1 = the obvious one), bring up logical
    // device + compute queue; nullptr on any failure (no driver, no
    // device, no kernel compiler). Until then: unimplemented, so
    // callers and auto-selection fail over.
    return nullptr;
}

void Device::run(const Kernel &, std::size_t, std::span<const std::byte>,
                 std::span<Buffer *const>) {
    // TODO(impl): bind the kernel's pipeline and push `scalars`
    // followed by each buffer's device address (BDA), positional;
    // dispatch `groups` workgroups. Free to batch submissions —
    // read() is the sync point.
    unimplemented("Device::run");
}

} // namespace gpud::vulkan
