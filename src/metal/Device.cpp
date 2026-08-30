// try_open (device bring-up) and run (dispatch). Becomes Device.mm
// (Objective-C++) when the real implementation lands — see this
// directory's CMakeLists.txt.

#include "Device.h"

#include <gpud/Metal.h>

namespace gpud::metal {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    // TODO(impl): MTLCopyAllDevices / MTLCreateSystemDefaultDevice,
    // pick by Options::device_index (-1 = the obvious one), create the
    // command queue; nullptr on any failure (not on Apple hardware, no
    // device). Until then: unimplemented, so callers and auto-selection
    // fail over.
    return nullptr;
}

Ticket Device::run(const Kernel &, std::size_t, std::span<const std::byte>,
                   std::span<Buffer *const>) {
    // TODO(impl): compute command encoder; pass `scalars` via
    // setBytes, bind buffers positionally via setBuffer(…, k);
    // dispatch `groups` threadgroups. Free to batch command buffers —
    // read() is the sync point.
    unimplemented("Device::run");
}

} // namespace gpud::metal
