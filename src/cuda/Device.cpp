// try_open (device bring-up), run (launch), and — once implemented —
// context/stream teardown ordering.

#include "Device.h"

#include <gpud/Cuda.h>

namespace gpud::cuda {

std::unique_ptr<::gpud::Device> try_open(const Options &) {
    // TODO(impl): cuInit, pick CUdevice by Options::device_index
    // (-1 = the obvious one), retain the primary context, create a
    // stream; nullptr on any failure (no driver, no NVIDIA device).
    // Until then: unimplemented, so callers and auto-selection fail
    // over.
    return nullptr;
}

void Device::run(const Kernel &, std::size_t, std::span<const std::byte>,
                 std::span<Buffer *const>) {
    // TODO(impl): cuLaunchKernel with `groups` blocks; kernel
    // parameters are the scalars (laid out as the dialect declared)
    // followed by each buffer's device pointer, positional. Launches
    // are async on the stream — read() is the sync point.
    unimplemented("Device::run");
}

} // namespace gpud::cuda
