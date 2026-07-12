#include "Buffer.h"
#include "Device.h"

namespace gpud::cuda {

Buffer Device::alloc(std::size_t) {
    // TODO(impl): cuMemAlloc; return
    // Buffer(std::make_unique<BufferImpl>(…), bytes).
    unimplemented("Device::alloc");
}

void Device::write(Buffer &, const void *, std::size_t) {
    // TODO(impl): cuMemcpyHtoD (or async on the stream — call order is
    // preserved either way).
    unimplemented("Device::write");
}

void Device::read(const Buffer &, void *, std::size_t) {
    // TODO(impl): the sync point — synchronize the stream so every
    // pending run() touching this buffer completes (ordering
    // contract), then cuMemcpyDtoH.
    unimplemented("Device::read");
}

} // namespace gpud::cuda
