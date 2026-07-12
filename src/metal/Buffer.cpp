#include "Buffer.h"
#include "Device.h"

namespace gpud::metal {

Buffer Device::alloc(std::size_t) {
    // TODO(impl): [device newBufferWithLength:options:]; return
    // Buffer(std::make_unique<BufferImpl>(…), bytes).
    unimplemented("Device::alloc");
}

void Device::write(Buffer &, const void *, std::size_t) {
    // TODO(impl): memcpy into [buffer contents] (+ didModifyRange for
    // managed storage).
    unimplemented("Device::write");
}

void Device::read(const Buffer &, void *, std::size_t) {
    // TODO(impl): the sync point — waitUntilCompleted on every pending
    // command buffer touching this buffer (ordering contract), then
    // memcpy out of [buffer contents].
    unimplemented("Device::read");
}

} // namespace gpud::metal
