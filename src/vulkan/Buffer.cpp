#include "Buffer.h"
#include "Device.h"

namespace gpud::vulkan {

Buffer Device::alloc(std::size_t) {
    // TODO(impl): create VkBuffer (storage + device-address usage),
    // bind memory, resolve the device address; return
    // Buffer(std::make_unique<BufferImpl>(…), bytes).
    unimplemented("Device::alloc");
}

void Device::write(Buffer &, const void *, std::size_t) {
    // TODO(impl): upload (staging copy or host-visible memcpy).
    unimplemented("Device::write");
}

void Device::read(const Buffer &, void *, std::size_t) {
    // TODO(impl): the sync point — wait for every pending run()
    // touching this buffer (ordering contract), then download.
    unimplemented("Device::read");
}

} // namespace gpud::vulkan
