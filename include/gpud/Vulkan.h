#pragma once

// Vulkan backend factory. Declaration only — no Vulkan (or any SDK)
// types appear here or in any other public header; the implementation
// links its SDK PRIVATE inside src/vulkan/.
//

#include "Device.h"

#include <memory>

namespace gpud::vulkan {

// nullptr if the backend can't come up (no driver, no device, no
// kernel compiler). dialect() of the returned device will be
// "slang-vulkan".
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

} // namespace gpud::vulkan
