#pragma once

// Vulkan backend factory. Declaration only — no Vulkan (or any SDK)
// types appear here or in any other public header; the implementation
// links its SDK PRIVATE inside src/vulkan/.
//
// STATUS: scaffolding only — try_open() always returns nullptr until
// the backend is implemented.

#include "Device.h"

#include <memory>

namespace gpud::vulkan {

// nullptr if the backend can't come up (no driver, no device, no
// kernel compiler — or, today, not implemented). dialect() of the
// returned device will be "slang-vulkan".
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

} // namespace gpud::vulkan
