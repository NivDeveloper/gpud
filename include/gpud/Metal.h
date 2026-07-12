#pragma once

// Metal backend factory. Declaration only — no Metal (or any SDK)
// types appear here or in any other public header; the implementation
// links its framework PRIVATE inside src/metal/.
//
// STATUS: scaffolding only — try_open() always returns nullptr until
// the backend is implemented.

#include "Device.h"

#include <memory>

namespace gpud::metal {

// nullptr if the backend can't come up (not on Apple hardware, no
// device — or, today, not implemented). dialect() of the returned
// device will be "metal".
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

} // namespace gpud::metal
