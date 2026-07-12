#pragma once

// CUDA backend factory. Declaration only — no CUDA (or any SDK) types
// appear here or in any other public header; the implementation links
// its toolkit PRIVATE inside src/cuda/.
//
// STATUS: scaffolding only — try_open() always returns nullptr until
// the backend is implemented.

#include "Device.h"

#include <memory>

namespace gpud::cuda {

// nullptr if the backend can't come up (no driver, no NVIDIA device —
// or, today, not implemented). dialect() of the returned device will
// be "cuda".
std::unique_ptr<::gpud::Device> try_open(const Options & = {});

} // namespace gpud::cuda
