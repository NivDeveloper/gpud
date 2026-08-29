#pragma once

// Auto-selection entry point. This header only declares open_default();
// the definition lives in the compiled gpud_auto target (CMake alias
// gpud::auto_ — the trailing underscore because plain `auto` is a C++
// keyword), which bakes in backend availability via the GPUD_HAS_*
// defines at *its* build time. Linking gpud::auto_ pulls in every
// backend that was built.

#include "Device.h"

#include <memory>

namespace gpud {

// Open the best available device:
//
//   1. If the environment variable GPUD_BACKEND is set to one of
//      vulkan|metal|cuda|sdl|mock, open exactly that backend — nullptr
//      if it is not compiled in or its try_open fails. No silent
//      fallback: asking for a specific backend and getting a different
//      one is worse than failing.
//   2. Otherwise walk the compiled-in backends native-first — CUDA,
//      then Metal, then Vulkan, then SDL — returning the first whose
//      try_open succeeds. (mock is env-opt-in only; it never wins
//      auto-selection.)
//   3. Otherwise nullptr.
std::unique_ptr<Device> open_default();

} // namespace gpud
