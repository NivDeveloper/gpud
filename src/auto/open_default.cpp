// Backend availability is baked in at THIS target's build time via the
// GPUD_HAS_* defines. Adding a backend is one #ifdef block in each of
// the two chains below (plus its include) — keep priority native-first:
// CUDA, then Metal, then Vulkan.

#include <gpud/Auto.h>
#include <gpud/Mock.h>

#ifdef GPUD_HAS_CUDA
#include <gpud/Cuda.h>
#endif
#ifdef GPUD_HAS_METAL
#include <gpud/Metal.h>
#endif
#ifdef GPUD_HAS_VULKAN
#include <gpud/Vulkan.h>
#endif

#include <cstdlib>
#include <string_view>

namespace gpud {

std::unique_ptr<Device> open_default() {
    // Env override: open exactly the named backend or fail — no fallback.
    if (const char *env = std::getenv("GPUD_BACKEND")) {
        const std::string_view want = env;
        if (want == "mock") return mock::try_open();
#ifdef GPUD_HAS_CUDA
        if (want == "cuda") return cuda::try_open();
#endif
#ifdef GPUD_HAS_METAL
        if (want == "metal") return metal::try_open();
#endif
#ifdef GPUD_HAS_VULKAN
        if (want == "vulkan") return vulkan::try_open();
#endif
        return nullptr;   // named backend unknown or not compiled in
    }

    // Priority chain, native first; each try_open may fail over to the
    // next. mock is deliberately absent — it is env-opt-in only.
#ifdef GPUD_HAS_CUDA
    if (auto d = cuda::try_open()) return d;
#endif
#ifdef GPUD_HAS_METAL
    if (auto d = metal::try_open()) return d;
#endif
#ifdef GPUD_HAS_VULKAN
    if (auto d = vulkan::try_open()) return d;
#endif
    return nullptr;
}

} // namespace gpud
