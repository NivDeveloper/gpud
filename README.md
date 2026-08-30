# gpud

A minimal GPU compute abstraction for C++20: one `Device` interface with
five operations — `alloc`, `write`, `read`, `compile`, `run`. Backends
are separate static libraries; your code never sees an SDK header, and
there is no global state — you open a `Device` and everything dies with it.

**Status: early.** The interface, a mock backend for testing,
auto-selection, the **Vulkan backend** (macOS included, via MoltenVK)
and the **SDL_GPU backend** work. CUDA and Metal are unimplemented
stubs. The SDL backend consumes slot-bound Slang ("slang-slot") and
exports its native SDL_GPUDevice/SDL_GPUBuffer handles, so a renderer
can share the device and read compute buffers zero-copy;
`BufferSource` + the `source_of` protocol are the pull-model
vocabulary producers and consumers exchange — `current()` runs on the
consumer's thread and holds nothing, so an off-thread producer
publishes through a role-swapping type on the consumer's side.

## Build

```sh
git clone https://github.com/NivDeveloper/gpud
cd gpud
make            # or: cmake -B build && cmake --build build
make test
```

Needs CMake ≥ 3.24 and any C++20 compiler — nothing else: the Vulkan
backend fetches its build dependencies (Khronos headers, volk) and
never links a Vulkan SDK. It builds by default where Vulkan dev files
exist; force it anywhere with `-DGPUD_BACKEND_VULKAN=ON` (`_METAL` /
`_CUDA` exist too, OFF while they are stubs). The SDL backend builds
where SDL3's CMake package is findable (a system install, or a prefix
on `CMAKE_PREFIX_PATH`) — found, never fetched.

**Running** on the GPU needs a Vulkan ≥ 1.2 driver with
buffer-device-address (macOS: `brew install molten-vk`) and `slangc`
on PATH; without them `try_open()`/`open_default()` just return null
(`GPUD_LOG=1` explains why). The SDL backend needs the same slangc
plus an SDL_GPU driver accepting SPIR-V (its Vulkan driver — try_open
points SDL at a /usr/local or /opt/homebrew loader when the
`SDL_VULKAN_LIBRARY` env is unset).

## Use in a project

```cmake
include(FetchContent)
FetchContent_Declare(gpud
  GIT_REPOSITORY https://github.com/NivDeveloper/gpud
  GIT_TAG v0.5.1)   # ← pin a release; bump this line to update
FetchContent_MakeAvailable(gpud)

target_link_libraries(app PRIVATE gpud::auto_)   # underscore: `auto` is a keyword
```

Versioning is [SemVer](https://semver.org) via git tags; pre-1.0, a
minor bump may break the interface, a patch bump never does.
`<gpud/Version.h>` (generated) carries `GPUD_VERSION_*` for
compile-time checks.

Only the application picks a backend (`gpud::auto_` or a specific one,
e.g. `gpud::vulkan`). Libraries just take a `gpud::Device&` and link
`gpud::gpud`; tests link `gpud::mock`.

## Example

```cpp
#include <gpud/Auto.h>

int main() {
    auto dev = gpud::open_default();   // best backend; GPUD_BACKEND=mock overrides
    if (!dev) return 1;

    gpud::Buffer buf = dev->alloc(4 * sizeof(float));
    float data[4] = {1, 2, 3, 4};
    dev->write(buf, data, sizeof data);

    static constexpr char src[] = "...kernel source in dev->dialect()...";
    const gpud::Kernel &k = dev->compile(src);   // cached per device

    gpud::Buffer *bufs[] = {&buf};               // positional: [0] = output
    dev->run(k, /*groups=*/1, /*scalars=*/{}, bufs);

    dev->read(buf, data, sizeof data);           // waits for the run
}
```

Full design rationale: [docs/design.md](docs/design.md).
