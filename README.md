# gpud

A minimal GPU compute abstraction for C++20: one `Device` interface with
five operations — `alloc`, `write`, `read`, `compile`, `run`. Backends
are separate static libraries; your code never sees an SDK header, and
there is no global state — you open a `Device` and everything dies with it.

**Status: early.** The interface, a mock backend for testing, and
auto-selection work. The CUDA / Metal / Vulkan backends are unimplemented
stubs.

## Build

```sh
git clone https://github.com/NivDeveloper/gpud
cd gpud
make            # or: cmake -B build && cmake --build build
make test
```

Needs CMake ≥ 3.24 and any C++20 compiler. Backends are opt-in at
configure time: `-DGPUD_BACKEND_VULKAN=ON` (also `_METAL`, `_CUDA`;
all OFF while they are stubs).

## Use in a project

```cmake
include(FetchContent)
FetchContent_Declare(gpud GIT_REPOSITORY https://github.com/NivDeveloper/gpud GIT_TAG main)
FetchContent_MakeAvailable(gpud)

target_link_libraries(app PRIVATE gpud::auto_)   # underscore: `auto` is a keyword
```

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
