# gpud

A minimal GPU compute abstraction: five operations behind one virtual
interface, backends as separately compiled libraries, no global state.

`gpud` is a working name. Current status: the interface, a header-only
mock backend, and the auto-selection skeleton. No real GPU backend is
implemented yet. Full rationale in [docs/design.md](docs/design.md);
conventions and how to add a backend in [CLAUDE.md](CLAUDE.md).

## The interface

One abstract class, `gpud::Device`
([include/gpud/Device.h](include/gpud/Device.h), std-only), with five
operations:

| operation | meaning |
|---|---|
| `alloc(bytes)` | make a device buffer (`gpud::Buffer`, move-only RAII handle) |
| `write(buf, src, bytes)` | host → device |
| `read(buf, dst, bytes)` | device → host; completes only after every prior `run` touching `buf` |
| `compile(source)` | kernel source (in `dialect()`) → `const gpud::Kernel&`; memoized per Device by source identity |
| `run(kernel, groups, scalars, buffers)` | launch `groups` workgroups; `buffers` is positional: `[0]` output, `[1+k]` input k |

Calls on one Device behave as if executed in call order; a backend may
batch internally and synchronize only at `read()`. Calls on one Device
are externally synchronized (v1). Handles must not outlive their Device.
There is no init/shutdown — you open a Device and everything dies with it.

## Quickstart

Only the application's composition root names a backend. Libraries that
use GPU compute accept `gpud::Device&` and link only `gpud::gpud`.

```cpp
#include <gpud/Auto.h>

int main() {
    auto dev = gpud::open_default();      // or a specific factory, e.g.
    if (!dev) return 1;                   //   gpud::mock::try_open()

    gpud::Buffer buf = dev->alloc(4 * sizeof(float));
    float data[4] = {1, 2, 3, 4};
    dev->write(buf, data, sizeof data);

    static constexpr char source[] = "...kernel in dev->dialect()...";
    const gpud::Kernel &k = dev->compile(source);   // cached per Device

    gpud::Buffer *buffers[] = {&buf};     // [0] = output
    dev->run(k, /*groups=*/1, /*scalars=*/{}, buffers);

    dev->read(buf, data, sizeof data);    // syncs with the run above
}
```

`GPUD_BACKEND=mock|vulkan|metal|cuda` in the environment overrides
auto-selection (exactly that backend, or failure — no silent fallback).

## Building

```sh
make            # configure + build (Release, system default compiler)
make test       # run the test suite
```

or plain CMake: `cmake -B build && cmake --build build && ctest --test-dir build`.
Requires C++20 and CMake ≥ 3.24; no other dependencies (tests fetch
googletest).

CMake targets: `gpud::gpud` (the interface, INTERFACE), `gpud::mock`
(header-only mock backend), `gpud::auto_` (compiled auto-selection —
trailing underscore because plain `auto` is a C++ keyword).
