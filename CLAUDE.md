# gpud — conventions

Minimal GPU compute abstraction. Namespace `gpud`; plain C++20
(std::span is the floor — no C++23/26, no reflection); must build with
the system default compiler (AppleClang). Design rationale:
docs/design.md. Current state: interface + header-only mock backend +
auto-selection skeleton; no real GPU backend exists yet.

## Invariants (do not break)

- **No SDK types in any public header.** Everything under include/gpud/
  uses std only. SDK headers (vulkan.h, Metal, cuda.h) are visible only
  inside their backend's src/<backend>/ sources; the virtual `Device`
  plus the `Buffer::Impl`/`Kernel::Impl` seam is the entire boundary.
- **No globals, no library init/shutdown.** Entry points are per-backend
  factories `gpud::<backend>::try_open(const Options& = {})` →
  `std::unique_ptr<Device>`, nullptr when the backend can't come up.
  All state hangs off the Device and dies with it (RAII).
- **Handles never outlive their Device**, and are only passed back to
  the Device that created them. Buffer/Kernel are move-only.
- **Ordering contract.** Calls on one Device behave as if executed in
  call order; read() returns only after every prior run() touching that
  buffer completed. Backends may batch and sync only at read().
- **External synchronization (v1).** Calls on one Device are externally
  synchronized; distinct Devices are independent.
- **Positional buffer ABI.** buffers[0] = output, buffers[1+k] = input
  leaf k. No reflection/metadata queries; caller and kernel-source
  generator agree on the order. How a buffer reaches the kernel (BDA
  push constants, setBuffer slot, kernel parameter) is per-backend,
  paired with its dialect() string. dialect() is the only introspection.
- **compile() memoizes on source.data() identity** (stable storage
  required of callers). compile() is public non-virtual on Device;
  backends implement protected do_compile() only.
- **Backends are separately compiled static libraries** linking their
  SDKs PRIVATE. The core `gpud` target stays INTERFACE and
  dependency-free. Mock is the one header-only backend so consumer
  tests need no SDK.
- **Auto-selection lives only in the gpud_auto target** (alias
  `gpud::auto_` — plain `auto` is a C++ keyword). Availability is baked
  in at gpud_auto's build time via GPUD_HAS_* defines; priority is
  native-first: CUDA, Metal, Vulkan. Env override GPUD_BACKEND picks
  exactly one backend or fails — never a silent fallback. mock never
  wins auto-selection; it is env-opt-in only.

## Layout

```
include/gpud/Device.h      the whole abstract interface + Options; std-only
include/gpud/Mock.h        header-only mock (gpud::mock::Device, try_open)
include/gpud/Auto.h        open_default() declaration
src/auto/open_default.cpp  env override + #ifdef GPUD_HAS_* priority chain
tests/                     gtest suite + Device.h standalone-compile check
docs/design.md             full design rationale
```

CMake: targets `gpud` (INTERFACE, alias `gpud::gpud`), `gpud_mock`
(INTERFACE, alias `gpud::mock`), `gpud_auto` (STATIC, alias
`gpud::auto_`). Options: `GPUD_BUILD_TESTS` (default ON at top level),
`GPUD_BACKEND_{VULKAN,METAL,CUDA}` (intended default: SDK auto-detect;
hard OFF until the backends exist — today they only feed GPUD_HAS_*
into gpud_auto).

## Build / test

```
make          # cmake -B build (Release, default compiler) + build
make test     # ctest --output-on-failure
make clean
```

Tests fetch googletest via FetchContent (needs network on first
configure). tests/device_h_standalone.cpp is a build-time check that
Device.h compiles alone under -std=c++20.

## Adding a backend (example: vulkan)

1. `include/gpud/Vulkan.h` — factory declaration ONLY, no SDK types:
   `namespace gpud::vulkan { std::unique_ptr<Device> try_open(const Options& = {}); }`
2. `src/vulkan/*.cpp` — subclass `Device`, `Buffer::Impl`,
   `Kernel::Impl`; keep teardown ordering, driver quirks, and compiler
   invocations private to these files.
3. CMake: `add_subdirectory(src/vulkan)` gated on `GPUD_BACKEND_VULKAN`
   (flip that option's default to SDK detection, e.g.
   `find_package(Vulkan QUIET)`); define target `gpud_vulkan` + alias
   `gpud::vulkan`, linking the SDK PRIVATE and `gpud::gpud` PUBLIC.
4. Wire gpud_auto: in the existing `if(GPUD_BACKEND_VULKAN)` block add
   `target_link_libraries(gpud_auto PRIVATE gpud::vulkan)` (the
   GPUD_HAS_VULKAN define is already plumbed); in
   src/auto/open_default.cpp the `#ifdef GPUD_HAS_VULKAN` arms already
   exist for new backends' pattern — one include + one arm in the env
   chain + one arm in the priority chain. Keep priority native-first:
   CUDA, Metal, Vulkan.
5. `dialect()` returns the kernel-source dialect the backend consumes
   ("slang-vulkan", "metal", "cuda"). Resist adding introspection
   beyond that string.
6. Tests against real hardware must skip cleanly when
   `try_open() == nullptr` (no driver/device on the machine or CI).
