# gpud — conventions

Minimal GPU compute abstraction. Namespace `gpud`; plain C++20
(std::span is the floor — no C++23/26, no reflection); must build with
the system default compiler (AppleClang). Design rationale:
docs/design.md. Current state: interface + header-only mock backend +
auto-selection skeleton + scaffolding stubs for the cuda/metal/vulkan
backends (their try_open always returns nullptr; no SDK is included,
located, or linked anywhere). No real GPU backend is implemented yet.

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
include/gpud/Cuda.h        cuda::try_open declaration only — never SDK types
include/gpud/Metal.h       metal::try_open declaration only
include/gpud/Vulkan.h      vulkan::try_open declaration only
src/auto/open_default.cpp  env override + #ifdef GPUD_HAS_* priority chain
src/{cuda,metal,vulkan}/   backend scaffolding — the include firewall:
  Device.h / Device.cpp      Device subclass; try_open, run, teardown
  Buffer.h / Buffer.cpp      BufferImpl (: Buffer::Impl) + alloc/write/read
  Kernel.h / Kernel.cpp      KernelImpl (: Kernel::Impl) + do_compile
  CMakeLists.txt             stub lib; comments show the PRIVATE SDK link lines
tests/                     gtest suite + Device.h standalone-compile check
docs/design.md             full design rationale
docs/backend-implementation.md  the plan for implementing backends
                           (dependency/error policy, step order,
                           conformance suite, per-backend specifics)
```

Backend-internal headers (src/*/[A-Z]*.h) are never installed and never
included from include/gpud/*; they are where SDK types will live. The
public Buffer/Kernel handle classes are defined once in Device.h —
backends implement only the Impl subclasses and the Device methods that
produce/consume them. Every stub body calls that backend's
`unimplemented()` (defined in its Device.h) and aborts; delete the
helper when the last stub is gone. Metal's sources are .cpp until the
real (Objective-C++) implementation renames them to .mm.

CMake: targets `gpud` (INTERFACE, alias `gpud::gpud`), `gpud_mock`
(INTERFACE, alias `gpud::mock`), `gpud_auto` (STATIC, alias
`gpud::auto_`), and per-backend static libs `gpud_cuda`/`gpud_metal`/
`gpud_vulkan` (aliases `gpud::cuda` etc.), each gated on its option.
Options: `GPUD_BUILD_TESTS` (default ON at top level),
`GPUD_BACKEND_{VULKAN,METAL,CUDA}` (intended default: SDK auto-detect;
hard OFF while the backends are stubs). Turning a backend option ON
builds its stub lib, defines GPUD_HAS_<backend> on gpud_auto, and links
it into the open_default() chain — where its try_open, returning
nullptr, fails over to the next backend.

## Build / test

```
make          # cmake -B build (Release, default compiler) + build
make test     # ctest --output-on-failure
make clean
```

Tests fetch googletest via FetchContent (needs network on first
configure). tests/device_h_standalone.cpp is a build-time check that
Device.h compiles alone under -std=c++20.

## Implementing a backend (example: vulkan)

The scaffolding already exists: factory header, the src/ dir with class
skeletons (Device subclass, BufferImpl, KernelImpl — every TODO(impl)
marks a hole) and its CMakeLists, the option gate, GPUD_HAS_* plumbing,
and the open_default arms. To implement:

1. In `src/vulkan/`: fill the TODO(impl) holes — SDK state into the
   headers' member slots, real bodies replacing the unimplemented()
   stubs (Device.cpp: try_open + run; Buffer.cpp: alloc/write/read;
   Kernel.cpp: do_compile). try_open constructs the Device (still
   nullptr when no driver/device/compiler). SDK headers are included
   ONLY in this dir — the public header must keep declaring the factory
   with no SDK types. Keep teardown ordering, driver quirks, and
   compiler invocations private to these files.
2. In `src/vulkan/CMakeLists.txt`: locate the SDK and link it PRIVATE
   (the top comment there shows the exact lines). Metal additionally
   becomes Objective-C++: Metal.cpp → Metal.mm + enable_language(OBJCXX).
3. In the top-level CMakeLists.txt: flip GPUD_BACKEND_VULKAN's default
   from OFF to SDK auto-detection (find_package(Vulkan QUIET) →
   ${Vulkan_FOUND}; Metal → ${APPLE}; CUDA → find_package(CUDAToolkit)).
4. `dialect()` returns the kernel-source dialect the backend consumes
   ("slang-vulkan", "metal", "cuda"). Resist adding introspection
   beyond that string.
5. Tests against real hardware must skip cleanly when
   `try_open() == nullptr` (no driver/device on the machine or CI).

Adding a *fourth* backend = replicate the scaffolding: factory header,
src/<backend>/ stub + CMakeLists, one gated block in the top-level
CMakeLists, one #ifdef GPUD_HAS_* include + env arm + priority-chain
arm in src/auto/open_default.cpp (priority stays native-first: CUDA,
Metal, Vulkan).
