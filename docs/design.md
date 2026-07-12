# gpud design: the Device interface, packaging, and backend selection

Status: accepted design, 2026-07. Extracted from the GPU-execution
design work of a host project (a CPU expression library that will
consume gpud), rewritten to be self-contained — gpud has no knowledge
of any consumer. `gpud` is a working name.

Four questions this document answers: what is the abstract interface,
are backends libraries, how is the library initiated/used, and how does
auto-selection work.

## Answers up front

1. **The interface is one abstract class with five operations** —
   `alloc`/`write`/`read` (storage), `compile` (build), `run` (launch) —
   plus a single introspection member, `dialect()`.
2. **Backends are separately compiled static libraries** — one per
   backend, each linking its SDK privately. The interface header stays
   dependency-free (std only).
3. **There is no library init.** You *open a Device*
   (`gpud::vulkan::try_open()`) and pass it around; every resource
   hangs off it and dies with it.
4. **Auto-selection is its own tiny compiled target** (`gpud_auto`,
   alias `gpud::auto_`) that tries the backends available at *its*
   build time in priority order — not static self-registration magic,
   not dlopen (yet).

## Lessons from prior art

An earlier in-house Vulkan integration was built around a global
`initGpu(ctx)` / `shutdownGpu()` pair, a static shader registry keyed
by type hashes, hook-function globals, and implicit dispatch with
per-object dirty-flag mirrors. Three lessons, each inverted here:

- **Globals that outlive the device force a shutdown call** and create
  teardown-ordering traps (driver deadlocks at exit). → All state hangs
  off the Device; RAII destroys it in order; no init/shutdown exists.
- **Implicit dispatch with silent CPU fallback** demands mutable mirror
  state and sync checks on every access. → The device is an explicit
  argument at whatever call site wants GPU execution; no ambient state,
  no fallback.
- **A type-erased registry is only needed when the dispatch site has
  lost track of what to run.** Callers here hold their kernel source at
  the point of use; a per-device memoizing `compile` replaces the
  registry.

The backend surface that integration actually consumed was tiny:
upload/update/download a buffer, build a kernel from source, and
execute (group count, a small scalar blob, a buffer list). That is the
whole interface below.

## The interface — five operations

`include/gpud/Device.h`, deliberately free of everything but std, so a
backend can be implemented (and shipped out-of-tree) without ever
seeing a consumer:

```cpp
namespace gpud {

// Backend-open options; -1 = "pick the obvious device". Kept minimal.
struct Options { int device_index = -1; };

// Move-only RAII handles; a backend subclasses the nested virtual Impl.
// A handle must not outlive its Device. Buffer carries its byte size.
class Buffer { /* unique_ptr<Impl>, size_t bytes */ };
class Kernel { /* unique_ptr<Impl> */ };

class Device {
  public:
    virtual ~Device() = default;

    virtual std::string_view dialect() const = 0;  // "slang-vulkan", "metal", "cuda", "mock"

    // ── storage ────────────────────────────────────────────────
    virtual Buffer alloc(size_t bytes) = 0;
    virtual void write(Buffer &dst, const void *src, size_t bytes) = 0;
    virtual void read(const Buffer &src, void *dst, size_t bytes) = 0;

    // ── build ──────────────────────────────────────────────────
    // Source in the dialect this backend consumes. Public and
    // NON-virtual: memoizes on source.data() identity, forwards to
    // the protected virtual do_compile.
    const Kernel &compile(std::string_view source);

    // ── run ────────────────────────────────────────────────────
    // Launch `groups` workgroups. `scalars` is the scalar section of
    // the kernel parameter data, laid out exactly as the dialect
    // declared it. `buffers` is positional: [0] = output,
    // [1 + k] = input leaf k.
    virtual void run(const Kernel &, size_t groups,
                     std::span<const std::byte> scalars,
                     std::span<Buffer *const> buffers) = 0;

  protected:
    virtual Kernel do_compile(std::string_view source) = 0;
};

} // namespace gpud
```

**Semantic contract** (replaces fences/queues/events in the interface):
calls on one `Device` behave as if executed in call order, and `read`
returns only after every prior `run` touching that buffer has
completed. A simple backend makes everything blocking; a Vulkan backend
is free to batch submissions and only sync at `read`.

**Positional buffer ABI.** The agreement about *order* is between the
caller and whatever generated the kernel source; gpud carries no
reflection metadata. How each buffer reaches the kernel is
dialect-paired backend business — buffer-device-address pointers in
push constants are a Vulkan dialect choice, Metal binds by `setBuffer`
slot, CUDA passes raw pointers as kernel parameters — which is exactly
why none of that appears in the interface.

**Caching.** `compile` is public and non-virtual; it memoizes on
`source.data()` identity in a per-device map and forwards to the
protected virtual `do_compile`. Backends stay dumb, the kernel cache
dies with the device (no shutdown-ordering trap), and calling `compile`
at every use is free after the first. Corollary for callers: source
storage must be stable at least as long as the Device (string literals
or static storage). Content equality is irrelevant — identity is the
key — so a caller whose sources are compile-time constants gets exact,
zero-hash caching for free.

**Threading (v1).** Calls on one Device are externally synchronized;
distinct Devices are independent. Internal locking can come later
without an interface change.

**Why these five and not more**: `alloc`/`write`/`read` is the minimum
to get data there and back; `compile`/`run` is literally "build and run
kernels". Deliberately excluded:

- device enumeration/selection — a backend factory's job
- buffer device addresses, mapping, raw pointers — dialect-specific
- descriptor sets, pipelines, barriers, command buffers — backend internals
- async handles/streams — covered by the ordering contract until
  profiling says otherwise
- shader-reflection/metadata queries — the ABI is positional

**Portability check** — how each target maps:

| Backend | compile                      | scalars        | buffers                              |
|---------|------------------------------|----------------|--------------------------------------|
| Vulkan  | slangc → SPIR-V → pipeline   | push constants | BDA addresses appended to push data  |
| Metal   | slang `-target metal` → PSO  | `setBytes`     | `setBuffer(k)` positional            |
| CUDA    | NVRTC / slang → PTX          | kernel params  | device pointers as params            |
| Mock    | records the call             | records the call | records the call                   |

The mock backend is the test double: consumers exercise their GPU glue
with no GPU and no SDK in CI.

## Repo shape: header-only core + one compiled library per backend

```
gpud/
  include/gpud/Device.h     the interface — no deps beyond std (INTERFACE target)
  include/gpud/Mock.h       mock backend, header-only — consumers' tests need no SDK
  include/gpud/Auto.h  + src/auto/*.cpp       open_default() — see below
  include/gpud/Vulkan.h + src/vulkan/*.cpp    (future) factory decl only / links Vulkan PRIVATE
  include/gpud/Metal.h  + src/metal/*.mm      (future, APPLE only)
  include/gpud/Cuda.h   + src/cuda/*.cpp      (future, CUDA toolkit present only)
```

CMake targets: `gpud::gpud` (interface), `gpud::mock`, `gpud::auto_`,
and later `gpud::vulkan` / `gpud::metal` / `gpud::cuda`. Each real
backend is gated on an option + SDK auto-detection
(`find_package(Vulkan)`, `if(APPLE)`, `find_package(CUDAToolkit)`);
what can't be built is simply absent, and exported `GPUD_HAS_*` defines
let application code compile conditionally.

Why compiled libraries rather than header-only backends:

- **Include firewall.** SDK headers (vulkan.h, Metal.framework, cuda.h)
  never enter a consumer's translation units; the virtual `Device` +
  pimpl'd `Buffer::Impl`/`Kernel::Impl` are the entire boundary. A
  header-only backend would drag its SDK into every includer and
  require the SDK installed even where that backend is never used.
- **Contained quirks.** Teardown ordering, driver workarounds, the
  kernel-compiler invocation — all private to one .cpp.
- **Independent toolchains.** A .mm file (Metal) or nvcc-adjacent code
  (CUDA) can't live in consumers' C++ builds anyway.

Static libraries by default (simplest, no rpath story); nothing
prevents building them shared.

## Usage semantics: no init function — open a Device

Deliberate non-goal: a `gpud::init()` / global context. The entry point
is a **factory per backend**:

```cpp
namespace gpud::vulkan {
// nullptr if the backend can't come up (no driver, no device, no compiler).
std::unique_ptr<Device> try_open(const Options & = {});
}
```

- **All state hangs off the Device.** Kernel cache, allocations, queues —
  destroyed with it, in the right order, by RAII. No shutdown call. The
  one rule: handles (`Buffer`, `Kernel`) must not outlive their Device.
- **Multiple devices coexist** (two GPUs, or a real backend + mock in
  one test).
- **Composition root pattern:** only the application's top level names
  a backend. Every library takes `gpud::Device&` as a parameter:

```cpp
// app/main.cpp — the ONLY file that knows which backends exist
#include <gpud/Auto.h>

int main() {
    auto dev = gpud::open_default();     // or gpud::vulkan::try_open()
    if (!dev) return run_cpu_only();
    run_pipeline(*dev);                  // libraries accept gpud::Device&
}
```

## Linking

- A consumer **library** links/includes only `gpud::gpud` — and only in
  its opt-in GPU code. It stays SDK-free; its tests link `gpud::mock`.
- The **application** links `gpud::gpud` + the backend targets it chose
  (or `gpud::auto_`, which pulls in every backend that was built).
- **Distribution: build from source** (FetchContent / submodule /
  find_package from an installed copy). A C++ virtual interface is not
  a stable binary ABI across compilers/standard libraries, so prebuilt
  backend binaries are out of scope. If shipping prebuilt plugins ever
  matters, the known path is a small C ABI shim + dlopen — deferred
  until there's a reason (see below).

## Auto-selection

One interface member exists because the backend can be chosen
dynamically: the consumer's *code generator* must know what to emit, so
`Device` carries `dialect()` — one string, nothing more.

Selection mechanism — `gpud_auto`, a tiny compiled target:

```cpp
// src/auto/open_default.cpp — availability baked in at ITS build time
std::unique_ptr<Device> open_default() {
    // 1. env override first: GPUD_BACKEND=cuda|metal|vulkan|mock —
    //    exactly that backend or nullptr, never a silent substitute.
    // 2. then priority order, native first, each try_open may fail → next:
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
```

- **Priority = native first** (CUDA where an NVIDIA stack exists, Metal
  on macOS, Vulkan as the broad fallback), with build-time availability
  from the `GPUD_HAS_*` defines and runtime availability from
  `try_open` failing over. The env-var override costs a few lines and
  pays for itself immediately in testing ("run this suite on mock /
  vulkan without rebuilding"). The mock backend never wins
  auto-selection; it is env-opt-in only.
- **Rejected: static self-registration** (each backend registering a
  factory from a global constructor). With static libraries the linker
  dead-strips unreferenced registrar objects unless every consumer
  remembers WHOLE_ARCHIVE — a classic silent failure. The explicit
  #ifdef chain in one .cpp is dumber and cannot break.
- **Deferred: dlopen plugins** (backends as shared libs found at
  runtime by naming convention). The only way to add a backend without
  relinking the app, but it demands a C ABI, symbol-visibility
  discipline, and a versioning story — worth it for a deployed runtime,
  not for a source-built dev dependency. The factory-function seam is
  exactly where it would slot in later, so nothing now forecloses it.

## Accepted gaps for v1

- **Per-call upload/download round-trips.** Correct but slow; the fix
  is buffer residency (keeping `Buffer`s alive across uses on the
  consumer side). The interface already supports it since buffers are
  first-class handles; it is follow-up work, not part of the minimum.
- **External synchronization only.** No internal locking, no streams;
  the ordering contract covers correctness, and either can be added
  later without changing the interface.
