# gpud — conventions

Minimal GPU compute abstraction. Namespace `gpud`; plain C++20
(std::span is the floor — no C++23/26, no reflection); must build with
the system default compiler (AppleClang). Design rationale:
docs/design.md; implementation plan: docs/backend-implementation.md.
Current state: interface + header-only mock + auto-selection + TWO
implemented backends. Vulkan: volk-loaded, BDA push-constant ABI
("slang-vulkan"), slangc do_compile (resolved lazily at first compile;
GPUD_SLANGC pins it), batched submission on a timeline semaphore with
deferred release and VMA allocation; works on MoltenVK with zero
Apple-specific code. v0.7 adds the renderer seam: submit() pumps the
open batch without blocking, try_open_on ADOPTS an app-created device
(one queue — the doctrine holds; buffers go CONCURRENT over the app's
share_families; optional queue-lock brackets for one-queue drivers;
teardown idles only its queue and never destroys what it adopted),
and native_buffer/native_timeline export VkBuffer and the timeline
semaphore, whose signaled value IS Ticket::value — that identity is
API. v0.8 makes submission eager (Options::batch, default 16 — 4x on
a free-running producer) and RECYCLES dead buffers through a pool
instead of freeing them (a driver may keep every live buffer resident
per encoded batch, so a free under load loses the device; MoltenVK
past depth 64). SDL_GPU: slot-bound ABI ("slang-slot"),
slangc → SPIR-V resolved lazily at first compile (bring-up needs no
toolchain), async run() on a fence ring behind the ticket timeline, native
SDL_GPUDevice/SDL_GPUBuffer handle export for renderer sharing;
BufferSource + the source_of ADL protocol (Device.h) are the
pull-model interchange vocabulary; current() runs on the consumer's
thread and holds nothing, so a producer on another thread publishes
through a role-swapping type on the consumer's side.
cuda/metal remain scaffolding stubs (try_open returns nullptr).

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
  call order; any operation on a buffer with queued work synchronizes
  first — read() *and* write(). Backends may batch and sync only where
  the host observes.
- **Device timeline.** submitted()/completed()/wait() expose one
  monotonic Ticket per run(), plus non-virtual flush(); run() returns
  the Ticket it occupies. Ticket is a one-field VALUE type (compare,
  never reuse) — not a handle, nothing to release. Virtual with
  defaults meaning "nothing is outstanding", which is accurate for a
  blocking backend — do not make them pure. The rejected shapes stay
  rejected: no Signal/Fence objects, no wait/signal lists on run().
  submit() is the non-blocking half of flush() — a timeline pump for
  external GPU-side waiters, not a sync object — and joins the
  carve-out. Submission is EAGER (0.8): a batch of Options::batch
  dispatches goes out when full, a shorter one waits for an observer
  or submit(); max_queued is only the bound and wants to be at least
  twice the batch. Every host wait is UNBOUNDED unless
  Options::wait_ms bounds it (GPUD_WAIT_MS overrides the field — a
  gate bounds a program that never set it): past the bound wait()
  throws the hung sentence — ticket, completed, submitted, last
  submitted, the two things it can mean — and teardown prints it and
  exits, since a destructor cannot throw and nothing it owns can be
  freed under a device that may still be using it. A lost device
  throws its own sentence from check(). Options::profile (GPUD_PROFILE=1
  wins) stamps every batch begin/end on the device clock and labels it
  "gpud batch"; take_timings() drains BatchTiming records (tickets,
  dispatches, ns) once each and joins the carve-out. Stamps are read in
  batch order and never waited for — MoltenVK publishes them from the
  completion handler, AFTER the semaphore — so an unpublished one waits
  for the next reclaim. Names ("gpud compute queue", "gpud timeline")
  whenever the instance enabled VK_EXT_debug_utils; the owned instance
  enables it when available.
- **Fresh allocations are unspecified.** alloc() may hand back memory a
  destroyed Buffer owned, so its contents are undefined — write before
  reading. Conversely a Buffer may be destroyed while work using it is
  queued; the backend keeps the memory alive. Mock still allocates fresh
  and zero-filled: it is a test double whose log consumers assert on.
- **External synchronization, with one carve-out.** Calls on one Device
  are externally synchronized, except submitted()/completed()/wait()
  and submit(), which are callable from any thread. Distinct Devices are independent.
- **BufferSource holds nothing.** current() is the consumer's call, on
  the consumer's thread, and the pointer it returns is borrowed until
  the consumer's last use of it. A producer must not replace or destroy
  that Buffer meanwhile; a producer on another thread publishes through
  a role-swapping type on the consumer's side. gpud carries the carrier
  and states the rule — it does not carry the sync type.
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
include/gpud/Vulkan.h      vulkan::try_open + try_open_on/AdoptDesc +
                           native_buffer/native_timeline — the second
                           SDK-naming carve-out (dispatchable handles
                           forward-declared, non-dispatchable uint64)
include/gpud/Sdl.h         sdl::try_open + native_device/native_buffer —
                           the first SDK-naming carve-out (two forward
                           declarations, no include): the renderer seam
src/auto/open_default.cpp  env override + #ifdef GPUD_HAS_* priority chain
src/{cuda,metal,vulkan,sdl}/  backends — the include firewall (SDK types
                           live only here; cuda/metal still stubs):
  Device.h / Device.cpp      Device subclass; try_open, run, timeline,
                             batching/deferred release, teardown
  Buffer.h / Buffer.cpp      BufferImpl (: Buffer::Impl) + alloc/write/read
  Kernel.h / Kernel.cpp      KernelImpl (: Kernel::Impl) + do_compile
  CMakeLists.txt             deps: fetch/find per backend, linked PRIVATE
  (vulkan only) Vma.cpp      VMA_IMPLEMENTATION, alone in its TU
tests/                     device_test (mock/open_default units),
                           conformance_test (same suite over every
                           backend; real ones skip when try_open fails),
                           Device.h standalone-compile check
docs/design.md             full design rationale
docs/backend-implementation.md  the plan for implementing backends
                           (dependency/error policy, step order,
                           conformance suite, per-backend specifics)
```

Vulkan specifics worth knowing before touching src/vulkan: no SDK is
linked — volk (fetched, compiled in) dlopens the loader at runtime,
headers come from find_package(Vulkan) or a fetched pinned
Vulkan-Headers, and VMA is fetched download-only the same way volk is;
requires driver bufferDeviceAddress (no descriptor-set fallback) and
timelineSemaphore, both enabled explicitly (an ADOPTED device's
creator must have enabled the same two); push data = scalar blob then
8-aligned buffer addresses, range fixed at 128 bytes; the
portability-enumeration/subset handling is generic Khronos portability
code, NOT MoltenVK-specific — keep it free of platform #ifdefs.
Adoption (try_open_on) seeds volk's PROCESS-GLOBAL tables through the
app's vkGetInstanceProcAddr — the documented limit stays one
vulkan-backend Device per process (volkLoadDeviceTable is the named
refactor if that ever pinches); a consumer recording a native_buffer
into its own GPU work owns the other half of the lifetime bargain,
because last_use sees only this Device's dispatches.

SDL specifics worth knowing before touching src/sdl: SDL3 is found,
never fetched (system package or CMAKE_PREFIX_PATH — this machine:
`CMAKE_PREFIX_PATH=~/Projects/toolchains/sdl3 make`); the dialect is
"slang-slot" (numbered [[vk::binding]] resources — read-only storage
set 0, read-write set 1, uniforms set 2, exactly SDL's SPIR-V compute
convention — scalars behind a named ConstantBuffer); SDL declares a
pipeline's resource counts and threadgroup size at creation and gpud
carries no reflection, so do_compile SCANS THE SOURCE for its
declarations — legitimate only because the dialect is self-describing,
and part of the dialect pact; run() is ASYNC (v0.6): submit + fence
onto a ring, the ticket claimed only AFTER a successful submit (an SDL
fence bakes in no value, so vulkan's failed-submit host-signal trap
does not arise), wait() looping on a condition variable because a
fence ring is not waiter-independent, completed() answering from an
atomic mirror behind a try_lock so it never blocks; write() drains
outstanding dispatches before its upload (SDL forbids overwriting
referenced data, and cycling would rotate the resource out from under
native_buffer() handles) while read() needs only its own fence on the
in-order queue; try_open points
SDL_VULKAN_LIBRARY at a /usr/local or /opt/homebrew loader when unset
(SDL dlopens by bare name and misses /usr/local/lib; a user's env
wins); SDL_InitSubSystem/SDL_QuitSubSystem are ref-counted and paired
inside the Device's lifetime — the RAII reading of "no library
init/shutdown". Native-Metal MSL is the planned phase 2 (slang
-target metal; the entry-naming and set→argument-table mapping risks
live there).

Teardown order in ~Device is load-bearing and easy to get subtly wrong:
idle OUR queue, bracketed (vkDeviceWaitIdle would stall an adopted
device's other queues) → drain deferred releases *unconditionally*
(ticket-checked would strand entries behind a never-signalled ticket)
→ clear_kernels() (the base cache destructs after the derived dtor,
and KernelImpls hold pipelines) → vmaDestroyAllocator (after the
drain, whose closures call vmaDestroyBuffer) → semaphore → command
pool → device → instance, the last two only when owned.

Three more traps in the batching code, all commented in place: the
throttle must use the internal wait_locked(), never the public wait(),
which would re-lock a non-recursive mutex; the ticket is claimed only
after the dispatch is recorded, with the lock never released in between;
and a failed submit host-signals the timeline, since the tickets were
already handed out and nothing else would ever signal them. Note that
synchronization validation cannot verify the compute→compute barrier
here — it tracks hazards via descriptor bindings and this ABI passes
device addresses in push constants — so do not treat a clean syncval run
as evidence about the barrier.

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

## Releasing

Versioning is SemVer on annotated git tags; consumers (e.g. the tensor
repo) pin `GIT_TAG vX.Y.Z` in FetchContent, so releasing is what makes
a change reachable. Pre-1.0 policy: MINOR may break the interface,
PATCH never does. The version lives in exactly one place —
`project(gpud VERSION X.Y.Z)` — and `<gpud/Version.h>` is generated
from it (never edit or commit a generated Version.h). To release:

1. Bump `project(gpud VERSION X.Y.Z)` in CMakeLists.txt.
2. Commit, then `git tag -a vX.Y.Z -m "<one-line summary>"`.
3. `git push --follow-tags`.

The tag and project() version must always match.

## Build / test

```
make          # cmake -B build (Release, default compiler) + build
make test     # ctest --output-on-failure
make clean
```

Tests fetch googletest via FetchContent (needs network on first
configure). tests/device_h_standalone.cpp is a build-time check that
Device.h compiles alone under -std=c++20.

## Implementing a backend (metal/cuda; vulkan and sdl are the worked references)

Follow docs/backend-implementation.md and mirror src/vulkan. The
scaffolding already exists: factory header, the src/ dir with class
skeletons (Device subclass, BufferImpl, KernelImpl — every TODO(impl)
marks a hole) and its CMakeLists, the option gate, GPUD_HAS_* plumbing,
the open_default arms, and the conformance suite (add the backend's
factory + saxpy kernel source to tests/conformance_test.cpp). To
implement:

1. In `src/<backend>/`: fill the TODO(impl) holes — SDK state into the
   headers' member slots, real bodies replacing the unimplemented()
   stubs (Device.cpp: try_open + run; Buffer.cpp: alloc/write/read;
   Kernel.cpp: do_compile). try_open constructs the Device (still
   nullptr when no driver/device/compiler, quietly; GPUD_LOG=1
   explains). SDK headers are included ONLY in this dir — the public
   header must keep declaring the factory with no SDK types. Keep
   teardown ordering, driver quirks, and compiler invocations private
   to these files. If Kernel impls hold device resources, the Device
   destructor must call clear_kernels() before releasing them.
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
