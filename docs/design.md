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
   plus a single introspection member, `dialect()`, and (since 0.2) the
   device timeline `submitted`/`completed`/`wait` for observing queued
   work without forcing it. Since 0.5, `BufferSource` — the pull-model
   carrier producers and consumers exchange, and the ONE piece of gpud
   whose reason to exist is another library.
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

// The pull-model carrier (0.5): a producer's ADL source_of(const P &)
// returns one; a consumer asks current() at the moment of use.
struct BufferSource { Buffer *(*fn)(void *); void *user; Buffer *current() const; };

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
calls on one `Device` behave as if executed in call order, and any
operation on a buffer with queued work synchronizes first — `read`
returns only after every prior `run` touching that buffer has completed,
and `write` waits rather than overwriting storage a queued dispatch
still reads. A simple backend makes everything blocking; the Vulkan
backend batches submissions and syncs only where the host observes
(see the device timeline below).

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

**Threading.** Calls on one Device are externally synchronized, with one
carve-out: `submitted`/`completed`/`wait` may be called from another
thread while one is inside a Device call, since observing the timeline
from elsewhere is the only reason to expose it. Distinct Devices are
independent. `BufferSource::current()` runs on the CONSUMER's thread
and holds nothing: the producer must not replace or destroy the
returned Buffer until the consumer's last use of it, which for a
renderer is the submit. A producer on the consumer's thread has that
by program order; one on its own thread publishes through a
role-swapping type on the consumer's side, which hands `current()` a
slot the producer is not writing. The native seam (`Sdl.h`) adds a
second, separate rule: what it hands out are SDL objects under SDL's
own threading rules, which is why a renderer records on one thread
while `run()` blocks on another.

**Why these five and not more**: `alloc`/`write`/`read` is the minimum
to get data there and back; `compile`/`run` is literally "build and run
kernels". Deliberately excluded:

- device enumeration/selection — a backend factory's job
- buffer device addresses, mapping, raw pointers — dialect-specific
- descriptor sets, pipelines, barriers, command buffers — backend internals
- async handles/streams — covered by the ordering contract until
  profiling says otherwise
- shader-reflection/metadata queries — the ABI is positional
- a producer/consumer sync type — the consumer's: it knows the frame,
  gpud does not, and `BufferSource` is deliberately only the carrier

**Portability check** — how each target maps:

| Backend | compile                      | scalars        | buffers                              |
|---------|------------------------------|----------------|--------------------------------------|
| Vulkan  | slangc → SPIR-V → pipeline   | push constants | BDA addresses appended to push data  |
| Metal   | slang `-target metal` → PSO  | `setBytes`     | `setBuffer(k)` positional            |
| CUDA    | NVRTC / slang → PTX          | kernel params  | device pointers as params            |
| SDL_GPU | slangc → SPIR-V → pipeline   | uniform slot 0 | numbered storage-buffer slots        |
| Mock    | records the call             | records the call | records the call                   |

The mock backend is the test double: consumers exercise their GPU glue
with no GPU and no SDK in CI.

## Repo shape: header-only core + one compiled library per backend

```
gpud/
  include/gpud/Device.h     the interface — no deps beyond std (INTERFACE target)
  include/gpud/Mock.h       mock backend, header-only — consumers' tests need no SDK
  include/gpud/Auto.h  + src/auto/*.cpp       open_default() — see below
  include/gpud/Vulkan.h + src/vulkan/*.cpp    factory decl only / implemented: volk + fetched headers, no SDK linked
  include/gpud/Metal.h  + src/metal/*.mm      APPLE only (stub today, .cpp until real)
  include/gpud/Cuda.h   + src/cuda/*.cpp      CUDA toolkit present only (stub today)
```

CMake targets: `gpud::gpud` (interface), `gpud::mock`, `gpud::auto_`,
and `gpud::vulkan` / `gpud::metal` / `gpud::cuda` (scaffolding stubs
until implemented). Each real backend is gated on an option + SDK
auto-detection
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
  later without changing the interface. *(Partly closed since: the
  device timeline below is built, and with it a narrow thread-safety
  carve-out. Streams remain deliberately absent — the answer to queue
  parallelism is still "open two Devices".)*
- **Blocking submission.** *(Closed for both real backends: vulkan
  batches on a timeline semaphore, sdl submits per dispatch onto a
  fence ring; see
  the device timeline below.)*

## Finer-grained synchronization: the device timeline (BUILT, 2026-08)

Built as designed below, together with batched submission and recycled
allocation — they are one mechanism, not three, because all of them
reduce to *has the work touching this memory completed?* What changed
against the note as originally written is recorded at the end of this
section.

Host-visible sync is a **device timeline** (tickets), not semaphore
objects.

**The cross-backend primitive is the timeline** — a monotonically
increasing 64-bit counter that work signals and anyone can wait on or
poll. Every backend has converged on it:

| Backend | timeline primitive                                | host wait                | poll                           |
|---------|---------------------------------------------------|--------------------------|--------------------------------|
| Vulkan  | timeline `VkSemaphore` (core 1.2, already required) | `vkWaitSemaphores`     | `vkGetSemaphoreCounterValue`   |
| Metal   | `MTLSharedEvent` (value-based signal/wait)        | `waitUntilSignaledValue` | `.signaledValue`               |
| CUDA    | in-order stream + `cuEvent` per value             | `cuEventSynchronize`     | `cuEventQuery`                 |
| Mock    | a counter                                         | no-op                    | `==`                           |

Vulkan's binary semaphores and fences are the legacy shapes; timelines
were introduced precisely because the counter model matches D3D12
fences, `MTLSharedEvent`, and CUDA's stream/event model.

**Why tickets and not semaphore objects.** A gpud Device is one
in-order execution context (one queue/stream), so ordering within a
device is already total — GPU→GPU semaphores between its own commands
add nothing. The only thing fine-grained sync buys here is letting the
host observe and wait on *positions in that order* without the big
hammer of `read()`. That needs no handle — one comparable value type:

```cpp
struct Ticket { std::uint64_t value; };   // one tick; compare to order

class Device {
    // ... the five ops; run() returns the Ticket it occupies ...

    // The device timeline: every run() (and, with batching, every
    // enqueued write/read) occupies one tick, in call order.
    virtual Ticket submitted() const;   // ticket of last enqueued work
    virtual Ticket completed() const;   // highest finished ticket (poll)
    virtual void wait(Ticket);          // block host until ticket done
};
```

Tickets *refine* the ordering contract rather than replacing it:
`read()` keeps its exact meaning and becomes, internally, "wait(last
ticket touching this buffer) + copy". No move-only handle with
lifetime rules, no wait/signal lists on run(), and values are never
reused, so the scheme is race-free by construction (the same property
that makes timelines strictly better than binary semaphores).

Rejected: explicit Signal/Fence objects with wait/signal lists on
run() (the VkSubmitInfo shape). More expressive — arbitrary DAGs,
cross-device edges — but on a single in-order queue a DAG degenerates
to submission order, so the ceremony buys nothing until gpud exposes
multiple queues, which it deliberately doesn't. If queue-level
parallelism ever matters, the design-consistent move is *open two
Devices*; the ticket model then extends with one primitive
(`wait_on(other_device, ticket)`). Cross-*backend* waits (Vulkan →
Metal) would require OS-level external semaphores and are out of
scope; host-side `wait` + `write` already bridges devices correctly.

Sequencing and contract notes, borne out in the build:

- **Land it with (or after) batching, not before.** Under v1's blocking
  run() every ticket is complete before run() returns — the API would
  be dead weight. Batching and tickets are the same feature seen from
  inside and outside: the submission counter batching needs internally
  is exactly what `submitted()` exposes.
- **The real contract change is threading, not the primitive.**
  wait()/completed() are only useful from another thread; v1 says
  calls on one Device are externally synchronized. All three backends
  support host wait/poll from any thread, so the carve-out ("wait and
  completed are thread-safe; everything else stays externally
  synchronized") is cheap — but must be explicit in Device.h.
- **Interop-grade sync** (handing a semaphore to a renderer — external
  semaphore handles, `MTLSharedEvent` export, `cuImportExternalSemaphore`)
  is the one case that would need an exported opaque handle. Separate,
  later decision; nothing in the ticket model forecloses it. The
  HOST-side half of that question — a producer on one thread, a
  renderer on another, sharing a buffer — is answered without it: a
  role-swapping type on the consumer's side (the Threading paragraph
  above), which needs no GPU sync at all for a producer that parks a
  fresh buffer per step.

### What the build added to the note

- **The trio is virtual with defaults, not pure.** `submitted()` and
  `completed()` return 0 and `wait()` does nothing, which is *accurate*
  for a backend that completes every call before returning. That keeps
  the stub backends and the mock's storage half conformant without
  touching them, and makes the ticket API additive for out-of-tree
  backends. `flush()` (= `wait(submitted())`) is public non-virtual for
  the same reason `compile()` is: a backend that got the other two right
  cannot get it wrong.
- **Deferred release is the second half.** A `Buffer` whose handle dies
  hands its teardown to a FIFO of closures keyed by ticket, drained
  oldest-first and stopping at the first that isn't complete. That is
  what lets a caller destroy a buffer straight after `run()`, and what
  makes recycled allocation safe — so the contract gained "a fresh
  buffer's contents are unspecified", which is what pays for it.
- **Boundedness is one knob, not two mechanisms.** `Options::max_queued`
  caps outstanding *tickets*, which bounds recorded commands, in-flight
  submissions and memory awaiting release together, at one stall per
  `max_queued` dispatches. Bounding *bytes* is a separate concern and
  belongs to `Options::pool_budget_bytes`.
- **The allocator is per-backend and off-the-shelf.** Vulkan uses VMA.
  Nothing about it belongs in the core: CUDA gets the same behaviour
  from `cudaMallocAsync`/`cudaMemPool_t` (stream-ordered allocation *is*
  this design, built into the API) and Metal from `MTLHeap`. Do not
  write a free list.
- **Batching needs one barrier, and its correctness rests on there being
  one queue.** A global compute→compute barrier before every dispatch
  orders it against everything earlier in submission order on that
  queue — across batch and submission boundaries alike — so no
  per-buffer tracking is needed. This is exactly the assumption that
  breaks if a Device ever owns more than one queue; the design-consistent
  answer there stays "open two Devices".
- **Verification caveat worth knowing.** Vulkan synchronization
  validation cannot check that barrier: it tracks hazards through
  descriptor bindings, and this dialect passes buffers as device
  addresses in push constants, so it sees no buffer accesses to order.
  Nor will a chain test catch a missing barrier on MoltenVK, which
  coalesces dispatches into a Metal encoder that orders them anyway. The
  barrier rests on the spec argument above; the tests cover chaining,
  batch boundaries and throttling, which is a different thing and worth
  not confusing with it.
- **The SDL backend joined (v0.6), and what forced it.** Beside a
  presenting renderer on the shared device a fence wait costs a
  display refresh — measured 15.26 ms per dispatch against 0.16 ms
  uncontended, which took a consumer's sim from 1020 sweeps/s to 9 —
  and the blocking run() paid it once per dispatch. run() now submits
  and keeps the fence on a ring; the ticket is claimed only after a
  SUCCESSFUL submit (an SDL fence bakes in no value, so the
  failed-submit host-signal trap does not arise); completed() answers
  from an atomic mirror behind a try_lock and never blocks; wait()
  loops on a condition variable, because a fence ring — unlike a
  timeline value — is not waiter-independent and another waiter may
  hold the prefix. No last_use and no defer_release: the one in-order
  queue plus SDL's pass-boundary hazard tracking order run→run and
  run→read, read() still waits its own fence, and buffer teardown
  rides SDL's deferred free, since everything gpud recorded was
  submitted. write() is the one that changed: it drains outstanding
  dispatches first — SDL's header forbids overwriting referenced
  data, and cycling would rotate the internal resource out from under
  native_buffer() handles.
