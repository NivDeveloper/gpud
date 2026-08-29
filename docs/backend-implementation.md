# Backend implementation plan

Status: plan, 2026-07. The generic template applies to every backend;
Vulkan is instantiated in full below and goes first. Metal and CUDA
follow the same template with their sections' specifics.

## Generic template

### Dependency policy — three tiers

1. **Build-time.** Acquired automatically at configure when the backend
   option is ON; never vendored into the repo, never a manual install
   where avoidable. Preference order:
   a. nothing (resolve the library at runtime via dlopen),
   b. find system first, FetchContent a pinned copy as fallback — only
      for small redistributables (headers, volk),
   c. find_package REQUIRED — only where unavoidable (CUDA toolkit).
2. **Link-time.** Avoid linking the vendor library where the platform
   allows (dlopen instead): a machine without it then gets
   `try_open() == nullptr` instead of a build failure.
3. **Runtime.** Driver + kernel compiler are probed in try_open, never
   assumed. ALL probing lives in try_open; after it succeeds,
   everything is assumed present.

A fresh machine with compiler + CMake + git must be able to *build*
any backend (CUDA excepted, see below). What it needs to *run* one is
documented per backend in the README (driver + kernel compiler, one
line each).

### Error policy

- try_open: any missing library/device/feature/tool → nullptr,
  quietly. Set `GPUD_LOG=1` in the environment to get a one-line
  stderr reason per failed probe (answers "why is my GPU not used?").
- After open: failures (kernel compile error, OOM, device loss) throw
  `std::runtime_error` with context — compiler diagnostics included
  verbatim. The interface has no other error channel; these are
  exceptional. Document this in Device.h when the first backend lands.

### Implementation order (each step compiles and is tested before the next)

1. **Bring-up** — try_open (all probes, device_index selection,
   feature checks) + Device constructor/destructor, teardown in
   reverse order. Test: open-or-skip, dialect(), clean destruction.
2. **Storage** — BufferImpl + alloc/write/read with the simplest
   correct memory strategy (host-visible/shared memory; staging and
   residency are post-v1). Test: round-trip.
3. **Kernels** — do_compile via the dialect toolchain. Test: trivial
   kernel compiles; a broken kernel throws with diagnostics.
4. **Execution** — run(), fully blocking in v1 (submit + wait before
   returning). The ordering contract explicitly permits this; batching
   with sync-at-read is a later optimization that lands with the
   conformance tests already in place to catch regressions.
5. **Wiring** — flip the backend option default from OFF to
   auto-detection, README runtime-requirements line, CLAUDE.md status.

### Conformance suite (write once, with Vulkan; every backend inherits)

tests/conformance_test.cpp, gtest value-parameterized over backend
factories `{name, open-fn, vector-add kernel source or empty}`:

- every backend (mock included): open-or-skip; dialect() nonempty;
  alloc/write/read round-trip; unwritten buffer reads zeros.
- real backends only (those with a kernel entry in the per-dialect
  source table): compile + run `out[i] = a[i] + s*b[i]` end-to-end and
  compare against the CPU loop — exercises scalars, positional
  buffers, and a multi-workgroup dispatch in one test.
- skip (GTEST_SKIP) whenever try_open returns nullptr, so the suite
  passes on driverless CI and fresh machines.

Adding a backend to the suite = one factory + one kernel-source table
entry.

### v1 non-goals (unchanged from docs/design.md)

Staging and buffer residency, streams, capability introspection beyond
dialect(). (Batching, recycled allocation and a narrow thread-safety
carve-out are no longer non-goals — see below.)

### Batching, tickets and the allocator (added 0.2, after Vulkan)

Step 4 below says run() may be fully blocking, and that is still the
right way to *start*. When a backend outgrows it, the three pieces
arrive together, because they are all the same question — *has the work
touching this memory completed?*

1. **A timeline**, implementing `submitted`/`completed`/`wait`. Every
   backend has the primitive (docs/design.md has the table): Vulkan a
   timeline `VkSemaphore`, Metal an `MTLSharedEvent`, CUDA an in-order
   stream plus an event per value. Leave the base-class defaults alone
   until then — they say "nothing is ever outstanding", which is exactly
   true of a blocking backend.
2. **Deferred release**: a FIFO of closures keyed by ticket, drained
   oldest-first and stopping at the first not complete. A Buffer handle
   may die while queued work still reads its memory; its Impl's
   destructor hands the teardown here instead of doing it inline. The
   Device destructor drains *unconditionally* after going idle — a
   ticket-checked drain strands everything behind a ticket that was
   never signalled.
3. **The platform allocator — do not write a free list.** Vulkan uses
   VMA. CUDA gets this free from `cudaMallocAsync`/`cudaMemPool_t`:
   stream-ordered allocation is precisely this design, built into the
   API, so points 2 and 3 largely collapse into the driver. Metal has
   `MTLHeap`. The allocator is per-backend and must not migrate into the
   core.

Two things that bit the Vulkan implementation and will bite the next
one. **Order of operations inside run():** claim the ticket only after
the dispatch is fully recorded and never drop the lock in between, or a
concurrent `wait()` can submit a batch promising work that was never
recorded. **Failed submission:** tickets are handed out at record time,
so if the submit fails nothing will ever signal them and every later
wait — teardown included — hangs; signal the timeline from the host
before rethrowing.

## Vulkan (first, in full) — IMPLEMENTED as planned

Everything below was built as written (plus one interface addition the
plan missed: base-class kernel cache members destruct after the derived
destructor, so gpud::Device grew a protected clear_kernels() that
backend destructors call before tearing down device state).

Works on macOS through MoltenVK with **zero MoltenVK-specific code**:
everything Apple-flavored below is the generic Khronos portability
pattern, driven by runtime extension queries — no `#ifdef __APPLE__`
anywhere in src/vulkan.

### Dependencies

| tier | what | how |
|---|---|---|
| compile | Vulkan-Headers | find_package(Vulkan QUIET) for headers; else FetchContent pinned Khronos Vulkan-Headers. `VK_NO_PROTOTYPES` everywhere. |
| link | none | volk via FetchContent (pinned; one .c/.h), compiled into gpud_vulkan. volkInitialize() dlopens libvulkan — on macOS it falls back to libMoltenVK.dylib itself, so `brew install molten-vk` alone suffices. |
| runtime | Vulkan ≥ 1.2 driver with bufferDeviceAddress | MoltenVK (macOS), mesa/vendor driver (Linux/Windows) |
| runtime | slangc | the "slang-vulkan" dialect compiler; slang release or LunarG SDK |

### try_open sequence (every step: fail → cleanup → nullptr)

1. volkInitialize (no loader → nullptr).
2. vkCreateInstance, apiVersion 1.2. Enumerate instance extensions
   first: if VK_KHR_portability_enumeration is present, enable it +
   VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR (generic
   portability-driver handling; a no-op elsewhere).
3. Enumerate physical devices; pick Options::device_index, or -1 =
   first discrete GPU, else first device.
4. Require VkPhysicalDeviceVulkan12Features::bufferDeviceAddress
   (MoltenVK has it on Metal-3 hardware; without it the BDA dialect
   can't work — nullptr, not a fallback path).
5. Create logical device + one compute queue; enable
   VK_KHR_portability_subset iff the device advertises it (spec
   requirement, again generic).
6. Command pool + one primary command buffer + one fence (reused by
   every run()).
7. Probe slangc: try /usr/local/bin/slangc, /opt/homebrew/bin/slangc,
   then bare `slangc` via `-h` exit code (std::system's shell may not
   have /usr/local/bin on PATH — vklib-proven). Cache the found path
   in the Device.

### Storage

One VkBuffer per Buffer: usage STORAGE | SHADER_DEVICE_ADDRESS,
HOST_VISIBLE | HOST_COHERENT, persistently mapped. write = memcpy in;
read = wait out the buffer's last ticket, then memcpy out.

*Superseded (0.2):* this originally used one dedicated
vkAllocateMemory + vkMapMemory per buffer, on the reasoning that it was
fine at this scale and saved a dependency. It was not — it cost ~52
us/MB, and the validation layer's best-practices checks flag every such
buffer. Allocation now goes through **VMA**, which suballocates from
large blocks; BufferImpl = { VmaAllocator, VkBuffer, VmaAllocation,
void* mapped, VkDeviceAddress, last_use ticket }. Two details are
load-bearing: the allocator needs
VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT (it is what tags the
backing memory for BDA), and HOST_VISIBLE|HOST_COHERENT must be
*required* rather than preferred, or USAGE_AUTO may hand back
host-cached non-coherent memory that needs flush/invalidate around every
copy. The zero-fill that used to give mock parity is gone with it: a
fresh buffer's contents are unspecified, which is what makes recycling
free.

### Kernels

do_compile: write source to fs::temp_directory_path(), shell out —

    slangc <src>.slang -target spirv -entry main -stage compute
           -fvk-use-entrypoint-name -emit-spirv-directly -o <out>.spv

(the exact vklib-proven invocation), capture stderr; failure → throw
std::runtime_error carrying the diagnostics. Then vkCreateShaderModule
→ pipeline layout with a single push-constant range (compute stage,
size 128 — the spec-guaranteed maxPushConstantsSize minimum, no
per-kernel reflection) → vkCreateComputePipelines. KernelImpl =
{ VkPipeline, VkPipelineLayout } (shader module destroyed right after
pipeline creation).

Push-data ABI (the dialect pact): the scalar blob as emitted, then one
VkDeviceAddress per buffer in positional order. run() asserts
scalars.size() + 8 * buffers.size() <= 128.

### run (v1: blocking; superseded 0.2)

Originally: begin the shared command buffer → bind pipeline →
vkCmdPushConstants(scalars ⧺ addresses) → vkCmdDispatch(groups, 1, 1)
→ end → submit with the shared fence → wait → reset. ~40 lines, correct
by construction, and it did prove the batching upgrade "only touches
run/read internals — interface and tests unchanged".

*Now:* run() records into a lazily-begun batch command buffer and
returns; submission happens where the host observes. Each dispatch is
preceded by a **global compute→compute VkMemoryBarrier**
(SHADER_WRITE → SHADER_READ|SHADER_WRITE). One barrier suffices because
a barrier's first synchronization scope covers everything earlier in
submission order *on that queue*, batch boundaries included — which
holds only because a gpud Device owns exactly one queue. Each batch also
ends with a compute→host barrier so device writes are available to the
host domain. Command buffers are recycled through a free list rather
than freed via the deferred queue, keeping every command-pool touch on
one code path (pools require external synchronization, and the deferred
queue can be drained by whichever thread calls wait()).

Note for anyone trying to test that barrier: synchronization validation
cannot see it. Syncval tracks hazards through descriptor bindings, and
this dialect passes buffers as device addresses in push constants, so it
observes no buffer accesses to order. Nor does a dependent-dispatch
chain catch its absence on MoltenVK, which coalesces dispatches into a
Metal encoder that orders them regardless.

### Teardown (~Device, reverse order)

Wait queue idle → drain deferred releases unconditionally → destroy
cached Kernels' pipelines/layouts (the base class map destructs
KernelImpls) → VMA allocator (after the drain, whose closures call
vmaDestroyBuffer) → semaphore → command pool → device → instance.
Handle destruction order relative to the Device is the caller's
contract (handles-don't-outlive-device); nothing to do here.

### Size estimate

~450 lines across the existing six files; bring-up (Device.cpp) is the
bulk. No new files needed beyond what's scaffolded.

## SDL_GPU — IMPLEMENTED (v1, blocking)

The portability-layer backend, and the renderer seam: a consumer can
claim a window on the exported native SDL_GPUDevice and read compute
buffers zero-copy (allocations carry GRAPHICS_STORAGE_READ). Sits LAST
in auto-selection — native backends first.

- Dependency tier: found, never fetched (`find_package(SDL3 CONFIG)`;
  a system package or a prefix on CMAKE_PREFIX_PATH). SDL is not a
  small redistributable, so the volk-style pinned fallback does not
  apply.
- dialect "slang-slot": numbered `[[vk::binding(k, set)]]` resources —
  read-only storage buffers dense from 0 in set 0, ONE read-write
  output at (0, 1), at most one uniform block `ConstantBuffer<Scalars>`
  at (0, 2) — which is SDL's documented SPIR-V compute convention.
  run(): scalars via SDL_PushGPUComputeUniformData slot 0, buffers[0]
  as the pass's read-write binding, buffers[1+k] bound read-only at
  slot k. std140 for a struct of 4-byte scalars equals the natural
  packing consumers already use, so the one blob serves both Slang
  dialects.
- do_compile: the vulkan backend's slangc invocation verbatim (the
  binding decorations are in the source), then a SOURCE-TEXT SCAN for
  the declared resource counts and [numthreads(...)] — SDL declares
  both at pipeline creation and gpud carries no reflection. Legitimate
  only because the dialect is self-describing; the scan is part of the
  dialect pact.
- v1 execution is FULLY BLOCKING (submit + fence wait in run, write
  and read), so the base timeline defaults stand and buffer teardown
  releases immediately. Batching = fences ring + tickets, the vulkan
  shape, when a consumer's frame loop needs it.
- try_open probes: slangc (same three paths), SDL_InitSubSystem
  (ref-counted, paired in ~Device), the SDL_VULKAN_LIBRARY hint filled
  from /usr/local//opt/homebrew when unset (SDL dlopens the loader by
  bare name and misses /usr/local/lib on macOS; env wins), then
  SDL_CreateGPUDevice(SPIRV).
- The public header carve-out: `include/gpud/Sdl.h` forward-declares
  SDL_GPUDevice/SDL_GPUBuffer (no include) for native_device /
  native_buffer — the ONE deliberate exception to the no-SDK-names
  rule, because the renderer seam is the backend's purpose.
- Phase 2, recorded: native MSL via slang -target metal +
  SDL_GPU_SHADERFORMAT_MSL, choosing the target from
  SDL_GetGPUShaderFormats — the entry-naming (MSL reserves `main`) and
  set→argument-table mapping risks live there.

## Metal (same template, later)

- Build deps: none to fetch or find — frameworks ship with Xcode/CLT;
  the option gate becomes `${APPLE}`. Sources → .mm +
  enable_language(OBJCXX); link Metal/Foundation PRIVATE (linking is
  fine here: always present on the only platform that can build it).
- try_open: MTLCopyAllDevices, pick by device_index (-1 →
  MTLCreateSystemDefaultDevice), command queue. No external kernel
  compiler to probe — MSL compiles in-process.
- Storage: MTLBuffer, storageModeShared; memcpy via contents.
- do_compile: newLibraryWithSource ("metal" dialect = MSL) → function
  → compute pipeline state; NSError text → the runtime_error.
- run ABI: buffers at [[buffer(0..N-1)]] via setBuffer positional,
  scalar blob via setBytes at [[buffer(N)]]. v1 blocks with
  waitUntilCompleted.

## CUDA (same template, later)

- The one backend where a build-time install is unavoidable:
  find_package(CUDAToolkit REQUIRED) for cuda.h + NVRTC (headers are
  not redistributable-fetchable the way Khronos' are). Option default
  = CUDAToolkit_FOUND, so fresh non-NVIDIA machines are untouched.
  Softening link-time via dlopen of libcuda/libnvrtc is possible later
  if it ever matters.
- try_open: cuInit(0) failure → nullptr (this is the no-driver probe);
  device by index; retain primary context; one stream.
- Storage: cuMemAlloc / cuMemcpyHtoD / cuMemcpyDtoH.
- do_compile: NVRTC in-process ("cuda" dialect) → PTX →
  cuModuleLoadData → cuModuleGetFunction; NVRTC log → the
  runtime_error.
- run ABI: kernel params = pointer to scalar blob fields per the
  dialect's declaration, then one CUdeviceptr per buffer positional;
  cuLaunchKernel(groups,1,1). v1 syncs the stream before returning.
