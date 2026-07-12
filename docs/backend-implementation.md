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

Performance (staging, batching, buffer residency), internal locking,
streams, capability introspection beyond dialect().

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

One VkBuffer per Buffer: usage STORAGE | TRANSFER_SRC | TRANSFER_DST |
SHADER_DEVICE_ADDRESS; one dedicated allocation, HOST_VISIBLE |
HOST_COHERENT, allocated with the DEVICE_ADDRESS flag, persistently
mapped. BufferImpl = { VkBuffer, VkDeviceMemory, void* mapped,
VkDeviceAddress }. write = memcpy in; read = memcpy out (runs are
blocking in v1, so read has nothing to wait for yet — it is still THE
designated sync point when batching lands). No VMA: one allocation per
buffer is fine at this scale and saves a dependency; unified memory
(Apple Silicon, iGPUs) makes host-visible storage buffers fast anyway.

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

### run (v1: blocking)

Reset the shared command buffer → begin → bind pipeline →
vkCmdPushConstants(scalars ⧺ addresses) → vkCmdDispatch(groups, 1, 1)
→ end → queue submit with the shared fence → wait → reset fence.
~40 lines, correct by construction. The batching upgrade later only
touches run/read internals — interface and tests unchanged.

### Teardown (~Device, reverse order)

Wait queue idle → destroy cached Kernels' pipelines/layouts (the base
class map destructs KernelImpls) → pool/fence → device → instance.
Handle destruction order relative to the Device is the caller's
contract (handles-don't-outlive-device); nothing to do here.

### Size estimate

~450 lines across the existing six files; bring-up (Device.cpp) is the
bulk. No new files needed beyond what's scaffolded.

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
