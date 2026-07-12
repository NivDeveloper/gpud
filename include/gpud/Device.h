#pragma once

// gpud — a minimal GPU compute abstraction.
//
// This header is the entire abstract interface and is deliberately free
// of any GPU SDK types: backends are separately compiled libraries that
// hide their SDK behind the Impl seam below. Libraries that *use* GPU
// compute include only this header and accept a Device&; only the
// application's composition root names a concrete backend (via a
// per-backend factory header, or Auto.h).
//
// Contract (v1):
//
//  1. Ordering. Calls on one Device behave as if executed in call order.
//     read() returns only after every prior run() touching that buffer
//     has completed. A backend may batch/queue work internally and
//     synchronize only at read(); a simple backend may make every call
//     blocking. There are no fences, events, or streams in the interface.
//
//  2. Positional buffers. run()'s buffer list is positional: buffers[0]
//     is the output, buffers[1 + k] is input leaf k. The caller and the
//     code generator that produced the kernel source agree on that
//     order; how each buffer reaches the kernel (device-address push
//     constants, setBuffer slot, kernel parameter) is the backend's
//     business, paired with its dialect(). gpud carries no reflection
//     metadata.
//
//  3. External synchronization. Calls on one Device must be externally
//     synchronized (v1). Distinct Devices are independent.
//
//  4. Lifetime. All state hangs off the Device and dies with it; there
//     is no library-wide init or shutdown. Handles (Buffer, Kernel) must
//     not outlive the Device that created them, and must only be passed
//     back to that same Device.

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace gpud {

// Backend-open options, consumed by the per-backend factories
// gpud::<backend>::try_open(const Options& = {}). Kept deliberately
// minimal.
struct Options {
    int device_index = -1;   // -1 = let the backend pick the obvious device
};

// Move-only RAII handle to a device allocation. Backends subclass Impl;
// consumers treat the handle as opaque. A default-constructed or
// moved-from handle is empty.
class Buffer {
  public:
    struct Impl {
        virtual ~Impl() = default;
    };

    Buffer() = default;
    Buffer(std::unique_ptr<Impl> impl, std::size_t bytes)
        : impl_(std::move(impl)), bytes_(bytes) {}

    Buffer(Buffer &&) = default;
    Buffer &operator=(Buffer &&) = default;

    // Size requested at alloc().
    std::size_t bytes() const { return bytes_; }

    // Backend-facing: the concrete Impl this handle wraps (null if empty).
    Impl *impl() const { return impl_.get(); }

    explicit operator bool() const { return impl_ != nullptr; }

  private:
    std::unique_ptr<Impl> impl_;
    std::size_t bytes_ = 0;
};

// Move-only RAII handle to a compiled kernel. Same rules as Buffer.
class Kernel {
  public:
    struct Impl {
        virtual ~Impl() = default;
    };

    Kernel() = default;
    explicit Kernel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

    Kernel(Kernel &&) = default;
    Kernel &operator=(Kernel &&) = default;

    Impl *impl() const { return impl_.get(); }

    explicit operator bool() const { return impl_ != nullptr; }

  private:
    std::unique_ptr<Impl> impl_;
};

class Device {
  public:
    virtual ~Device() = default;

    // Kernel-source dialect this device consumes, e.g. "slang-vulkan",
    // "metal", "cuda", "mock". The one introspection member: it lets a
    // code generator pick what to emit when the backend was chosen at
    // runtime (open_default()).
    virtual std::string_view dialect() const = 0;

    // ── storage ──────────────────────────────────────────────────────
    virtual Buffer alloc(std::size_t bytes) = 0;
    virtual void write(Buffer &dst, const void *src, std::size_t bytes) = 0;
    virtual void read(const Buffer &src, void *dst, std::size_t bytes) = 0;

    // ── build ────────────────────────────────────────────────────────
    // Memoizes on source.data() *identity*, not content: callers must
    // pass sources whose storage is stable for the Device's lifetime
    // (string literals / static storage). Equal text at a different
    // address compiles again, and a dangling data() pointer could alias
    // a cache entry. The first call per identity forwards to
    // do_compile(); the cache dies with the Device.
    const Kernel &compile(std::string_view source) {
        if (auto it = kernels_.find(source.data()); it != kernels_.end())
            return it->second;
        return kernels_.emplace(source.data(), do_compile(source))
            .first->second;
    }

    // ── run ──────────────────────────────────────────────────────────
    // Launch `groups` workgroups. `scalars` is the scalar section of the
    // kernel's parameter data, laid out exactly as the dialect declared
    // it. `buffers` is positional — see contract note 2 above.
    virtual void run(const Kernel &kernel, std::size_t groups,
                     std::span<const std::byte> scalars,
                     std::span<Buffer *const> buffers) = 0;

  protected:
    virtual Kernel do_compile(std::string_view source) = 0;

  private:
    std::unordered_map<const void *, Kernel> kernels_;
};

} // namespace gpud
