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
//     Any operation on a buffer with queued work synchronizes first:
//     read() returns only after every prior run() touching that buffer
//     has completed, and write() likewise waits rather than overwriting
//     storage a queued dispatch still reads. A backend may batch/queue
//     work internally and synchronize only where the host observes it; a
//     simple backend may make every call blocking. There are no fences,
//     events, or streams in the interface — to watch progress without
//     forcing it, see the device timeline on Device below.
//
//  2. Positional buffers. run()'s buffer list is positional: buffers[0]
//     is the output, buffers[1 + k] is input leaf k. The caller and the
//     code generator that produced the kernel source agree on that
//     order; how each buffer reaches the kernel (device-address push
//     constants, setBuffer slot, kernel parameter) is the backend's
//     business, paired with its dialect(). gpud carries no reflection
//     metadata.
//
//  3. External synchronization, with one carve-out. Calls on one Device
//     must be externally synchronized — except submitted(), completed()
//     and wait(), which are safe to call from another thread while one
//     is inside a Device call. Observing the timeline from elsewhere is
//     the entire point of exposing it, so it would be useless otherwise.
//     Distinct Devices are independent.
//
//  4. Lifetime. All state hangs off the Device and dies with it; there
//     is no library-wide init or shutdown. Handles (Buffer, Kernel) must
//     not outlive the Device that created them, and must only be passed
//     back to that same Device. A Buffer may be destroyed while work
//     using it is still queued: the backend keeps the memory alive as
//     long as anything queued still needs it, so callers need not track
//     that themselves. In exchange, the contents of a freshly alloc()ed
//     buffer are UNSPECIFIED — it may well be backed by memory a
//     destroyed Buffer used to own — so write before you read. (The mock
//     zero-fills, being a test double; no backend promises it.)
//
//  5. Errors. Backend factories (gpud::<backend>::try_open) never throw:
//     they return nullptr when the backend can't come up, for any reason
//     (set GPUD_LOG=1 for a stderr line saying why). Once a Device is
//     open, failed operations throw std::runtime_error — kernel compile
//     errors carry the compiler diagnostics verbatim.

#include <compare>
#include <cstddef>
#include <cstdint>
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

    // Ceiling on dispatches queued but not yet completed. Reaching it
    // makes run() submit what it has and wait for the oldest excess
    // ticket before recording more, which is what stops queued commands,
    // in-flight submissions and memory awaiting release from growing
    // without bound. Costs one stall per max_queued dispatches; raise it
    // to let the host run further ahead, lower it to cap latency and
    // retained memory. Values below 1 are treated as 1.
    std::uint32_t max_queued = 64;

    // Ceiling in bytes on the device memory a backend's allocator may
    // hold; 0 = no limit. Past it an alloc() fails rather than the pool
    // growing indefinitely. Backends that allocate per buffer, with no
    // pool to bound, ignore it.
    std::size_t pool_budget_bytes = 0;
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

// The pull-model interchange: where a logical array currently lives.
// A producer library advertises a type P as a producer by an
// ADL-findable `gpud::BufferSource source_of(const P &)`; a consumer
// calls current() at the moment of use. nullptr means nothing is
// resident right now — consumers skip, they do not fail. The source is
// valid while the producer object lives at its address; the returned
// Buffer obeys the normal handle rules (same Device, do not outlive it).
//
// Threads. current() runs on the CONSUMER's thread and hands back a
// borrowed pointer with no hold on it: the producer must not replace
// or destroy that Buffer while a consumer holds the pointer — from
// the call to the last use, which for a renderer is the submit. A
// producer driven on the consumer's thread satisfies that by program
// order. A producer on another thread publishes through a role-
// swapping type on the consumer's side, one that hands current() a
// slot the producer is not writing; gpud carries only the carrier.
struct BufferSource {
    Buffer *(*fn)(void *) = nullptr;
    void *user = nullptr;
    explicit operator bool() const { return fn != nullptr; }
    Buffer *current() const { return fn(user); }
};

// One tick of the device timeline. Every run() occupies the next tick,
// in call order, and returns it; completed() >= t means tick t's
// dispatch has finished on the device. Value 0 is "before any work".
// A value, not a handle: nothing to release, never reused; compare
// tickets to order work.
struct Ticket {
    std::uint64_t value = 0;
    friend auto operator<=>(const Ticket &, const Ticket &) = default;
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
    // it. `buffers` is positional — see contract note 2 above. Returns
    // the Ticket this dispatch occupies on the device timeline.
    virtual Ticket run(const Kernel &kernel, std::size_t groups,
                       std::span<const std::byte> scalars,
                       std::span<Buffer *const> buffers) = 0;

    // ── the device timeline ──────────────────────────────────────────
    // Every run() occupies one Ticket, in call order, and hands it
    // back. Tickets refine the ordering contract rather than replacing
    // it: read() still means what it always did, and is internally
    // "wait(last ticket touching this buffer) + copy".
    //
    // The defaults below describe a backend that completes every call
    // before returning (the stub backends): such a device never has
    // outstanding work, so a timeline pinned at zero is accurate.
    // Backends that queue work override all three.
    //
    // submitted()/completed() poll and never block; wait() blocks until
    // the ticket has completed. Waiting on a ticket beyond submitted()
    // is a caller error and returns immediately rather than hanging.

    virtual Ticket submitted() const { return {}; }   // last enqueued
    virtual Ticket completed() const { return {}; }   // highest done
    virtual void wait(Ticket /*ticket*/) {}

    // Block until everything enqueued so far has completed. Non-virtual:
    // it is exactly wait(submitted()), and a backend that got those two
    // right cannot get this wrong.
    void flush() { wait(submitted()); }

  protected:
    virtual Kernel do_compile(std::string_view source) = 0;

    // Base members are destroyed only after the derived destructor has
    // run — a backend whose Kernel impls reference device state must
    // clear the memoization cache before tearing that state down.
    void clear_kernels() { kernels_.clear(); }

  private:
    std::unordered_map<const void *, Kernel> kernels_;
};

} // namespace gpud
