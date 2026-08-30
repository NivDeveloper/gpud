#pragma once

// Header-only mock backend — the test double for consumer projects.
// Exercises GPU glue with no GPU and no SDK: every call is recorded in
// an inspectable log, and read() returns the bytes previously written
// to that buffer (zeros if never written).
//
// Two ways in:
//   - gpud::mock::try_open()   — the uniform factory, for code paths
//     that open backends generically (open_default, composition roots).
//   - gpud::mock::Device dev;  — direct construction, for tests that
//     want to assert on dev.log afterwards.

#include "Device.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace gpud::mock {

class Device final : public ::gpud::Device {
  public:
    struct Write {
        int buffer;
        std::size_t bytes;
    };
    struct Read {
        int buffer;
        std::size_t bytes;
    };
    struct Run {
        int kernel;                       // index into log.compiled
        std::size_t groups;
        std::vector<std::byte> scalars;   // copy of the scalar blob
        std::vector<int> buffers;         // buffer ids, in positional order
    };
    struct Log {
        std::vector<std::size_t> allocs;    // byte size of each alloc, in call order
        std::vector<Write> writes;
        std::vector<Read> reads;
        std::vector<std::string> compiled;  // sources handed to do_compile
        std::vector<Run> runs;
    };

    Log log;

    // The id a handle was assigned at creation, for log assertions.
    static int id(const Buffer &b) { return state(b).id; }
    static int id(const Kernel &k) {
        return static_cast<const KernelImpl *>(k.impl())->id;
    }

    std::string_view dialect() const override { return "mock"; }

    Buffer alloc(std::size_t bytes) override {
        log.allocs.push_back(bytes);
        auto impl = std::make_unique<BufferImpl>();
        impl->id = next_buffer_id_++;
        impl->data.resize(bytes);   // zero-filled: unwritten reads are zeros
        return Buffer(std::move(impl), bytes);
    }

    void write(Buffer &dst, const void *src, std::size_t bytes) override {
        BufferImpl &b = state(dst);
        assert(bytes <= b.data.size());
        std::memcpy(b.data.data(), src, bytes);
        log.writes.push_back({b.id, bytes});
    }

    void read(const Buffer &src, void *dst, std::size_t bytes) override {
        const BufferImpl &b = state(src);
        assert(bytes <= b.data.size());
        std::memcpy(dst, b.data.data(), bytes);
        log.reads.push_back({b.id, bytes});
    }

    Ticket run(const Kernel &kernel, std::size_t groups,
               std::span<const std::byte> scalars,
               std::span<Buffer *const> buffers) override {
        Run r;
        r.kernel = id(kernel);
        r.groups = groups;
        r.scalars.assign(scalars.begin(), scalars.end());
        for (Buffer *b : buffers) r.buffers.push_back(id(*b));
        log.runs.push_back(std::move(r));
        return {ticket_.fetch_add(1, std::memory_order_release) + 1};
    }

    // The timeline, mock-style: run() ticks the counter and the work is
    // done by the time run() returns, so completed() == submitted() and
    // wait() has nothing to wait for. Consumers can exercise their
    // ticket bookkeeping against this without a GPU. Atomic because the
    // contract makes these three callable from another thread while one
    // is inside a Device call — the mock owes that like any backend.
    Ticket submitted() const override {
        return {ticket_.load(std::memory_order_acquire)};
    }
    Ticket completed() const override {
        return {ticket_.load(std::memory_order_acquire)};
    }
    void wait(Ticket ticket) override {
        // Debug-only, and compiled out of the default Release build:
        // tests must not depend on it firing.
        assert(ticket <= submitted() && "waiting on an unissued ticket");
        (void)ticket;
    }

  protected:
    Kernel do_compile(std::string_view source) override {
        auto impl = std::make_unique<KernelImpl>();
        impl->id = static_cast<int>(log.compiled.size());
        log.compiled.emplace_back(source);
        return Kernel(std::move(impl));
    }

  private:
    struct BufferImpl final : Buffer::Impl {
        int id = 0;
        std::vector<std::byte> data;
    };
    struct KernelImpl final : Kernel::Impl {
        int id = 0;
    };

    static BufferImpl &state(const Buffer &b) {
        return *static_cast<BufferImpl *>(b.impl());
    }

    int next_buffer_id_ = 0;
    std::atomic<std::uint64_t> ticket_{0};
};

inline std::unique_ptr<::gpud::Device> try_open(const Options & = {}) {
    return std::make_unique<Device>();
}

} // namespace gpud::mock
