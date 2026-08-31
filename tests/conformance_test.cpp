// Backend conformance suite: the same assertions run against every
// backend, through the gpud::Device interface only. A backend that
// can't open on this machine (no driver, no compiler) skips rather
// than fails, so the suite stays green on fresh machines and CI.
// Backends without a saxpy kernel entry (mock) run the storage half
// only.
//
// Adding a backend = one entry in kBackends (+ its dialect's saxpy
// source, for real backends).

#include <gpud/Device.h>
#include <gpud/Mock.h>
#ifdef GPUD_HAS_VULKAN
#include <gpud/Vulkan.h>
#endif
#ifdef GPUD_HAS_SDL
#include <gpud/Sdl.h>
#endif

#include <gtest/gtest.h>

#include <ostream>

namespace gpud {
inline void PrintTo(const Ticket &t, std::ostream *os) {
    *os << "Ticket{" << t.value << "}";
}
} // namespace gpud

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Backend {
    const char *name;
    std::unique_ptr<gpud::Device> (*open)();
    std::unique_ptr<gpud::Device> (*open_with)(const gpud::Options &);
    // Kernel source in the backend's dialect computing
    //   out[i] = in0[i] + s * in1[i]   for i < n
    // with scalars {float s; uint32 n} and 64-wide workgroups.
    // nullptr = storage-only backend (mock).
    const char *saxpy;
};

void PrintTo(const Backend &b, std::ostream *os) { *os << b.name; }

#ifdef GPUD_HAS_VULKAN
// Mirrors the shape consumers' code generators emit for "slang-vulkan":
// Buf_* wrapper structs, scalars-then-pointers push constants.
constexpr char vulkan_saxpy[] = R"(
struct Buf_float { float data[1]; };
struct PC {
  float s0;
  uint s1;
  Buf_float* out_buf;
  Buf_float* in0;
  Buf_float* in1;
};
[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if (i >= pc.s1) return;
  pc.out_buf.data[i] = pc.in0.data[i] + pc.s0 * pc.in1.data[i];
}
)";
#endif

#ifdef GPUD_HAS_VULKAN
// The same saxpy with a deliberately expensive inner loop, so a
// dispatch takes long enough for batches to queue up behind it.
constexpr char vulkan_saxpy_heavy[] = R"(
struct Buf_float { float data[1]; };
struct PC {
  float s0;
  uint s1;
  Buf_float* out_buf;
  Buf_float* in0;
  Buf_float* in1;
};
[[vk::push_constant]] PC pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if (i >= pc.s1) return;
  float acc = pc.in0.data[i];
  for (uint k = 0; k < 512; ++k)
    acc = sin(acc) + pc.s0 * cos(pc.in1.data[i] + float(k));
  pc.out_buf.data[i] = acc;
}
)";
#endif

#ifdef GPUD_HAS_SDL
// Mirrors the shape consumers' code generators emit for "slang-slot":
// numbered [[vk::binding]] resources, scalars behind a ConstantBuffer.
// std140 for a struct of 4-byte scalars matches SaxpyScalars' natural
// layout, so the one scalar blob serves both dialects.
constexpr char sdl_saxpy[] = R"(
struct Scalars {
  float s0;
  uint s1;
};
[[vk::binding(0, 2)]] ConstantBuffer<Scalars> sc;
[[vk::binding(0, 1)]] RWStructuredBuffer<float> out_buf;
[[vk::binding(0, 0)]] StructuredBuffer<float> in0;
[[vk::binding(1, 0)]] StructuredBuffer<float> in1;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
  uint i = tid.x;
  if (i >= sc.s1) return;
  out_buf[i] = in0[i] + sc.s0 * in1[i];
}
)";
#endif

const Backend kBackends[] = {
    {"mock", [] { return gpud::mock::try_open(); },
     [](const gpud::Options &o) { return gpud::mock::try_open(o); }, nullptr},
#ifdef GPUD_HAS_VULKAN
    {"vulkan", [] { return gpud::vulkan::try_open(); },
     [](const gpud::Options &o) { return gpud::vulkan::try_open(o); },
     vulkan_saxpy},
#endif
#ifdef GPUD_HAS_SDL
    {"sdl", [] { return gpud::sdl::try_open(); },
     [](const gpud::Options &o) { return gpud::sdl::try_open(o); },
     sdl_saxpy},
#endif
};

// The saxpy kernel's scalar section: {float s; uint32 n}, laid out as
// the dialect declared it.
struct SaxpyScalars {
    float s;
    std::uint32_t n;
};

std::span<const std::byte> blob_of(const SaxpyScalars &sc) {
    return {reinterpret_cast<const std::byte *>(&sc), sizeof sc};
}

class Conformance : public ::testing::TestWithParam<Backend> {
  protected:
    void SetUp() override {
        dev_ = GetParam().open();
        if (!dev_)
            GTEST_SKIP() << GetParam().name
                         << ": try_open returned nullptr on this machine";
    }
    std::unique_ptr<gpud::Device> dev_;
};

INSTANTIATE_TEST_SUITE_P(
    Backends, Conformance, ::testing::ValuesIn(kBackends),
    [](const ::testing::TestParamInfo<Backend> &info) {
        return std::string(info.param.name);
    });

TEST_P(Conformance, OpensWithDialect) {
    EXPECT_FALSE(dev_->dialect().empty());
}

TEST_P(Conformance, WriteReadRoundTrip) {
    gpud::Buffer buf = dev_->alloc(16 * sizeof(float));
    EXPECT_EQ(buf.bytes(), 16 * sizeof(float));
    std::array<float, 16> in, out{};
    for (int i = 0; i < 16; ++i) in[std::size_t(i)] = 0.5f * float(i);
    dev_->write(buf, in.data(), sizeof in);
    dev_->read(buf, out.data(), sizeof out);
    EXPECT_EQ(std::memcmp(in.data(), out.data(), sizeof in), 0);
}

// No FreshBufferReadsZeros here on purpose: a freshly alloc()ed buffer's
// contents are unspecified (contract note 4), because a backend is free
// to hand back memory a destroyed Buffer used to own. The mock still
// zero-fills — it is a test double, and consumers assert on its log —
// which MockDevice.UnwrittenBufferReadsAsZeros covers.

TEST_P(Conformance, SaxpyEndToEnd) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 1000;   // deliberately not a multiple of 64
    constexpr float S = 2.5f;
    std::vector<float> a(N), b(N), out(N, -1.0f);
    for (std::uint32_t i = 0; i < N; ++i) {
        a[i] = float(i);
        b[i] = float(N - i);
    }

    gpud::Buffer ba = dev.alloc(N * sizeof(float));
    gpud::Buffer bb = dev.alloc(N * sizeof(float));
    gpud::Buffer bo = dev.alloc(N * sizeof(float));
    dev.write(ba, a.data(), N * sizeof(float));
    dev.write(bb, b.data(), N * sizeof(float));

    const gpud::Kernel &k = dev.compile(GetParam().saxpy);

    struct {
        float s;
        std::uint32_t n;
    } scalars{S, N};
    const std::span<const std::byte> blob{
        reinterpret_cast<const std::byte *>(&scalars), sizeof scalars};
    gpud::Buffer *buffers[] = {&bo, &ba, &bb};   // [0]=out, [1+k]=input k

    dev.run(k, (N + 63) / 64, blob, buffers);

    dev.read(bo, out.data(), N * sizeof(float));
    for (std::uint32_t i = 0; i < N; ++i)
        ASSERT_EQ(out[i], a[i] + S * b[i]) << "index " << i;
}

// ── the device timeline ──────────────────────────────────────────────
// Every assertion here has to hold for three quite different backends:
// one keeping Device's base-class defaults (completes inline, reports a
// timeline pinned at zero), the mock (completed() == submitted()), and
// real hardware, where the GPU may well finish work before the host
// thinks to ask. So the invariants are one-sided — completed() never
// runs ahead of submitted(), neither ever goes backwards, and flush()
// settles them — and never "completed() lags at this moment", which no
// backend owes anyone.
TEST_P(Conformance, TimelineNeverRunsAheadAndSettlesAtFlush) {
    gpud::Device &dev = *dev_;
    EXPECT_LE(dev.completed(), dev.submitted());

    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";

    constexpr std::uint32_t N = 256;
    gpud::Buffer ba = dev.alloc(N * sizeof(float));
    gpud::Buffer bb = dev.alloc(N * sizeof(float));
    gpud::Buffer bo = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    SaxpyScalars sc{1.0f, N};
    gpud::Buffer *buffers[] = {&bo, &ba, &bb};

    const gpud::Ticket before = dev.submitted();
    gpud::Ticket seen_submitted = before, seen_completed = dev.completed();
    for (int i = 0; i < 3; ++i) {
        dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        EXPECT_GE(dev.submitted(), seen_submitted) << "submitted() went back";
        EXPECT_GE(dev.completed(), seen_completed) << "completed() went back";
        EXPECT_LE(dev.completed(), dev.submitted());
        seen_submitted = dev.submitted();
        seen_completed = dev.completed();
    }
    const gpud::Ticket after = dev.submitted();

    // One ticket per run() — but only for a backend that issues tickets
    // at all; the base-class defaults leave this at zero and stay
    // conformant.
    if (after != before) EXPECT_EQ(after.value, before.value + 3);

    dev.flush();
    EXPECT_EQ(dev.completed(), dev.submitted());
    EXPECT_EQ(dev.submitted(), after) << "flush() must not enqueue work";

    // Waiting past the last ticket issued is a caller error that must
    // return rather than block on a value nothing will ever signal.
    dev.wait(gpud::Ticket{after.value + 1000});
}

// The contract's one thread-safety carve-out: submitted(), completed()
// and wait() may be called from another thread while this one is inside
// a Device call. Everything the poller observes must still be
// consistent — each counter non-decreasing, and completed() never ahead
// of a submitted() sampled afterwards.
TEST_P(Conformance, TimelineIsSafeToPollFromAnotherThread) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 4096;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &a};

    std::atomic<bool> stop{false};
    std::atomic<int> samples{0};
    std::thread poller([&] {
        gpud::Ticket last_completed, last_submitted;
        while (!stop.load(std::memory_order_relaxed)) {
            // completed() first: a submitted() sampled after it can only
            // have grown, so completed <= submitted must hold.
            const gpud::Ticket c = dev.completed();
            const gpud::Ticket s = dev.submitted();
            EXPECT_GE(c, last_completed) << "completed() went backwards";
            EXPECT_GE(s, last_submitted) << "submitted() went backwards";
            EXPECT_LE(c, s) << "completed() ran ahead of submitted()";
            last_completed = c;
            last_submitted = s;
            samples.fetch_add(1, std::memory_order_relaxed);
        }
    });

    for (int i = 0; i < 300; ++i) dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    dev.flush();

    stop.store(true, std::memory_order_relaxed);
    poller.join();
    EXPECT_GT(samples.load(), 0) << "poller never ran";
}

// max_queued bounds the tickets outstanding: submitted minus completed
// never exceeds it. One-sided on purpose — a fast device may never
// accumulate the bound, so the assertion is "never exceeds", and the
// throttle's firing was proven by removing it once (commit message).
TEST_P(Conformance, ThrottleBoundsOutstanding) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    auto capped = GetParam().open_with(gpud::Options{.max_queued = 4});
    if (!capped) GTEST_SKIP() << "reopen with options failed";
    gpud::Device &dev = *capped;

    constexpr std::uint32_t N = 1u << 20; // enough work to stay queued
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &a};

    for (int i = 0; i < 20; ++i) {
        dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        EXPECT_LE(dev.submitted().value - dev.completed().value, 4u)
            << "outstanding exceeded max_queued at dispatch " << i;
    }
    dev.flush();
    EXPECT_EQ(dev.completed(), dev.submitted());
}

// Two threads wait() overlapping tickets while a third dispatches: a
// wait(t) must not return before t completes, even when another
// waiter holds the fence prefix covering t — a fence, unlike a
// timeline value, can be waited only by whoever holds it, and the
// backend must bridge that gap.
TEST_P(Conformance, ConcurrentWaitersAllObserveCompletion) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 1u << 18;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &a};

    std::atomic<std::uint64_t> latest{0};
    std::atomic<bool> stop{false};
    auto waiter = [&] {
        while (!stop.load(std::memory_order_relaxed)) {
            const std::uint64_t t = latest.load(std::memory_order_acquire);
            if (!t) continue;
            dev.wait(gpud::Ticket{t});
            EXPECT_GE(dev.completed(), gpud::Ticket{t});
        }
    };
    std::thread w1(waiter), w2(waiter);
    for (int i = 0; i < 200; ++i) {
        const gpud::Ticket t = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        latest.store(t.value, std::memory_order_release);
    }
    dev.flush();
    stop.store(true, std::memory_order_relaxed);
    w1.join();
    w2.join();
}

// run() names the tick it occupies: equal to submitted() right after,
// strictly increasing, and completed() reaches it once waited.
TEST_P(Conformance, RunReturnsItsTicket) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 256;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &a};

    const gpud::Ticket t1 = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    EXPECT_EQ(t1, dev.submitted());
    const gpud::Ticket t2 = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    EXPECT_LT(t1, t2);
    dev.wait(t2);
    EXPECT_GE(dev.completed(), t2);
}

// Requirement: a Buffer whose handle dies must not take its memory with
// it while queued work still references it. The inputs are destroyed
// straight after run(), before anything has forced completion.
TEST_P(Conformance, BufferDestroyedAfterRunIsSafe) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 1000;
    constexpr float S = 2.5f;
    std::vector<float> a(N), b(N), out(N, -1.0f);
    for (std::uint32_t i = 0; i < N; ++i) {
        a[i] = float(i);
        b[i] = float(N - i);
    }

    gpud::Buffer bo = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    SaxpyScalars sc{S, N};
    {
        gpud::Buffer ba = dev.alloc(N * sizeof(float));
        gpud::Buffer bb = dev.alloc(N * sizeof(float));
        dev.write(ba, a.data(), N * sizeof(float));
        dev.write(bb, b.data(), N * sizeof(float));
        gpud::Buffer *buffers[] = {&bo, &ba, &bb};
        dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    }   // ba and bb die here, with the dispatch that reads them possibly
        // still queued

    dev.read(bo, out.data(), N * sizeof(float));
    for (std::uint32_t i = 0; i < N; ++i)
        ASSERT_EQ(out[i], a[i] + S * b[i]) << "index " << i;
}

// A chain where every dispatch reads what the previous one wrote — the
// least-covered path today, and the one a backend that queues work can
// get silently wrong rather than loudly. saxpy with s = 1 and both
// inputs aimed at the same buffer is a doubling (out = in + 1*in), so
// after L links out[i] == i * 2^L: exact in float for i < 2^24, since
// doubling only moves the exponent.
//
// The chain is deliberately longer than any plausible queue-depth cap,
// so it also covers the batch boundaries, throttling and command-buffer
// reuse a batching backend has to get right. (It is NOT a barrier
// detector on its own — a driver that happens to serialize dispatches
// passes it either way. Synchronization validation is what checks the
// barrier; see docs.)
TEST_P(Conformance, DispatchChainClosedForm) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 1000;
    constexpr int LINKS = 100;   // 999 * 2^100 is far below FLT_MAX
    constexpr int MID = 50;

    std::vector<float> seed(N), out(N);
    for (std::uint32_t i = 0; i < N; ++i) seed[i] = float(i);

    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    dev.write(a, seed.data(), N * sizeof(float));

    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};

    gpud::Buffer *src = &a, *dst = &b;
    for (int link = 1; link <= LINKS; ++link) {
        gpud::Buffer *buffers[] = {dst, src, src};
        dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        std::swap(src, dst);   // *src now holds this link's output

        // One read partway through, which must force the queued half of
        // the chain to complete and hand back exactly this link's
        // value — neither the seed nor the final answer.
        if (link == MID) {
            dev.read(*src, out.data(), N * sizeof(float));
            const float scale = std::ldexp(1.0f, MID);
            for (std::uint32_t i = 0; i < N; ++i)
                ASSERT_EQ(out[i], float(i) * scale)
                    << "index " << i << " at link " << MID;
        }
    }

    dev.read(*src, out.data(), N * sizeof(float));
    const float scale = std::ldexp(1.0f, LINKS);
    for (std::uint32_t i = 0; i < N; ++i)
        ASSERT_EQ(out[i], float(i) * scale) << "index " << i;
}

// Writing into a buffer that a queued dispatch still reads must not
// race it: the first run's output has to reflect the data that was in
// the buffer when it was issued, not what the host put there after.
TEST_P(Conformance, WriteAfterRunSynchronizes) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 1000;
    const std::vector<float> first(N, 1.0f), second(N, 8.0f);
    std::vector<float> out1(N, -1.0f), out2(N, -1.0f);

    gpud::Buffer in = dev.alloc(N * sizeof(float));
    gpud::Buffer o1 = dev.alloc(N * sizeof(float));
    gpud::Buffer o2 = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};   // out = in + in

    dev.write(in, first.data(), N * sizeof(float));
    gpud::Buffer *b1[] = {&o1, &in, &in};
    dev.run(k, (N + 63) / 64, blob_of(sc), b1);

    dev.write(in, second.data(), N * sizeof(float));   // waits out run 1
    gpud::Buffer *b2[] = {&o2, &in, &in};
    dev.run(k, (N + 63) / 64, blob_of(sc), b2);

    dev.read(o1, out1.data(), N * sizeof(float));
    dev.read(o2, out2.data(), N * sizeof(float));
    for (std::uint32_t i = 0; i < N; ++i) {
        ASSERT_EQ(out1[i], 2.0f) << "index " << i;
        ASSERT_EQ(out2[i], 16.0f) << "index " << i;
    }
}

// Allocate, use and destroy in a tight loop with nothing read until the
// end, so releases pile up behind work that is still queued and the
// allocator is free to hand the same memory back out. A release that
// ran too early — or memory recycled into a later alloc() before the
// dispatch reading it had finished — corrupts the arithmetic.
TEST_P(Conformance, RecycleWhileQueued) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 256;
    constexpr int ROUNDS = 200;
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    const SaxpyScalars sc{1.0f, N};

    const auto seed_of = [](int round, std::uint32_t i) {
        return float(round * 1000 + int(i));
    };

    std::vector<float> in(N), out(N);
    std::vector<gpud::Buffer> outputs;
    outputs.reserve(ROUNDS);

    for (int round = 0; round < ROUNDS; ++round) {
        for (std::uint32_t i = 0; i < N; ++i) in[i] = seed_of(round, i);

        gpud::Buffer o = dev.alloc(N * sizeof(float));
        {
            gpud::Buffer src = dev.alloc(N * sizeof(float));
            dev.write(src, in.data(), N * sizeof(float));
            gpud::Buffer *buffers[] = {&o, &src, &src};
            dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        }   // src dies here, its dispatch very likely still queued
        outputs.push_back(std::move(o));
    }

    for (int round = 0; round < ROUNDS; ++round) {
        dev.read(outputs[std::size_t(round)], out.data(), N * sizeof(float));
        for (std::uint32_t i = 0; i < N; ++i)
            ASSERT_EQ(out[i], 2.0f * seed_of(round, i))
                << "round " << round << " index " << i;
    }
}

// submit() is the non-blocking half of flush(): after it, the work
// must reach completed() with NO further host call — the property an
// external GPU-side waiter depends on. completed() polls and never
// submits, so for a batching backend this loop terminates only if
// submit() really pushed the batch.
TEST_P(Conformance, SubmitMakesProgressWithoutHostWaits) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    gpud::Device &dev = *dev_;

    constexpr std::uint32_t N = 256;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};

    const gpud::Ticket t = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    dev.submit();

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (dev.completed() < t) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "submit() did not make the work reach completed()";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(dev.completed(), t);
}

// Submission is eager: a batch of Options::batch dispatches reaches
// the device with NO host call at all — not a wait, not a read, not a
// submit(). completed() polls and never submits, so this loop ends
// only if run() itself pushed the batch out.
TEST_P(Conformance, FullBatchReachesCompletionWithoutHostCalls) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    constexpr std::uint32_t kBatch = 4;
    auto dev_ptr = GetParam().open_with(gpud::Options{.batch = kBatch});
    if (!dev_ptr) GTEST_SKIP() << "open_with returned nullptr";
    gpud::Device &dev = *dev_ptr;

    constexpr std::uint32_t N = 256;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(GetParam().saxpy);
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};

    gpud::Ticket last;
    for (std::uint32_t i = 0; i < kBatch; ++i)
        last = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (dev.completed() < last) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "a full batch must reach the device without a host call";
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

TEST_P(Conformance, BadKernelThrowsWithDiagnostics) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    static constexpr char garbage[] = "this is not a kernel {";
    EXPECT_THROW((void)dev_->compile(garbage), std::runtime_error);
}

} // namespace

#ifdef GPUD_HAS_SDL
#include <SDL3/SDL.h>

// The visualizer seam: the native handles are live SDL objects on the
// very device the compute runs on; a foreign Device yields nullptr.
TEST(SdlInterop, NativeHandlesAreLive) {
    auto dev = gpud::sdl::try_open();
    if (!dev) GTEST_SKIP() << "sdl: try_open returned nullptr on this machine";
    SDL_GPUDevice *nd = gpud::sdl::native_device(*dev);
    ASSERT_NE(nd, nullptr);
    EXPECT_NE(SDL_GetGPUShaderFormats(nd) & SDL_GPU_SHADERFORMAT_SPIRV, 0u);

    gpud::Buffer buf = dev->alloc(64);
    EXPECT_NE(gpud::sdl::native_buffer(buf), nullptr);
    gpud::Buffer empty;
    EXPECT_EQ(gpud::sdl::native_buffer(empty), nullptr);   // not a null deref

    auto foreign = gpud::mock::try_open();
    EXPECT_EQ(gpud::sdl::native_device(*foreign), nullptr);
}

#endif

#ifdef GPUD_HAS_VULKAN

#include <volk.h>

// A raw device of the tests' own, to hand to try_open_on: the minimal
// subset of the backend's bring-up (portability handled, first compute
// family, the two features adoption documents as required).
namespace vkraw {

struct Raw {
    VkInstance inst{};
    VkPhysicalDevice phys{};
    VkDevice dev{};
    VkQueue q{};
    std::uint32_t fam = ~0u;
};

bool bring_up(Raw &r) {
    if (volkInitialize() != VK_SUCCESS) return false;
    std::uint32_t n = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> ie(n);
    vkEnumerateInstanceExtensionProperties(nullptr, &n, ie.data());
    std::vector<const char *> exts;
    VkInstanceCreateFlags fl = 0;
    for (const auto &e : ie)
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_enumeration")) {
            exts.push_back("VK_KHR_portability_enumeration");
            fl |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.flags = fl;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = std::uint32_t(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    if (vkCreateInstance(&ici, nullptr, &r.inst) != VK_SUCCESS) return false;
    volkLoadInstance(r.inst);

    n = 0;
    vkEnumeratePhysicalDevices(r.inst, &n, nullptr);
    if (!n) return false;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(r.inst, &n, devs.data());
    r.phys = devs[0];

    n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &n, nullptr);
    std::vector<VkQueueFamilyProperties> fams(n);
    vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &n, fams.data());
    for (std::uint32_t i = 0; i < n; ++i)
        if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            r.fam = i;
            break;
        }
    if (r.fam == ~0u) return false;

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = r.fam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;
    n = 0;
    vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> de(n);
    vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &n, de.data());
    std::vector<const char *> dexts;
    for (const auto &e : de)
        if (!std::strcmp(e.extensionName, "VK_KHR_portability_subset"))
            dexts.push_back("VK_KHR_portability_subset");
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = std::uint32_t(dexts.size());
    dci.ppEnabledExtensionNames = dexts.data();
    if (vkCreateDevice(r.phys, &dci, nullptr, &r.dev) != VK_SUCCESS)
        return false;
    volkLoadDevice(r.dev);
    vkGetDeviceQueue(r.dev, r.fam, 0, &r.q);
    return true;
}

void destroy(Raw &r) {
    if (r.dev) vkDestroyDevice(r.dev, nullptr);
    if (r.inst) vkDestroyInstance(r.inst, nullptr);
    r = {};
}

gpud::vulkan::AdoptDesc desc_of(const Raw &r) {
    gpud::vulkan::AdoptDesc d;
    d.instance = r.inst;
    d.physical = r.phys;
    d.device = r.dev;
    d.queue = r.q;
    d.queue_family = r.fam;
    d.get_instance_proc_addr = vkGetInstanceProcAddr;
    return d;
}

void saxpy_roundtrip(gpud::Device &dev) {
    constexpr std::uint32_t N = 256;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    std::vector<float> ha(N), hb(N);
    for (std::uint32_t i = 0; i < N; ++i) ha[i] = float(i), hb[i] = 2.0f * i;
    dev.write(a, ha.data(), N * sizeof(float));
    dev.write(b, hb.data(), N * sizeof(float));
    const gpud::Kernel &k = dev.compile(vulkan_saxpy);
    SaxpyScalars sc{3.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};
    dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    std::vector<float> out(N);
    dev.read(o, out.data(), N * sizeof(float));
    for (std::uint32_t i : {0u, 1u, 255u})
        ASSERT_FLOAT_EQ(out[i], float(i) + 3.0f * (2.0f * i));
}

} // namespace vkraw

// The pool: a dead buffer's device objects come back on the next
// same-sized alloc() — the same VkBuffer — once its last reader has
// completed; while a dispatch still reads it, alloc() gets a different
// one. A producer allocating per step therefore creates nothing.
TEST(VulkanPool, RecyclesOnceTheLastReaderCompleted) {
    auto dev_ptr = gpud::vulkan::try_open();
    if (!dev_ptr) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    gpud::Device &dev = *dev_ptr;

    constexpr std::uint32_t N = 1 << 16;
    const gpud::Kernel &heavy = dev.compile(vulkan_saxpy_heavy);
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));

    std::uint64_t first = 0;
    {
        gpud::Buffer a = dev.alloc(N * sizeof(float));
        first = gpud::vulkan::native_buffer(a);
        gpud::Buffer *buffers[] = {&o, &a, &b};
        dev.run(heavy, (N + 63) / 64, blob_of(sc), buffers);
        dev.submit();
    }   // a dies with its dispatch in flight
    gpud::Buffer busy = dev.alloc(N * sizeof(float));
    EXPECT_NE(gpud::vulkan::native_buffer(busy), first)
        << "a buffer a queued dispatch still reads must not be reused";
    dev.flush();
    gpud::Buffer other = dev.alloc(2 * N * sizeof(float));
    EXPECT_NE(gpud::vulkan::native_buffer(other), first) << "sizes differ";
    gpud::Buffer again = dev.alloc(N * sizeof(float));
    EXPECT_EQ(gpud::vulkan::native_buffer(again), first)
        << "the same-sized buffer comes back once its reader completed";
}

// The other half of eager submission, pinned so nobody "fixes" it: a
// batch SHORTER than Options::batch stays open — completed() does not
// reach it on its own — until submit() pushes it out. A backend that
// submits per dispatch (sdl's fence ring) has no open batch and no such
// promise, so this is the batching backend's alone.
TEST(VulkanBatching, ShortBatchStaysOpenUntilSubmit) {
    auto dev_ptr = gpud::vulkan::try_open(gpud::Options{.batch = 8});
    if (!dev_ptr) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    gpud::Device &dev = *dev_ptr;

    constexpr std::uint32_t N = 256;
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    const gpud::Kernel &k = dev.compile(vulkan_saxpy);
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};

    gpud::Ticket last;
    for (int i = 0; i < 7; ++i)   // one short of the batch
        last = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_LT(dev.completed(), last) << "a short batch must not go out by itself";

    dev.submit();
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(15);
    while (dev.completed() < last) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    dev.flush();
}

// A free-running producer's shape: fresh buffers for every dispatch,
// dropped as soon as the dispatch is recorded, at a depth that keeps
// many batches in flight. What it caught: MoltenVK makes EVERY live
// device-address buffer resident in a batch at encode time, so a
// buffer freed the moment its own last dispatch completed was still
// referenced by later batches — "Invalid Resource", the device lost.
// The release is two-phase now (the VkBuffer at last_use, its memory
// once everything submitted by then has completed), and this is the
// test that turns red without it.
TEST(VulkanBatching, FreshBuffersPerDispatchAtDepthSurvive) {
    auto dev_ptr = gpud::vulkan::try_open(
        gpud::Options{.max_queued = 256, .batch = 16});
    if (!dev_ptr) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    gpud::Device &dev = *dev_ptr;

    // Heavy enough that batches queue up unscheduled, and a CHAIN: each
    // dispatch reads the previous output, which then dies — the Sync's
    // slot rotation, in miniature.
    constexpr std::uint32_t N = 1 << 16;
    const gpud::Kernel &k = dev.compile(vulkan_saxpy_heavy);
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer prev = dev.alloc(N * sizeof(float));
    try {
        for (int i = 0; i < 3000; ++i) {
            gpud::Buffer out = dev.alloc(N * sizeof(float));
            gpud::Buffer b = dev.alloc(N * sizeof(float));
            gpud::Buffer *buffers[] = {&out, &prev, &b};
            dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
            prev = std::move(out);   // the old prev dies here
        }
        dev.flush();
    } catch (const std::runtime_error &e) {
        FAIL() << "the device was lost: " << e.what();
    }
    EXPECT_EQ(dev.completed(), dev.submitted());
}

// The export seam: the timeline whose signaled value is Ticket::value
// verbatim, and per-buffer VkBuffers — 0 for empty or foreign handles.
// Adoption supplies the VkDevice the counter query needs.
TEST(VulkanInterop, NativeHandlesAreLiveAndTicketsAreSemaphoreValues) {
    vkraw::Raw r{};
    if (!vkraw::bring_up(r)) {
        vkraw::destroy(r);
        GTEST_SKIP() << "vulkan: no adoptable device on this machine";
    }
    {
        auto dev = gpud::vulkan::try_open_on(vkraw::desc_of(r));
        ASSERT_NE(dev, nullptr);

        const std::uint64_t sem = gpud::vulkan::native_timeline(*dev);
        ASSERT_NE(sem, 0u);

        std::uint64_t counter = ~0ull;
        ASSERT_EQ(vkGetSemaphoreCounterValue(
                      r.dev, reinterpret_cast<VkSemaphore>(sem), &counter),
                  VK_SUCCESS);
        EXPECT_EQ(counter, dev->completed().value) << "identity before work";

        vkraw::saxpy_roundtrip(*dev);   // read() inside forces completion
        ASSERT_EQ(vkGetSemaphoreCounterValue(
                      r.dev, reinterpret_cast<VkSemaphore>(sem), &counter),
                  VK_SUCCESS);
        EXPECT_EQ(counter, dev->completed().value) << "identity after work";
        EXPECT_EQ(counter, dev->submitted().value)
            << "the semaphore signals Ticket::value verbatim";
        EXPECT_GE(counter, 1u);

        gpud::Buffer b = dev->alloc(64);
        const std::uint64_t nb1 = gpud::vulkan::native_buffer(b);
        EXPECT_NE(nb1, 0u);
        EXPECT_EQ(nb1, gpud::vulkan::native_buffer(b))
            << "the handle must be stable across calls";

        gpud::Buffer empty;
        EXPECT_EQ(gpud::vulkan::native_buffer(empty), 0u);

        auto foreign = gpud::mock::try_open();
        EXPECT_EQ(gpud::vulkan::native_timeline(*foreign), 0u);
        gpud::Buffer fb = foreign->alloc(16);
        EXPECT_EQ(gpud::vulkan::native_buffer(fb), 0u);
    }
    vkraw::destroy(r);
}

TEST(VulkanAdopt, RefusesNullHandles) {
    EXPECT_EQ(gpud::vulkan::try_open_on({}), nullptr);
}

// The app-owns-the-device shape: gpud computes on a queue it was
// handed and never destroys what it adopted.
TEST(VulkanAdopt, SaxpyRunsOnAForeignDevice) {
    vkraw::Raw r{};
    if (!vkraw::bring_up(r)) {
        vkraw::destroy(r);
        GTEST_SKIP() << "vulkan: no adoptable device on this machine";
    }
    {
        auto dev = gpud::vulkan::try_open_on(vkraw::desc_of(r));
        ASSERT_NE(dev, nullptr);
        EXPECT_EQ(dev->dialect(), "slang-vulkan");
        vkraw::saxpy_roundtrip(*dev);
    }
    vkraw::destroy(r);
}

// The teardown half of non-owning: destroying the gpud Device leaves
// the adopted VkDevice fully usable.
TEST(VulkanAdopt, TeardownLeavesTheDeviceUsable) {
    vkraw::Raw r{};
    if (!vkraw::bring_up(r)) {
        vkraw::destroy(r);
        GTEST_SKIP() << "vulkan: no adoptable device on this machine";
    }
    {
        auto dev = gpud::vulkan::try_open_on(vkraw::desc_of(r));
        ASSERT_NE(dev, nullptr);
        vkraw::saxpy_roundtrip(*dev);
    }
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 64;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer probe{};
    EXPECT_EQ(vkCreateBuffer(r.dev, &bci, nullptr, &probe), VK_SUCCESS)
        << "the adopted device must survive the gpud Device";
    if (probe) vkDestroyBuffer(r.dev, probe, nullptr);
    vkraw::destroy(r);
}

// share_families => buffers go CONCURRENT over compute + the app's
// family; the whole pipeline must still round-trip.
TEST(VulkanAdopt, ConcurrentSharingRoundTrips) {
    vkraw::Raw r{};
    if (!vkraw::bring_up(r)) {
        vkraw::destroy(r);
        GTEST_SKIP() << "vulkan: no adoptable device on this machine";
    }
    std::uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &n, nullptr);
    std::uint32_t other = ~0u;
    for (std::uint32_t i = 0; i < n && other == ~0u; ++i)
        if (i != r.fam) other = i;
    if (other == ~0u) {
        vkraw::destroy(r);
        GTEST_SKIP() << "vulkan: single queue family — nothing to share with";
    }
    {
        auto d = vkraw::desc_of(r);
        d.share_families = &other;
        d.share_family_count = 1;
        auto dev = gpud::vulkan::try_open_on(d);
        ASSERT_NE(dev, nullptr);
        vkraw::saxpy_roundtrip(*dev);
    }
    vkraw::destroy(r);
}

// The compiler resolves at FIRST compile, not in try_open: a consumer
// that only shares buffers and the timeline needs no shader toolchain.
// GPUD_SLANGC pins a (non-)compiler, which is what lets this assert on
// a machine where slangc is installed.
TEST(VulkanLazySlangc, OpensWithoutCompilerAndCompileThrowsByName) {
    auto plain = gpud::vulkan::try_open();
    if (!plain)
        GTEST_SKIP() << "vulkan: try_open returned nullptr on this machine";
    plain.reset();

#ifdef _WIN32
    _putenv_s("GPUD_SLANGC", "gpud-no-such-compiler");
#else
    setenv("GPUD_SLANGC", "/gpud-no-such-compiler", 1);
#endif
    auto dev = gpud::vulkan::try_open();
    ASSERT_NE(dev, nullptr) << "try_open must not need the compiler";

    gpud::Buffer b = dev->alloc(64);
    const float v[4] = {1, 2, 3, 4};
    dev->write(b, v, sizeof v);   // storage works without a compiler

    try {
        (void)dev->compile("void main() {}");
        FAIL() << "compile without a compiler must throw";
    } catch (const std::runtime_error &e) {
        EXPECT_NE(std::string(e.what()).find("slangc"), std::string::npos);
    }
#ifdef _WIN32
    _putenv_s("GPUD_SLANGC", "");
#else
    unsetenv("GPUD_SLANGC");
#endif
}

namespace {

// Queue kDepth heavy dispatches (8 of them measured 9-14 ms on an Apple
// M4 Pro through MoltenVK; seconds on lavapipe) and wait for the last
// with the bound the device was opened with. `expect` names the bound
// the sentence must quote.
void expect_hung_wait(gpud::Device &dev, unsigned expect_ms) {
    constexpr std::uint32_t N = 1 << 16;
    constexpr int kDepth = 8;
    const gpud::Kernel &heavy = dev.compile(vulkan_saxpy_heavy);
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};
    gpud::Ticket last;
    for (int i = 0; i < kDepth; ++i)
        last = dev.run(heavy, (N + 63) / 64, blob_of(sc), buffers);

    try {
        dev.wait(last);
        FAIL() << "a " << expect_ms << " ms bound on " << kDepth
               << " heavy dispatches must trip";
    } catch (const std::runtime_error &e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("waited " + std::to_string(expect_ms) +
                            " ms for ticket " + std::to_string(last.value)),
                  std::string::npos)
            << what;
        EXPECT_NE(what.find("GPUD_WAIT_MS"), std::string::npos) << what;
    }

    // The throw promised nothing about the work, which still completes;
    // the timeline settles on its own and the Device stays usable. The
    // bound is still 1 ms, so every later wait is polled to completion
    // first — a plain saxpy's submit-to-signal alone can exceed 1 ms.
    const auto settle = [&](gpud::Ticket t) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (dev.completed() < t) {
            ASSERT_LT(std::chrono::steady_clock::now(), deadline)
                << "the dispatches the wait gave up on never completed";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    settle(last);
    const gpud::Kernel &k = dev.compile(vulkan_saxpy);
    std::vector<float> in(N, 1.0f), out(N);
    dev.write(a, in.data(), N * sizeof(float));
    dev.write(b, in.data(), N * sizeof(float));
    const gpud::Ticket t = dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    dev.submit();
    settle(t);
    dev.read(o, out.data(), N * sizeof(float));
    EXPECT_EQ(out[0], 3.0f) << "the Device must work after a tripped wait";
    EXPECT_EQ(out[N - 1], 3.0f);
}

} // namespace

// A host wait is unbounded unless Options::wait_ms bounds it; past the
// bound wait() throws a sentence naming the ticket and the bound, and
// the Device survives the throw.
TEST(VulkanBounded, WaitPastTheBoundThrowsAndTheDeviceSurvives) {
    auto dev = gpud::vulkan::try_open(gpud::Options{.wait_ms = 1});
    if (!dev) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    expect_hung_wait(*dev, 1);
}

// GPUD_WAIT_MS overrides the field — a gate bounds a program that never
// set it.
TEST(VulkanBounded, EnvironmentOverridesTheField) {
#ifdef _WIN32
    _putenv_s("GPUD_WAIT_MS", "1");
#else
    setenv("GPUD_WAIT_MS", "1", 1);
#endif
    auto dev = gpud::vulkan::try_open(gpud::Options{.wait_ms = 0});
#ifdef _WIN32
    _putenv_s("GPUD_WAIT_MS", "");
#else
    unsetenv("GPUD_WAIT_MS");
#endif
    if (!dev) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    expect_hung_wait(*dev, 1);
}

namespace {

// What the death test's child runs: heavy work in flight, then a
// teardown bounded at 1 ms. Braces inside a macro argument are not
// parenthesized, hence a function.
void teardown_past_the_bound() {
    auto dev = gpud::vulkan::try_open(gpud::Options{.wait_ms = 1});
    constexpr std::uint32_t N = 1 << 16;
    const gpud::Kernel &heavy = dev->compile(vulkan_saxpy_heavy);
    gpud::Buffer a = dev->alloc(N * sizeof(float));
    gpud::Buffer b = dev->alloc(N * sizeof(float));
    gpud::Buffer o = dev->alloc(N * sizeof(float));
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};
    for (int i = 0; i < 8; ++i)
        dev->run(heavy, (N + 63) / 64, blob_of(sc), buffers);
    dev->submit();
    dev.reset();
}

} // namespace

namespace {

// Eight dispatches in batches of four, profiled: two timings, tickets
// contiguous, every end after its begin, batches in order, and nothing
// left behind once drained.
void expect_two_batch_timings(gpud::Device &dev) {
    constexpr std::uint32_t N = 1 << 16;
    const gpud::Kernel &k = dev.compile(vulkan_saxpy);
    gpud::Buffer a = dev.alloc(N * sizeof(float));
    gpud::Buffer b = dev.alloc(N * sizeof(float));
    gpud::Buffer o = dev.alloc(N * sizeof(float));
    SaxpyScalars sc{2.0f, N};
    gpud::Buffer *buffers[] = {&o, &a, &b};
    const std::uint64_t before = dev.submitted().value;
    for (int i = 0; i < 8; ++i) dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
    dev.flush();
    // No further run(): take_timings polls for itself, so a producer
    // that stopped still hands over its last batches (stamps a driver
    // publishes late arrive on a later call).
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    std::array<gpud::Device::BatchTiming, 8> got{};
    std::size_t n = dev.take_timings(got);
    for (int tries = 0; n < 2 && tries < 100; ++tries) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        n += dev.take_timings(std::span(got).subspan(n));
    }
    ASSERT_GE(n, std::size_t(2)) << "two full batches must report";
    EXPECT_EQ(got[0].first.value, before + 1);
    EXPECT_EQ(got[0].last.value, before + 4);
    EXPECT_EQ(got[0].dispatches, 4u);
    EXPECT_EQ(got[1].first.value, before + 5);
    EXPECT_EQ(got[1].last.value, before + 8);
    EXPECT_EQ(got[1].dispatches, 4u);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_GT(got[i].gpu_end_ns, got[i].gpu_begin_ns)
            << "batch " << i << " must end after it began";
        EXPECT_LT(got[i].gpu_end_ns - got[i].gpu_begin_ns, 1000000000ull)
            << "batch " << i << " lasted over a second";
    }
    EXPECT_GE(got[1].gpu_begin_ns, got[0].gpu_begin_ns)
        << "the second batch must not begin before the first";
    EXPECT_EQ(dev.take_timings(got), std::size_t(0))
        << "a timing is handed out once";
}

} // namespace

// Options::profile: every batch reports its tickets and a begin/end
// pair on the device clock, drained once through take_timings.
TEST(VulkanProfile, BatchTimingsArriveInOrderAndCoverTheDispatches) {
    auto dev = gpud::vulkan::try_open(gpud::Options{.batch = 4, .profile = true});
    if (!dev) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    expect_two_batch_timings(*dev);
}

// GPUD_PROFILE=1 overrides the field, and off by default means EMPTY.
TEST(VulkanProfile, EnvironmentOverridesTheFieldAndOffMeansEmpty) {
    {
        auto plain = gpud::vulkan::try_open(gpud::Options{.batch = 4});
        if (!plain) GTEST_SKIP() << "vulkan: try_open returned nullptr";
        constexpr std::uint32_t N = 256;
        const gpud::Kernel &k = plain->compile(vulkan_saxpy);
        gpud::Buffer a = plain->alloc(N * 4), b = plain->alloc(N * 4),
                     o = plain->alloc(N * 4);
        SaxpyScalars sc{2.0f, N};
        gpud::Buffer *buffers[] = {&o, &a, &b};
        for (int i = 0; i < 8; ++i) plain->run(k, 4, blob_of(sc), buffers);
        plain->flush();
        std::array<gpud::Device::BatchTiming, 4> got{};
        EXPECT_EQ(plain->take_timings(got), std::size_t(0))
            << "profiling is opt-in";
    }
#ifdef _WIN32
    _putenv_s("GPUD_PROFILE", "1");
#else
    setenv("GPUD_PROFILE", "1", 1);
#endif
    auto dev = gpud::vulkan::try_open(gpud::Options{.batch = 4});
#ifdef _WIN32
    _putenv_s("GPUD_PROFILE", "");
#else
    unsetenv("GPUD_PROFILE");
#endif
    ASSERT_NE(dev, nullptr);
    expect_two_batch_timings(*dev);
}

// Teardown cannot throw: past the bound it prints the sentence and
// exits, because nothing it owns can be destroyed while the device may
// still be using it. A threadsafe death test re-runs this binary, so the
// child brings up its own device.
TEST(VulkanBoundedDeath, TeardownPastTheBoundAbortsWithTheSentence) {
    {
        auto probe = gpud::vulkan::try_open();
        if (!probe) GTEST_SKIP() << "vulkan: try_open returned nullptr";
    }
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH(teardown_past_the_bound(),
                 "gpud/vulkan: waited 1 ms for ticket");
}

#endif
