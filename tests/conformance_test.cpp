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

#include <gtest/gtest.h>

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
#include <thread>
#include <utility>
#include <vector>

namespace {

struct Backend {
    const char *name;
    std::unique_ptr<gpud::Device> (*open)();
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

const Backend kBackends[] = {
    {"mock", [] { return gpud::mock::try_open(); }, nullptr},
#ifdef GPUD_HAS_VULKAN
    {"vulkan", [] { return gpud::vulkan::try_open(); }, vulkan_saxpy},
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

    const std::uint64_t before = dev.submitted();
    std::uint64_t seen_submitted = before, seen_completed = dev.completed();
    for (int i = 0; i < 3; ++i) {
        dev.run(k, (N + 63) / 64, blob_of(sc), buffers);
        EXPECT_GE(dev.submitted(), seen_submitted) << "submitted() went back";
        EXPECT_GE(dev.completed(), seen_completed) << "completed() went back";
        EXPECT_LE(dev.completed(), dev.submitted());
        seen_submitted = dev.submitted();
        seen_completed = dev.completed();
    }
    const std::uint64_t after = dev.submitted();

    // One ticket per run() — but only for a backend that issues tickets
    // at all; the base-class defaults leave this at zero and stay
    // conformant.
    if (after != before) EXPECT_EQ(after, before + 3);

    dev.flush();
    EXPECT_EQ(dev.completed(), dev.submitted());
    EXPECT_EQ(dev.submitted(), after) << "flush() must not enqueue work";

    // Waiting past the last ticket issued is a caller error that must
    // return rather than block on a value nothing will ever signal.
    dev.wait(after + 1000);
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
        std::uint64_t last_completed = 0, last_submitted = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            // completed() first: a submitted() sampled after it can only
            // have grown, so completed <= submitted must hold.
            const std::uint64_t c = dev.completed();
            const std::uint64_t s = dev.submitted();
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

TEST_P(Conformance, BadKernelThrowsWithDiagnostics) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    static constexpr char garbage[] = "this is not a kernel {";
    EXPECT_THROW((void)dev_->compile(garbage), std::runtime_error);
}

} // namespace
