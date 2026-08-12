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
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
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

TEST_P(Conformance, FreshBufferReadsZeros) {
    gpud::Buffer buf = dev_->alloc(32);
    std::array<std::byte, 32> out;
    out.fill(std::byte{0xff});
    dev_->read(buf, out.data(), out.size());
    for (std::byte b : out) EXPECT_EQ(int(b), 0);
}

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

TEST_P(Conformance, BadKernelThrowsWithDiagnostics) {
    if (!GetParam().saxpy) GTEST_SKIP() << "storage-only backend";
    static constexpr char garbage[] = "this is not a kernel {";
    EXPECT_THROW((void)dev_->compile(garbage), std::runtime_error);
}

} // namespace
