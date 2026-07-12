// Exercises the mock backend strictly through the gpud::Device
// interface; assertions inspect gpud::mock::Device::log.

#include <gpud/Auto.h>
#include <gpud/Device.h>
#include <gpud/Mock.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {

// Handles are move-only RAII.
static_assert(std::is_move_constructible_v<gpud::Buffer>);
static_assert(!std::is_copy_constructible_v<gpud::Buffer>);
static_assert(!std::is_copy_assignable_v<gpud::Buffer>);
static_assert(std::is_move_constructible_v<gpud::Kernel>);
static_assert(!std::is_copy_constructible_v<gpud::Kernel>);
static_assert(!std::is_copy_assignable_v<gpud::Kernel>);

TEST(MockDevice, TryOpenYieldsMockDialect) {
    auto dev = gpud::mock::try_open();
    ASSERT_NE(dev, nullptr);
    EXPECT_EQ(dev->dialect(), "mock");
}

TEST(MockDevice, AllocWriteReadRoundTrip) {
    gpud::mock::Device mock;
    gpud::Device &dev = mock;

    gpud::Buffer buf = dev.alloc(4 * sizeof(float));
    EXPECT_TRUE(buf);
    EXPECT_EQ(buf.bytes(), 4 * sizeof(float));

    const float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    dev.write(buf, src, sizeof src);

    float dst[4] = {};
    dev.read(buf, dst, sizeof dst);
    EXPECT_EQ(std::memcmp(src, dst, sizeof src), 0);

    ASSERT_EQ(mock.log.allocs.size(), 1u);
    EXPECT_EQ(mock.log.allocs[0], 4 * sizeof(float));
    ASSERT_EQ(mock.log.writes.size(), 1u);
    EXPECT_EQ(mock.log.writes[0].buffer, gpud::mock::Device::id(buf));
    EXPECT_EQ(mock.log.writes[0].bytes, sizeof src);
    ASSERT_EQ(mock.log.reads.size(), 1u);
    EXPECT_EQ(mock.log.reads[0].buffer, gpud::mock::Device::id(buf));
    EXPECT_EQ(mock.log.reads[0].bytes, sizeof dst);
}

TEST(MockDevice, UnwrittenBufferReadsAsZeros) {
    gpud::mock::Device mock;
    gpud::Device &dev = mock;

    gpud::Buffer buf = dev.alloc(8);
    std::array<std::byte, 8> dst;
    dst.fill(std::byte{0xff});
    dev.read(buf, dst.data(), dst.size());
    for (std::byte b : dst) EXPECT_EQ(int(b), 0);
}

TEST(MockDevice, CompileMemoizesOnSourceIdentity) {
    gpud::mock::Device mock;
    gpud::Device &dev = mock;

    static constexpr char source[] = "__kernel__ add(out, a, b)";
    const gpud::Kernel &k1 = dev.compile(source);
    const gpud::Kernel &k2 = dev.compile(source);
    EXPECT_EQ(&k1, &k2);                       // same Kernel object
    EXPECT_EQ(mock.log.compiled.size(), 1u);   // one do_compile call
    EXPECT_EQ(mock.log.compiled[0], source);

    // The cache keys on data() identity, not content: equal text at a
    // different address compiles again.
    const std::string copy(source);
    const gpud::Kernel &k3 = dev.compile(copy);
    EXPECT_NE(&k1, &k3);
    EXPECT_EQ(mock.log.compiled.size(), 2u);
}

TEST(MockDevice, RunRecordsGroupsScalarsAndPositionalBuffers) {
    gpud::mock::Device mock;
    gpud::Device &dev = mock;

    gpud::Buffer out = dev.alloc(64);
    gpud::Buffer in0 = dev.alloc(32);
    gpud::Buffer in1 = dev.alloc(16);

    static constexpr char source[] = "__kernel__ fma(out, in0, in1, s)";
    const gpud::Kernel &kernel = dev.compile(source);

    const float scale = 2.5f;
    std::array<std::byte, sizeof scale> scalars;
    std::memcpy(scalars.data(), &scale, sizeof scale);

    gpud::Buffer *buffers[] = {&out, &in0, &in1};   // [0]=output, [1+k]=input k
    dev.run(kernel, 7, scalars, buffers);

    ASSERT_EQ(mock.log.runs.size(), 1u);
    const auto &run = mock.log.runs[0];
    EXPECT_EQ(run.kernel, gpud::mock::Device::id(kernel));
    EXPECT_EQ(run.groups, 7u);
    ASSERT_EQ(run.scalars.size(), sizeof scale);
    EXPECT_EQ(std::memcmp(run.scalars.data(), &scale, sizeof scale), 0);

    const std::vector<int> want = {gpud::mock::Device::id(out),
                                   gpud::mock::Device::id(in0),
                                   gpud::mock::Device::id(in1)};
    EXPECT_EQ(run.buffers, want);
    EXPECT_NE(want[0], want[1]);   // distinct buffers, distinct ids
    EXPECT_NE(want[1], want[2]);
}

TEST(OpenDefault, HonorsGpudBackendMock) {
    ::setenv("GPUD_BACKEND", "mock", /*overwrite=*/1);
    auto dev = gpud::open_default();
    ::unsetenv("GPUD_BACKEND");
    ASSERT_NE(dev, nullptr);
    EXPECT_EQ(dev->dialect(), "mock");
}

TEST(OpenDefault, RequestedBackendNotCompiledInYieldsNull) {
#ifdef GPUD_HAS_VULKAN
    GTEST_SKIP() << "vulkan backend is compiled in";
#else
    ::setenv("GPUD_BACKEND", "vulkan", /*overwrite=*/1);
    EXPECT_EQ(gpud::open_default(), nullptr);
    ::unsetenv("GPUD_BACKEND");
#endif
}

TEST(OpenDefault, NoEnvNoBackendsYieldsNull) {
#if defined(GPUD_HAS_CUDA) || defined(GPUD_HAS_METAL) || defined(GPUD_HAS_VULKAN)
    GTEST_SKIP() << "a real backend is compiled in";
#else
    ::unsetenv("GPUD_BACKEND");
    EXPECT_EQ(gpud::open_default(), nullptr);   // mock is env-opt-in only
#endif
}

} // namespace
