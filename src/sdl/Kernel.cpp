// KernelImpl + do_compile: slangc → SPIR-V (the vulkan backend's
// proven invocation), then the counts scan and pipeline creation.
// Called once per distinct source — memoization already happened in
// the base class.

#include "Kernel.h"

#include "Device.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace gpud::sdl {
namespace {

std::string slurp(const std::filesystem::path &p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

// std::system's shell may not have /usr/local/bin or /opt/homebrew/bin
// on PATH — probe the common install locations (vklib-proven). Called
// lazily from do_compile so device bring-up needs no shader toolchain.
std::string find_slangc() {
    for (const char *c :
         {"/usr/local/bin/slangc", "/opt/homebrew/bin/slangc", "slangc"}) {
        const std::string probe = std::string(c) + " -h > /dev/null 2>&1";
        if (std::system(probe.c_str()) == 0) return c;
    }
    return {};
}

// Slang source → SPIR-V via the slangc CLI. The slot dialect's binding
// decorations are in the source, so the invocation is the vulkan
// backend's, verbatim. Throws with the compiler's diagnostics.
std::string compile_slang(const std::string &slangc, std::string_view source) {
    namespace fs = std::filesystem;
    static std::atomic<unsigned> counter{0};
    const std::string stem = "gpud-" + std::to_string(::getpid()) + "-" +
                             std::to_string(counter.fetch_add(1));
    const fs::path src = fs::temp_directory_path() / (stem + ".slang");
    const fs::path spv = fs::temp_directory_path() / (stem + ".spv");
    const fs::path err = fs::temp_directory_path() / (stem + ".err");

    {
        std::ofstream f(src, std::ios::binary);
        f.write(source.data(), std::streamsize(source.size()));
    }
    const std::string cmd = slangc + " \"" + src.string() + "\"" +
                            " -target spirv -entry main -stage compute"
                            " -fvk-use-entrypoint-name -emit-spirv-directly"
                            " -o \"" + spv.string() + "\""
                            " 2> \"" + err.string() + "\"";
    const int rc = std::system(cmd.c_str());
    std::string spirv = slurp(spv);
    const std::string diag = slurp(err);
    std::error_code ec;
    fs::remove(src, ec), fs::remove(spv, ec), fs::remove(err, ec);

    if (rc != 0 || spirv.size() < 4 || spirv.size() % 4 != 0)
        throw std::runtime_error("gpud/sdl: slangc failed:\n" + diag);
    return spirv;
}

unsigned count_of(std::string_view s, std::string_view needle) {
    unsigned n = 0;
    for (std::size_t at = s.find(needle); at != std::string_view::npos;
         at = s.find(needle, at + needle.size()))
        ++n;
    return n;
}

// SDL declares a compute pipeline's resource counts and threadgroup
// size at creation, and gpud carries no reflection metadata — but the
// slot dialect is self-describing: bindings are dense from 0 per set,
// one read-write output, at most one uniform block. Scanning the
// SOURCE for its declarations is part of the dialect pact.
struct Declared {
    Uint32 readonly = 0, readwrite = 0, uniforms = 0, tx = 64;
};

Declared scan_declarations(std::string_view source) {
    Declared d;
    d.readonly = count_of(source, ")]] StructuredBuffer<");
    d.readwrite = count_of(source, ")]] RWStructuredBuffer<");
    d.uniforms = source.find("ConstantBuffer<") != std::string_view::npos;
    if (const auto at = source.find("[numthreads(");
        at != std::string_view::npos)
        d.tx = Uint32(std::atoi(source.data() + at + 12));
    return d;
}

} // namespace

Kernel Device::do_compile(std::string_view source) {
    // Resolved here, not at open: a re-probe per failed compile means
    // a slangc installed mid-session starts working.
    if (s_.slangc.empty()) s_.slangc = find_slangc();
    if (s_.slangc.empty())
        throw std::runtime_error(
            "gpud/sdl: slangc not found (looked in /usr/local/bin, "
            "/opt/homebrew/bin, PATH)");
    const std::string spirv = compile_slang(s_.slangc, source);
    const Declared d = scan_declarations(source);

    SDL_GPUComputePipelineCreateInfo ci{};
    ci.code_size = spirv.size();
    ci.code = reinterpret_cast<const Uint8 *>(spirv.data());
    ci.entrypoint = "main";
    ci.format = SDL_GPU_SHADERFORMAT_SPIRV;
    ci.num_readonly_storage_buffers = d.readonly;
    ci.num_readwrite_storage_buffers = d.readwrite;
    ci.num_uniform_buffers = d.uniforms;
    ci.threadcount_x = d.tx;
    ci.threadcount_y = 1;
    ci.threadcount_z = 1;
    SDL_GPUComputePipeline *p = SDL_CreateGPUComputePipeline(s_.dev, &ci);
    if (!p)
        throw std::runtime_error(std::string("gpud/sdl: pipeline: ") +
                                 SDL_GetError());

    auto impl = std::make_unique<KernelImpl>();
    impl->dev = s_.dev;
    impl->pipeline = p;
    impl->n_readonly = d.readonly;
    impl->has_uniform = d.uniforms != 0;
    return Kernel(std::move(impl));
}

} // namespace gpud::sdl
