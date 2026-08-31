#include "Device.h"
#include "Kernel.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#ifdef _WIN32
#include <process.h>
#define GPUD_GETPID _getpid
#else
#include <unistd.h>
#define GPUD_GETPID getpid
#endif

namespace gpud::vulkan {
namespace {

// std::system's shell may not have /usr/local/bin or /opt/homebrew/bin
// on PATH — probe the common install locations (vklib-proven). Resolved
// HERE, at first compile, not in try_open: a consumer that only shares
// buffers and the timeline never needs a shader toolchain (the sdl
// backend made the same move).
std::string find_slangc() {
    // GPUD_SLANGC pins the compiler outright — and pins its absence,
    // which is what makes the lazy resolution testable on a machine
    // that has slangc installed.
    if (const char *pin = std::getenv("GPUD_SLANGC")) {
#ifdef _WIN32
        const std::string probe = std::string(pin) + " -h > NUL 2>&1";
#else
        const std::string probe = std::string(pin) + " -h > /dev/null 2>&1";
#endif
        return std::system(probe.c_str()) == 0 ? pin : "";
    }
    for (const char *c :
         {"/usr/local/bin/slangc", "/opt/homebrew/bin/slangc", "slangc"}) {
#ifdef _WIN32
        const std::string probe = std::string(c) + " -h > NUL 2>&1";
#else
        const std::string probe = std::string(c) + " -h > /dev/null 2>&1";
#endif
        if (std::system(probe.c_str()) == 0) return c;
    }
    return {};
}

std::string slurp(const std::filesystem::path &p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), {}};
}

// Slang source → SPIR-V via the slangc CLI (invocation proven against
// MoltenVK in the predecessor project). Throws with the compiler's
// diagnostics on failure.
std::string compile_slang(const std::string &slangc, std::string_view source) {
    namespace fs = std::filesystem;
    static std::atomic<unsigned> counter{0};
    const std::string stem = "gpud-" + std::to_string(GPUD_GETPID()) + "-" +
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
        throw std::runtime_error("gpud/vulkan: slangc failed:\n" + diag);
    return spirv;
}

} // namespace

Kernel Device::do_compile(std::string_view source) {
    // do_compile is externally synchronized (only the timeline trio is
    // thread-safe), so caching the resolved path needs no lock.
    if (s.slangc.empty()) {
        s.slangc = find_slangc();
        if (s.slangc.empty())
            throw std::runtime_error(
                "gpud/vulkan: compile: no slangc at /usr/local/bin, "
                "/opt/homebrew/bin or on PATH — kernels need the Slang "
                "compiler; buffers and the timeline do not");
    }
    const std::string spirv = compile_slang(s.slangc, source);

    auto impl = std::make_unique<KernelImpl>();
    impl->device = s.device;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size();
    smci.pCode = reinterpret_cast<const std::uint32_t *>(spirv.data());
    VkShaderModule module;
    check(vkCreateShaderModule(s.device, &smci, nullptr, &module),
          "vkCreateShaderModule");

    // One push-constant range of 128 bytes (the spec-guaranteed
    // minimum) — no per-kernel reflection; run() checks the fit.
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    VkResult r = vkCreatePipelineLayout(s.device, &plci, nullptr, &impl->layout);
    if (r != VK_SUCCESS) vkDestroyShaderModule(s.device, module, nullptr);
    check(r, "vkCreatePipelineLayout");

    VkComputePipelineCreateInfo cpci{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = module;
    cpci.stage.pName = "main";
    cpci.layout = impl->layout;
    r = vkCreateComputePipelines(s.device, VK_NULL_HANDLE, 1, &cpci, nullptr,
                                 &impl->pipeline);
    vkDestroyShaderModule(s.device, module, nullptr);
    check(r, "vkCreateComputePipelines");

    return Kernel(std::move(impl));
}

} // namespace gpud::vulkan
