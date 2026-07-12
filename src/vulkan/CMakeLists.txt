# Vulkan backend — scaffolding: no SDK is located or linked yet.
# When the real implementation lands, locate and link the SDK PRIVATE
# (the include firewall — never PUBLIC/INTERFACE):
#   find_package(Vulkan REQUIRED)
#   target_link_libraries(gpud_vulkan PRIVATE Vulkan::Vulkan)
# and flip GPUD_BACKEND_VULKAN's default (top-level CMakeLists.txt) to
# SDK auto-detection.
add_library(gpud_vulkan STATIC
    Device.h Device.cpp     # try_open, run, teardown ordering
    Buffer.h Buffer.cpp     # BufferImpl + alloc/write/read
    Kernel.h Kernel.cpp)    # KernelImpl + do_compile
add_library(gpud::vulkan ALIAS gpud_vulkan)
target_link_libraries(gpud_vulkan PUBLIC gpud::gpud)
set_target_properties(gpud_vulkan PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_vulkan PRIVATE -Wall -Wextra)
