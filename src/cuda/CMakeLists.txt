# CUDA backend — scaffolding: no toolkit is located or linked yet.
# When the real implementation lands, locate and link the toolkit
# PRIVATE (the include firewall — never PUBLIC/INTERFACE):
#   find_package(CUDAToolkit REQUIRED)
#   target_link_libraries(gpud_cuda PRIVATE CUDA::cuda_driver CUDA::nvrtc)
# and flip GPUD_BACKEND_CUDA's default (top-level CMakeLists.txt) to
# SDK auto-detection.
add_library(gpud_cuda STATIC
    Device.h Device.cpp     # try_open, run, context/stream teardown
    Buffer.h Buffer.cpp     # BufferImpl + alloc/write/read
    Kernel.h Kernel.cpp)    # KernelImpl + do_compile
add_library(gpud::cuda ALIAS gpud_cuda)
target_link_libraries(gpud_cuda PUBLIC gpud::gpud)
set_target_properties(gpud_cuda PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(gpud_cuda PRIVATE -Wall -Wextra)
endif()
