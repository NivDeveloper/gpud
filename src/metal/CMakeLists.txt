# Metal backend — scaffolding: no framework is linked yet.
# When the real implementation lands, the sources become Objective-C++
# (rename each .cpp to .mm) and the framework links PRIVATE (the
# include firewall — never PUBLIC/INTERFACE):
#   enable_language(OBJCXX)
#   target_link_libraries(gpud_metal PRIVATE "-framework Metal" "-framework Foundation")
# and flip GPUD_BACKEND_METAL's default (top-level CMakeLists.txt) to
# ${APPLE}.
add_library(gpud_metal STATIC
    Device.h Device.cpp     # try_open, run
    Buffer.h Buffer.cpp     # BufferImpl + alloc/write/read
    Kernel.h Kernel.cpp)    # KernelImpl + do_compile
add_library(gpud::metal ALIAS gpud_metal)
target_link_libraries(gpud_metal PUBLIC gpud::gpud)
set_target_properties(gpud_metal PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_metal PRIVATE -Wall -Wextra)
