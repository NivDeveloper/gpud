# SDL3 is found, never fetched (it is not a small redistributable):
# a system package or a prefix on CMAKE_PREFIX_PATH. The top-level
# gate defaults this backend ON exactly when find_package(SDL3)
# succeeds there.
find_package(SDL3 REQUIRED CONFIG)

add_library(gpud_sdl STATIC
    Device.h Device.cpp     # try_open, run, native handles
    Buffer.h Buffer.cpp     # BufferImpl + alloc/write/read
    Kernel.h Kernel.cpp)    # KernelImpl + do_compile
add_library(gpud::sdl ALIAS gpud_sdl)
target_link_libraries(gpud_sdl PUBLIC gpud::gpud PRIVATE SDL3::SDL3)
set_target_properties(gpud_sdl PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(gpud_sdl PRIVATE -Wall -Wextra)
endif()
