cmake_minimum_required(VERSION 3.24)
project(gpud VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Minimal GPU compute abstraction: five operations behind one virtual interface")

option(GPUD_BUILD_TESTS "Build gpud tests" ${PROJECT_IS_TOP_LEVEL})

# Per-backend options. The intended default is SDK auto-detection, but no
# backend is implemented yet, so every default resolves to OFF. When a
# backend lands, switch its default to detection:
#   Vulkan:  find_package(Vulkan QUIET)       -> default ${Vulkan_FOUND}
#   Metal:   Apple platforms                  -> default ${APPLE}
#   CUDA:    find_package(CUDAToolkit QUIET)  -> default ${CUDAToolkit_FOUND}
option(GPUD_BACKEND_VULKAN "Build the Vulkan backend" OFF)
option(GPUD_BACKEND_METAL "Build the Metal backend" OFF)
option(GPUD_BACKEND_CUDA "Build the CUDA backend" OFF)

# ── core: the dependency-free interface ─────────────────────────────────
add_library(gpud INTERFACE)
add_library(gpud::gpud ALIAS gpud)
target_compile_features(gpud INTERFACE cxx_std_20)
target_include_directories(gpud INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ── mock backend (header-only: include/gpud/Mock.h) ─────────────────────
# Its own target so consumer tests can link `gpud::mock` like any other
# backend, and so it could grow a compiled part without consumers changing.
add_library(gpud_mock INTERFACE)
add_library(gpud::mock ALIAS gpud_mock)
target_link_libraries(gpud_mock INTERFACE gpud)

# ── real backends hook in here when they land ───────────────────────────
# Each is a separately compiled static library linking its SDK PRIVATE,
# added inside its option gate, e.g.:
#   if(GPUD_BACKEND_VULKAN)
#     add_subdirectory(src/vulkan)   # defines gpud_vulkan + alias gpud::vulkan
#   endif()
# (src/vulkan, src/metal, src/cuda do not exist yet — nothing to add.)

# ── auto-selection: gpud_auto (alias gpud::auto_; `auto` is a C++ keyword) ─
add_library(gpud_auto STATIC src/auto/open_default.cpp)
add_library(gpud::auto_ ALIAS gpud_auto)
target_link_libraries(gpud_auto PUBLIC gpud)
set_target_properties(gpud_auto PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_auto PRIVATE -Wall -Wextra)

# Backend availability is baked into gpud_auto at its build time via the
# GPUD_HAS_* defines (PUBLIC so application code can compile conditionally
# on them too). When a backend lands, its block also links the backend
# library, e.g. target_link_libraries(gpud_auto PRIVATE gpud::cuda).
if(GPUD_BACKEND_CUDA)
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_CUDA)
endif()
if(GPUD_BACKEND_METAL)
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_METAL)
endif()
if(GPUD_BACKEND_VULKAN)
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_VULKAN)
endif()

if(GPUD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
