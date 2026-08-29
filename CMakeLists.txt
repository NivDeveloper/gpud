cmake_minimum_required(VERSION 3.24)
project(gpud VERSION 0.4.0 LANGUAGES CXX
        DESCRIPTION "Minimal GPU compute abstraction: five operations behind one virtual interface")

option(GPUD_BUILD_TESTS "Build gpud tests" ${PROJECT_IS_TOP_LEVEL})

# Per-backend options.
#
# Vulkan is implemented; it defaults ON where Vulkan development files
# already exist and can be forced ON anywhere — building needs no
# preinstalled SDK (headers and volk are fetched; the driver is probed
# at runtime by try_open).
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  set(GPUD_VULKAN_DEFAULT ON)
else()
  set(GPUD_VULKAN_DEFAULT OFF)
endif()
option(GPUD_BACKEND_VULKAN "Build the Vulkan backend" ${GPUD_VULKAN_DEFAULT})

# Metal and CUDA are scaffolding stubs (try_open always returns
# nullptr), so their defaults stay OFF. When a real implementation
# lands, switch to detection:
#   Metal:   Apple platforms                  -> default ${APPLE}
#   CUDA:    find_package(CUDAToolkit QUIET)  -> default ${CUDAToolkit_FOUND}
option(GPUD_BACKEND_METAL "Build the Metal backend" OFF)
option(GPUD_BACKEND_CUDA "Build the CUDA backend" OFF)

# SDL_GPU is implemented; it defaults ON where SDL3's CMake package is
# findable (system package or CMAKE_PREFIX_PATH). find-only, no
# FetchContent fallback: SDL is not a small redistributable.
find_package(SDL3 QUIET CONFIG)
if(SDL3_FOUND)
  set(GPUD_SDL_DEFAULT ON)
else()
  set(GPUD_SDL_DEFAULT OFF)
endif()
option(GPUD_BACKEND_SDL "Build the SDL_GPU backend" ${GPUD_SDL_DEFAULT})

# ── core: the dependency-free interface ─────────────────────────────────
# <gpud/Version.h> is generated into the build tree from the project
# version above — bumping the version means editing project() only.
configure_file(include/gpud/Version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/gpud/Version.h @ONLY)

add_library(gpud INTERFACE)
add_library(gpud::gpud ALIAS gpud)
target_compile_features(gpud INTERFACE cxx_std_20)
target_include_directories(gpud INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# ── mock backend (header-only: include/gpud/Mock.h) ─────────────────────
# Its own target so consumer tests can link `gpud::mock` like any other
# backend, and so it could grow a compiled part without consumers changing.
add_library(gpud_mock INTERFACE)
add_library(gpud::mock ALIAS gpud_mock)
target_link_libraries(gpud_mock INTERFACE gpud)

# ── auto-selection: gpud_auto (alias gpud::auto_; `auto` is a C++ keyword) ─
add_library(gpud_auto STATIC src/auto/open_default.cpp)
add_library(gpud::auto_ ALIAS gpud_auto)
target_link_libraries(gpud_auto PUBLIC gpud)
set_target_properties(gpud_auto PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(NOT MSVC)
    target_compile_options(gpud_auto PRIVATE -Wall -Wextra)
endif()

# ── real backends (scaffolding: stub try_open, no SDK located/linked) ───
# One block per backend, in auto-selection priority order. Each is a
# separately compiled static library that links its SDK PRIVATE inside
# its own directory. Availability is baked into gpud_auto at its build
# time via the GPUD_HAS_* defines (PUBLIC so application code can
# compile conditionally on them too).
if(GPUD_BACKEND_CUDA)
  add_subdirectory(src/cuda)     # gpud_cuda + alias gpud::cuda
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_CUDA)
  target_link_libraries(gpud_auto PRIVATE gpud::cuda)
endif()
if(GPUD_BACKEND_METAL)
  add_subdirectory(src/metal)    # gpud_metal + alias gpud::metal
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_METAL)
  target_link_libraries(gpud_auto PRIVATE gpud::metal)
endif()
if(GPUD_BACKEND_VULKAN)
  add_subdirectory(src/vulkan)   # gpud_vulkan + alias gpud::vulkan
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_VULKAN)
  target_link_libraries(gpud_auto PRIVATE gpud::vulkan)
endif()
if(GPUD_BACKEND_SDL)
  add_subdirectory(src/sdl)      # gpud_sdl + alias gpud::sdl
  target_compile_definitions(gpud_auto PUBLIC GPUD_HAS_SDL)
  target_link_libraries(gpud_auto PRIVATE gpud::sdl)
endif()

if(GPUD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
