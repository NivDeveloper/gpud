# Vulkan backend. Building needs NO preinstalled SDK:
#   headers — the system's when find_package sees them, else a pinned
#             FetchContent of Khronos Vulkan-Headers
#   loader  — never linked: volk resolves it at runtime via dlopen (on
#             macOS it also tries libMoltenVK.dylib directly), so a
#             machine without a driver builds fine and try_open()
#             returns nullptr
# Running needs a Vulkan >= 1.2 driver with bufferDeviceAddress
# (macOS: `brew install molten-vk`) and slangc.
include(FetchContent)
enable_language(C)   # volk.c

add_library(gpud_vulkan STATIC
    Device.h Device.cpp     # try_open, run, teardown
    Buffer.h Buffer.cpp     # BufferImpl + alloc/write/read
    Kernel.h Kernel.cpp     # KernelImpl + do_compile (slangc)
    Vma.cpp)                # the VMA implementation, alone in its TU
add_library(gpud::vulkan ALIAS gpud_vulkan)
target_link_libraries(gpud_vulkan PUBLIC gpud::gpud)
if(CMAKE_DL_LIBS)
  target_link_libraries(gpud_vulkan PRIVATE ${CMAKE_DL_LIBS})
endif()
set_target_properties(gpud_vulkan PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
# -Wno-missing-field-initializers: `VkFoo f{VK_STRUCTURE_TYPE_...}` is
# the standard Vulkan idiom — the omitted fields zero-initialize.
if(NOT MSVC)
    target_compile_options(gpud_vulkan PRIVATE -Wall -Wextra
        -Wno-missing-field-initializers)
endif()
# VMA has to agree with volk about how Vulkan is resolved, and every TU
# including its header must see the same configuration as the one that
# compiles it — hence on the target rather than in Vma.cpp.
target_compile_definitions(gpud_vulkan PRIVATE VK_NO_PROTOTYPES
    VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=1)
# VMA's implementation is not warning-clean under -Wall -Wextra; keeping
# it quiet here stops it burying warnings from our own sources.
set_source_files_properties(Vma.cpp PROPERTIES COMPILE_OPTIONS "-w")

find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  target_include_directories(gpud_vulkan PRIVATE ${Vulkan_INCLUDE_DIRS})
else()
  FetchContent_Declare(VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers
    GIT_TAG v1.4.356)
  FetchContent_MakeAvailable(VulkanHeaders)
  target_link_libraries(gpud_vulkan PRIVATE Vulkan::Headers)
endif()

# volk is compiled directly into this target; SOURCE_SUBDIR points at a
# directory with no CMakeLists.txt so MakeAvailable only downloads —
# volk's own build system stays out of the picture.
FetchContent_Declare(volk
  GIT_REPOSITORY https://github.com/zeux/volk
  GIT_TAG vulkan-sdk-1.4.350.1
  SOURCE_SUBDIR sources-only-no-cmake)
FetchContent_MakeAvailable(volk)
target_sources(gpud_vulkan PRIVATE ${volk_SOURCE_DIR}/volk.c)
target_include_directories(gpud_vulkan PRIVATE ${volk_SOURCE_DIR})

# VMA (header-only): suballocates from large blocks, which is what
# removes the per-buffer vkAllocateMemory/vkMapMemory that dominated
# alloc(). Fetched download-only by the same SOURCE_SUBDIR trick as volk
# — its own CMakeLists would drag in its options and a find_package(Vulkan)
# requirement this backend deliberately does without.
FetchContent_Declare(VulkanMemoryAllocator
  GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
  GIT_TAG v3.3.0
  SOURCE_SUBDIR sources-only-no-cmake)
FetchContent_MakeAvailable(VulkanMemoryAllocator)
target_include_directories(gpud_vulkan PRIVATE
    ${vulkanmemoryallocator_SOURCE_DIR}/include)
