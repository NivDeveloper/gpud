// The Vulkan Memory Allocator implementation, alone in its own
// translation unit — it is a large header and compiling it once keeps it
// out of the sources that merely call it.
//
// volk resolves Vulkan at runtime, so VMA must not bind static function
// pointers; it imports what it needs through the two getters handed to
// vmaCreateAllocator in Device.cpp. The two defines that arrange this
// (VMA_STATIC_VULKAN_FUNCTIONS=0, VMA_DYNAMIC_VULKAN_FUNCTIONS=1) live
// on the target, so every TU including the header agrees with this one.
//
// volk.h comes first: with VK_NO_PROTOTYPES in force it is what declares
// the Vulkan types VMA is written against.

#include <volk.h>

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
