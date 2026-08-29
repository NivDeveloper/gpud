include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)   # MSVC runtime; no-op elsewhere
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Device.h must be self-contained under plain C++20 — compile it alone,
# with no other includes in the translation unit.
add_library(gpud_device_h_standalone OBJECT device_h_standalone.cpp)
target_link_libraries(gpud_device_h_standalone PRIVATE gpud::gpud)
set_target_properties(gpud_device_h_standalone PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_device_h_standalone PRIVATE -Wall -Wextra)

add_executable(gpud_tests device_test.cpp conformance_test.cpp)
target_link_libraries(gpud_tests PRIVATE gpud::mock gpud::auto_ GTest::gtest_main)
# The conformance suite calls backend factories directly (gpud_auto only
# links them PRIVATE); GPUD_HAS_* defines arrive via gpud::auto_.
if(GPUD_BACKEND_VULKAN)
  target_link_libraries(gpud_tests PRIVATE gpud::vulkan)
endif()
if(GPUD_BACKEND_SDL)
  # SDL3 itself too: the interop test makes one real SDL call on the
  # exported native handle.
  target_link_libraries(gpud_tests PRIVATE gpud::sdl SDL3::SDL3)
endif()
set_target_properties(gpud_tests PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_tests PRIVATE -Wall -Wextra)

include(GoogleTest)
gtest_discover_tests(gpud_tests)
