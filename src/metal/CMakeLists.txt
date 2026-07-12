# Metal backend — scaffolding: no framework is linked yet.
# When the real implementation lands, this becomes Objective-C++:
#   enable_language(OBJCXX)
#   add_library(gpud_metal STATIC Metal.mm)
#   target_link_libraries(gpud_metal PRIVATE "-framework Metal" "-framework Foundation")
# (PRIVATE — the include firewall) and flip GPUD_BACKEND_METAL's default
# (top-level CMakeLists.txt) to ${APPLE}.
add_library(gpud_metal STATIC Metal.cpp)
add_library(gpud::metal ALIAS gpud_metal)
target_link_libraries(gpud_metal PUBLIC gpud::gpud)
set_target_properties(gpud_metal PROPERTIES
    CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
target_compile_options(gpud_metal PRIVATE -Wall -Wextra)
