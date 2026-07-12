# Convenience wrapper around the CMake build.
#
#   make        configure (once) + build everything
#   make test   run the test suite
#   make clean  remove the build tree

BUILD := build

.PHONY: all test clean

all: $(BUILD)/CMakeCache.txt
	cmake --build $(BUILD)

$(BUILD)/CMakeCache.txt:
	cmake -B $(BUILD) -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

test: all
	ctest --test-dir $(BUILD) --output-on-failure

clean:
	rm -rf $(BUILD)
