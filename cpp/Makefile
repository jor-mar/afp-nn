CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS  ?=

BUILD := build

.PHONY: all test bench clean

all: $(BUILD)/quantize_and_benchmark $(BUILD)/test_afp

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/quantize_and_benchmark: src/quantize_and_benchmark.cpp include/afp/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/quantize_and_benchmark.cpp -o $@ $(LDFLAGS)

$(BUILD)/test_afp: tests/test_afp.cpp include/afp/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_afp.cpp -o $@ $(LDFLAGS)

test: $(BUILD)/test_afp
	./$(BUILD)/test_afp

bench: $(BUILD)/quantize_and_benchmark
	./$(BUILD)/quantize_and_benchmark --synthetic

clean:
	rm -rf $(BUILD)
