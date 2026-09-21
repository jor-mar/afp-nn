CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O3 -march=native -Wall -Wextra -Iinclude
LDFLAGS  ?=

BUILD := build

.PHONY: all test test-afp test-hybrid bench clean

all: $(BUILD)/quantize_and_benchmark $(BUILD)/test_afp $(BUILD)/test_hybrid

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/quantize_and_benchmark: src/quantize_and_benchmark.cpp include/afp/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) src/quantize_and_benchmark.cpp -o $@ $(LDFLAGS)

$(BUILD)/test_afp: tests/test_afp.cpp include/afp/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_afp.cpp -o $@ $(LDFLAGS)

$(BUILD)/test_hybrid: tests/test_hybrid.cpp include/afp/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_hybrid.cpp -o $@ $(LDFLAGS)

# Runs both suites; either can also be run individually (test-afp / test-hybrid).
test: test-afp test-hybrid

test-afp: $(BUILD)/test_afp
	./$(BUILD)/test_afp

test-hybrid: $(BUILD)/test_hybrid
	./$(BUILD)/test_hybrid

bench: $(BUILD)/quantize_and_benchmark
	./$(BUILD)/quantize_and_benchmark --synthetic

clean:
	rm -rf $(BUILD)
