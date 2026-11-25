CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra -mavx2 -mfma -fopenmp -Iinclude

BIN_DIR  := bin

CORE_SRCS := src/hnsw.cpp src/mmap_hnsw.cpp src/distributed.cpp

CORE_TARGETS := \
	$(BIN_DIR)/hnsw_main \
	$(BIN_DIR)/hnsw_benchmark \
	$(BIN_DIR)/hnsw_recall_benchmark \
	$(BIN_DIR)/hnsw_cluster \
	$(BIN_DIR)/simd_pq_benchmark

BENCH_TARGETS := \
	$(BIN_DIR)/benchmark_engine \
	$(BIN_DIR)/benchmark_engine_angular

.PHONY: all benchmark python uring test clean

all: $(CORE_TARGETS) $(BENCH_TARGETS)

benchmark: $(BENCH_TARGETS)

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BIN_DIR)/hnsw_main: src/main.cpp $(CORE_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/hnsw_benchmark: src/benchmark.cpp $(CORE_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/hnsw_recall_benchmark: src/recall_benchmark.cpp $(CORE_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/hnsw_cluster: src/cluster_demo.cpp $(CORE_SRCS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/simd_pq_benchmark: src/simd_pq_benchmark.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN_DIR)/benchmark_engine: src/sift_benchmark.cpp src/hnsw.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR)/benchmark_engine_angular: src/sift_benchmark.cpp src/hnsw.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_ANGULAR_DISTANCE $^ -o $@

# Optional io_uring demo (requires liburing-dev or LIBURING_INC / LIBURING_LIB)
LIBURING_INC ?=
LIBURING_LIB ?= -luring
uring: $(BIN_DIR)/uring_demo
$(BIN_DIR)/uring_demo: src/uring_demo.cpp src/hnsw.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(if $(LIBURING_INC),-I$(LIBURING_INC)) $^ $(LIBURING_LIB) -o $@

# Optional pybind11 module (requires pybind11 Python package or PYBIND11_INC)
PYTHON ?= python3
PY_EXT := $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo .so)
PY_INC := $(shell $(PYTHON)-config --includes 2>/dev/null)
PYBIND11_INC ?= $(shell $(PYTHON) -c "import pybind11; print('-I' + pybind11.get_include())" 2>/dev/null)

python: custom_hnsw$(PY_EXT)
custom_hnsw$(PY_EXT): src/bindings.cpp src/hnsw.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC $(PY_INC) $(PYBIND11_INC) $^ -o $@

test: $(CORE_TARGETS)
	./$(BIN_DIR)/hnsw_main
	./$(BIN_DIR)/hnsw_benchmark
	./$(BIN_DIR)/hnsw_cluster
	./$(BIN_DIR)/simd_pq_benchmark
	./$(BIN_DIR)/hnsw_recall_benchmark

clean:
	rm -rf $(BIN_DIR) build *.so
