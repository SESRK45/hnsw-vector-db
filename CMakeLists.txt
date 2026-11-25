cmake_minimum_required(VERSION 3.14)
project(HNSW_Vector_DB VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -mavx2 -mfma -fopenmp")

include_directories(include)

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module (custom_hnsw)" ON)

# Core C++ test and demo executables
add_executable(hnsw_main src/main.cpp src/hnsw.cpp src/mmap_hnsw.cpp src/distributed.cpp)
add_executable(hnsw_benchmark src/benchmark.cpp src/hnsw.cpp src/mmap_hnsw.cpp src/distributed.cpp)
add_executable(hnsw_recall_benchmark src/recall_benchmark.cpp src/hnsw.cpp src/mmap_hnsw.cpp src/distributed.cpp)
add_executable(hnsw_cluster src/cluster_demo.cpp src/hnsw.cpp src/mmap_hnsw.cpp src/distributed.cpp)

# Standard dataset benchmarks (Euclidean L2 and Normalized Angular)
add_executable(benchmark_engine src/sift_benchmark.cpp src/hnsw.cpp)
add_executable(benchmark_engine_angular src/sift_benchmark.cpp src/hnsw.cpp)
target_compile_definitions(benchmark_engine_angular PRIVATE USE_ANGULAR_DISTANCE)

# Standalone AVX2 PQ ADC gather microbenchmark
add_executable(simd_pq_benchmark src/simd_pq_benchmark.cpp)

# Optional io_uring async server demo (requires liburing-dev)
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBURING QUIET liburing)
endif()
if(LIBURING_FOUND)
    add_executable(uring_demo src/uring_demo.cpp src/hnsw.cpp)
    target_include_directories(uring_demo PRIVATE ${LIBURING_INCLUDE_DIRS})
    target_link_libraries(uring_demo PRIVATE ${LIBURING_LIBRARIES})
endif()

# Optional Python bindings via pybind11
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(custom_hnsw src/bindings.cpp src/hnsw.cpp)
endif()
