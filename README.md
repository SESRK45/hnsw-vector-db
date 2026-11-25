# HNSW Vector Database

A C++17 HNSW (Hierarchical Navigable Small World) vector database engine built from scratch with AVX2 FMA3 SIMD distance kernels, SQ8/PQ compression, dynamic graph healing, concurrent reader-writer locking, out-of-core `mmap` search, and `pybind11` Python bindings.

---

## Key Features

- **Multi-Layer HNSW Graph (`src/hnsw.cpp`, `include/hnsw.h`):** Skip-list hierarchy with top-down greedy routing and dual-heap (`MinHeap` candidates + bounded `MaxHeap` top-k) Layer 0 beam search.
- **Vamana-Style Relaxed Pruning (`src/hnsw.cpp`):** Applies `alpha = 1.2` pruning and adaptive degree expansion on Layer 0 to preserve diverse paths in clustered distributions.
- **AVX2 FMA3 SIMD Kernels (`include/distance.h`):** Vectorized Euclidean (L2) and normalized Angular dot-product (`-DUSE_ANGULAR_DISTANCE`) kernels using `_mm256_fmadd_ps`.
- **Scalar & Product Quantization (`include/node.h`, `include/pq.h`):**
  - **SQ8:** Per-vector min/max 8-bit quantization during graph traversal paired with exact Float32 re-ranking.
  - **PQ + ADC:** Lloyd's K-Means codebook training and lookup-table Asymmetric Distance Computation (`_mm256_i32gather_ps` microbenchmark in `src/simd_pq_benchmark.cpp`).
- **Dynamic Deletions via Bypass Bridge (`HNSWGraph::remove`):** Marks deleted nodes as tombstones, strips dangling edges, and bridges starved neighbors (`degree < M_limit / 2`) without rebuilding the index.
- **Concurrency & Memory Layout (`include/arena.h`, `include/hnsw.h`):** Lock-free `std::atomic` entry-point updates, per-node `std::shared_mutex` reader-writer locks, and a contiguous `ArenaAllocator` for adjacency spans.
- **Persistence & Systems Extensions:**
  - Binary index serialization (`save` / `load`)
  - Packed `DiskNode` `mmap` traversal with `MADV_RANDOM` (`src/mmap_hnsw.cpp`)
  - Scatter-gather TCP shard server/router (`src/distributed.cpp`, `src/cluster_demo.cpp`)
  - Asynchronous `io_uring` socket server demo (`src/uring_demo.cpp`)
  - NumPy-compatible `pybind11` bindings (`src/bindings.cpp`)

---

## Repository Structure

```text
.
├── CMakeLists.txt
├── Makefile
├── requirements.txt
├── include/
│   ├── arena.h             # Contiguous bump allocator and ArenaSpan
│   ├── distance.h          # AVX2 L2, Angular dot-product, and SQ8 kernels
│   ├── distributed.h       # TCP shard server and scatter-gather router
│   ├── hnsw.h              # HNSWGraph declarations and locks
│   ├── mmap_hnsw.h         # Packed DiskNode and mmap reader
│   ├── node.h              # Node storage (Float32, SQ8, PQ, ArenaSpan edges)
│   ├── pq.h                # Lloyd's K-Means Product Quantizer & ADC LUT
│   └── priority_queue.h    # MinHeap / MaxHeap typedefs
├── src/
│   ├── hnsw.cpp            # Core HNSW build, search, PQ, healing, persistence
│   ├── bindings.cpp        # pybind11 module (custom_hnsw)
│   ├── benchmark.cpp       # End-to-end feature suite + benchmarks/results.csv
│   ├── recall_benchmark.cpp# Clustered 20K Recall@10 vs QPS sweep
│   ├── sift_benchmark.cpp  # Multi-dataset .fvecs/.ivecs benchmark runner
│   ├── simd_pq_benchmark.cpp # Isolated AVX2 gather ADC microbenchmark
│   ├── main.cpp            # Binary save/load verification test
│   ├── mmap_hnsw.cpp       # Out-of-core mmap traversal
│   ├── distributed.cpp     # TCP shard worker & cluster router
│   ├── cluster_demo.cpp    # 3-shard local cluster test
│   └── uring_demo.cpp      # Optional liburing async query server
├── scripts/
│   ├── download_datasets.py# Download & convert 5 ANN-Benchmarks datasets
│   ├── prep_cifar.py       # Prepare CIFAR-100 (3072-D) + exact L2 ground truth
│   ├── benchmark_faiss.py  # Single-core FAISS HNSW baseline runner
│   ├── run_benchmarks.sh   # Run C++ engine across all 6 datasets
│   ├── run_all_benchmarks.py # Orchestrate C++ Engine vs FAISS runs
│   └── benchmark_python.py # Exercise custom_hnsw Python bindings
└── benchmarks/
    ├── analyze.py          # Summarize benchmarks/results.csv
    ├── benchmark_results.json # Recorded C++ Engine vs FAISS metrics
    └── results.csv         # Per-query latency & Recall@10 log
```

---

## Building & Running

### Prerequisites
- Linux x86_64 CPU with AVX2 + FMA support
- `g++` (C++17) and `make` (or `cmake >= 3.14`)
- Python 3.10+ for dataset prep, FAISS baseline, and Python bindings (`pip install -r requirements.txt`)
- Optional: `liburing-dev` for building `uring_demo`

### 1. Build and Run Core C++ Suite

```bash
make all
make test
```

Individual executables built into `bin/`:
- `./bin/hnsw_main`: Verifies binary `save()` / `load()` round-trip equivalence (`10,000` vectors, `128-D`).
- `./bin/hnsw_benchmark`: Exercises multi-threaded insert, Recall@10 logging (`benchmarks/results.csv`), Bypass Bridge deletion, `mmap` export/search, metadata filtering, PQ ADC, and concurrent read throughput.
- `./bin/hnsw_recall_benchmark`: Sweeps `ef_search = {10, 20, 40, 80, 160, 320}` against exact KNN on `20,000` clustered vectors.
- `./bin/hnsw_cluster`: Runs a 3-shard local TCP cluster query test.
- `./bin/simd_pq_benchmark`: Measures single-core AVX2 `_mm256_i32gather_ps` PQ ADC scan speed over 1M 64-byte codes.

### 2. Optional Targets (`uring_demo` and Python Bindings)

```bash
# Build io_uring server demo (requires liburing-dev)
make uring && ./bin/uring_demo

# Build pybind11 Python extension and run Python test
make python
python3 scripts/benchmark_python.py
```

Or build everything via CMake:

```bash
cmake -B build && cmake --build build -j
```

### 3. Reproducing the 6-Dataset Benchmarks

Download the 5 official [ann-benchmarks.com](http://ann-benchmarks.com/) datasets (`mnist`, `fashion-mnist`, `sift-128`, `glove-25`, `nytimes-256`) and prepare CIFAR-100 (`3072-D`):

```bash
python3 scripts/download_datasets.py
python3 scripts/prep_cifar.py
```

Run the C++ Engine and FAISS side-by-side (`taskset -c 0` / Core 0 isolated query loop):

```bash
python3 scripts/run_all_benchmarks.py
```

---

## Verified Test & Benchmark Results

All measurements below were recorded on **12th Gen Intel Core i5-12500H (16 GB RAM, Ubuntu Linux, `g++ -O3 -march=native -mavx2 -mfma -fopenmp`)** with `M = 16`, `efConstruction = 200`.

### Core Suite Verification (Current Code, `1x ef_search`)
- **Binary Serialization (`./bin/hnsw_main`):** Save `6.23 ms`, load `7.60 ms`, identical Top-5 before and after reload.
- **Lifecycle Suite (`./bin/hnsw_benchmark` + `benchmarks/analyze.py`):** `99.85%` Mean Recall@10 (`169.99 us` mean latency, `5,883 QPS`), Bypass Bridge deletion in `8.03 us`, `54,825 QPS` concurrent OpenMP reads.
- **Clustered 20K Sweep (`./bin/hnsw_recall_benchmark`):** `0.9580` Recall@10 at `ef=10` (`19,432.2 QPS`), reaching `1.0000` at `ef=40` (`10,565.1 QPS`).
- **PQ ADC Microbenchmark (`./bin/simd_pq_benchmark`):** `10.75 ms` per query over 1M 64-byte PQ vectors (`33.45` CPU cycles/vector) on Core 0.
- **MNIST-784 (`1x ef_search`, Core 0 isolated query, 3-run mean):**

| `ef_search` | C++ Engine Recall@10 | C++ Engine QPS | Latency (ms) | FAISS Recall@10 | FAISS QPS |
| --- | --- | --- | --- | --- | --- |
| 10 | **0.9655** | 5,328.0 | 0.188 | 0.9166 | 20,691.0 |
| 20 | **0.9918** | 3,508.3 | 0.285 | 0.9747 | 13,320.8 |
| 40 | **0.9983** | 2,214.1 | 0.452 | 0.9920 | 8,494.7 |
| 80 | **0.9995** | 1,341.6 | 0.745 | 0.9983 | 4,972.9 |
| 160 | **1.0000** | 834.3 | 1.199 | 0.9997 | 2,935.9 |
| 320 | **1.0000** | 502.6 | 1.990 | 0.9999 | 1,630.0 |

### 6-Dataset Comparison Baseline (`benchmarks/benchmark_results.json`)
Recorded across `2,643,514` vectors (`ef_search = 10`, single-core `taskset -c 0`, 3-run averages; baseline recorded on the earlier `2x` SQ8 oversampled search configuration prior to switching `search()` to `1x ef_search`):

| Dataset | Vectors | Dim | Metric | C++ Recall@10 | C++ QPS | FAISS Recall@10 | FAISS QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | 60,000 | 784 | L2 | **0.9537** | 4,172.7 | 0.9261 | 23,931.4 |
| Fashion-MNIST | 60,000 | 784 | L2 | **0.9620** | 4,687.8 | 0.9319 | 25,397.1 |
| SIFT-1M | 1,000,000 | 128 | L2 | 0.4428 | 7,974.5 | **0.7157** | 39,921.5 |
| GloVe-25 | 1,183,514 | 25 | Angular | **0.8772** | 11,843.8 | 0.7249 | 64,920.5 |
| NYTimes-256 | 290,000 | 256 | Angular | **0.6093** | 5,778.8 | 0.6082 | 23,466.5 |
| CIFAR-100 | 50,000 | 3072 | L2 | 0.6916 | 611.8 | **0.8069** | 4,445.8 |

*(Full `ef = {10, 20, 40, 80, 160, 320}` sweeps are available in `benchmarks/benchmark_results.json`.)*
