#!/usr/bin/env python3
"""Single-core FAISS HNSW baseline benchmark across the 6 evaluation datasets."""

import argparse
import math
import time
from pathlib import Path

import faiss
import h5py
import numpy as np

ALL_DATASETS = [
    "mnist-784-euclidean",
    "fashion-mnist-784-euclidean",
    "sift-128-euclidean",
    "glove-25-angular",
    "nytimes-256-angular",
    "cifar-100-euclidean",
]


def safe_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return arr / norms


def run_faiss_dataset(hdf5_file: Path, max_queries: int = 1000) -> dict:
    dataset_name = hdf5_file.stem
    print(f"\n=== FAISS HNSW Baseline: {dataset_name} ===")

    with h5py.File(hdf5_file, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"])

    if max_queries > 0:
        test = test[:max_queries]
        neighbors = neighbors[:max_queries]

    d = train.shape[1]
    M = 16
    ef_construction = 200

    if "angular" in dataset_name:
        train = safe_normalize(train)
        test = safe_normalize(test)
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_L2)

    index.hnsw.efConstruction = ef_construction
    faiss.omp_set_num_threads(1)

    t0 = time.perf_counter()
    index.add(train)
    build_s = time.perf_counter() - t0
    print(f"Vectors: {len(train)} | Queries: {len(test)} | Dim: {d} | Build time (1 thread): {build_s:.2f}s")

    print(f"{'ef_search':<12}{'Recall@10':<14}{'QPS':<14}{'Latency(ms)':<14}{'StdDev(%)':<12}")
    print("-" * 66)

    results = {}
    for ef in [10, 20, 40, 80, 160, 320]:
        index.hnsw.efSearch = ef
        recalls = []
        qps_runs = []

        for _ in range(3):
            start = time.perf_counter()
            _, I = index.search(test, 10)
            elapsed = time.perf_counter() - start

            matches = 0
            for i in range(len(test)):
                gt = set(neighbors[i][:10])
                pred = set(I[i])
                matches += len(gt.intersection(pred))

            recalls.append(matches / (len(test) * 10.0))
            qps_runs.append(len(test) / elapsed)

        mean_recall = sum(recalls) / 3.0
        mean_qps = sum(qps_runs) / 3.0
        var = sum((x - mean_qps) ** 2 for x in qps_runs) / 3.0
        std_pct = (math.sqrt(var) / mean_qps) * 100.0 if mean_qps > 0 else 0.0
        latency_ms = 1000.0 / mean_qps if mean_qps > 0 else 0.0

        print(f"{ef:<12}{mean_recall:<14.4f}{mean_qps:<14.1f}{latency_ms:<14.3f}{std_pct:<12.2f}")
        results[str(ef)] = {
            "recall": round(mean_recall, 4),
            "qps": round(mean_qps, 1),
            "latency_ms": round(latency_ms, 3),
            "std_pct": round(std_pct, 2),
        }

    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run single-core FAISS HNSW baseline")
    parser.add_argument("dataset", nargs="?", help="Dataset stem (e.g. mnist-784-euclidean)")
    parser.add_argument("--all", action="store_true", help="Run all 6 datasets")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "dataset")
    parser.add_argument("--max-queries", type=int, default=1000)
    args = parser.parse_args()

    targets = ALL_DATASETS if args.all else ([args.dataset] if args.dataset else ["mnist-784-euclidean"])
    for ds in targets:
        hdf5_path = args.dataset_dir / f"{ds}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {hdf5_path}")
        run_faiss_dataset(hdf5_path, max_queries=args.max_queries)


if __name__ == "__main__":
    main()
