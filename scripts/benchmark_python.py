#!/usr/bin/env python3
"""Test and benchmark Python bindings (custom_hnsw) using NumPy arrays."""

import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import custom_hnsw


def main() -> None:
    num_vectors = 10000
    num_queries = 100
    dim = 128
    k = 10

    rng = np.random.default_rng(42)
    data = rng.random((num_vectors, dim), dtype=np.float32)
    queries = rng.random((num_queries, dim), dtype=np.float32)

    graph = custom_hnsw.HNSWGraph(M=16, ef_construction=200)

    t0 = time.perf_counter()
    graph.build_index(data)
    build_s = time.perf_counter() - t0
    print(f"Built index for {graph.size()} vectors ({dim}D) in {build_s:.3f}s")

    t0 = time.perf_counter()
    results = graph.search_batch(queries, k=k, ef_search=50)
    batch_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Batch search ({num_queries} queries, ef=50): {batch_ms:.2f} ms ({batch_ms / num_queries:.3f} ms/query)")
    print(f"Sample query 0 top-5 IDs: {results[0][:5].tolist()}")

    # Test dynamic Bypass Bridge deletion via Python
    target_id = int(results[0][0])
    graph.remove(target_id)
    after_remove = graph.search(queries[0], k=k, ef_search=50)
    assert target_id not in after_remove, f"Deleted node {target_id} still appeared in search results!"
    print(f"Bypass Bridge deletion verified for node {target_id}.")

    # Test PQ training & ADC batch search
    graph.build_pq(data)
    pq_results = graph.search_pq_batch(queries, k=k, ef_search=50)
    print(f"PQ ADC sample query 0 top-5 IDs: {pq_results[0][:5].tolist()}")


if __name__ == "__main__":
    main()
