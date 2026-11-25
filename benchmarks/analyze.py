#!/usr/bin/env python3
"""Analyze per-query Recall@10 and latency from benchmarks/results.csv."""

import csv
from pathlib import Path


def load_and_analyze() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    csv_file = repo_root / "benchmarks" / "results.csv"
    if not csv_file.exists():
        print(f"File {csv_file} not found! Run ./bin/hnsw_benchmark first.")
        return

    recalls = []
    times = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exact = set(row["exact_nn"].strip().split())
            hnsw_res = set(row["hnsw_nn"].strip().split())
            recalls.append(len(exact.intersection(hnsw_res)) / 10.0)
            times.append(float(row["time_us"]))

    mean_recall = (sum(recalls) / len(recalls)) * 100.0
    mean_time = sum(times) / len(times)
    qps = 1_000_000.0 / mean_time if mean_time > 0 else 0.0

    print("\n======================================")
    print("      HNSW VECTOR DB BENCHMARK SUMMARY")
    print("======================================")
    print(f"Queries Evaluated: {len(recalls)}")
    print(f"Mean Recall@10:    {mean_recall:.2f}%")
    print(f"Mean Latency:      {mean_time:.2f} us")
    print(f"Throughput:        {qps:.0f} QPS")
    print("======================================\n")


if __name__ == "__main__":
    load_and_analyze()
