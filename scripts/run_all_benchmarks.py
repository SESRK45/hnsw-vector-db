#!/usr/bin/env python3
"""Run head-to-head C++ HNSW Engine vs FAISS benchmarks across all 6 datasets."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

DATASETS = [
    ("mnist-784-euclidean", "L2", 60000, 784),
    ("fashion-mnist-784-euclidean", "L2", 60000, 784),
    ("sift-128-euclidean", "L2", 1000000, 128),
    ("glove-25-angular", "Angular", 1183514, 25),
    ("nytimes-256-angular", "Angular", 290000, 256),
    ("cifar-100-euclidean", "L2", 50000, 3072),
]

ROW_RE = re.compile(r"^\s*(10|20|40|80|160|320)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def parse_table(output: str) -> dict:
    parsed = {}
    for line in output.splitlines():
        m = ROW_RE.match(line)
        if m:
            ef, rec, qps, lat, std = m.groups()
            parsed[ef] = {
                "recall": float(rec),
                "qps": float(qps),
                "latency_ms": float(lat),
                "std_pct": float(std),
            }
    return parsed


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Run full C++ Engine vs FAISS benchmark suite")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "dataset")
    parser.add_argument("--output", type=Path, default=repo_root / "benchmarks" / "benchmark_results.json")
    parser.add_argument("--skip-faiss", action="store_true")
    args = parser.parse_args()

    subprocess.run(["make", "benchmark"], cwd=repo_root, check=True)

    all_results = {}
    for ds, metric, count, dim in DATASETS:
        print(f"\n>>> Benchmarking {ds} ({count} vectors, {dim}D, {metric})")
        binary = repo_root / "bin" / ("benchmark_engine_angular" if metric == "Angular" else "benchmark_engine")
        cmd = [
            str(binary),
            str(args.dataset_dir / f"{ds}_base.fvecs"),
            str(args.dataset_dir / f"{ds}_query.fvecs"),
            str(args.dataset_dir / f"{ds}_groundtruth.ivecs"),
        ]
        cpp_res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
        print(cpp_res.stdout, flush=True)
        ds_entry = {"metric": metric, "vectors": count, "dim": dim, "cpp": parse_table(cpp_res.stdout)}

        if not args.skip_faiss:
            faiss_cmd = [
                "taskset", "-c", "0", sys.executable,
                str(repo_root / "scripts" / "benchmark_faiss.py"),
                ds, "--dataset-dir", str(args.dataset_dir), "--max-queries", "1000",
            ]
            faiss_res = subprocess.run(faiss_cmd, cwd=repo_root, capture_output=True, text=True, check=True)
            print(faiss_res.stdout, flush=True)
            ds_entry["faiss"] = parse_table(faiss_res.stdout)

        all_results[ds] = ds_entry
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2) + "\n")

    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
