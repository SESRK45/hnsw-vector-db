#!/usr/bin/env python3
"""Download the 5 official ANN-Benchmarks HDF5 datasets and convert to fvecs/ivecs."""

import argparse
import os
import struct
import urllib.request
from pathlib import Path

import h5py
import numpy as np

DATASETS = [
    ("mnist-784-euclidean", "http://ann-benchmarks.com/mnist-784-euclidean.hdf5"),
    ("fashion-mnist-784-euclidean", "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"),
    ("sift-128-euclidean", "http://ann-benchmarks.com/sift-128-euclidean.hdf5"),
    ("glove-25-angular", "http://ann-benchmarks.com/glove-25-angular.hdf5"),
    ("nytimes-256-angular", "http://ann-benchmarks.com/nytimes-256-angular.hdf5"),
]


def write_fvecs(filename: Path, data: np.ndarray) -> None:
    with open(filename, "wb") as f:
        for vec in data:
            f.write(struct.pack("i", len(vec)))
            f.write(vec.astype("float32").tobytes())


def write_ivecs(filename: Path, data: np.ndarray) -> None:
    with open(filename, "wb") as f:
        for vec in data:
            f.write(struct.pack("i", len(vec)))
            f.write(vec.astype("int32").tobytes())


def safe_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return arr / norms


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Download and prepare ANN-Benchmarks datasets")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "dataset")
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    for name, url in DATASETS:
        hdf5_path = args.dataset_dir / f"{name}.hdf5"
        if not hdf5_path.exists():
            print(f"Downloading {name} from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(hdf5_path, "wb") as out:
                while chunk := resp.read(1024 * 1024):
                    out.write(chunk)

        base_path = args.dataset_dir / f"{name}_base.fvecs"
        query_path = args.dataset_dir / f"{name}_query.fvecs"
        gt_path = args.dataset_dir / f"{name}_groundtruth.ivecs"

        if not (base_path.exists() and query_path.exists() and gt_path.exists()):
            print(f"Converting {name} to fvecs/ivecs...")
            with h5py.File(hdf5_path, "r") as f:
                train = np.array(f["train"], dtype=np.float32)
                test = np.array(f["test"], dtype=np.float32)
                neighbors = np.array(f["neighbors"], dtype=np.int32)

                if "angular" in name:
                    train = safe_normalize(train)
                    test = safe_normalize(test)

                write_fvecs(base_path, train)
                write_fvecs(query_path, test)
                write_ivecs(gt_path, neighbors)

        print(f"Ready: {name}")


if __name__ == "__main__":
    main()
