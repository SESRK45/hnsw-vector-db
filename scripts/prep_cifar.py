#!/usr/bin/env python3
"""Download CIFAR-100 (3072-D RGB) and generate exact L2 ground truth in HDF5/fvecs/ivecs format."""

import argparse
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path

import faiss
import h5py
import numpy as np

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


def unpickle(file_path: Path) -> dict:
    with open(file_path, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


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


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Prepare CIFAR-100 Euclidean benchmark dataset")
    parser.add_argument("--dataset-dir", type=Path, default=repo_root / "dataset")
    args = parser.parse_args()

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.dataset_dir / "cifar-100-python.tar.gz"

    if not archive_path.exists():
        print(f"Downloading CIFAR-100 from {CIFAR_URL}...")
        urllib.request.urlretrieve(CIFAR_URL, archive_path)

    extracted_dir = args.dataset_dir / "cifar-100-python"
    if not extracted_dir.exists():
        print("Extracting CIFAR-100...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=args.dataset_dir)

    train_dict = unpickle(extracted_dir / "train")
    test_dict = unpickle(extracted_dir / "test")

    train_data = train_dict[b"data"].astype(np.float32)
    test_data = test_dict[b"data"].astype(np.float32)

    print(f"Computing exact top-100 L2 ground truth for {len(test_data)} queries against {len(train_data)} vectors...")
    index = faiss.IndexFlatL2(train_data.shape[1])
    index.add(train_data)
    distances, neighbors = index.search(test_data, 100)

    hdf5_path = args.dataset_dir / "cifar-100-euclidean.hdf5"
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("train", data=train_data)
        f.create_dataset("test", data=test_data)
        f.create_dataset("distances", data=distances)
        f.create_dataset("neighbors", data=neighbors.astype(np.int32))

    write_fvecs(args.dataset_dir / "cifar-100-euclidean_base.fvecs", train_data)
    write_fvecs(args.dataset_dir / "cifar-100-euclidean_query.fvecs", test_data)
    write_ivecs(args.dataset_dir / "cifar-100-euclidean_groundtruth.ivecs", neighbors.astype(np.int32))
    print("CIFAR-100 preparation complete.")


if __name__ == "__main__":
    main()
