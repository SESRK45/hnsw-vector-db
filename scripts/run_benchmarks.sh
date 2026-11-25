#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${1:-${REPO_ROOT}/dataset}"

cd "${REPO_ROOT}"
make benchmark

EUCLIDEAN_DATASETS=(
    "mnist-784-euclidean"
    "fashion-mnist-784-euclidean"
    "sift-128-euclidean"
    "cifar-100-euclidean"
)

ANGULAR_DATASETS=(
    "glove-25-angular"
    "nytimes-256-angular"
)

for ds in "${EUCLIDEAN_DATASETS[@]}"; do
    echo "=========================================================="
    echo "Running C++ Engine (Euclidean L2) on: ${ds}"
    echo "=========================================================="
    ./bin/benchmark_engine \
        "${DATASET_DIR}/${ds}_base.fvecs" \
        "${DATASET_DIR}/${ds}_query.fvecs" \
        "${DATASET_DIR}/${ds}_groundtruth.ivecs"
    echo ""
done

for ds in "${ANGULAR_DATASETS[@]}"; do
    echo "=========================================================="
    echo "Running C++ Engine (Angular Dot-Product) on: ${ds}"
    echo "=========================================================="
    ./bin/benchmark_engine_angular \
        "${DATASET_DIR}/${ds}_base.fvecs" \
        "${DATASET_DIR}/${ds}_query.fvecs" \
        "${DATASET_DIR}/${ds}_groundtruth.ivecs"
    echo ""
done
