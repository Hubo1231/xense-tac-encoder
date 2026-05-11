#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_PATH="${1:-data/file-000_000000.png}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-1000}"

for config in configs/*.yaml; do
    echo
    echo "==> Benchmarking ${config}"
    python src/utils/benchmark_feature_extractors.py \
        --config "$config" \
        --warmup "$WARMUP" \
        --iters "$ITERS" \
        --image "$IMAGE_PATH"
done
