#!/usr/bin/env bash
# Full ABL-004 ablation entry point. Downloaded data/results stay under ignored results/.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-results/datasets/abl004}"
OUTDIR="${OUTDIR:-results/abl004_ablation_full}"
MAX_SIDE="${MAX_SIDE:-768}"
DEVICE="${DEVICE:-cuda}"

python scripts/prepare_abl004_images.py --outdir "$DATA_ROOT"

mapfile -t IMAGES < "$DATA_ROOT/images.txt"
python -m benchmarks.ablation "${IMAGES[@]}" \
  --budgets 2000 5000 10000 20000 \
  --seeds 0 1 2 \
  --iters 1500 \
  --target-psnr 35 \
  --max-side "$MAX_SIDE" \
  --resume \
  --device "$DEVICE" \
  --outdir "$OUTDIR"
