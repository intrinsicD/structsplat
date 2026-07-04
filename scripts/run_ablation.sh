#!/usr/bin/env bash
# Convenience wrapper for the ABL-001 sweep. Usage: scripts/run_ablation.sh <images-or-dir>
set -euo pipefail
IMAGES="${1:?usage: run_ablation.sh <images-or-dir>}"
python -m benchmarks.ablation "$IMAGES" \
  --budgets 2000 5000 10000 20000 \
  --iters 1500 --target-psnr 35 \
  --max-side "${MAX_SIDE:-768}" \
  --resume \
  --outdir "${OUTDIR:-results/ablation}"
