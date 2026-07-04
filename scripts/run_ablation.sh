#!/usr/bin/env bash
# Convenience wrapper for the ABL-001 sweep. Usage: scripts/run_ablation.sh <images-or-dir>
set -euo pipefail
IMAGES="${1:?usage: run_ablation.sh <images-or-dir>}"
EXTRA_ARGS=()
if [[ -n "${MAX_NEW_CELLS:-}" ]]; then
  EXTRA_ARGS+=(--max-new-cells "$MAX_NEW_CELLS")
fi

python -m benchmarks.ablation "$IMAGES" \
  --budgets 2000 5000 10000 20000 \
  --iters 1500 --target-psnr 35 \
  --max-side "${MAX_SIDE:-768}" \
  --resume \
  "${EXTRA_ARGS[@]}" \
  --outdir "${OUTDIR:-results/ablation}"
