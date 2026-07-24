#!/usr/bin/env bash
# Per-stage influence study (ADR-0010): one-factor-at-a-time paired deltas around the
# ADR-0009 baseline, across all stage alternatives. Writes influence.md with
# ΔPSNR / ΔMS-SSIM / ΔAUC / Δiters-to-target / Δseconds per stage option.
# Usage: deprecated_scripts/run_stage_influence.sh <images-or-dir>
set -euo pipefail

IMAGES="${1:?usage: run_stage_influence.sh <images-or-dir>}"
python -m benchmarks.stage_search "$IMAGES" \
  --mode influence \
  --budgets 1024 4096 \
  --seeds 0 1 2 \
  --iters 500 \
  --max-side 320 \
  --target-psnr 30 \
  --outdir results/stage_influence
