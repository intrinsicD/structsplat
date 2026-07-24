#!/usr/bin/env bash
# Cheap first-pass search across StructSplat stage alternatives.
# Usage: deprecated_scripts/run_stage_search_screening.sh <images-or-dir>
set -euo pipefail

IMAGES="${1:?usage: run_stage_search_screening.sh <images-or-dir>}"
python -m benchmarks.stage_search "$IMAGES" \
  --budgets 1024 2048 \
  --seeds 0 \
  --iters 300 \
  --max-side 320 \
  --target-psnr 30 \
  --strategies aniso_flanking aniso_onedge iso_blue_noise \
  --tensor-operators central scharr \
  --tensor-colors luma rgb \
  --density-modes structure hybrid variance \
  --sampling-modes wse dart_throwing halton \
  --orientation-modes tensor \
  --color-modes bilinear local_mean two_sided \
  --scale-modes spacing uniform \
  --opacity-modes none constant \
  --renderers normalized \
  --pixel-losses l1 charbonnier \
  --optimizers adam adamw \
  --lr-schedules none cosine \
  --refine-modes none residual_add prune_residual_add \
  --pyramid-modes single pyramid \
  --split-count 128 \
  --max-configs 256 \
  --shuffle-configs \
  --config-seed 0 \
  --outdir results/stage_search_screening
