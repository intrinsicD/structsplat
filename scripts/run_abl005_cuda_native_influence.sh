#!/usr/bin/env bash
# ABL-005 fair-regime influence shard for CUDA-native fitter knobs only.
#
# Usage:
#   scripts/run_abl005_cuda_native_influence.sh <kodim01> <kodim04> ...
#
# This intentionally excludes color_basis=affine because renderer=cuda falls back to the
# reference renderer for affine color gradients, confounding fit-time deltas.
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: run_abl005_cuda_native_influence.sh <image> [<image> ...]" >&2
  exit 2
fi

OUTDIR="${OUTDIR:-results/abl005_cuda_native_influence}"
DEVICE="${DEVICE:-cuda}"
RENDERER="${RENDERER:-cuda}"
MAX_SIDE="${MAX_SIDE:-768}"
ITERS="${ITERS:-1500}"
BUDGETS="${BUDGETS:-2000 5000 10000}"
SEEDS="${SEEDS:-0 1}"
TARGET_PSNRS="${TARGET_PSNRS:-28 30 32}"

read -r -a BUDGET_ARGS <<< "$BUDGETS"
read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a TARGET_ARGS <<< "$TARGET_PSNRS"

EXTRA_ARGS=()
if [[ -n "${MAX_NEW_CELLS:-}" ]]; then
  EXTRA_ARGS+=(--max-new-cells "$MAX_NEW_CELLS")
fi

python -m benchmarks.stage_search "$@" \
  --mode influence \
  --budgets "${BUDGET_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --iters "$ITERS" \
  --max-side "$MAX_SIDE" \
  --target-psnrs "${TARGET_ARGS[@]}" \
  --strategies quadtree_wse \
  --tensor-operators central \
  --tensor-colors luma \
  --density-modes structure variance \
  --sampling-modes wse \
  --orientation-modes tensor \
  --color-modes bilinear \
  --scale-modes spacing \
  --scale-cap-modes none \
  --background-modes off \
  --opacity-modes none constant \
  --renderers "$RENDERER" \
  --aa-dilations 0.0 \
  --color-basis-modes constant \
  --color-solve-modes none every10 \
  --pixel-losses l1 charbonnier \
  --loss-weight-modes none \
  --optimizers adam \
  --lr-schedules none cosine \
  --refine-modes none moment_preserving \
  --state-seed-modes off \
  --row-temper-modes off \
  --support-fade-modes off \
  --pyramid-modes single \
  --resume \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
