#!/usr/bin/env bash
# ABL-005 fair-regime influence shard for CUDA-native fitter knobs only.
#
# Usage:
#   deprecated_scripts/run_abl005_cuda_native_influence.sh <kodim01> <kodim04> ...
#
# This intentionally excludes color_basis=affine because renderer=cuda falls back to the
# reference renderer for affine color gradients, confounding fit-time deltas.
#
# Axis env overrides keep long fair runs shardable by knob group, e.g.
#   COLOR_SOLVE_MODES=none PIXEL_LOSSES="l1 charbonnier" LR_SCHEDULES="none cosine" ...
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
DENSITY_MODES="${DENSITY_MODES:-structure variance}"
OPACITY_MODES="${OPACITY_MODES:-none constant}"
COLOR_SOLVE_MODES="${COLOR_SOLVE_MODES:-none every10}"
PIXEL_LOSSES="${PIXEL_LOSSES:-l1 charbonnier}"
LR_SCHEDULES="${LR_SCHEDULES:-none cosine}"
REFINE_MODES="${REFINE_MODES:-none moment_preserving}"

read -r -a BUDGET_ARGS <<< "$BUDGETS"
read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a TARGET_ARGS <<< "$TARGET_PSNRS"
read -r -a DENSITY_ARGS <<< "$DENSITY_MODES"
read -r -a OPACITY_ARGS <<< "$OPACITY_MODES"
read -r -a COLOR_SOLVE_ARGS <<< "$COLOR_SOLVE_MODES"
read -r -a PIXEL_LOSS_ARGS <<< "$PIXEL_LOSSES"
read -r -a LR_SCHEDULE_ARGS <<< "$LR_SCHEDULES"
read -r -a REFINE_ARGS <<< "$REFINE_MODES"

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
  --density-modes "${DENSITY_ARGS[@]}" \
  --sampling-modes wse \
  --orientation-modes tensor \
  --color-modes bilinear \
  --scale-modes spacing \
  --scale-cap-modes none \
  --background-modes off \
  --opacity-modes "${OPACITY_ARGS[@]}" \
  --renderers "$RENDERER" \
  --aa-dilations 0.0 \
  --color-basis-modes constant \
  --color-solve-modes "${COLOR_SOLVE_ARGS[@]}" \
  --pixel-losses "${PIXEL_LOSS_ARGS[@]}" \
  --loss-weight-modes none \
  --optimizers adam \
  --lr-schedules "${LR_SCHEDULE_ARGS[@]}" \
  --refine-modes "${REFINE_ARGS[@]}" \
  --state-seed-modes off \
  --row-temper-modes off \
  --support-fade-modes off \
  --pyramid-modes single \
  --resume \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
