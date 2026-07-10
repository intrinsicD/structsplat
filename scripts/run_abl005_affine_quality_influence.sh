#!/usr/bin/env bash
# ABL-005 affine color-basis quality-only influence shard.
#
# Usage:
#   scripts/run_abl005_affine_quality_influence.sh <kodim01> <kodim04> ...
#
# The affine arm must not be used for speed/default-promotion claims until native CUDA affine
# backward exists. This script pins renderer=normalized by default so constant and affine rows use
# the same exact reference renderer semantics.
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "usage: run_abl005_affine_quality_influence.sh <image> [<image> ...]" >&2
  exit 2
fi

OUTDIR="${OUTDIR:-results/abl005_affine_quality_influence}"
DEVICE="${DEVICE:-cuda}"
RENDERER="${RENDERER:-normalized}"
MAX_SIDE="${MAX_SIDE:-768}"
ITERS="${ITERS:-1500}"
BUDGETS="${BUDGETS:-2000 5000 10000}"
SEEDS="${SEEDS:-0 1}"
TARGET_PSNRS="${TARGET_PSNRS:-28 30 32}"
COLOR_BASIS_MODES="${COLOR_BASIS_MODES:-constant affine}"

read -r -a BUDGET_ARGS <<< "$BUDGETS"
read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a TARGET_ARGS <<< "$TARGET_PSNRS"
read -r -a COLOR_BASIS_ARGS <<< "$COLOR_BASIS_MODES"

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
  --density-modes structure \
  --sampling-modes wse \
  --orientation-modes tensor \
  --color-modes bilinear \
  --scale-modes spacing \
  --scale-cap-modes none \
  --background-modes off \
  --opacity-modes none \
  --renderers "$RENDERER" \
  --aa-dilations 0.0 \
  --color-basis-modes "${COLOR_BASIS_ARGS[@]}" \
  --color-solve-modes none \
  --pixel-losses l1 \
  --loss-weight-modes none \
  --optimizers adam \
  --lr-schedules none \
  --refine-modes none \
  --state-seed-modes off \
  --row-temper-modes off \
  --support-fade-modes off \
  --pyramid-modes single \
  --resume \
  --outdir "$OUTDIR" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
