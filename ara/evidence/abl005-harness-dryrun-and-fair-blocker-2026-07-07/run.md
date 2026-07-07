# ABL-005 Harness Dry Run And Fair-Run Blocker

Date: 2026-07-07.

## What Ran

After BENCH-004, ABL-005 was next in the backlog. The requested seven-knob influence command was
dry-run on `kodim01` at 64 px / 12 iters / budget 64 with the exact axes pinned:

- baseline plus `density=variance`
- `opacity=constant`
- `color_basis=affine`
- `color_solve=every10`
- `loss=charbonnier`
- `lr_schedule=cosine`
- `refine=moment_preserving`

The dry run completed 8/8 cells under `renderer=cuda`; see `dryrun_influence.md`.

## Harness Fixes Made

- `color_solve=every10` now supports normalized CUDA renderer modes (`cuda`,
  `cuda_normalized`, `cuda_tiled`, `cuda_tiled_normalized`).
- `color_basis=affine` under normalized CUDA modes falls back to the exact differentiable
  PyTorch reference equation on CUDA because the custom CUDA extension has no affine-color
  backward kernel.
- `stage-search` now has `--resume` and `--max-new-cells` so fair-regime runs can be sharded
  from `stage_search.jsonl`.

## Fair Shard Result

Started the ABL-005 fair command with:

```bash
python -m benchmarks.stage_search results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim04.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim10.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim16.png \
  results/datasets/abl004/kodak24/kodim19.png \
  results/datasets/abl004/kodak24/kodim22.png \
  --mode influence --budgets 2000 5000 10000 --seeds 0 1 \
  --iters 1500 --max-side 768 --strategies aniso_flanking \
  --tensor-operators central --tensor-colors luma \
  --density-modes structure variance --sampling-modes wse \
  --orientation-modes tensor --color-modes bilinear --scale-modes spacing \
  --scale-cap-modes none --opacity-modes none constant --renderers cuda \
  --aa-dilations 0 --color-basis-modes constant affine \
  --color-solve-modes none every10 --pixel-losses l1 charbonnier \
  --optimizers adam --lr-schedules none cosine \
  --refine-modes none moment_preserving --pyramid-modes single \
  --chunk 512 --ssim-weight 0.3 --target-psnrs 28 30 32 \
  --split-count 64 --outdir results/abl005_fitter_knob_influence_fair_2026_07_07 \
  --device cuda --resume --max-new-cells 16
```

The first three normal CUDA 2k cells completed in about 5 seconds of fit time each. The fourth cell
(`color_basis=affine`) used the reference fallback and had not completed after roughly 3 minutes;
the shard was interrupted at 229.97 wall seconds. The partial rows are in
`fair_shard_partial.jsonl`.

## Decision

Do not retire ABL-005 yet. The required seven-knob fair run is blocked by native CUDA affine-color
support, or the protocol must be split into:

1. six CUDA-native knobs at the ABL-005 fair regime, and
2. a separate affine quality run that explicitly excludes speed claims until native CUDA affine
   exists.

Continuing the current all-seven command would make the `color_basis=affine` speed delta mostly an
implementation fallback measurement, not an algorithmic fitter-knob measurement.
