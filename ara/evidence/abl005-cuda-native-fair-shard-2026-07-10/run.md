# ABL-005 CUDA-native fair-regime shard — 2026-07-10

Purpose: start the decision-grade ABL-005 CUDA-native influence run on the fixed Kodak fair regime,
verify resumable `index.html` artifact generation, and separate fast knob groups from the slow
`color_solve=every10` arm.

## Commands

Initial broad shard, interrupted when `color_solve=every10` proved too slow for a 21-cell mixed
shard:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
MAX_NEW_CELLS=21 OUTDIR=results/abl005_cuda_native_influence DEVICE=cuda \
scripts/run_abl005_cuda_native_influence.sh \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim04.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim10.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim16.png \
  results/datasets/abl004/kodak24/kodim19.png \
  results/datasets/abl004/kodak24/kodim22.png
```

Fast-axis resumes after adding per-axis env overrides:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
MAX_NEW_CELLS=3 OUTDIR=results/abl005_cuda_native_influence DEVICE=cuda COLOR_SOLVE_MODES=none \
scripts/run_abl005_cuda_native_influence.sh ...

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
MAX_NEW_CELLS=6 OUTDIR=results/abl005_cuda_native_influence DEVICE=cuda COLOR_SOLVE_MODES=none \
scripts/run_abl005_cuda_native_influence.sh ...

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
MAX_NEW_CELLS=6 OUTDIR=results/abl005_cuda_native_influence DEVICE=cuda \
BUDGETS=2000 SEEDS="0 1" COLOR_SOLVE_MODES=none \
scripts/run_abl005_cuda_native_influence.sh ...
```

All resume commands used the same 8 Kodak screen images listed in the initial command.

## Result

- Completed 18 full fair-regime CUDA rows: `kodim01` budget 2000 seeds 0 and 1, plus `kodim04`
  budget 2000 seed 0, with 1500 iterations, max-side 768, exact CUDA renderer.
- Current coverage is 18/288 fast-axis rows, or 18/336 CUDA-native rows when the slow
  `color_solve=every10` arm is included.
- Wrote `results/abl005_cuda_native_influence/index.html`, `influence.md`, `summary.md`,
  `stage_search.csv`, and `config.json`; copied the same artifacts into this evidence directory.
- The mixed shard was interrupted at `color_solve=every10`; that arm should be resumed separately
  with a small `MAX_NEW_CELLS` because it runs much slower than ordinary CUDA fit arms.
- This is not promotion evidence: only two images, one budget, three paired cells, and no completed
  color-solve arm.

## Paired deltas vs baseline

- `density=variance` (3 pairs): ΔPSNR +0.3730, ΔMS-SSIM +0.00787, ΔAUC +0.0191, Δfit -0.431s
- `opacity=constant` (3 pairs): ΔPSNR +1.1558, ΔMS-SSIM +0.01875, ΔAUC +0.3108, Δfit +0.654s
- `loss=charbonnier` (3 pairs): ΔPSNR +0.3048, ΔMS-SSIM +0.00319, ΔAUC -0.0157, Δfit +0.214s
- `lr_schedule=cosine` (3 pairs): ΔPSNR +0.7819, ΔMS-SSIM +0.01224, ΔAUC +0.1109, Δfit +0.046s
- `refine_site=residual|refine_primitive=moment_preserving` (3 pairs): ΔPSNR +0.4064, ΔMS-SSIM +0.00539, ΔAUC -0.0338, Δfit -0.636s
Baseline row: `quadtree_wse`, structure density, no opacity, `loss=l1`, no LR schedule, no refine,
constant color basis, `renderer=cuda`.
