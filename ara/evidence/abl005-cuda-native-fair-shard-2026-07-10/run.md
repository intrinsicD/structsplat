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

Fast-axis resume after adding per-axis env overrides:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
MAX_NEW_CELLS=3 OUTDIR=results/abl005_cuda_native_influence DEVICE=cuda COLOR_SOLVE_MODES=none \
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

## Result

- Completed 6 full fair-regime CUDA rows: `kodim01`, budget 2000, seed 0, 1500 iterations,
  max-side 768, exact CUDA renderer.
- Wrote `results/abl005_cuda_native_influence/index.html`, `influence.md`, `summary.md`,
  `stage_search.csv`, and `config.json`; copied the same artifacts into this evidence directory.
- The mixed shard was interrupted at `color_solve=every10`; that arm should be resumed separately
  with a small `MAX_NEW_CELLS` because it runs much slower than ordinary CUDA fit arms.
- This is not promotion evidence: one image, one budget, one seed, and no completed color-solve arm.

## Paired deltas vs baseline

- `density=variance`: ΔPSNR -0.0203, ΔMS-SSIM +0.00640, ΔAUC -0.2181, Δfit -0.777s
- `opacity=constant`: ΔPSNR +1.5168, ΔMS-SSIM +0.02756, ΔAUC +0.3783, Δfit +1.411s
- `loss=charbonnier`: ΔPSNR +0.3367, ΔMS-SSIM +0.00278, ΔAUC -0.0910, Δfit +1.021s
- `lr_schedule=cosine`: ΔPSNR +0.9614, ΔMS-SSIM +0.01644, ΔAUC +0.0778, Δfit +0.245s
- `refine_site=residual|refine_primitive=moment_preserving`: ΔPSNR +0.4769, ΔMS-SSIM +0.00812, ΔAUC -0.0019, Δfit -0.238s
Baseline row: `quadtree_wse`, structure density, no opacity, `loss=l1`, no LR schedule, no refine,
constant color basis, `renderer=cuda`.
