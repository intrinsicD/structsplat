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
```

Both resume commands used the same 8 Kodak screen images listed in the initial command.

## Result

- Completed 12 full fair-regime CUDA rows: `kodim01`, budget 2000, seeds 0 and 1, 1500
  iterations, max-side 768, exact CUDA renderer.
- Wrote `results/abl005_cuda_native_influence/index.html`, `influence.md`, `summary.md`,
  `stage_search.csv`, and `config.json`; copied the same artifacts into this evidence directory.
- The mixed shard was interrupted at `color_solve=every10`; that arm should be resumed separately
  with a small `MAX_NEW_CELLS` because it runs much slower than ordinary CUDA fit arms.
- This is not promotion evidence: one image and one budget only, and no completed color-solve arm.

## Paired deltas vs baseline

- `density=variance` (2 pairs): ΔPSNR +0.4040, ΔMS-SSIM +0.01007, ΔAUC -0.0547, Δfit -0.728s
- `opacity=constant` (2 pairs): ΔPSNR +1.5633, ΔMS-SSIM +0.02641, ΔAUC +0.3759, Δfit +0.319s
- `loss=charbonnier` (2 pairs): ΔPSNR +0.3719, ΔMS-SSIM +0.00387, ΔAUC -0.0469, Δfit +0.166s
- `lr_schedule=cosine` (2 pairs): ΔPSNR +1.1433, ΔMS-SSIM +0.01822, ΔAUC +0.1595, Δfit -0.237s
- `refine_site=residual|refine_primitive=moment_preserving` (2 pairs): ΔPSNR +0.5602, ΔMS-SSIM +0.00787, ΔAUC -0.0377, Δfit -1.011s
Baseline row: `quadtree_wse`, structure density, no opacity, `loss=l1`, no LR schedule, no refine,
constant color basis, `renderer=cuda`.
