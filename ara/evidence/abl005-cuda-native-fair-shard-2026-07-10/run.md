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

All resume commands used the same 8 Kodak screen images listed in the initial command. The final
command was repeated to complete `kodim04` seed 1.

## Result

- Completed 24 full fair-regime CUDA rows: `kodim01` and `kodim04`, budget 2000, seeds 0 and 1,
  with 1500 iterations, max-side 768, exact CUDA renderer.
- Current coverage is 24/288 fast-axis rows, or 24/336 CUDA-native rows when the slow
  `color_solve=every10` arm is included.
- Wrote `results/abl005_cuda_native_influence/index.html`, `influence.md`, `summary.md`,
  `stage_search.csv`, and `config.json`; copied the same artifacts into this evidence directory.
- The mixed shard was interrupted at `color_solve=every10`; that arm should be resumed separately
  with a small `MAX_NEW_CELLS` because it runs much slower than ordinary CUDA fit arms.
- This is not promotion evidence: only two images, one budget, four paired cells, and no completed
  color-solve arm.

## Paired deltas vs baseline

- `density=variance` (4 pairs): ΔPSNR +0.1632, ΔMS-SSIM +0.00504, ΔAUC -0.0016, Δfit -0.525s
- `opacity=constant` (4 pairs): ΔPSNR +0.9082, ΔMS-SSIM +0.01472, ΔAUC +0.2617, Δfit +0.773s
- `loss=charbonnier` (4 pairs): ΔPSNR +0.1037, ΔMS-SSIM +0.00156, ΔAUC -0.0512, Δfit +0.160s
- `lr_schedule=cosine` (4 pairs): ΔPSNR +0.5757, ΔMS-SSIM +0.00906, ΔAUC +0.0738, Δfit +0.019s
- `refine_site=residual|refine_primitive=moment_preserving` (4 pairs): ΔPSNR +0.2973, ΔMS-SSIM +0.00366, ΔAUC -0.0443, Δfit -0.507s
Baseline row: `quadtree_wse`, structure density, no opacity, `loss=l1`, no LR schedule, no refine,
constant color basis, `renderer=cuda`.
