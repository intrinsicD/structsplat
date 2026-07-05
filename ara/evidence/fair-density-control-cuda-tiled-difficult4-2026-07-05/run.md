# Fair Density-Control cuda_tiled Test

Purpose: assess whether `renderer=cuda_tiled` is ready to accelerate fair/ABL training sweeps.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python benchmarks/fair_density_control_compare.py \
  --outdir results/fair_density_control_cuda_tiled_difficult4 \
  --methods structsplat_onedge_residual structsplat_onedge_tensor \
            structsplat_quadtree_wse_residual structsplat_quadtree_wse_tensor \
  --renderer cuda_tiled --resume
```

Scope: four current finalist rows, four difficult Kodak images, budgets {2000,5000,10000}, seed 0,
max-side 768, 1500 iters. Completed 48/48 ok cells and wrote a local HTML overview.

Paired result against exact `renderer=cuda`:

- Mean final PSNR delta: -0.1328 dB; wins 18/48.
- Mean AUC delta: +0.0009; wins 24/48.
- Mean fit-time delta: +17.63 s; mean ratio 1.69x slower.
- Slowest image: `kodim19`, +24.05 s mean, 1.86x slower.
- High-budget rows remained 1.68x slower on average.

Decision: keep exact CUDA for fair/ABL training sweeps. Treat tiled backward reductions and tighter
ellipse-tile bounds as prerequisites before using `cuda_tiled` as an acceleration path.

Live artifacts: `results/fair_density_control_cuda_tiled_difficult4/index.html`,
`results/fair_density_control_cuda_tiled_difficult4/summary.md`,
`results/fair_density_control_cuda_tiled_difficult4/metrics.jsonl`.
