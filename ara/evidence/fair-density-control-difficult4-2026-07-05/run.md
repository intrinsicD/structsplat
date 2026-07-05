# Fair Density-Control Difficult-4 Comparison

Purpose: answer the density-control-aware comparison question against
GaussianImage/GaussianImage++/Image-GS/Instant-GI style baselines under matched
StructSplat fitter conditions.

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
STRUCTSPLAT_INSTANT_GI=/home/alex/Documents/Instant-GI/quard_image.py \
PYTHONPATH=src:. \
python -m benchmarks.fair_density_control_compare \
  --outdir results/fair_density_control_difficult4 \
  --resume
```

Protocol:

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19` from Kodak24.
- Budgets: 2000, 5000, 10000 final Gaussians.
- Seeds: 0.
- Methods: 11 rows, including GaussianImage fixed, GaussianImage++ residual,
  Image-GS residual, Instant-GI quadtree fallback, StructSplat on-edge,
  flanking, quadtree-WSE, quadtree-hybrid, and Floyd controls.
- Fitter: exact CUDA renderer, 1500 iterations, max side 768, shared loss,
  shared target tracking, start fraction 0.5 for growth rows, 4 growth waves.
- Result shape: 4 images x 3 budgets x 11 methods = 132 cells.

Artifacts:

- Root: `results/fair_density_control_difficult4/`
- HTML overview: `results/fair_density_control_difficult4/index.html`
- Summary: `results/fair_density_control_difficult4/summary.md`
- Metrics: `results/fair_density_control_difficult4/metrics.csv`,
  `metrics.json`, `metrics.jsonl`
- Visuals: `plots/`, `grids/by_image/`, `grids/by_budget/`,
  `reconstructions/`

Completion:

- 132/132 cells finished with status `ok`.
- Sum of recorded per-cell total time: 55.95 minutes.
- Reconstructions written: 132 PNGs.
- Visual grids written: four by-image grids and three by-budget grids.

Headline PSNR means:

| Method | Mean PSNR | Mean MS-SSIM |
|---|---:|---:|
| SS qt-WSE + residual | 28.7143 | 0.92545 |
| SS on-edge + residual | 28.7012 | 0.92726 |
| SS qt-WSE + tensor | 28.6933 | 0.92641 |
| SS on-edge + tensor | 28.6719 | 0.92714 |
| SS flanking + tensor | 28.5561 | 0.92718 |
| Image-GS residual | 28.4006 | 0.92553 |
| GaussianImage++ residual | 27.6875 | 0.92183 |
| GaussianImage fixed | 25.4045 | 0.92228 |
| Instant-GI quadtree | 21.1642 | 0.80184 |

Paired mean PSNR deltas:

- Best StructSplat row vs GaussianImage++ residual: +1.0269 dB
  (`SS qt-WSE + residual`).
- Best StructSplat row vs Image-GS residual: +0.3137 dB
  (`SS qt-WSE + residual`).
- Top five PSNR means are all StructSplat structured-placement rows.

Winner counts:

- PSNR: StructSplat rows won all 12 image/budget slices.
- MS-SSIM: mixed. StructSplat rows won 9/12 slices; GaussianImage++ residual
  won 2/12 and GaussianImage fixed won 1/12.

Interpretation:

This is positive evidence for the broader structured-placement/density-control
direction under a fair same-start/same-cap growth protocol. The strongest
rows are not the original flanking-specific hypothesis; they are quadtree-WSE
or on-edge initialization combined with ordinary residual growth. Tensor-aware
growth remains competitive but is not clearly superior to residual_add here.

Caveats:

- This is a matched-policy analogue benchmark inside StructSplat's fitter and
  renderer, not a native external-repository execution.
- The Instant-GI row is the local quadtree/Delaunay fallback, not evidence about
  a learned Instant-GI checkpoint.
- The subset is hard-selected and seed-0 only, so it should guide the next
  confirmation plan but not replace the full 28-image x 3-seed confirmation set.
