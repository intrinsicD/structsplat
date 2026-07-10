# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS best default | best-default | Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM. |
| SS best + GCR 0.015 | best-geometry-loss | Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.015. |
| SS best + GCR 0.030 | best-geometry-loss | Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.030. |
| SS best + GCR 0.060 | best-geometry-loss | Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.060. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 8 | 27.6135 | 3.8337 | 0.97010 | 0.01939 | 24.831 | 0.1008 | 0.054 | 1.228 | 1.282 |
| SS best + GCR 0.015 | 8 | 27.8021 | 3.7376 | 0.97096 | 0.01867 | 24.944 | 0.0961 | 0.052 | 1.317 | 1.369 |
| SS best + GCR 0.030 | 8 | 27.7832 | 3.7623 | 0.97084 | 0.01817 | 24.940 | 0.0945 | 0.051 | 1.316 | 1.367 |
| SS best + GCR 0.060 | 8 | 27.7213 | 3.8857 | 0.97040 | 0.01925 | 24.920 | 0.0966 | 0.054 | 1.327 | 1.380 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + GCR 0.015 | 8 / 4 | +0.1887 [+0.0853, +0.3179] | +0.00086 [+0.00017, +0.00156] | +0.1132 [+0.1001, +0.1312] | -0.0888 [-0.2083, +0.1097] | -0.0870 [-0.2084, +0.1152] | +0.0047 [+0.0003, +0.0108] | tradeoff | tradeoff |
| SS best + GCR 0.030 | 8 / 4 | +0.1697 [+0.0833, +0.3274] | +0.00074 [-0.00002, +0.00213] | +0.1093 [+0.0948, +0.1296] | -0.0879 [-0.2133, +0.0959] | -0.0851 [-0.2127, +0.1029] | +0.0064 [+0.0007, +0.0151] | tradeoff | tradeoff |
| SS best + GCR 0.060 | 8 / 4 | +0.1078 [-0.0172, +0.1887] | +0.00031 [-0.00021, +0.00091] | +0.0897 [+0.0358, +0.1437] | -0.0987 [-0.1830, +0.0602] | -0.0983 [-0.1898, +0.0674] | +0.0042 [-0.0004, +0.0103] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + GCR 0.015 | 8 | +0.1887 | +0.00086 | +0.1132 | +0.0888 | +0.0870 | 6/8 | 4/8 | 8/8 | 1/8 | no |
| SS best + GCR 0.030 | 8 | +0.1697 | +0.00074 | +0.1093 | +0.0879 | +0.0851 | 6/8 | 4/8 | 8/8 | 1/8 | no |
| SS best + GCR 0.060 | 8 | +0.1078 | +0.00031 | +0.0897 | +0.0987 | +0.0983 | 6/8 | 5/8 | 7/8 | 1/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.831 | 17.090 | 23.906 | 23.301 | 26.766 | 27.613 |
| SS best + GCR 0.015 | 24.944 | 17.090 | 23.995 | 23.228 | 26.934 | 27.802 |
| SS best + GCR 0.030 | 24.940 | 17.090 | 24.035 | 23.390 | 26.902 | 27.783 |
| SS best + GCR 0.060 | 24.920 | 17.090 | 24.090 | 23.109 | 26.893 | 27.721 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 108.0 | 25% | 196.5 | 25% | 323.0 |
| SS best + GCR 0.015 | 25% | 106.0 | 25% | 190.5 | 25% | 311.0 |
| SS best + GCR 0.030 | 25% | 105.0 | 25% | 188.5 | 25% | 310.5 |
| SS best + GCR 0.060 | 25% | 102.0 | 25% | 187.0 | 25% | 307.0 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | SS best default | 320 | 640 | 27.6135 | 3.8337 | 0.97010 | 24.831 | 1.228 |
| 640 | SS best + GCR 0.015 | 320 | 640 | 27.8021 | 3.7376 | 0.97096 | 24.944 | 1.317 |
| 640 | SS best + GCR 0.030 | 320 | 640 | 27.7832 | 3.7623 | 0.97084 | 24.940 | 1.316 |
| 640 | SS best + GCR 0.060 | 320 | 640 | 27.7213 | 3.8857 | 0.97040 | 24.920 | 1.327 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best + GCR 0.060 (27.645) | SS best + GCR 0.030 (0.98392) |
| COCO_train2014_000000000025 | 640 | SS best + GCR 0.060 (25.465) | SS best + GCR 0.015 (0.96883) |
| COCO_train2014_000000000030 | 640 | SS best default (34.113) | SS best default (0.99130) |
| COCO_train2014_000000000034 | 640 | SS best + GCR 0.015 (24.414) | SS best default (0.94561) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
