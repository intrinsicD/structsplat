# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS best default | best-default | Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM. |
| SS best + GCR 0.015 | best-geometry-loss | Best default plus ground-truth-gradient-weighted Sobel consistency at weight 0.015. |
| SS best + GCR 0.015 every2 | best-geometry-loss | GCR 0.015 evaluated every 2 steps with active weight scaled to preserve its mean. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 4 | 20.3305 | 2.4376 | 0.77254 | 0.08626 | 21.523 | 0.4807 | 0.793 | 20.816 | 21.609 |
| SS best + GCR 0.015 | 4 | 20.5303 | 2.3498 | 0.78270 | 0.08669 | 21.698 | 0.4654 | 0.569 | 23.005 | 23.575 |
| SS best + GCR 0.015 every2 | 4 | 20.3567 | 2.5921 | 0.77274 | 0.09470 | 21.499 | 0.4756 | 0.246 | 18.878 | 19.124 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + GCR 0.015 | 4 / 4 | +0.1998 [-0.1109, +0.5105] | +0.01016 [-0.00257, +0.02497] | +0.1756 [+0.0140, +0.3025] | -2.1892 [-3.3450, -0.5918] | -1.9655 [-3.0182, -0.4496] | +0.0153 [+0.0005, +0.0302] | tradeoff | tradeoff |
| SS best + GCR 0.015 every2 | 4 / 4 | +0.0262 [-0.3639, +0.4739] | +0.00020 [-0.01541, +0.01598] | -0.0239 [-0.2756, +0.3809] | +1.9388 [+1.6299, +2.2477] | +2.4850 [+2.1267, +2.8434] | +0.0051 [-0.0090, +0.0134] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + GCR 0.015 | 4 | +0.1998 | +0.01016 | +0.1756 | +2.1892 | +1.9655 | 3/4 | 3/4 | 3/4 | 1/4 | no |
| SS best + GCR 0.015 every2 | 4 | +0.0262 | +0.00020 | -0.0239 | -1.9388 | -2.4850 | 1/4 | 2/4 | 1/4 | 4/4 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 1499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@375 | PSNR@750 | PSNR@1124 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 21.523 | 17.729 | 22.709 | 18.932 | 20.769 | 20.331 |
| SS best + GCR 0.015 | 21.698 | 17.729 | 22.766 | 19.277 | 21.223 | 20.530 |
| SS best + GCR 0.015 every2 | 21.499 | 17.729 | 22.883 | 18.961 | 20.905 | 20.357 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 0% | - | 0% | - | 0% | - |
| SS best + GCR 0.015 | 0% | - | 0% | - | 0% | - |
| SS best + GCR 0.015 every2 | 0% | - | 0% | - | 0% | - |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | SS best default | 1000 | 2000 | 20.3305 | 2.4376 | 0.77254 | 21.523 | 20.816 |
| 2000 | SS best + GCR 0.015 | 1000 | 2000 | 20.5303 | 2.3498 | 0.78270 | 21.698 | 23.005 |
| 2000 | SS best + GCR 0.015 every2 | 1000 | 2000 | 20.3567 | 2.5921 | 0.77274 | 21.499 | 18.878 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| kodim01 | 2000 | SS best + GCR 0.015 (18.732) | SS best + GCR 0.015 (0.73267) |
| kodim07 | 2000 | SS best + GCR 0.015 (22.640) | SS best + GCR 0.015 (0.88663) |
| kodim13 | 2000 | SS best + GCR 0.015 (17.696) | SS best default (0.67438) |
| kodim19 | 2000 | SS best + GCR 0.015 every2 (23.963) | SS best + GCR 0.015 every2 (0.85883) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
