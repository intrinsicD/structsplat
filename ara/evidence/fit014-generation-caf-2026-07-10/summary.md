# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS best default | best-default | Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM. |
| SS best + generation CAF | best-covariance-filter | Best default plus GaussianImage++-style birth-cohort covariance filtering with s=min(300, HW/(9*pi*N_after)) px^2. |
| SS best + generation CAF 18pi | best-covariance-filter | Weaker generation-density covariance filter with alpha=18*pi (half the native GaussianImage++ variance). |
| SS best + generation CAF 36pi | best-covariance-filter | Mild generation-density covariance filter with alpha=36*pi (quarter the native GaussianImage++ variance). |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 8 | 27.7115 | 3.7961 | 0.97080 | 0.01847 | 24.837 | 0.1033 | 0.059 | 1.261 | 1.320 |
| SS best + generation CAF | 8 | 25.3528 | 3.2770 | 0.95993 | 0.02529 | 23.276 | 0.2066 | 0.056 | 1.260 | 1.316 |
| SS best + generation CAF 18pi | 8 | 26.6316 | 3.4243 | 0.96709 | 0.02043 | 24.100 | 0.1439 | 0.055 | 1.259 | 1.313 |
| SS best + generation CAF 36pi | 8 | 27.2753 | 3.5347 | 0.96975 | 0.01886 | 24.549 | 0.1160 | 0.053 | 1.190 | 1.243 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + generation CAF | 8 / 4 | -2.3587 [-3.0772, -1.6167] | -0.01087 [-0.01879, -0.00438] | -1.5616 [-1.9527, -1.0427] | +0.0011 [-0.1307, +0.1822] | +0.0044 [-0.1340, +0.1978] | -0.1033 [-0.1347, -0.0732] | tradeoff | tradeoff |
| SS best + generation CAF 18pi | 8 / 4 | -1.0799 [-1.5378, -0.7131] | -0.00371 [-0.00572, -0.00193] | -0.7372 [-1.0013, -0.4556] | +0.0027 [-0.0215, +0.0268] | +0.0070 [-0.0150, +0.0290] | -0.0406 [-0.0571, -0.0241] | tradeoff | tradeoff |
| SS best + generation CAF 36pi | 8 / 4 | -0.4362 [-0.7519, -0.2556] | -0.00106 [-0.00153, -0.00058] | -0.2883 [-0.4290, -0.1643] | +0.0716 [-0.0363, +0.2355] | +0.0778 [-0.0359, +0.2517] | -0.0127 [-0.0208, -0.0072] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + generation CAF | 8 | -2.3587 | -0.01087 | -1.5616 | -0.0011 | -0.0044 | 0/8 | 0/8 | 0/8 | 2/8 | no |
| SS best + generation CAF 18pi | 8 | -1.0799 | -0.00371 | -0.7372 | -0.0027 | -0.0070 | 0/8 | 0/8 | 0/8 | 5/8 | no |
| SS best + generation CAF 36pi | 8 | -0.4362 | -0.00106 | -0.2883 | -0.0716 | -0.0778 | 0/8 | 0/8 | 0/8 | 5/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.837 | 17.090 | 23.903 | 23.099 | 26.819 | 27.712 |
| SS best + generation CAF | 23.276 | 16.926 | 22.668 | 22.522 | 24.613 | 25.353 |
| SS best + generation CAF 18pi | 24.100 | 17.007 | 23.277 | 22.926 | 25.791 | 26.632 |
| SS best + generation CAF 36pi | 24.549 | 17.048 | 23.625 | 23.300 | 26.381 | 27.275 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 108.0 | 25% | 196.5 | 25% | 325.0 |
| SS best + generation CAF | 25% | 198.5 | 25% | 421.5 | 0% | - |
| SS best + generation CAF 18pi | 25% | 132.5 | 25% | 284.0 | 25% | 470.5 |
| SS best + generation CAF 36pi | 25% | 116.0 | 25% | 220.0 | 25% | 381.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | SS best default | 320 | 640 | 27.7115 | 3.7961 | 0.97080 | 24.837 | 1.261 |
| 640 | SS best + generation CAF | 320 | 640 | 25.3528 | 3.2770 | 0.95993 | 23.276 | 1.260 |
| 640 | SS best + generation CAF 18pi | 320 | 640 | 26.6316 | 3.4243 | 0.96709 | 24.100 | 1.259 |
| 640 | SS best + generation CAF 36pi | 320 | 640 | 27.2753 | 3.5347 | 0.96975 | 24.549 | 1.190 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best default (27.515) | SS best default (0.98337) |
| COCO_train2014_000000000025 | 640 | SS best default (25.307) | SS best default (0.96746) |
| COCO_train2014_000000000030 | 640 | SS best default (34.147) | SS best default (0.99142) |
| COCO_train2014_000000000034 | 640 | SS best default (24.240) | SS best default (0.94265) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
