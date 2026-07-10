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
| SS best + GCR 0.015 every4 | best-geometry-loss | GCR 0.015 evaluated every 4 steps with active weight scaled to preserve its mean. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 8 | 27.6595 | 3.6792 | 0.97103 | 0.01791 | 24.832 | 0.0958 | 0.054 | 1.378 | 1.431 |
| SS best + GCR 0.015 | 8 | 27.7453 | 3.7666 | 0.97123 | 0.01832 | 24.910 | 0.0969 | 0.059 | 1.614 | 1.673 |
| SS best + GCR 0.015 every2 | 8 | 27.8099 | 3.7082 | 0.97117 | 0.01767 | 24.900 | 0.1029 | 0.066 | 1.486 | 1.552 |
| SS best + GCR 0.015 every4 | 8 | 27.7054 | 3.8023 | 0.97004 | 0.01973 | 24.860 | 0.0973 | 0.065 | 1.556 | 1.621 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + GCR 0.015 | 8 / 4 | +0.0859 [-0.0359, +0.2076] | +0.00020 [-0.00041, +0.00087] | +0.0771 [+0.0277, +0.1091] | -0.2365 [-0.5883, +0.0857] | -0.2420 [-0.6074, +0.0889] | -0.0011 [-0.0034, +0.0013] | tradeoff | tradeoff |
| SS best + GCR 0.015 every2 | 8 / 4 | +0.1504 [+0.0380, +0.2848] | +0.00015 [-0.00077, +0.00107] | +0.0673 [+0.0370, +0.0979] | -0.1088 [-0.3881, +0.1581] | -0.1210 [-0.4276, +0.1609] | -0.0070 [-0.0205, -0.0002] | tradeoff | tradeoff |
| SS best + GCR 0.015 every4 | 8 / 4 | +0.0459 [-0.1351, +0.1761] | -0.00098 [-0.00331, +0.00050] | +0.0274 [-0.1068, +0.1335] | -0.1782 [-0.6002, +0.1675] | -0.1893 [-0.6382, +0.1717] | -0.0015 [-0.0049, +0.0015] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + GCR 0.015 | 8 | +0.0859 | +0.00020 | +0.0771 | +0.2365 | +0.2420 | 5/8 | 4/8 | 7/8 | 1/8 | no |
| SS best + GCR 0.015 every2 | 8 | +0.1504 | +0.00015 | +0.0673 | +0.1088 | +0.1210 | 7/8 | 5/8 | 8/8 | 2/8 | no |
| SS best + GCR 0.015 every4 | 8 | +0.0459 | -0.00098 | +0.0274 | +0.1782 | +0.1893 | 6/8 | 5/8 | 6/8 | 1/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.832 | 17.090 | 23.907 | 23.277 | 26.774 | 27.659 |
| SS best + GCR 0.015 | 24.910 | 17.090 | 23.989 | 23.153 | 26.891 | 27.745 |
| SS best + GCR 0.015 every2 | 24.900 | 17.090 | 23.942 | 23.553 | 26.851 | 27.810 |
| SS best + GCR 0.015 every4 | 24.860 | 17.090 | 23.890 | 23.371 | 26.816 | 27.705 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 107.5 | 25% | 198.5 | 25% | 357.0 |
| SS best + GCR 0.015 | 25% | 106.0 | 25% | 192.0 | 25% | 315.5 |
| SS best + GCR 0.015 every2 | 25% | 106.5 | 25% | 193.0 | 25% | 312.5 |
| SS best + GCR 0.015 every4 | 25% | 107.5 | 25% | 193.5 | 25% | 307.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | SS best default | 320 | 640 | 27.6595 | 3.6792 | 0.97103 | 24.832 | 1.378 |
| 640 | SS best + GCR 0.015 | 320 | 640 | 27.7453 | 3.7666 | 0.97123 | 24.910 | 1.614 |
| 640 | SS best + GCR 0.015 every2 | 320 | 640 | 27.8099 | 3.7082 | 0.97117 | 24.900 | 1.486 |
| 640 | SS best + GCR 0.015 every4 | 320 | 640 | 27.7054 | 3.8023 | 0.97004 | 24.860 | 1.556 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best + GCR 0.015 every2 (27.790) | SS best + GCR 0.015 (0.98402) |
| COCO_train2014_000000000025 | 640 | SS best + GCR 0.015 every2 (25.456) | SS best + GCR 0.015 (0.96775) |
| COCO_train2014_000000000030 | 640 | SS best + GCR 0.015 every4 (34.001) | SS best + GCR 0.015 (0.99126) |
| COCO_train2014_000000000034 | 640 | SS best + GCR 0.015 every2 (24.367) | SS best + GCR 0.015 every2 (0.94534) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
