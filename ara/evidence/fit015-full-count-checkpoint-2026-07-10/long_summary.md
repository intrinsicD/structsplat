# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS best default | best-default | Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM. |
| SS best + full-count checkpoint | best-long-horizon | Best default with post-transition PSNR checkpoint selection restricted to states at the terminal Gaussian count. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 8 | 26.6016 | 5.5464 | 0.95475 | 0.03181 | 24.764 | 0.0981 | 0.058 | 13.915 | 13.973 |
| SS best + full-count checkpoint | 8 | 27.2031 | 4.8617 | 0.96393 | 0.02392 | 24.731 | 0.0865 | 0.056 | 13.356 | 13.412 |

## Within-Trajectory Checkpoint Audit

Each delta compares the selected state with that same run's terminal state; both have the terminal Gaussian count. Positive means the selected state is better, including LPIPS (positive means lower). This avoids attributing independent CUDA trajectory divergence to checkpoint selection. Per-cell values are in `checkpoint_selection.csv`.

| Runs | Earlier states selected | PSNR gain | SSIM gain | MS-SSIM gain | LPIPS gain |
|---:|---:|---:|---:|---:|---:|
| 8 | 7 | 0.7702 | 0.00669 | 0.00892 | 0.0076 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + full-count checkpoint | 8 / 4 | +0.6016 [-0.2185, +1.4420] | +0.00918 [+0.00049, +0.01787] | -0.0334 [-0.3144, +0.2631] | +0.5584 [-2.1052, +3.2219] | +0.5605 [-2.1049, +3.2258] | +0.0115 [-0.0033, +0.0264] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + full-count checkpoint | 8 | +0.6016 | +0.00918 | -0.0334 | -0.5584 | -0.5605 | 5/8 | 6/8 | 3/8 | 5/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 4999-iteration budget.

| Method | AUC | PSNR@0 | PSNR@1250 | PSNR@2500 | PSNR@3749 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.764 | 17.090 | 23.717 | 21.784 | 26.336 | 26.602 |
| SS best + full-count checkpoint | 24.731 | 17.090 | 24.065 | 22.656 | 26.052 | 27.203 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 128.5 | 25% | 866.0 | 25% | 1745.0 |
| SS best + full-count checkpoint | 25% | 128.5 | 25% | 864.5 | 25% | 1723.0 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | SS best + full-count checkpoint | 320 | 640 | 27.2031 | 4.8617 | 0.96393 | 24.731 | 13.356 |
| 640 | SS best default | 320 | 640 | 26.6016 | 5.5464 | 0.95475 | 24.764 | 13.915 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best + full-count checkpoint (26.943) | SS best + full-count checkpoint (0.97827) |
| COCO_train2014_000000000025 | 640 | SS best + full-count checkpoint (24.420) | SS best + full-count checkpoint (0.95803) |
| COCO_train2014_000000000030 | 640 | SS best default (36.014) | SS best default (0.99552) |
| COCO_train2014_000000000034 | 640 | SS best + full-count checkpoint (22.931) | SS best + full-count checkpoint (0.93360) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
