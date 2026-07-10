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
| SS best default | 8 | 27.6994 | 3.7632 | 0.97137 | 0.01737 | 24.854 | 0.0985 | 0.055 | 1.337 | 1.392 |
| SS best + full-count checkpoint | 8 | 27.6689 | 3.7301 | 0.97084 | 0.01780 | 24.832 | 0.0983 | 0.054 | 1.226 | 1.279 |

## Within-Trajectory Checkpoint Audit

Each delta compares the selected state with that same run's terminal state; both have the terminal Gaussian count. Positive means the selected state is better, including LPIPS (positive means lower). This avoids attributing independent CUDA trajectory divergence to checkpoint selection. Per-cell values are in `checkpoint_selection.csv`.

| Runs | Earlier states selected | PSNR gain | SSIM gain | MS-SSIM gain | LPIPS gain |
|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 0.0066 | -0.00013 | 0.00017 | -0.0010 |

## Paired Strict-Dominance Audit

Every delta is a candidate gain over `SS best default`; positive is better, including fit/total-time gains (positive means faster) and LPIPS gain (positive means lower LPIPS). Confidence intervals bootstrap source images after averaging correlated seeds/budgets within each image. A tradeoff is not a dominance result, and over-budget rows are not comparable. Displayed intervals are marginal 95% image-bootstrap intervals; the relation column uses Bonferroni-adjusted bounds for 95% familywise coverage across the five core metrics. Full rows are in `default_dominance.csv`.

| Candidate | Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | LPIPS gain [95% CI] | Sample relation | Familywise 95% relation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| SS best + full-count checkpoint | 8 / 4 | -0.0305 [-0.1147, +0.0824] | -0.00053 [-0.00115, -0.00012] | -0.0215 [-0.0555, +0.0093] | +0.1110 [+0.0104, +0.3001] | +0.1125 [+0.0097, +0.3061] | +0.0001 [-0.0020, +0.0030] | tradeoff | tradeoff |

The strict-dominance core uses PSNR, MS-SSIM, AUC, fit seconds, and total seconds. LPIPS is reported when enabled but is not silently imputed or added to the gate. These rows compare policies under StructSplat's harness; analogue labels do not make them native external-repository results.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + full-count checkpoint | 8 | -0.0305 | -0.00053 | -0.0215 | -0.1110 | -0.1125 | 4/8 | 2/8 | 3/8 | 6/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.854 | 17.090 | 23.905 | 23.321 | 26.809 | 27.699 |
| SS best + full-count checkpoint | 24.832 | 17.090 | 23.902 | 23.217 | 26.768 | 27.669 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 107.5 | 25% | 195.0 | 25% | 320.5 |
| SS best + full-count checkpoint | 25% | 107.5 | 25% | 199.0 | 25% | 343.0 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | SS best + full-count checkpoint | 320 | 640 | 27.6689 | 3.7301 | 0.97084 | 24.832 | 1.226 |
| 640 | SS best default | 320 | 640 | 27.6994 | 3.7632 | 0.97137 | 24.854 | 1.337 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best + full-count checkpoint (27.423) | SS best default (0.98263) |
| COCO_train2014_000000000025 | 640 | SS best default (25.356) | SS best default (0.96722) |
| COCO_train2014_000000000030 | 640 | SS best + full-count checkpoint (33.989) | SS best default (0.99116) |
| COCO_train2014_000000000034 | 640 | SS best default (24.360) | SS best default (0.94678) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
