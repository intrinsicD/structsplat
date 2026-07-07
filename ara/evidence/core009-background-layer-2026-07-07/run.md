# CORE-009 background layer

Date: 2026-07-07

Goal: test whether reserving a counted, frozen-geometry low-frequency Gaussian layer improves
image fitting by freeing the remaining detail Gaussians from broad background coverage.

## Protocol

Fair-regime difficult-four exact-CUDA slice:

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budgets: 1000, 2000, 5000 total Gaussians.
- Strategies: `aniso_onedge`, `quadtree_wse`.
- Seed 0, max-side 768, 1500 iterations, `renderer=cuda`, `loss=l1`,
  `density=structure`, `sampling=wse`, `color=bilinear`, `scale=spacing`,
  `scale_cap=none`, no refine, no color solve, no loss weighting, single pyramid level.
- Background modes: `off`, `frac0.05_grid8`, `frac0.10_grid16`.

The background rows count against `num_gaussians`. Their means/scales/rotations are frozen during
fit, while colors remain learnable. Actual reserved rows are logged because the count is capped by
the grid size: for example `frac0.05_grid8` reserves 50/64/64 rows at budgets 1000/2000/5000.

## Result

Paired deltas versus matching `background=off` rows:

| background | pairs | dPSNR | PSNR wins | dMS-SSIM | MS-SSIM wins | dAUC | AUC wins | dFit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `frac0.05_grid8` | 24 | +1.0152 | 22/24 | +0.01412 | 24/24 | +0.1192 | 11/24 | +2.76 |
| `frac0.10_grid16` | 24 | +0.9564 | 17/24 | +0.01223 | 22/24 | +0.0828 | 11/24 | +2.57 |

By budget:

| budget | background | pairs | dPSNR | PSNR wins | dMS-SSIM | dAUC | AUC wins |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1000 | `frac0.05_grid8` | 8 | +1.8768 | 8/8 | +0.02210 | +0.5765 | 7/8 |
| 1000 | `frac0.10_grid16` | 8 | +1.9156 | 8/8 | +0.02074 | +0.6341 | 7/8 |
| 2000 | `frac0.05_grid8` | 8 | +1.1183 | 8/8 | +0.01371 | +0.1111 | 4/8 |
| 2000 | `frac0.10_grid16` | 8 | +1.0116 | 6/8 | +0.01072 | +0.0281 | 4/8 |
| 5000 | `frac0.05_grid8` | 8 | +0.0504 | 6/8 | +0.00655 | -0.3301 | 0/8 |
| 5000 | `frac0.10_grid16` | 8 | -0.0579 | 3/8 | +0.00525 | -0.4139 | 0/8 |

Baseline broad-support share was already small under the current init recipes:

| budget | baseline support > 1/4 image area |
|---:|---:|
| 1000 | 2.55% |
| 2000 | 1.14% |
| 5000 | 0.54% |

The background modes increased detail-row broad-support diagnostics after optimization
(+0.071 to +0.073 mean detail fraction overall), so the observed win is not cleanly explained as
"freeing many baseline broad splats." It is better described as a useful low-frequency color prior
that improves final PSNR/MS-SSIM at low budgets while slowing convergence.

## Verdict

Rung 1 is positive enough to keep as a stage-searchable initialization option, with
`frac0.05_grid8` the safer candidate. It is not promoted as a shipped default because the 5000-row
slice loses AUC in every pair and edge MAE worsens there.

Rung 2 is not implemented in this task. A true additive background/compositing layer would still
need a new ADR and larger confirmation, because this evidence does not prove the stronger renderer
semantics are worth changing.

## Artifacts

- `stage_search.csv`, `stage_search.json`, `summary.md`, `index.html`: raw benchmark outputs.
- `paired_deltas.csv`: row-level paired deltas versus `background=off`.
- `aggregate_deltas.csv`: grouped deltas by overall, budget, strategy, and image.
- `baseline_large_support.csv`: broad-support diagnostics for baseline rows.

## Verification

- Focused tests before benchmark: `python -m pytest tests/test_gaussians.py tests/test_init_stages.py tests/test_fit_dynamics.py tests/test_stage_search.py -q` passed 130 tests.
