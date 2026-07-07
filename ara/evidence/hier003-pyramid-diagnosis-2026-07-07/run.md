# HIER-003 pyramid diagnosis

Date: 2026-07-07

Goal: separate a failed pyramid idea from a bad schedule by comparing equal-budget difficult-four
arms at the fair regime.

## Protocol

Common settings:

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budgets: 2000 and 5000.
- Strategies: `aniso_onedge`, `quadtree_wse`.
- Seed 0, max-side 768, exact CUDA renderer, 1500 nominal iterations, `loss=l1`,
  `color=bilinear`, `scale=spacing`, `scale_cap=none`, no color solve, no loss weighting.

Arms:

1. `single_1500`: single-stage baseline, 1500 iterations.
2. `pyramid_split_1500`: two-level pyramid, fractions 0.35/0.65, 750 iterations per level.
3. `pyramid_fullfield_iters`: two-level pyramid, fractions 0.35/0.65, 1500 iterations per level
   (3000 total), measuring the ceiling if coarse-level extra compute is free.
4. `pyramid_frac10_cosine`: two-level pyramid, fractions 0.1/0.9, 750 iterations per level, cosine
   LR over the pyramid run.
5. `refine_twin`: single-stage residual sampled-add schedule matched to the 0.35/0.65 level
   budgets: 700->2000 at iteration 750 and 1750->5000 at iteration 750.

The wrapper ran six resumable `stage_search` slices under
`results/hier003_pyramid_diagnosis_2026_07_07/`, then joined rows by image/budget/seed/strategy.

## Result

All 80 cells completed with `status=ok`.

Paired deltas vs `single_1500`:

| arm | pairs | dPSNR mean | dMS-SSIM mean | dEdge-MAE mean | dAUC mean | dFit sec mean | iterations |
|---|---:|---:|---:|---:|---:|---:|---:|
| pyramid_split_1500 | 16 | +1.0000 | +0.01399 | -0.000562 | -1.3540 | +0.2499 | +0 |
| pyramid_frac10_cosine | 16 | +0.4441 | +0.00595 | +0.003396 | -3.0657 | +0.1457 | +0 |
| pyramid_fullfield_iters | 16 | +0.0794 | +0.00354 | +0.001081 | -1.5716 | +5.2312 | +1500 |
| refine_twin | 16 | -0.4537 | -0.02392 | +0.003038 | -1.9926 | -0.0063 | +0 |

Key observations:

- The old “pyramid loses final quality” assumption is stale after HIER-002 and current settings:
  `pyramid_split_1500` won final PSNR in 16/16 pairs and improved edge MAE in 11/16 pairs.
- It is still a bad convergence/AUC tradeoff: every pyramid arm lost AUC in 16/16 pairs.
- Extra coarse-level iterations do not explain the result. The 3000-iteration ceiling arm barely
  improved final PSNR (+0.0794 dB) and still lost AUC, so simply giving the coarse prefix more
  training is not the fix.
- The 10/90 + cosine schedule helps final PSNR less than the current 35/65 split and worsens edge
  MAE, so it is not the schedule fix.
- The matched residual-add refine twin loses final PSNR and AUC, so the positive result is not
  merely “progressive capacity” in any form; the pyramid's residual re-tensoring/re-init ceremony
  matters for final quality.

## Verdict

HIER-001 should not be retired. The pyramid is a real final-quality arm now, but it should not be a
default yet because it loses convergence/AUC badly. The next task is to repair or bound that AUC
cost and confirm whether `pyramid_split_1500` is an offline-quality mode or a true default
candidate.

## Artifacts

- `combined_rows.csv`: all 80 rows with an `arm` column.
- `paired_deltas_vs_single.csv`: paired deltas by image/budget/strategy.
- `aggregate_deltas.csv`: aggregate deltas overall, by budget, and by strategy.
- `arm_means.csv`: absolute mean metrics by arm.
- `index.html`: scalar overview with links to all raw artifacts.
- Per-arm subdirectories include `config.json`, `stage_search.csv`, `summary.md`, and `index.html`.
