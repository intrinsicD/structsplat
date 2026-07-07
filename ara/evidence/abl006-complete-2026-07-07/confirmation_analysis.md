# ABL-006 Successive-Halving Analysis

Expected cells: 728
Completed cells: 728
Missing cells: 0
Baseline for paired deltas: `aniso_onedge`

## Leaderboard

| Budget | Rank | Strategy | Runs | PSNR | PSNR std | MS-SSIM | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean total s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1 | aniso_onedge | 84/84 | 26.5552 | 4.2266 | 0.90963 | 39/84 | 44.3 | 30/84 | 152.8 | 6/84 | 170.2 | 5.33 |
| 2000 | 2 | quadtree_wse | 84/84 | 26.5064 | 4.2822 | 0.90726 | 39/84 | 46.0 | 30/84 | 140.6 | 6/84 | 159.0 | 5.64 |
| 2000 | 3 | quadtree_hybrid | 56/56 | 26.4181 | 4.2372 | 0.90887 | 26/56 | 50.5 | 20/56 | 175.1 | 4/56 | 184.8 | 5.27 |
| 2000 | 4 | aniso_flanking | 56/56 | 26.3681 | 4.2565 | 0.90650 | 26/56 | 46.5 | 20/56 | 154.9 | 4/56 | 175.8 | 5.27 |
| 2000 | 5 | iso_blue_noise | 56/56 | 26.0755 | 4.2418 | 0.90292 | 26/56 | 94.6 | 19/56 | 229.5 | 4/56 | 264.0 | 5.05 |
| 2000 | 6 | floyd_steinberg | 56/56 | 23.6040 | 4.8684 | 0.88192 | 22/56 | 81.5 | 14/56 | 132.4 | 2/56 | 107.5 | 5.39 |
| 5000 | 1 | quadtree_wse | 84/84 | 29.8172 | 4.6439 | 0.95324 | 57/84 | 93.9 | 46/84 | 137.7 | 36/84 | 111.8 | 6.90 |
| 5000 | 2 | aniso_onedge | 84/84 | 29.7243 | 4.5984 | 0.95311 | 56/84 | 99.9 | 45/84 | 112.3 | 36/84 | 111.6 | 6.55 |
| 5000 | 3 | aniso_flanking | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 4 | quadtree_hybrid | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 5 | iso_blue_noise | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 6 | floyd_steinberg | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 1 | quadtree_wse | 84/84 | 32.6211 | 4.9161 | 0.97418 | 69/84 | 39.4 | 60/84 | 66.6 | 48/84 | 103.1 | 8.86 |
| 10000 | 2 | aniso_onedge | 84/84 | 32.5854 | 4.8293 | 0.97495 | 69/84 | 38.0 | 60/84 | 66.6 | 48/84 | 100.0 | 9.24 |
| 10000 | 3 | aniso_flanking | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 4 | quadtree_hybrid | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 5 | iso_blue_noise | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 6 | floyd_steinberg | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |

## Paired Deltas Vs Baseline

Positive means `left - baseline` is better. Confidence intervals bootstrap image x seed units.

| Budget | Left | Right | Pairs | PSNR delta | 95% CI | PSNR wins | MS-SSIM delta | 95% CI |
|---:|---|---|---:|---:|---|---:|---:|---|
| 2000 | aniso_flanking | aniso_onedge | 56 | -0.1791 | [-0.3054, -0.0471] | 13/56 | -0.00330 | [-0.00458, -0.00210] |
| 2000 | quadtree_wse | aniso_onedge | 84 | -0.0488 | [-0.1832, 0.0746] | 37/84 | -0.00237 | [-0.00364, -0.00122] |
| 2000 | quadtree_hybrid | aniso_onedge | 56 | -0.1291 | [-0.2631, 0.0191] | 14/56 | -0.00093 | [-0.00207, 0.00029] |
| 2000 | iso_blue_noise | aniso_onedge | 56 | -0.4717 | [-0.6229, -0.3236] | 5/56 | -0.00688 | [-0.00881, -0.00504] |
| 2000 | floyd_steinberg | aniso_onedge | 56 | -2.9432 | [-3.8009, -2.1402] | 6/56 | -0.02788 | [-0.03547, -0.02129] |
| 5000 | aniso_flanking | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | quadtree_wse | aniso_onedge | 84 | 0.0930 | [0.0168, 0.1700] | 52/84 | 0.00013 | [-0.00035, 0.00060] |
| 5000 | quadtree_hybrid | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | iso_blue_noise | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | floyd_steinberg | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | aniso_flanking | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | quadtree_wse | aniso_onedge | 84 | 0.0357 | [-0.0041, 0.0778] | 48/84 | -0.00077 | [-0.00102, -0.00052] |
| 10000 | quadtree_hybrid | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | iso_blue_noise | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | floyd_steinberg | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |

## Rank Stability

| Budget | Strategy | Complete groups | Wins | Mean rank | Median rank | Rank std |
|---:|---|---:|---:|---:|---:|---:|
| 2000 | aniso_onedge | 56 | 26 | 2.107 | 2.000 | 1.372 |
| 2000 | quadtree_wse | 56 | 18 | 2.464 | 2.000 | 1.375 |
| 2000 | quadtree_hybrid | 56 | 4 | 3.321 | 3.000 | 1.311 |
| 2000 | aniso_flanking | 56 | 4 | 3.357 | 3.000 | 1.231 |
| 2000 | iso_blue_noise | 56 | 1 | 4.482 | 5.000 | 1.134 |
| 2000 | floyd_steinberg | 56 | 3 | 5.268 | 6.000 | 1.433 |
| 5000 | quadtree_wse | 84 | 52 | 1.381 | 1.000 | 0.486 |
| 5000 | aniso_onedge | 84 | 32 | 1.619 | 2.000 | 0.486 |
| 5000 | aniso_flanking | 84 | 0 | - | - | - |
| 5000 | floyd_steinberg | 84 | 0 | - | - | - |
| 5000 | iso_blue_noise | 84 | 0 | - | - | - |
| 5000 | quadtree_hybrid | 84 | 0 | - | - | - |
| 10000 | quadtree_wse | 84 | 48 | 1.429 | 1.000 | 0.495 |
| 10000 | aniso_onedge | 84 | 36 | 1.571 | 2.000 | 0.495 |
| 10000 | aniso_flanking | 84 | 0 | - | - | - |
| 10000 | floyd_steinberg | 84 | 0 | - | - | - |
| 10000 | iso_blue_noise | 84 | 0 | - | - | - |
| 10000 | quadtree_hybrid | 84 | 0 | - | - | - |

## Files

- `abl006_plan.csv`: expected cells.
- `missing_cells.csv`: cells not yet present in `ablation.jsonl` / `ablation.json`.
- `leaderboard.csv`: aggregate budget x strategy means.
- `paired_deltas_vs_baseline.csv`: strategy minus baseline paired deltas.
- `pairwise_deltas.csv`: all pairwise strategy deltas.
- `paired_units_vs_baseline.csv`: per image x seed paired rows for failure inspection.
- `rank_stability.csv`: per-budget winner counts and rank distribution.
- `abl006_elimination_decisions.json`: frozen rule and stage survivor decisions.
- `abl006_elimination_trail.csv`: eliminated arms and decision statistics.
- `abl006_run_groups.json`: cartesian shards executed by `halving-run`.
