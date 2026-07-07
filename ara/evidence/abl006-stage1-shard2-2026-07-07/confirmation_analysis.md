# ABL-006 Successive-Halving Analysis

Expected cells: 336
Completed cells: 24
Missing cells: 312
Baseline for paired deltas: `aniso_onedge`

## Leaderboard

| Budget | Rank | Strategy | Runs | PSNR | PSNR std | MS-SSIM | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean total s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1 | aniso_onedge | 4/56 | 27.3511 | 3.8951 | 0.90271 | 2/4 | 21.0 | 2/4 | 64.0 | 0/4 | - | 5.41 |
| 2000 | 2 | quadtree_wse | 4/56 | 27.2392 | 3.9395 | 0.90257 | 2/4 | 20.5 | 2/4 | 65.0 | 0/4 | - | 6.05 |
| 2000 | 3 | floyd_steinberg | 4/56 | 26.9542 | 4.0469 | 0.89761 | 2/4 | 44.0 | 2/4 | 124.0 | 0/4 | - | 5.90 |
| 2000 | 4 | quadtree_hybrid | 4/56 | 26.9497 | 4.0505 | 0.89870 | 2/4 | 21.0 | 2/4 | 65.5 | 0/4 | - | 5.20 |
| 2000 | 5 | aniso_flanking | 4/56 | 26.8131 | 4.3357 | 0.89474 | 2/4 | 20.5 | 2/4 | 64.0 | 0/4 | - | 5.63 |
| 2000 | 6 | iso_blue_noise | 4/56 | 26.4690 | 4.1618 | 0.89153 | 2/4 | 52.5 | 2/4 | 107.0 | 0/4 | - | 5.39 |
| 5000 | 1 | aniso_onedge | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 2 | aniso_flanking | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 3 | quadtree_wse | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 4 | quadtree_hybrid | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 5 | iso_blue_noise | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 5000 | 6 | floyd_steinberg | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 1 | aniso_onedge | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 2 | aniso_flanking | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 3 | quadtree_wse | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 4 | quadtree_hybrid | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 5 | iso_blue_noise | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |
| 10000 | 6 | floyd_steinberg | 0/0 | - | - | - | 0/0 | - | 0/0 | - | 0/0 | - | - |

## Paired Deltas Vs Baseline

Positive means `left - baseline` is better. Confidence intervals bootstrap image x seed units.

| Budget | Left | Right | Pairs | PSNR delta | 95% CI | PSNR wins | MS-SSIM delta | 95% CI |
|---:|---|---|---:|---:|---|---:|---:|---|
| 2000 | aniso_flanking | aniso_onedge | 4 | -0.5379 | [-0.9797, -0.0961] | 1/4 | -0.00797 | [-0.01535, -0.00059] |
| 2000 | quadtree_wse | aniso_onedge | 4 | -0.1119 | [-0.2804, 0.0724] | 1/4 | -0.00014 | [-0.00186, 0.00190] |
| 2000 | quadtree_hybrid | aniso_onedge | 4 | -0.4014 | [-0.6167, -0.2374] | 0/4 | -0.00400 | [-0.00872, -0.00066] |
| 2000 | iso_blue_noise | aniso_onedge | 4 | -0.8820 | [-1.4093, -0.4283] | 0/4 | -0.01117 | [-0.02193, -0.00448] |
| 2000 | floyd_steinberg | aniso_onedge | 4 | -0.3968 | [-0.8432, -0.0464] | 1/4 | -0.00510 | [-0.00989, -0.00127] |
| 5000 | aniso_flanking | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | quadtree_wse | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | quadtree_hybrid | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | iso_blue_noise | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 5000 | floyd_steinberg | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | aniso_flanking | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | quadtree_wse | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | quadtree_hybrid | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | iso_blue_noise | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |
| 10000 | floyd_steinberg | aniso_onedge | 0 | - | [-, -] | 0/0 | - | [-, -] |

## Rank Stability

| Budget | Strategy | Complete groups | Wins | Mean rank | Median rank | Rank std |
|---:|---|---:|---:|---:|---:|---:|
| 2000 | aniso_onedge | 4 | 2 | 1.750 | 1.500 | 0.829 |
| 2000 | quadtree_wse | 4 | 1 | 2.500 | 2.500 | 1.118 |
| 2000 | aniso_flanking | 4 | 0 | 3.750 | 3.500 | 1.479 |
| 2000 | floyd_steinberg | 4 | 1 | 3.750 | 4.500 | 1.639 |
| 2000 | quadtree_hybrid | 4 | 0 | 4.000 | 4.000 | 1.581 |
| 2000 | iso_blue_noise | 4 | 0 | 5.250 | 5.500 | 0.829 |
| 5000 | aniso_flanking | 0 | 0 | - | - | - |
| 5000 | aniso_onedge | 0 | 0 | - | - | - |
| 5000 | floyd_steinberg | 0 | 0 | - | - | - |
| 5000 | iso_blue_noise | 0 | 0 | - | - | - |
| 5000 | quadtree_hybrid | 0 | 0 | - | - | - |
| 5000 | quadtree_wse | 0 | 0 | - | - | - |
| 10000 | aniso_flanking | 0 | 0 | - | - | - |
| 10000 | aniso_onedge | 0 | 0 | - | - | - |
| 10000 | floyd_steinberg | 0 | 0 | - | - | - |
| 10000 | iso_blue_noise | 0 | 0 | - | - | - |
| 10000 | quadtree_hybrid | 0 | 0 | - | - | - |
| 10000 | quadtree_wse | 0 | 0 | - | - | - |

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
