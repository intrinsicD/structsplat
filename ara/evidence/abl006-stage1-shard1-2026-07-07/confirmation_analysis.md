# ABL-006 Successive-Halving Analysis

Expected cells: 336
Completed cells: 12
Missing cells: 324
Baseline for paired deltas: `aniso_onedge`

## Leaderboard

| Budget | Rank | Strategy | Runs | PSNR | PSNR std | MS-SSIM | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean total s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 1 | aniso_onedge | 2/56 | 23.4572 | 0.0670 | 0.87031 | 0/2 | - | 0/2 | - | 0/2 | - | 5.00 |
| 2000 | 2 | quadtree_wse | 2/56 | 23.3023 | 0.1570 | 0.86927 | 0/2 | - | 0/2 | - | 0/2 | - | 5.97 |
| 2000 | 3 | floyd_steinberg | 2/56 | 22.9219 | 0.4854 | 0.86424 | 0/2 | - | 0/2 | - | 0/2 | - | 5.85 |
| 2000 | 4 | quadtree_hybrid | 2/56 | 22.9047 | 0.2325 | 0.86297 | 0/2 | - | 0/2 | - | 0/2 | - | 4.72 |
| 2000 | 5 | aniso_flanking | 2/56 | 22.4775 | 0.0365 | 0.85496 | 0/2 | - | 0/2 | - | 0/2 | - | 5.19 |
| 2000 | 6 | iso_blue_noise | 2/56 | 22.3247 | 0.4867 | 0.85356 | 0/2 | - | 0/2 | - | 0/2 | - | 5.17 |
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
| 2000 | aniso_flanking | aniso_onedge | 2 | -0.9797 | [-1.0832, -0.8762] | 0/2 | -0.01535 | [-0.01896, -0.01174] |
| 2000 | quadtree_wse | aniso_onedge | 2 | -0.1550 | [-0.2449, -0.0650] | 0/2 | -0.00104 | [-0.00219, 0.00011] |
| 2000 | quadtree_hybrid | aniso_onedge | 2 | -0.5526 | [-0.7180, -0.3871] | 0/2 | -0.00734 | [-0.01108, -0.00360] |
| 2000 | iso_blue_noise | aniso_onedge | 2 | -1.1325 | [-1.6862, -0.5788] | 0/2 | -0.01676 | [-0.02710, -0.00641] |
| 2000 | floyd_steinberg | aniso_onedge | 2 | -0.5353 | [-1.0876, 0.0171] | 1/2 | -0.00607 | [-0.01229, 0.00016] |
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
| 2000 | aniso_onedge | 2 | 1 | 1.500 | 1.500 | 0.500 |
| 2000 | quadtree_wse | 2 | 0 | 2.500 | 2.500 | 0.500 |
| 2000 | floyd_steinberg | 2 | 1 | 3.000 | 3.000 | 2.000 |
| 2000 | quadtree_hybrid | 2 | 0 | 4.000 | 4.000 | 1.000 |
| 2000 | aniso_flanking | 2 | 0 | 5.000 | 5.000 | 1.000 |
| 2000 | iso_blue_noise | 2 | 0 | 5.000 | 5.000 | 1.000 |
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
