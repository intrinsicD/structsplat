# HIER-004 pyramid convergence repair

Date: 2026-07-07

Goal: keep the HIER-003 pyramid final-quality gain while reducing the AUC/convergence loss.

## Protocol

Controls reused from HIER-003:

- `single_1500`: single-stage, 1500 iterations.
- `pyramid_split_1500`: two-level pyramid, 0.35/0.65 Gaussian budget, 750/750 iterations.

New candidates used the same difficult-four exact-CUDA slice:

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budgets: 2000 and 5000.
- Strategies: `aniso_onedge`, `quadtree_wse`.
- Seed 0, max-side 768, exact CUDA renderer, total 1500 iterations, `loss=l1`,
  `color=bilinear`, `scale=spacing`, `scale_cap=none`, no color solve, no loss weighting.
- Gaussian budget fractions stayed 0.35/0.65; only level iteration counts changed.

Candidates:

| arm | level iters |
|---|---:|
| `pyramid_leveliters_150_1350` | 150 / 1350 |
| `pyramid_leveliters_200_1300` | 200 / 1300 |
| `pyramid_leveliters_300_1200` | 300 / 1200 |
| `pyramid_leveliters_375_1125` | 375 / 1125 |
| `pyramid_leveliters_500_1000` | 500 / 1000 |

## Result

The candidate run completed 80/80 cells with `status=ok`; including reused controls, the combined
analysis covers 112 ok rows.

Promotion rule from the task: final PSNR within 0.1 dB of `pyramid_split_1500` and AUC loss no
worse than -0.25 dB versus `single_1500`.

| arm | dPSNR vs split | dAUC vs single | dEdge-MAE vs single | promotion |
|---|---:|---:|---:|---|
| `pyramid_leveliters_150_1350` | +0.0601 | +0.0011 | -0.000893 | pass |
| `pyramid_leveliters_200_1300` | -0.0429 | -0.1319 | -0.000967 | pass |
| `pyramid_leveliters_300_1200` | -0.0562 | -0.3567 | -0.000913 | fail AUC |
| `pyramid_leveliters_375_1125` | -0.0531 | -0.5126 | -0.000978 | fail AUC |
| `pyramid_leveliters_500_1000` | -0.1543 | -0.8244 | -0.000533 | fail PSNR/AUC |

Absolute means:

| arm | PSNR | MS-SSIM | edge MAE | AUC |
|---|---:|---:|---:|---:|
| `pyramid_leveliters_150_1350` | 27.0033 | 0.92351 | 0.049380 | 26.2095 |
| `pyramid_split_1500` | 26.9433 | 0.92331 | 0.049711 | 24.8544 |
| `pyramid_leveliters_200_1300` | 26.9004 | 0.92310 | 0.049306 | 26.0765 |
| `single_1500` | 25.9432 | 0.90932 | 0.050273 | 26.2084 |

## Verdict

`pyramid_level_iters=[150, 1350]` repairs the AUC problem while preserving the final-quality gain
on this slice. It beats single-stage by +1.0601 dB final PSNR, matches single-stage AUC
(+0.0011), improves edge MAE, and slightly beats the HIER-003 750/750 pyramid final PSNR.

Decision: expose explicit per-level pyramid iteration schedules and use 150/1350 as the pyramid
quality candidate. Keep `pyramid=single` as the shipped/default schedule until a larger
multi-seed confirmation promotes the pyramid candidate.

## Artifacts

- `combined_rows_with_controls.csv`: controls plus all HIER-004 candidate rows.
- `paired_deltas_vs_controls.csv`: candidate deltas vs `single_1500` and `pyramid_split_1500`.
- `candidate_summary.csv`: aggregate promotion checks.
- `arm_means.csv`: absolute means by arm.
- Per-arm subdirectories include `config.json`, `stage_search.csv`, `summary.md`, and `index.html`.

## Verification

- `python -m pytest tests/test_pyramid.py tests/test_stage_search.py tests/test_cli.py -q`
  passed 38 tests.
- `python -m pytest -q` passed 293 tests.
