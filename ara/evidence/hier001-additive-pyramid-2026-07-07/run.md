# HIER-001 additive pyramid comparison

Date: 2026-07-07

Goal: close the optional HIER-001 question: whether running the progressive pyramid under additive
composition, closer to true residual summation, changes the pyramid decision versus the normalized
renderer path used by HIER-003/HIER-004.

## Protocol

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budget: 512 Gaussians.
- Init: `strategy=quadtree_wse`, seed 0.
- Fit: 400 total iterations, max-side 384, `loss=l1`, no refinement, no color solve, no opacity.
- Renderers: `cuda` and `cuda_additive`.
- Pyramid modes: `single` and two-level `pyramid`.
- Pyramid schedule: fractions `0.35/0.65`, level iterations `40/360`, matching the HIER-004
  delayed-full-field schedule at this smaller iteration count.

Command used the stage-search harness with every unrelated axis explicitly pinned, crossing only
`renderer in {cuda, cuda_additive}` and `pyramid in {single, pyramid}`.

## Result

Absolute means over four images:

| renderer | pyramid | PSNR | MS-SSIM | AUC | edge MAE |
|---|---|---:|---:|---:|---:|
| `cuda` | `single` | 23.7735 | 0.89348 | 22.9270 | 0.065525 |
| `cuda` | `pyramid` | 23.7756 | 0.89513 | 22.4453 | 0.068105 |
| `cuda_additive` | `single` | 23.5424 | 0.88272 | 19.6698 | 0.070067 |
| `cuda_additive` | `pyramid` | 23.1681 | 0.87405 | 17.8166 | 0.075009 |

Paired deltas:

| comparison | pairs | dPSNR | PSNR wins | dMS-SSIM | dAUC | dEdge MAE |
|---|---:|---:|---:|---:|---:|---:|
| `cuda pyramid` vs `cuda single` | 4 | +0.0021 | 1/4 | +0.00165 | -0.4817 | +0.002579 |
| `cuda_additive pyramid` vs `cuda_additive single` | 4 | -0.3743 | 0/4 | -0.00866 | -1.8532 | +0.004942 |
| `cuda_additive single` vs `cuda single` | 4 | -0.2311 | 1/4 | -0.01076 | -3.2572 | +0.004542 |
| `cuda_additive pyramid` vs `cuda pyramid` | 4 | -0.6075 | 0/4 | -0.02108 | -4.6287 | +0.006904 |

## Verdict

The existing pyramid implementation already accepts additive renderers; a focused regression test
covers `fit_pyramid(..., renderer="additive")`.

Additive composition does not improve the HIER-001 decision on this matched slice. It is worse than
the normalized renderer overall, and additive+pyramid is worse than additive+single in every PSNR
pair. Keep HIER-004's normalized `level_iters=[150, 1350]` result as the pyramid quality candidate.
Do not pursue additive residual summation as part of HIER-001 unless a new task changes the renderer
semantics or training objective more broadly.

## Artifacts

- `stage_search.csv`, `stage_search.json`, `stage_search.jsonl`, `summary.md`, `index.html`: raw
  16-row comparison.
- `arm_means.csv`: absolute grouped means.
- `paired_deltas.csv`: row-level paired deltas.
- `paired_summary.csv`: grouped paired deltas used above.
