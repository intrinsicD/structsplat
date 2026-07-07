# FIT-012: Edge-weighted pixel loss (structure-tensor loss weighting)

**Status: done.** Implemented as an opt-in stage-search axis and evaluated on the fair-regime
difficult-four slice. The tensor-weighted pixel loss is searchable, but it is not promoted as a
default because the result is strategy-dependent and loses AUC on average.

## Context
StructSplat spends its Gaussian budget where the structure tensor says detail lives, but the
fitter's L1+SSIM objective weights every pixel equally, so the optimizer can trade edge fidelity
for flat-region polish — exactly the error the diff maps in the fair-density run show
concentrating on edges. The cross-repo matrix already tracks `edge MAE` as a metric; nothing
optimizes it.

## Goal
An opt-in per-pixel loss weight derived from the init-time structure-tensor energy, as a new
stage-search axis, evaluated for both quality (does edge MAE drop at equal PSNR?) and convergence.

## Design
- `FitConfig.loss_weighting in {none, tensor}` — a *separate* axis so it composes with
  `loss in {l1, l2, charbonnier}`.
- Weight map `w = 1 + beta * E_norm` where `E_norm` is the init-time NumPy tensor energy
  normalized to [0,1], resampled to the fit resolution, computed once and passed as a constant
  tensor (invariant 1: the NumPy/torch split is untouched — energy comes from the existing
  init-path output).
- Applies to the pixel-loss term only; the SSIM term stays unweighted in v1. `beta` default 1.0,
  configurable.
- The *metric protocol is unchanged*: PSNR/MS-SSIM/LPIPS stay unweighted (BENCH-001); only the
  training objective changes. State this in the run configs to avoid one-metric-convention drift.

## Acceptance criteria
- [x] Weighted loss implemented with finite gradients; `loss_weighting=none` is bit-identical to
      today (regression test).
- [x] Stage-search axis + metadata; canonicalization marks the axis inert for `density=uniform` +
      `strategy=random/grid` cells only if the weight map is genuinely constant there (it is not
      in general — do not over-dedupe).
- [x] Fair-regime difficult-4 slice: report PSNR, MS-SSIM, edge MAE, AUC vs unweighted baseline
      at budgets {2000, 5000}; hypothesis is edge-MAE down and AUC up at low budget, PSNR neutral
      or slightly down (weighted training optimizes a different objective than the PSNR metric —
      judge on the metric bundle, not PSNR alone).
- [x] Evidence committed; promoted or parked in `ara/logic/claims.md`.

## Outcome

Evidence: `ara/evidence/fit012-edge-weighted-loss-2026-07-07/`.

Fair-regime difficult-four exact-CUDA slice:

- Images: `kodim01`, `kodim07`, `kodim13`, `kodim19`.
- Budgets: 2000 and 5000.
- Strategies: `aniso_onedge`, `quadtree_wse`.
- Matched settings: seed 0, 1500 iterations, max-side 768, `renderer=cuda`, `loss=l1`,
  `refine_site=residual_tensor`, `refine_primitive=sampled_add`, split every 300 with 250
  insertions, same init/fitter settings. Only `loss_weight` changes between `none` and `tensor`.

Tensor weighting vs matching unweighted rows over 16 paired cells:

| metric | mean delta | wins |
|---|---:|---:|
| PSNR | +0.0061 dB | 10/16 |
| MS-SSIM | -0.00068 | 8/16 |
| edge MAE | +0.000018 | 10/16 lower-is-better wins |
| PSNR AUC | -0.0107 | 6/16 |

By strategy, the effect is split:

- `aniso_onedge`: +0.2661 dB mean PSNR, +0.00129 MS-SSIM, edge MAE improves in 7/8 pairs, and
  AUC improves in 5/8 pairs.
- `quadtree_wse`: -0.2538 dB mean PSNR, -0.00266 MS-SSIM, edge MAE improves in only 3/8 pairs,
  and AUC improves in only 1/8 pairs.

Decision: keep `loss_weight=tensor` as a stage-search axis and default it off. It may be useful
with `aniso_onedge`, but the aggregate does not support a global default or a general claim that
structure-tensor loss weighting is better than the uniform pixel loss.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`src/structsplat/pyramid.py`, `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`,
`tests/test_stage_search.py`.

## Depends on
FIT-001, INIT-001, BENCH-001.
