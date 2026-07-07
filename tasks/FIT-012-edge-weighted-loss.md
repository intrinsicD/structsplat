# FIT-012: Edge-weighted pixel loss (structure-tensor loss weighting)

**Status: todo.** Placement is structure-tensor-aware; the loss is uniform. Align the
optimization pressure with the placement prior.

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
- [ ] Weighted loss implemented with finite gradients; `loss_weighting=none` is bit-identical to
      today (regression test).
- [ ] Stage-search axis + metadata; canonicalization marks the axis inert for `density=uniform` +
      `strategy=random/grid` cells only if the weight map is genuinely constant there (it is not
      in general — do not over-dedupe).
- [ ] Fair-regime difficult-4 slice: report PSNR, MS-SSIM, edge MAE, AUC vs unweighted baseline
      at budgets {2000, 5000}; hypothesis is edge-MAE down and AUC up at low budget, PSNR neutral
      or slightly down (weighted training optimizes a different objective than the PSNR metric —
      judge on the metric bundle, not PSNR alone).
- [ ] Evidence committed; promoted or parked in `ara/logic/claims.md`.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/init.py` (energy
plumb-through), `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`.

## Depends on
FIT-001, INIT-001, BENCH-001.
