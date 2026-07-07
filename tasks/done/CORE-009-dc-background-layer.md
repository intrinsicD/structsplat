# CORE-009: DC / background layer under the detail Gaussians

**Status: done.** Rung 1 is implemented as an opt-in, counted initialization layer and evaluated
on the difficult-four fair-regime slice. Keep it searchable and default off; do not proceed to the
true compositing rung without a new ADR and larger confirmation.

## Context
At low budgets the normalized renderer must cover every pixel, so a meaningful share of the
budget becomes huge low-frequency splats whose only job is holding the background together —
capacity that detail regions need. The support-fade and feature-cap results both trace back to
this tension (caps forbid the big splats flat regions need; fade helps early coverage). A cheap
dedicated DC layer would free the entire Gaussian budget for structure.

## Goal
An opt-in background layer beneath the Gaussian field, evaluated at equal *total* parameter
count (the layer's parameters count against the budget — BENCH-002 spirit: no hidden capacity).

## Design (two rungs, in order)
1. **No-renderer-change rung**: reserve a small fraction of the budget as a jittered-grid of
   large, near-isotropic Gaussians initialized from a heavily downsampled image, with positions
   and scales frozen (colors learnable). Pure init-strategy composition — no ADR needed, and it
   isolates "does a guaranteed base help?".
2. **Compositing rung**: a true fixed low-resolution bilinear RGB grid `B` with
   `out = B + splat_correction`, where the splat term is the *additive* mode over a
   residual-like target. This changes compositing semantics (invariant 3 / ADR-0003), so it
   requires a new ADR before implementation and must cite rung-1 evidence to justify itself.

Budget accounting for both rungs: a 16x16x3 grid = 768 floats ≈ 96 Gaussians' worth of
parameters (8 floats each); the comparison deducts that from the Gaussian budget.

## Acceptance criteria
- [x] Rung 1 as an init composition flag (`background_fraction`, `background_grid`), NumPy-safe,
      stage-searchable; equal-parameter accounting enforced and logged in each result row.
- [x] Fair-regime difficult-4 slice at budgets {1000, 2000, 5000}: hypothesis is a low-budget
      win shrinking to neutral at 5000. Report PSNR/MS-SSIM/AUC + the share of budget the
      baseline spends on Gaussians with support > 1/4 image area (measures the freed capacity).
- [x] Rung 2 decision recorded: no additive-over-background renderer in CORE-009; any true
      compositing/background layer still requires a new ADR first.
- [x] Outcome recorded in `ara/logic/claims.md`.

## Outcome

Evidence: `ara/evidence/core009-background-layer-2026-07-07/`.

The benchmark completed 72/72 cells: 4 images x 3 budgets x 2 strategies x 3 background modes,
seed 0, max-side 768, 1500 iterations, exact CUDA renderer, no refine, no color solve, and equal
total Gaussian counts. Background rows count against `num_gaussians`; their geometry is frozen and
their colors are learnable.

Paired deltas versus matching `background=off` rows:

| background | pairs | dPSNR | PSNR wins | dMS-SSIM | MS-SSIM wins | dAUC | AUC wins | dFit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `frac0.05_grid8` | 24 | +1.0152 | 22/24 | +0.01412 | 24/24 | +0.1192 | 11/24 | +2.76 |
| `frac0.10_grid16` | 24 | +0.9564 | 17/24 | +0.01223 | 22/24 | +0.0828 | 11/24 | +2.57 |

The gain is strongly budget-dependent. At 1000 rows, both background modes win PSNR in every pair
and improve AUC in 7/8 pairs. At 2000 rows, `frac0.05_grid8` still wins PSNR in every pair, but
AUC is split. At 5000 rows, the 5%/8x8 mode is only +0.0504 dB PSNR on average and loses AUC in
all pairs; the 10%/16x16 mode is slightly negative on PSNR and also loses all AUC pairs.

Baseline large-support rows were already a small share of the current recipes: 2.55% at 1000,
1.14% at 2000, and 0.54% at 5000. The background modes increase the detail-row broad-support
diagnostic after fitting, so the observed win is not simply "freeing broad splats"; it behaves
more like a low-frequency color prior.

Decision: keep `background_fraction`/`background_grid` as stage-searchable controls and default
them off. `frac0.05_grid8` is the local quality candidate for low-budget screens. Do not implement
rung 2 in this task; a true additive background layer changes compositing semantics and needs a
separate ADR plus confirmation that the 5000-row convergence cost is worth accepting.

## Interfaces touched
`src/structsplat/init.py`, `src/structsplat/config.py`, `src/structsplat/gaussians.py`,
`src/structsplat/fit.py`, `src/structsplat/cli.py`, `benchmarks/stage_search.py`, `tests/`.

## Depends on
CORE-001, ADR-0003/0006 (compositing semantics), BENCH-002. Related: HIER-003 (both attack
low-frequency coverage).
