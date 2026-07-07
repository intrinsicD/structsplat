# CORE-009: DC / background layer under the detail Gaussians

**Status: todo (research prototype).** Stop spending scarce Gaussians on flat sky.

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
- [ ] Rung 1 as an init composition flag (`background_fraction`, `background_grid`), NumPy-safe,
      stage-searchable; equal-parameter accounting enforced and logged in each result row.
- [ ] Fair-regime difficult-4 slice at budgets {1000, 2000, 5000}: hypothesis is a low-budget
      win shrinking to neutral at 5000. Report PSNR/MS-SSIM/AUC + the share of budget the
      baseline spends on Gaussians with support > 1/4 image area (measures the freed capacity).
- [ ] Rung 2 only if rung 1 wins: ADR written first, additive-over-background renderer variant
      behind a flag, codec/save-load rejects or versions the layer explicitly.
- [ ] Negative result at rung 1 parks the whole idea in `ara/logic/claims.md`.

## Interfaces touched
`src/structsplat/init.py`, `src/structsplat/config.py`, rung 2: `src/structsplat/render.py` +
new ADR, `benchmarks/stage_search.py`, `tests/`.

## Depends on
CORE-001, ADR-0003/0006 (compositing semantics), BENCH-002. Related: HIER-003 (both attack
low-frequency coverage).
