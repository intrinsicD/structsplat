# FIT-011: Split-recovery micro-levers (Adam moment seeding, LR warmup, scheduled support fade)

**Status: todo.** Convergence-rate work targeting the measured post-split dip and the
early-vs-late support-fade tradeoff.

## Context
Every growth mode dips -1 to -2 dB at a split with ~6-iteration recovery (FIT-006/007 smokes).
`_carry_adam_state` currently zero-pads the moments of newly inserted rows: with the global step
count carried, fresh second moments give new rows abnormally large first updates — a plausible
mechanical cause of the dip that `moment_preserving` only partially masks. Separately, support
fade won 38/48 AUC pairs and +0.42 dB at budget 2000 but lost -0.37 to -0.46 dB at 5k/10k final
PSNR (`fair-density-control-supportfade-difficult4-2026-07-05`) — an early-helps / late-hurts
signature begging for a schedule rather than a toggle.

## Goal
Three small, independently testable fitter changes that flatten the split dip and capture support
fade's early-AUC win without its late-PSNR cost.

## Design
1. **Adam moment seeding**: new rows inherit `exp_avg`/`exp_avg_sq` from their parent row (split
   primitives know the parent; residual-sampled adds seed from the field median) instead of
   zeros. One-line change inside `_carry_adam_state`'s padding path, behind a config flag.
2. **Post-insert update tempering**: scale the effective step of rows younger than W iterations
   by a ramp w0→1. Adam's moment normalization makes naive gradient scaling wrong; temper the
   *update* (e.g. lerp new-row params toward their pre-step values by the ramp factor after
   `optimizer.step()`), which is exact and optimizer-agnostic.
3. **Scheduled support fade**: `support_fade_until_frac` — fade active for the first fraction of
   iterations, off afterward. Measure the PSNR discontinuity at the toggle (renderer output
   changes); if the step is visible in history, cross-fade over ~10 iterations.

## Acceptance criteria
- [ ] Each lever is a separate config flag, default off, individually stage-searchable.
- [ ] FIT-007-protocol smoke reports post-split delta and recovery iters for: baseline,
      moment-seeding, tempering, both — composed with `duplicate` and `moment_preserving`
      primitives (the levers must help the cheap primitive, not just the good one).
- [ ] Scheduled-fade smoke on the support-fade protocol shows AUC ≥ fade-on and final PSNR
      ≥ fade-off (within noise) at 5k/10k; otherwise record the negative and park it.
- [ ] No behavior change with all flags off (regression test on a fixed fixture).

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/render.py` (fade
schedule plumb-through), `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`.

## Depends on
FIT-004, FIT-007, CORE-005 (support fade). Composes with FIT-009's primitive axis.
