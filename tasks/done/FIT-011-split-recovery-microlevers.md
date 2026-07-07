# FIT-011: Split-recovery micro-levers (Adam moment seeding, LR warmup, scheduled support fade)

**Status: done.** Completed 2026-07-07. Convergence-rate work targeting the measured post-split
dip and the early-vs-late support-fade tradeoff.

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
- [x] Each lever is a separate config flag, default off, individually stage-searchable.
- [x] FIT-007-protocol smoke reports post-split delta and recovery iters for: baseline,
      moment-seeding, tempering, both — composed with `duplicate` and `moment_preserving`
      primitives (the levers must help the cheap primitive, not just the good one).
- [x] Scheduled-fade smoke on the support-fade protocol shows AUC ≥ fade-on and final PSNR
      ≥ fade-off (within noise) at 5k/10k; otherwise record the negative and park it.
- [x] No behavior change with all flags off (regression test on a fixed fixture).

## Outcome

- Added `FitConfig.seed_new_row_optimizer_state`, default off. Duplicate-style split children can
  inherit carried optimizer moment rows from their parent; sampled-add children seed from the
  carried-row median. Default zero-padding remains unchanged.
- Added `FitConfig.new_row_temper_iters` / `new_row_temper_start`, default off. Young new or
  relocated rows get their post-`optimizer.step()` parameter update interpolated by a per-row
  ramp, which avoids optimizer-specific gradient hacks.
- Added scheduled fade controls: `support_fade_until_frac` and
  `support_fade_crossfade_iters`. Existing `support_fade=True` remains static fade-on; the
  schedule is opt-in and can ramp the fade amount down over several iterations.
- Stage-search axes are available as `--state-seed-modes`, `--row-temper-modes`, and
  `--support-fade-modes`. Rows report support-fade alpha, temper counts, post-split delta, and
  split-recovery iterations.
- No lever is promoted as a default. The split-recovery smoke was negative for state seeding and
  warmup. Scheduled fade preserved or beat fade-off final PSNR in the 5k/10k budget smoke, but did
  not reach fade-on AUC. Keep all three as searchable controls for larger runs.

## Evidence

- Focused/expanded tests: `python -m pytest tests/test_fit_dynamics.py tests/test_stage_search.py tests/test_render.py -q`
  passed 113/113 with one CUDA-extension warning.
- Split-recovery smoke and scheduled support-fade smoke:
  `ara/evidence/fit011-split-recovery-microlevers-2026-07-07/run.md`.

Split-recovery aggregate:

| primitive | state seed | row temper | mean PSNR | mean AUC | post-split delta | recovery iters |
|---|---|---|---:|---:|---:|---:|
| duplicate | off | off | 23.6248 | 21.8625 | +0.1747 | 1.00 |
| duplicate | on | off | 23.4191 | 21.8203 | +0.1747 | 1.00 |
| duplicate | off | warmup5 | 23.5868 | 21.8475 | +0.1747 | 1.00 |
| duplicate | on | warmup5 | 23.3882 | 21.8128 | +0.1746 | 1.00 |
| moment_preserving | off | off | 23.3584 | 21.7844 | -0.0014 | 1.50 |
| moment_preserving | on | off | 23.2439 | 21.7589 | -0.0015 | 1.50 |

Support-fade schedule budget-smoke aggregate:

| budget | support fade | mean PSNR | mean AUC |
|---:|---|---:|---:|
| 5000 | off | 47.8790 | 39.4450 |
| 5000 | on | 47.9707 | 39.7002 |
| 5000 | until0.5 | 47.9724 | 39.6297 |
| 10000 | off | 49.1020 | 40.4712 |
| 10000 | on | 49.3857 | 40.8143 |
| 10000 | until0.5 | 49.3396 | 40.7349 |

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/render.py` (fade
schedule plumb-through), `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`.

## Depends on
FIT-004, FIT-007, CORE-005 (support fade). Composes with FIT-009's primitive axis.
