# FIT-010: Cheap color-solve schedules (init / final / on-split)

**Status: todo.** `every10` buys +0.5 dB at 2.2x fit time; most of that value is probably in a
handful of solves placed at the right moments.

## Context
FIT-005's exact color solve (matrix-free CG on the normal equations, fixed geometry) measured
+0.5285 dB mean final PSNR and +0.31 AUC at `color_solve_every=10`, but with +122% fit time;
`every=25/50` captured almost nothing (`fit005-color-solve-smoke-2026-07-06`). The scheduling
hypothesis: solves pay off at moments when colors are far from the least-squares optimum for the
current geometry — right after init (before Adam ever runs) and right after a split/growth wave
(where the measured -1 to -2 dB dip is partly a color problem). A fixed period mostly buys solves
at moments where Adam had already converged the colors.

## Goal
Event-based color-solve schedules that capture most of `every10`'s quality delta at a small
fraction of its fit-time cost.

## Design
`FitConfig.color_solve_schedule in {none, every_k, init, final, on_split}` (composable set, e.g.
`init+on_split`). `on_split` runs one CG solve immediately after each split/relocate/growth event,
then resets the colors' Adam state (mechanism already exists:
`_reset_optimizer_state_for_param`). `init` runs at iteration 0 — this also upgrades every init
strategy into "placement + least-squares colors" for free, which matters for iters-to-target.

## Acceptance criteria
- [ ] Schedules implemented and threaded through stage-search as axis values; events recorded in
      `history["color_solve_events"]` as today.
- [ ] CPU smoke on the FIT-005 protocol comparing `none / every10 / init / on_split /
      init+on_split`: report ΔPSNR, ΔAUC, Δfit-s per schedule.
- [ ] Target: an event-based schedule reaches ≥70% of `every10`'s ΔPSNR at ≤30% of its extra fit
      time on that protocol; if none does, record the negative result and keep `every_k` only.
- [ ] Split-dip interaction measured: post-split delta and recovery iters with and without
      `on_split` (same fixtures as FIT-007's smoke, so numbers are comparable).
- [ ] Constant-color + normalized-renderer restriction of FIT-005 unchanged and enforced by test.

## Interfaces touched
`src/structsplat/config.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`,
`tests/test_fit_dynamics.py`.

## Depends on
FIT-005. Pairs with FIT-009 (split events) and ABL-005 (the `color_solve` influence arm should
re-run with the winning schedule).
