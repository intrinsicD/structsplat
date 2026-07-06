# FIT-005: Exact / alternating color solve

**Status: done.** Low-risk convergence experiment from the 2026-07 SOTA review.

## Context
For fixed means, scales, rotations, and opacities, the current weighted-sum renderers are linear in
per-Gaussian RGB colors. The fitter currently optimizes colors with Adam alongside geometry, so it
can spend many iterations learning a linear subproblem that can be solved directly.

## Goal
Add an optional alternating color-solve step that periodically solves the RGB colors exactly, or
approximately with CG/LSQR, while the normal optimizer handles geometry.

## Approach
1. Freeze non-color parameters every `color_solve_every` iterations.
2. Build an implicit linear operator `A` mapping Gaussian colors to rendered RGB pixels for the
   active renderer equation.
3. Solve `min_c ||A c - target||^2 + lambda ||c - c_prev||^2` per channel, using CG/LSQR without
   materializing the dense pixel-by-Gaussian matrix.
4. Write the solved colors back into `GaussianField.colors` and reset/carry optimizer state so the
   next Adam step does not immediately undo the solve.

## Acceptance criteria
- [x] `FitConfig.color_solve_every`, `color_solve_lambda`, and `color_solve_maxiter` are exposed
      through CLI and benchmark config logging.
- [x] Works for the normalized renderer first; additive/CUDA variants either work or fail closed
      with a clear error.
- [x] Unit test on a fixed tiny field recovers known colors to tight tolerance when geometry is
      held fixed.
- [x] Fit smoke test shows no NaNs and no optimizer-state crash after a color-solve event.
- [x] Benchmark slice: same init/seed/budget, color solve every {10, 25, 50} iterations vs Adam-only,
      reporting PSNR/AUC/iters-to-target and wall time.
- [x] Stage-search axis added only if the slice shows a measurable convergence win.

## Notes

- 2026-07-06: Implemented an opt-in normalized-renderer color solve using implicit CG on
  `(A.T A + lambda I)c = A.T target + lambda c_prev`, with no dense pixel-by-Gaussian matrix.
  The fit loop runs it after optimizer steps, resets color optimizer moments, logs
  `history["color_solve_events"]`, and fails closed for non-normalized renderers.
- 2026-07-06 smoke slice recorded in
  `ara/evidence/fit005-color-solve-smoke-2026-07-06/run.md`: on two 64px COCO crops, 64
  Gaussians, 60 iters, `every10` improved mean final PSNR 20.5118 -> 21.0403 and mean PSNR AUC
  19.1648 -> 19.4761, but increased mean fit time 0.6263s -> 1.3917s. `every25`/`every50` were
  near neutral. Added `color_solve_modes=["none", "every10"]` to influence-mode stage search;
  keep the default off.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`benchmarks/stage_search.py`, `tests/test_fit*.py`.

## Depends on
FIT-001, CORE-001.
