# CORE-006: Linear color basis per Gaussian

**Status: done (experimental).** Expands the primitive attribute model beyond constant RGB.

## Context
Constant RGB per Gaussian is inefficient for smooth ramps and low-frequency shading. A local linear
or low-degree color model, such as `c0 + cx dx + cy dy`, can represent gradients with fewer
primitives while keeping colors linear for FIT-005-style solves.

## Goal
Add an optional affine color basis per Gaussian and evaluate whether it reduces Gaussian count or
fit iterations at equal quality.

## Approach
1. Extend `GaussianField` with optional color coefficients for constant, x-gradient, and y-gradient
   terms.
2. In the renderer, evaluate color from local Gaussian coordinates or image-space deltas before
   accumulation.
3. Regularize gradient coefficients so they do not become a high-frequency escape hatch.
4. Keep serialization/codec behavior explicit: either reject affine fields in v1 codec paths or
   encode the extra coefficients with a versioned header.

## Acceptance criteria
- [x] `FitConfig.color_basis={"constant","affine"}` and matching CLI flag.
- [x] Reference renderer supports affine color in normalized mode with gradient tests.
- [x] FIT-005 color-solve operator can include affine terms, or the task documents why the first
      implementation optimizes them with Adam only.
- [x] Codec and NPZ save/load round-trip either support affine coefficients or raise a clear
      unsupported-format error.
- [x] Benchmark slice compares constant vs affine at fixed N and at matched PSNR with lower N.
- [x] If useful, add a stage-search axis value and document the expected use cases.

## Notes
Completed 2026-07-06. The first implementation adds optional `GaussianField.color_grads`
coefficients and renders affine colors from scale-normalized local Gaussian coordinates. Affine
coefficients are optimized by Adam with `FitConfig.color_grad_l2`; FIT-005's exact color solve
fails closed for affine fields because the current implicit operator only covers constant RGB.

NPZ save/load round-trips `color_grads`. Codec v1 rejects affine fields explicitly rather than
dropping the extra coefficients.

Smoke evidence: `ara/evidence/core006-affine-color-smoke-2026-07-06/run.md`. On a tiny two-image,
two-budget CPU slice, affine improved mean PSNR from 20.4391 to 21.5029 and mean AUC from 19.1186
to 19.8572, with mean fit time increasing from 0.1056 s to 0.1390 s. Keep `color_basis=constant`
as the default and use `color_basis=affine` as a stage-search axis until larger evidence exists.

## Interfaces touched
`src/structsplat/gaussians.py`, `src/structsplat/render.py`, `src/structsplat/fit.py`,
`src/structsplat/codec.py`, `src/structsplat/config.py`, tests covering render and round-trip.

## Depends on
CORE-001, FIT-001. Pairs naturally with FIT-005.
