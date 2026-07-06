# CORE-006: Linear color basis per Gaussian

**Status: todo (experimental).** Expands the primitive attribute model beyond constant RGB.

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
- [ ] `FitConfig.color_basis={"constant","affine"}` and matching CLI flag.
- [ ] Reference renderer supports affine color in normalized mode with gradient tests.
- [ ] FIT-005 color-solve operator can include affine terms, or the task documents why the first
      implementation optimizes them with Adam only.
- [ ] Codec and NPZ save/load round-trip either support affine coefficients or raise a clear
      unsupported-format error.
- [ ] Benchmark slice compares constant vs affine at fixed N and at matched PSNR with lower N.
- [ ] If useful, add a stage-search axis value and document the expected use cases.

## Interfaces touched
`src/structsplat/gaussians.py`, `src/structsplat/render.py`, `src/structsplat/fit.py`,
`src/structsplat/codec.py`, `src/structsplat/config.py`, tests covering render and round-trip.

## Depends on
CORE-001, FIT-001. Pairs naturally with FIT-005.
