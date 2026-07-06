# CORE-007: Boundary-gated Gaussians

**Status: todo (research prototype).** Edge-contamination control for normalized splats.

## Context
Flanking placement helps avoid centering Gaussians across discontinuities, but normalized Gaussian
weights can still blend colors across strong boundaries. Contour-aware 2DGS work points in the same
direction: use contour or segmentation priors to prevent low-budget splats from crossing object or
edge boundaries.

## Goal
Add an optional per-Gaussian soft half-plane or segmentation gate so a Gaussian can cover one side
of a boundary without leaking weight across it.

## Approach
1. Start with a soft oriented half-plane gate tied to the local structure-tensor edge normal.
2. Multiply the Gaussian weight by `sigmoid(k * signed_distance)` before normalized accumulation.
3. Optionally support two-sided colors for a single boundary primitive after the one-sided gate is
   stable.
4. Keep the feature opt-in and benchmarked against flanking-only placement.

## Acceptance criteria
- [ ] Field representation stores optional gate normal, offset, sharpness, and enabled mask.
- [ ] Reference renderer supports gated normalized accumulation with finite gradients.
- [ ] Initializer can assign gates from tensor/edge evidence for edge-classified Gaussians.
- [ ] Synthetic step-edge test shows lower cross-boundary color bleed at equal N.
- [ ] Benchmark slice compares on-edge, flanking, and boundary-gated variants under low budgets.
- [ ] Codec/save-load behavior is versioned or explicitly rejects gated fields.

## Interfaces touched
`src/structsplat/gaussians.py`, `src/structsplat/render.py`, `src/structsplat/init.py`,
`src/structsplat/config.py`, `tests/test_render.py`, benchmark configs.

## Depends on
INIT-004, CORE-001.
