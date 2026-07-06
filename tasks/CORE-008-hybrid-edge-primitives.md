# CORE-008: Hybrid Gaussian + edge primitives

**Status: todo (stretch).** Non-pure-2DGS experiment for sharp contours and textures.

## Context
Some high-frequency edges and textures are inefficient for blob-only Gaussian bases. A Gaussian
envelope multiplied by a signed edge, DoG, or Gabor-like component could represent contours and
oriented texture with far fewer primitives, at the cost of leaving the pure 2DGS model class.

## Goal
Prototype a `hybrid_gaussian_edge` primitive family and benchmark whether it improves quality per
primitive on sharp synthetic and natural-image edge cases.

## Approach
1. Add a separate primitive type instead of overloading normal Gaussians.
2. Implement a reference renderer path for Gaussian envelope times signed edge/wave component.
3. Initialize edge primitives from tensor-classified edge pixels or residual edge evidence.
4. Compare hybrid fields against pure Gaussian fields at equal parameter count and equal byte
   budget.

## Acceptance criteria
- [ ] Primitive type is represented explicitly in field data and renderer dispatch.
- [ ] Reference forward/backward path passes finite-gradient tests.
- [ ] Initialization produces plausible edge primitive orientation, scale, phase/sign, and color.
- [ ] Benchmark includes synthetic hard edges, thin lines, and at least one natural-image crop.
- [ ] Results report both quality and complexity: PSNR/MS-SSIM, primitive count, parameters, and
      encoded bytes if codec support exists.
- [ ] README/benchmark notes clearly label this as a hybrid baseline, not core 2DGS.

## Interfaces touched
`src/structsplat/gaussians.py`, `src/structsplat/render.py`, `src/structsplat/init.py`,
`src/structsplat/fit.py`, `benchmarks/stage_search.py`, tests for render/init.

## Depends on
CORE-001, INIT-001, FIT-001. Optional: CORE-007.
