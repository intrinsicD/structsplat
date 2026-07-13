# CORE-007: Segmentation-free responsibility boundary flux

**Status: needs re-scope before implementation.** The original broad gate formulation is directly
threatened by Contour-Aware 2DGS and must not be implemented as an unqualified novelty claim.

## Context
Normalized Gaussian weights can blend colors across strong boundaries. Contour-Aware 2DGS already
uses segmentation-region-constrained rasterization to prevent cross-boundary mixing, especially at
small Gaussian counts. A per-Gaussian half-plane or segmentation gate is therefore a direct
baseline, not the remaining research contribution.

## Goal
Test whether **segmentation-free responsibility flux** derived from the image structure tensor can
reduce cross-boundary mixing at actual low rates without transmitting external masks or duplicating
per-splat boundary metadata.

## Approach
1. First implement diagnostics only: normalized responsibility mass crossing tensor-normal ridges,
   signed cross-edge bleed, and edge/texture-band distortion.
2. Use Contour-Aware 2DGS or the closest official implementation as the native direct control.
   If code is unavailable, keep its result in the literature table and label any reproduction
   `local_contour_gate_control`.
3. Compare a no-new-parameter flux penalty with a tensor-derived soft half-plane control. Count
   every gate parameter or boundary representation in actual stream bytes.
4. Run only after BENCH-007 identifies an edge-band failure at 0.5/1.0 bpp. Keep the feature opt-in.

## Acceptance criteria
- [ ] BENCH-007 establishes a reproducible cross-boundary failure at actual low rate.
- [ ] Responsibility-flux diagnostic is validated on synthetic step, junction, thin-line, and
      texture controls without changing the renderer.
- [ ] The no-parameter penalty is isolated from the parameterized gate control.
- [ ] Synthetic and eight-image tests compare no gate, flux penalty, local contour gate, and native
      Contour-Aware evidence where executable.
- [ ] Every side parameter/mask is counted in a self-contained stream and cold decoded.
- [ ] Promotion requires at least +0.20 dB edge-band benefit at 0.5 and 1.0 bpp, no worse than
      -0.05 dB whole-image PSNR, no texture-band regression, and an image-bootstrap interval above
      zero. Otherwise close the exact formulation.

## Interfaces touched
Begin in benchmark diagnostics. Only after the gate passes may it touch
`src/structsplat/gaussians.py`, `src/structsplat/render.py`, `src/structsplat/config.py`,
codec versioning, and render/codec tests.

## Depends on
INIT-004, CORE-001, BENCH-007.
