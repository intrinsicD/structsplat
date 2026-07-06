# FIT-007: Moment-preserving split / clone

**Status: todo.** Follow-up to FIT-004 function-preserving growth.

## Context
FIT-004 added function-preserving duplication by splitting opacity/weight and offsetting children
along the major axis. That reduces insertion shocks, but it does not explicitly preserve the
parent kernel's first and second spatial moments. A moment-preserving replacement can make the
rendered function and local footprint smoother immediately after a split.

## Goal
Replace one parent Gaussian with 2-4 children whose summed kernel approximately preserves the
parent's mass, mean, and covariance, minimizing the loss spike at split time.

## Approach
1. Derive child offsets and covariance shrinkage in parent principal-axis coordinates.
2. Preserve total opacity/weight and color under the normalized renderer.
3. Support a two-child major-axis split first; optionally add four-child covariance-preserving
   splits for isotropic/corner cases.
4. Carry optimizer state from parent to children in the same style as FIT-004.

## Acceptance criteria
- [ ] `split_mode="moment_preserving"` implemented behind config/CLI.
- [ ] Synthetic test verifies mass, mean, and covariance of the child mixture approximate the
      parent within tolerance.
- [ ] Split-iteration PSNR drop is below the FIT-004 `fp_duplicate` drop on the same smoke case.
- [ ] Works for normalized and additive renderers or explicitly documents the math difference.
- [ ] Benchmark slice vs `fp_duplicate` and `ranked_wave`, reporting post-split dip and
      iters-to-recovery.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/gaussians.py`, `src/structsplat/config.py`,
`tests/test_fit_dynamics.py`.

## Depends on
FIT-004.
