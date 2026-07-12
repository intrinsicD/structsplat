# Theory notes

## Why initialize on structure, with measured placement defaults
- A Gaussian's position gradient comes only from render error inside its support. One dropped in a
  flat region far from structure sees ~zero gradient and never migrates — stranded capacity.
- The original flanking hypothesis was that edge Gaussians should be pushed off the ridge to avoid
  averaging both sides of a discontinuity. ABL-006 did not support that as a default.
- Resolution: put density where structure is; orient by the structure tensor; use `quadtree_wse` as
  the high-budget PSNR default, keep `aniso_onedge` for low-budget/MS-SSIM-sensitive runs, and keep
  flanking as a control arm.

## Structure tensor
`J = G_rho * (grad I grad I^T)`, eigenvalues `lam1 >= lam2`:
- flat: `lam1 ≈ lam2 ≈ 0`; edge: `lam1 ≫ lam2`; corner: `lam1 ≈ lam2 ≫ 0`.
- `energy = lam1 + lam2` → density; `coherence = ((lam1-lam2)/(lam1+lam2))^2` → axis ratio;
  major eigenvector = across-edge (gradient) direction; tangent = +90°.

## Anisotropic blue noise
Poisson-disk conflict measured in the Mahalanobis metric `M(x)` from the tensor. Unit-area `M`
keeps counts comparable to isotropic. Blue-noise in the warped space ⇒ dense-across / sparse-along
in image space. Realized with Weighted Sample Elimination for exact N (Yuksel 2015).

## Hierarchy
WSE guarantees the selected terminal set, not candidate-index prefixes. INIT-009 adds Yuksel's
recursive halving/reverse-shell permutation behind `wse_progressive_order=True`; with that opt-in,
each within-WSE prefix is blue-noise-like at its density while the complete Gaussian tuples remain
unchanged. The default stays off because row order is part of saved-artifact and GPU-reduction
provenance. This within-level sequence is distinct from pyramid append order: finer pyramid levels
are driven by the residual structure tensor. Under a normalized renderer this is densification
(add + re-fit), not additive residual summation (ADR-0003). The Morton-sorted codec does not yet
preserve either order as a progressive bitstream.
For `quadtree_wse`, the permutation is global over the union of locally selected leaf sets; it does
not turn them into one globally selected WSE set. With a frozen background layer, background rows
remain first and the progressive guarantee applies to the following detail suffix, not arbitrary
prefixes of the complete field.

## Empirical answer
The optimizer discovers anisotropy itself, so the useful question became which structured placement
policy is worth shipping. ABL-006 completed the Kodak-24 + COCO4 successive-halving confirmation:
`quadtree_wse` is the significant 5000-Gaussian PSNR winner, has the best 10000-Gaussian mean PSNR,
and `aniso_onedge` remains competitive at 2000 and stronger on 10000-Gaussian MS-SSIM. Flanking did
not earn a default; see `ara/evidence/abl006-complete-2026-07-07/` and ADR-0013.

## References
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · AIR / Fast-2DGS (2025) · GaussianVision
(structured init) · Li & Wei, Anisotropic Blue Noise Sampling (SIGGRAPH Asia 2010) · Yuksel,
Sample Elimination (EGSR 2015) · Gaussian Blue Noise (2022).
