# Theory notes

## Why initialize ON corners/blobs and FLANKING edges (never away)
- A Gaussian's position gradient comes only from render error inside its support. One dropped in a
  flat region far from structure sees ~zero gradient and never migrates — stranded capacity.
- A Gaussian centered exactly on a step edge must average both sides → a baked-in blurred seam and
  optimizer thrash (an RBF centered on a discontinuity has an ill-defined target).
- Resolution: put density where structure is; orient by the structure tensor; for edges place two
  flanking chains parallel to the edge (elongated along the tangent, narrow across), corners get one
  compact Gaussian, flats get sparse large ones.

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
Progressive maximal blue-noise ordering: any prefix is a valid blue-noise set at its density = a
free LOD stack. Finer levels driven by the residual structure tensor. Under a normalized renderer
this is densification (add + re-fit), not additive residual summation (ADR-0003).

## Open empirical question (what ABL-001 tests)
The optimizer discovers anisotropy itself, so flanking/tensor init mainly buys convergence speed and
low-budget quality. Hypothesis: `aniso_flanking` wins clearly at low budgets; the gap shrinks as the
budget grows. If it never wins, prefer the simpler strategy and record that.

## References
GaussianImage (ECCV 2024) · Image-GS (SIGGRAPH 2025) · AIR / Fast-2DGS (2025) · GaussianVision
(structured init) · Li & Wei, Anisotropic Blue Noise Sampling (SIGGRAPH Asia 2010) · Yuksel,
Sample Elimination (EGSR 2015) · Gaussian Blue Noise (2022).
