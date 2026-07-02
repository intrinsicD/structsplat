# ADR-0005: Anisotropy via a Mahalanobis metric inside Weighted Sample Elimination

## Context
Isotropic blue noise wants equal spacing in all directions, which fights edges (we want dense
across an edge, sparse along it). We need blue-noise spectra *and* feature-adapted anisotropy, with
exact control over the Gaussian count for fair budget comparisons.

## Decision
Use Weighted Sample Elimination (Yuksel 2015) for exact-N blue noise. Encode density via per-point
target radius, and anisotropy via a per-point **unit-area metric tensor** `M(x)` from the structure
tensor: the WSE conflict distance is Mahalanobis in `M`. Samples are blue-noise in the warped metric
→ in image space they pack across edges and spread along them, no clumping, no grid aliasing.
(Anisotropic blue noise: Li & Wei, SIGGRAPH Asia 2010.)

## Consequences
+ One sampler yields all strategies: metric=None → isotropic; metric from tensor → anisotropic.
+ Exact N → clean `strategy x budget` ablation rows.
+ Prior art exists in pieces (anisotropic blue noise; structured 2D-GS init; error-driven
  densification) but not combined into a progressive 2D-Gaussian image codec — the novelty seam.
- Metric normalization (how axis-ratio maps to coherence) is a tuning surface; unit-area keeps
  counts comparable but the exact mapping is an ablation knob (`INIT-003`, `INIT-004`).
- Reference WSE is O(M) removals with grid neighbors; fine for init, not a hot loop.
