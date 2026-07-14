# ADR-0005: Anisotropy via a Mahalanobis metric inside Weighted Sample Elimination

**2026-07-14 scope update:** the sampler decision remains accepted as an interpretable engineering
option, but its original broad novelty rationale is superseded. Structure-Guided Allocation,
Image-GS, P-GSVC, and related work occupy structure-aware/progressive Gaussian territory, and the
specific tensor-metric WSE actual-rate formulation failed BENCH-007's development gate. No
held-out compression claim follows from this ADR.

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
+ The combination is a useful interpretable mechanism and control surface; no blanket novelty or
  progressive-codec claim follows from it.
- Metric normalization (how axis-ratio maps to coherence) is a tuning surface; unit-area keeps
  counts comparable but the exact mapping is an ablation knob (`INIT-003`, `INIT-004`).
- Reference WSE is O(M) removals with grid neighbors; fine for init, not a hot loop.
