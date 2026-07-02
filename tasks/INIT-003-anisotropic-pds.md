# INIT-003: Anisotropic blue-noise sampling (WSE + metric)

**Status: partial.** Reference works (`sampling.py`, ADR-0005); tuning + rigor open.

## Goal
Exact-N, density-adaptive, anisotropic blue noise via Weighted Sample Elimination with a
per-point Mahalanobis metric from the structure tensor.

## Acceptance criteria
- [x] Exact N; density adaptivity via per-point radius; unit-area anisotropy metric.
- [x] Validated: returns exactly N; blue-noise min-distance > 0; denser near features.
- [ ] Spectral check: radial power spectrum / pair-correlation shows blue-noise trough
      (isotropic) and the expected anisotropic signature near edges.
- [ ] Map from `coherence` → axis ratio calibrated (currently linear); expose + sweep.
- [x] Performance pass: conflict-graph construction and initial weights are now vectorized over
      grid-cell offsets; only the greedy heap removal stays in Python. ~30x faster at N=20k
      (candidate M=120k) vs the original per-pair loops. Also fixes a correctness bug: the old
      `cell = 2*r_max` 3x3 scan missed long-range along-edge conflicts under an anisotropic metric;
      the reach is now bounded per receiver by the metric's minimum eigenvalue.

## Depends on
INIT-001, INIT-002.
