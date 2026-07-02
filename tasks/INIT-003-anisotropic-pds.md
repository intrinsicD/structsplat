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
- [ ] Performance pass: current WSE is O(M) removals with a Python heap — profile at N~50k.

## Depends on
INIT-001, INIT-002.
