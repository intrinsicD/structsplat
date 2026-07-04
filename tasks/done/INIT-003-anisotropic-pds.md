# INIT-003: Anisotropic blue-noise sampling (WSE + metric)

**Status: done.** Reference works (`sampling.py`, ADR-0005); calibration tooling now covers
spectral, pair-correlation, edge-anisotropy, and coherence-to-axis-ratio sweeps.

## Goal
Exact-N, density-adaptive, anisotropic blue noise via Weighted Sample Elimination with a
per-point Mahalanobis metric from the structure tensor.

## Acceptance criteria
- [x] Exact N; density adaptivity via per-point radius; unit-area anisotropy metric.
- [x] Validated: returns exactly N; blue-noise min-distance > 0; denser near features.
- [x] Spectral check tooling: `benchmarks/init_spectral_analysis.py` writes radial power spectra,
      pair-correlation/nearest-neighbor summaries, and edge-local directional spacing metrics for
      isotropic and anisotropic initializers. Full calibration runs still need to be archived as
      evidence for a paper-grade claim.
- [x] Map from `coherence` → axis ratio is exposed through `max_axis_ratio` and
      `coherence_power`; `benchmarks/init_spectral_analysis.py` sweeps both and reports realized
      edge scale ratios plus RMSE against the mapping.
- [x] Performance pass: conflict-graph construction and initial weights are now vectorized over
      grid-cell offsets; only the greedy heap removal stays in Python. ~30x faster at N=20k
      (candidate M=120k) vs the original per-pair loops. Also fixes a correctness bug: the old
      `cell = 2*r_max` 3x3 scan missed long-range along-edge conflicts under an anisotropic metric;
      the reach is now bounded per receiver by the metric's minimum eigenvalue.

## Depends on
INIT-001, INIT-002.
