# INIT-008: Feature-relative scale caps (fix the cap-scaling failure)

**Status: todo.** The feature-cap *idea* predates its resolution-scaling failure; retest with
caps derived from measured feature scale, not image size.

## Context
Feature caps (ADR-0012) won the small max-side-160 COCO screens, but the fair-density finalist
test scaled the 12 px cap linearly with resolution (12 px @ 160 → 57.6 px @ 768) and lost
-2.05 dB mean, -4.18 dB at budget 2000, 4/48 wins
(`fair-density-control-featurecap-difficult4-2026-07-05`). A cap that grows with *resolution*
rather than with *content* stops being a feature cap: at high resolution it stops capping
features and at low budget it forbids the large low-frequency splats that flat regions need.

## Goal
Cap each Gaussian's scale relative to a local feature-scale estimate the pipeline already
computes, so the cap is resolution-invariant and only binds where features are actually small.

## Design
Candidate local feature scales, in order of preference (all already available at init time):
1. the per-point WSE spacing radius `r_i` (density-adaptive — small where detail is dense);
2. the quadtree leaf side for quadtree strategies;
3. a tensor-derived length scale (e.g. `sigma_rho / sqrt(coherence)`), fallback for non-WSE.

`scale_cap=feature_rel` caps `s_across <= gamma_across * r_i` and `s_along <= gamma_along * r_i`
(anisotropy-aware: along-edge elongation must stay allowed — the -2 dB failure partly came from
capping the long axis). Flat-classified Gaussians get a much looser or no cap.

## Acceptance criteria
- [ ] `feature_rel` cap in init + fit (cap enforced through reparameterization or clamp with
      subgradient, matching how ADR-0012 caps are enforced today), NumPy init math torch-free.
- [ ] Resolution-invariance test: caps computed at max-side 160 and 768 for the same image bind
      the same *fraction* of Gaussians within tolerance.
- [ ] Re-run the exact protocol that produced the -2.05 dB failure
      (`fair_density_control_compare`, difficult-4, budgets {2k,5k,10k}): acceptance is
      no-loss (> -0.1 dB) at every budget and a win somewhere; otherwise record the negative and
      close ADR-0012's candidacy honestly (update claims C02).
- [ ] Stage-search `scale_cap` axis gains `feature_rel`; evidence committed.

## Interfaces touched
`src/structsplat/init.py`, `src/structsplat/sampling.py` (expose `r_i`),
`src/structsplat/gaussians.py` / `fit.py` (cap enforcement), `benchmarks/stage_search.py`,
`tests/test_init_stages.py`.

## Depends on
ADR-0012, INIT-003 (WSE radii), fair-density featurecap evidence. Updates claim C02 either way.
