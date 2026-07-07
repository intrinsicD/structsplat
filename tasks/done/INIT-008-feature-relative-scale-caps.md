# INIT-008: Feature-relative scale caps (fix the cap-scaling failure)

**Status: done.** `feature_rel` is implemented and searchable, but the fair-density difficult-four
protocol rejected it as a default. It repairs most of the old resolution-scaled cap failure versus
`feature`, but still averages -0.3733 dB PSNR versus matching uncapped rows and loses badly at
budget 2000. Keep `scale_cap=feature_rel` searchable and default off.

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
- [x] `feature_rel` cap in init + fit (cap enforced through reparameterization or clamp with
      subgradient, matching how ADR-0012 caps are enforced today), NumPy init math torch-free.
- [x] Resolution-invariance test: caps computed at max-side 160 and 768 for the same image bind
      the same *fraction* of Gaussians within tolerance.
- [x] Re-run the exact protocol that produced the -2.05 dB failure
      (`fair_density_control_compare`, difficult-4, budgets {2k,5k,10k}): acceptance is
      no-loss (> -0.1 dB) at every budget and a win somewhere; otherwise record the negative and
      close ADR-0012's candidacy honestly (update claims C02).
- [x] Stage-search `scale_cap` axis gains `feature_rel`; evidence committed.

## Outcome
Evidence: `ara/evidence/init008-feature-relative-scale-caps-2026-07-07/`.

Exact difficult-four result (4 images x budgets 2k/5k/10k x four method families):

- `feature_rel` vs matching uncapped: 48 paired cells, mean dPSNR -0.3733 dB, min -2.1119 dB,
  14/48 wins.
- Budget 2000 rejects promotion in all four method families: mean dPSNR ranges from -0.7764 to
  -1.1165 and min dPSNR reaches -2.1119.
- Budget 10000 has small positives in residual rows, but the task acceptance rule required
  no-loss at every budget.
- `feature_rel` vs old absolute `feature` cap is a clear repair: +0.0793 to +3.3051 dB mean PSNR
  depending on method/budget, with the largest gains at 2000.

Decision: `feature_rel` remains a stage-search axis, not a default. ADR-0012's feature-cap default
candidacy is closed for the current fair-density protocol.

## Interfaces touched
`src/structsplat/init.py`, `src/structsplat/sampling.py` (expose `r_i`),
`src/structsplat/gaussians.py` / `fit.py` (cap enforcement), `benchmarks/stage_search.py`,
`tests/test_init_stages.py`.

## Depends on
ADR-0012, INIT-003 (WSE radii), fair-density featurecap evidence. Updates claim C02 either way.
