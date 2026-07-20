# FIT-018: Responsibility-normalized error-density densification

## Status

Implemented/screened and rejected by the frozen mechanism guard. The opt-in scorer remains a
research control, but the exact alpha-0.7 lineage is stopped. No larger proxy, default change,
compression claim, or held-out run is authorized.

## Context

The normalized renderer already defines a soft per-pixel ownership distribution, but the current
growth-site rules ignore its denominator. `support` averages residual under each Gaussian's raw
kernel, while center-residual rules sample error only at the mean. Neither distinguishes a broad
kernel that overlaps a pixel from the Gaussian that actually owns that pixel after normalization.

Soft Anisotropic Diagrams (SAD, SIGGRAPH 2026) uses soft responsibility mass
`m_i = sum_p gamma_ip`, responsibility-weighted error `E_i = sum_p gamma_ip e_p`, and
`E_i / max(m_i, eps)^alpha` for densification. This task is a controlled N2-T transfer of that
known mechanism into StructSplat's normalized Gaussian renderer, not a standalone novelty claim.
FIT-017 explicitly left a denominator-aware intervention as a new hypothesis after its
kernel-matched residual score failed recovery.

## Goal

Add an opt-in `refine_site="responsibility"` rule and a logged
`responsibility_mass_alpha` (default `0.7`). With normalized responsibilities
`gamma_ip = w_ip / (sum_j w_j + eps)`, rank existing Gaussians by

`score_i = sum_p gamma_ip * mean_c((render_p - target_p)^2) / max(sum_p gamma_ip, eps)^0.7`.

Hold the renderer, starting field, final Gaussian count, split primitive, and recovery optimizer
fixed so the experiment isolates only the ownership-aware site score.

## Preregistered mechanism guard

- Deterministic CPU; the four tracked COCO fixtures; seeds 0 and 1; max-side 64.
- Fit one shared `quadtree_wse` field per image/seed at N=64 for 40 steps.
- Clone it into `residual`, `support`, `responsibility_alpha1`, and
  `responsibility_alpha0.7` site-score arms. The alpha-1 diagnostic isolates denominator-aware
  ownership from SAD's sublinear mass exponent; alpha 0.7 is the preregistered donor arm.
- Each arm applies one identical moment-preserving 64 -> 80 split and is measured immediately,
  after 20 recovery steps, and after 100 independently replayed recovery steps.
- Record score time, total time, final N, PSNR, and paired deltas. These reused fixtures are only a
  mechanism smoke; they cannot support publication, default, SOTA, or compression claims.

Freeze `alpha in {1.0, 0.7}` before running: 1.0 is the causal denominator diagnostic and 0.7 is
the donor mechanism. Do not add values or tune the residual norm, split primitive, or support on
these pairs after seeing the result.

The guard survives only if the preregistered `responsibility_alpha0.7` arm, against the stronger
of the two existing arms:

1. gains at least +0.10 dB mean post-20 PSNR;
2. is positive on at least 6/8 image/seed pairs at post-20;
3. is no worse than -0.05 dB mean at post-100;
4. adds no more than 15% to total 100-step wall time; and
5. preserves exact equal final count with finite scores and renders.

Stop this exact lineage if any quality/recovery gate fails. A pass only authorizes a disjoint,
larger confirmation with PSNR/MS-SSIM/LPIPS/AUC and actual stream bytes; it is not promotion.

## Acceptance criteria

- [x] `FitConfig` and stage-search/CLI expose `refine_site="responsibility"` and a finite
      `responsibility_mass_alpha` in `(0, 1]` behind opt-in flags; all defaults and legacy aliases
      are unchanged.
- [x] A direct dense responsibility oracle validates mass, error, opacity, and overlap behavior.
- [x] The implementation uses the same compact support and opacity semantics as the normalized
      renderer and rejects additive/gsplat modes where ownership is not the same equation.
- [x] Tests cover finite zero-coverage handling, exact growth count, background-row masking, and
      unchanged behavior when the new site mode is not selected.
- [x] The shared-start benchmark writes config/environment provenance, per-pair CSV, aggregate
      JSON, a decision, and a concise Markdown summary.
- [x] Targeted tests and the full suite pass under the repository's documented system-libstdc++
      preload.
- [x] The measured decision and negative result, if any, are recorded without post-hoc tuning.

## Measured decision (2026-07-15)

The frozen 32-arm-row run completed all four images x two seeds x four site scores with exact
N=80 and finite scores/renders. `support` was the stronger existing comparator by mean post-20
PSNR.

| Site score | Immediate PSNR | Post-20 PSNR | Post-100 PSNR | Score s | Total-100 s |
|---|---:|---:|---:|---:|---:|
| `residual` | 20.3105 | 21.5835 | 23.0925 | 0.000096 | 0.9962 |
| `support` | 20.3093 | 21.6255 | 23.2022 | 0.003097 | 0.9939 |
| `responsibility_alpha1` | 20.3210 | 21.6274 | 23.2444 | 0.005861 | 0.9815 |
| `responsibility_alpha0.7` | 20.2470 | 21.6057 | 23.1611 | 0.005718 | 1.0122 |

Against `support`, the preregistered alpha-0.7 donor arm was `-0.0198 dB` at post-20, positive on
only `4/8` pairs, and `-0.0411 dB` at post-100, with `+1.8%` total-100 time. It passed the
post-100, time, count, and numerical guards but failed both post-20 quality gates. The guard is
therefore **rejected** and the exact donor lineage stops without tuning these fixtures.

The alpha-1 diagnostic was essentially tied at post-20 (`+0.0019 dB`) and modestly higher at
post-100 (`+0.0422 dB`) than support. That observation motivates a separate duplication/gauge
invariance question; it does not rescue alpha 0.7 and is not a promotion result.

The benchmark enforced one CPU thread and PyTorch deterministic algorithms after an audit found
sub-millidecibel drift under parallel CPU reduction. A second source-frozen replay matched every
non-timing aggregate exactly. Raw non-ARA artifacts are in
`results/fit018_responsibility_split_guard/`; relevant-source SHA-256 is
`32035c6988e66c3ec8a0c9a088433ab4a0833a66c2d5adc6a95dcd66b67d992b`. The research-manager
epilogue binds the durable evidence copy to the run provenance.

## Interfaces touched

`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/cli.py`,
`benchmarks/stage_search.py`, `benchmarks/responsibility_split_compare.py`,
`benchmarks/README.md`, and focused tests.

## Depends on

FIT-004, FIT-009, FIT-017, BENCH-002, ADR-0010.
