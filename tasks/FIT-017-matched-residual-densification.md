# FIT-017: Kernel-matched signed-residual densification

## Status

Implemented and screened; rejected by the preregistered COCO4/two-seed guard. Keep the score axis
for reproducibility, leave `legacy_abs` as the shipped default, and do not spend the planned
160-pixel or fair-regime budgets on this exact isotropic score.

## Context

Sampled-add growth currently ranks pixels by pointwise absolute RGB error, optionally mixed with a
fixed 3x3 average. That proxy ignores the intervention: a child affects a footprint, so nearby
residuals with opposite signs can cancel even when their absolute error is large. Matching pursuit
and adaptive error estimation suggest scoring the signed residual after applying the planned
child's kernel before deciding where to add capacity.

This is a transfer and mechanism test, not a standalone novelty claim. The implementation keeps
StructSplat's normalized renderer, target-color initialization, tensor-aligned child geometry,
budget, and optimization schedule fixed. It changes only the birth-site score.

## Goal

Add `sampled_add_score=signed_gaussian`, which scores each candidate pixel as the RGB norm of the
signed residual convolved with an isotropic Gaussian at the planned child scale. The controlled
arm retains `refine_site=residual_tensor`, `refine_primitive=sampled_add`, tensor-aligned geometry,
and the existing color policy.

## Acceptance criteria

- [x] `FitConfig` and both CLIs expose `signed_gaussian` as an opt-in sampled-add score without
      changing the `legacy_abs` default or the `residual_tensor_add` alias.
- [x] The score uses signed per-channel residuals, a normalized separable Gaussian kernel, and the
      same planned base scale used to initialize the child cohort.
- [x] A direct 2D-convolution oracle plus synthetic tests cover signed cancellation, coherent
      retention, tiny images, growth count, and axis wiring.
- [x] Existing `residual` and `residual_tensor` paths remain behaviorally unchanged when the new
      score stays at its `legacy_abs` default.
- [x] A reproducible equal-count screen records immediate, post-20, and post-100 PSNR plus score
      and total runtime for legacy, magnitude-blur, and signed-blur controls.
- [x] Documentation states the mechanism, primary prior-art lineage, evidence, and default-off
      decision.

## Preregistered screen and decision rule

1. Mechanism guard: deterministic CPU, COCO4, max-side 64, N=64 -> 80, 40 pre-growth steps,
   target-color children, seeds 0/1. Compare `legacy_abs`, signed Gaussian, and a same-width
   Gaussian blur of residual magnitude at immediate, post-20, and post-100 recovery checkpoints.
2. If the guard survives, confirm at max-side 160, N=320 -> 640 using the normal five-wave CUDA
   proxy before considering a fair-regime run.

The guard survives only if signed scoring gains at least +0.10 dB mean post-20 PSNR, is positive
on at least 75% of image/seed pairs, has no worse than -0.05 dB mean post-100 PSNR, and adds no
more than 15% to total fit time. Promotion still requires independent images, multiple seeds, the
project's fixed-count fair regime, and the standard PSNR/MS-SSIM/LPIPS/AUC gates.

Abandon this exact score if the post-20 gain is below +0.10 dB, reverses by post-100, or exceeds the
runtime gate. Do not tune kernel width on the same guard after observing it; a different kernel is
a new hypothesis.

## Evidence and decision (2026-07-13)

Command:

```bash
python -m benchmarks.sampled_add_score_compare \
  --outdir results/fit017_sampled_add_score_guard \
  --seeds 0 1 --max-side 64 --start-count 64 --add-count 16 \
  --pre-iters 40 --device cpu --render-chunk 512
```

All 24 arms completed (four images x two seeds x three scores). Relative to `legacy_abs`:

| Score | Immediate delta | Post-20 delta | Post-100 delta | Positive post-20 pairs | Total-100 overhead |
|---|---:|---:|---:|---:|---:|
| `gaussian_abs` | +0.6070 dB | -0.0116 dB | -0.1664 dB | 3/8 | +0.1% |
| `signed_gaussian` | +0.5199 dB | -0.0318 dB | -0.2301 dB | 3/8 | -2.1% |

The wider kernels improve the immediate render, but both lose after optimization. Signed
cancellation is also weaker than magnitude blur at every reported horizon, so the original
one-seed pilot did not isolate the claimed mechanism. The signed arm fails three of four gates:
post-20 gain, sign agreement, and post-100 retention. Stop at stage 1 and do not tune width on
these eight pairs. A future anisotropic or denominator-aware intervention is a new hypothesis.

## Interfaces touched

`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/cli.py`,
`benchmarks/stage_search.py`, `benchmarks/sampled_add_score_compare.py`, `benchmarks/README.md`,
`tests/test_fit_dynamics.py`, `tests/test_sampled_add_score_compare.py`,
`tests/test_stage_search.py`, `tests/test_cli.py`.

## Depends on

FIT-004, FIT-009, BENCH-002, ABL-004.

## Research provenance

The candidate and alternatives are audited in
`ara/evidence/research-portfolio-2026-07-13.md`. The pre-implementation four-image/one-seed pilot
favored a one-scale matched score by +0.2998 dB after 20 recovery steps, while a naive
denominator-exact color intervention was rejected because it lost -0.7403 dB against the current
unit-residual color after 20 steps. Those numbers are screening evidence only.
