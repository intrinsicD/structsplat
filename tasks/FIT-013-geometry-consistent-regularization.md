# FIT-013: Geometry-consistent image regularization

## Status

Partial. The loss, CLI/config plumbing, fair-harness candidates, unit tests, proxy sweep, schedule
sweep, and a held-out Kodak slice are complete. The evidence supports an opt-in quality candidate,
not a new default: dense evaluation improves quality and convergence but has a measurable cost at
the larger resolution, while intermittent evaluation gives back most of the quality gain.

## Goal

Test whether ground-truth-gradient-weighted rendered-gradient supervision improves edges and
convergence at a fixed Gaussian budget. Keep this distinct from StructSplat's tensor-weighted pixel
loss: that mechanism reweights RGB residuals spatially, whereas this loss directly penalizes Sobel
gradient mismatch.

The mechanism follows the geometry-consistent loss in
[Structure-Guided Allocation](https://arxiv.org/abs/2512.24018):

```text
Dx = (Sx(render) - Sx(target))^2
Dy = (Sy(render) - Sy(target))^2
Lgeometry = mean_HW(sum_RGB(|Sx(target)| * Dx + |Sy(target)| * Dy))
L = Lbase + geometry_loss_weight * Lgeometry
```

Target gradients and grouped Sobel kernels are cached. `geometry_loss_every=N` can evaluate the
term intermittently; its active weight is multiplied by `N` to retain the requested mean weight.
The shipped default remains disabled (`geometry_loss_weight=0`).

## Completed 2026-07-10

- Added `FitConfig.geometry_loss_weight`, `FitConfig.geometry_loss_every`, matching CLI flags,
  fit-history values, output metadata, validation, exact-loss/differentiability tests, and a
  global-schedule-offset test.
- Benchmark configs now stamp the StructSplat commit, dirty state, tracked-diff SHA-256, and
  untracked file list. Dominance relations use complete pairs and Bonferroni-adjusted bounds for
  95% familywise coverage; displayed per-metric intervals remain marginal 95% intervals.
- Added fair candidates at weights 0.015, 0.030, and 0.060 plus every-two/every-four schedules.
- COCO review proxy: four images x seeds 0/1, max-side 160, cap 640, 500 steps, LPIPS enabled.
  Dense 0.015 versus the paired default gains +0.1887 dB PSNR (image-bootstrap 95% CI
  [+0.0853, +0.3179]), +0.00086 MS-SSIM [+0.00017, +0.00156], +0.1132 AUC
  [+0.1001, +0.1312], and +0.0047 LPIPS gain [+0.0003, +0.0108]. Its mean fit-time gain is
  -0.0888 seconds, so the result is a tradeoff and fails default promotion. Weights 0.030 and
  0.060 did not improve that balance. Artifact:
  `results/fit013_geometry_consistency_coco4_proxy/index.html`.
- Schedule sweep on the same proxy: every-two retains +0.1504 dB PSNR and +0.0673 AUC but has an
  inconclusive MS-SSIM delta and worsens LPIPS by 0.0070. Every-four is weaker. Separate CUDA runs
  are paired only within their own artifact because exact values vary slightly across reruns.
  Artifact: `results/fit013_gcr_schedule_coco4_proxy/index.html`.
- Held-out Kodak slice: kodim01/07/13/19, native 768 x 512 resolution, cap 2000, 1500 steps,
  seed 0. Dense 0.015 gains +0.1998 dB PSNR, +0.01016 MS-SSIM, +0.1756 AUC, and +0.01533 LPIPS
  while costing 2.189 seconds of fit time. Quality wins occur on three of four images, but the
  four-image quality CIs still cross zero except for AUC and LPIPS. Every-two is 1.939 seconds
  faster than default but is nearly quality-neutral (+0.0262 dB PSNR, +0.00020 MS-SSIM,
  -0.0239 AUC). Artifact: `results/fit013_gcr015_kodak4_b2000/index.html`.
- A manual Sobel stencil was tested as a kernel-launch optimization and rejected: multiple sliced
  elementwise operations were substantially slower than two cached grouped convolutions on CUDA.

## Evidence limits

- The schedule sweep is an independent CUDA rerun, not a clean replication: identically seeded
  default cells differed by as much as 0.61 dB across artifacts. Only within-artifact paired deltas
  are interpreted, and the dense 0.015 result needs a larger randomized confirmation.
- Kodak has four images, one seed, and one budget. COCO and Kodak also use very different
  Gaussians-per-pixel densities, so dataset, resolution, and density effects are confounded.
- Current fair-harness `fit_seconds` synchronizes regularly through logged metrics but excludes
  target-gradient setup and final evaluation; method order is fixed. Timing is sufficient to block
  promotion when dense GCR is clearly slower, not to support a fine-grained speed claim.
- Cadence scaling preserves the arithmetic mean coefficient, not Adam update equivalence. The
  every-two/every-four arms are distinct impulsive objectives, not free approximations of dense
  training.

## Decision

Keep `structsplat_best_default` unchanged. Retain `structsplat_best_gcr015` as an explicit
quality/convergence candidate for larger confirmation runs. Do not promote the intermittent
schedules based on the current evidence.

## Next actions

1. Confirm dense 0.015 on all Kodak images with at least three seeds and three matched
   Gaussians-per-pixel budgets; randomize/interleave method order and add resized controls.
2. Profile/fuse the two Sobel passes into one CUDA or compiled operation; rerun the promotion gate
   after removing loss-computation overhead rather than weakening the objective.
3. Add edge-binned reconstruction error and gradient-domain validation metrics so the claimed
   mechanism is tested directly, not inferred only from aggregate image metrics.
4. If the fused dense loss remains useful, test a late-phase schedule that enables it after the
   final growth wave, plus an unscaled every-two arm, instead of assuming cadence scaling is
   optimization-equivalent.

## Interfaces

`src/structsplat/config.py`, `src/structsplat/fit.py`, `src/structsplat/cli.py`,
`benchmarks/fair_density_control_compare.py`, `tests/test_fit_dynamics.py`,
`tests/test_fair_density_control_compare.py`.

## Depends on

FIT-005, FIT-006, FIT-007, ABL-004.
