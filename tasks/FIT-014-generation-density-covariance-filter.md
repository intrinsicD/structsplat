# FIT-014: Generation-density covariance filtering

## Status

Implemented and screened; experimental default-off. The faithful GaussianImage++ cohort formula
and two weaker controls all lost quality/convergence on the COCO4 proxy, so none is promotable.

## Motivation

GaussianImage++ adds an isotropic covariance floor per birth cohort to reduce holes while a sparse
Gaussian set grows. The released rule is not an elapsed-age decay:

`s = min(max_variance, H*W / (alpha*N_after))`, with `alpha=9*pi` and max `300 px^2`.

Every cohort retains its assigned variance; pruning subsets it, sampled-add rows receive the
post-growth value, duplicate children inherit their parent, and relocated rows are reborn at the
current density.

## Implementation

- `FitConfig`/CLI expose `covariance_filter_mode`, `covariance_filter_alpha`, and the variance cap.
- `GaussianField.filter_variance` survives detach/subset/append/NPZ and is included in effective
  conics/radii/scales for normalized, custom CUDA, and gsplat render paths.
- Codec/QAT paths materialize the effective covariance into RS scales; no extra stream is claimed.
- Pyramid/generation render paths preserve effective covariance. Nonuniform generation resizes
  apply the exact pixel transform `D Sigma D` and re-diagonalize to RS form; scaling local axes
  directly is only valid for uniform resize or axis-aligned Gaussians.
- Default mode `none` allocates no filter metadata and preserves prior behavior.

Focused lifecycle, renderer, gradient, relocation, NPZ, and codec tests live in
`tests/test_covariance_filter.py`.

## Proxy result (2026-07-10)

Artifact: `results/structsplat_generation_caf_proxy/index.html`.

Protocol: four pinned COCO images, max-side 160, cap 640/start 320, 500 iterations, seeds 0/1,
CUDA renderer, pinned StructSplat best-default geometry/growth/loss. Candidate gains below are
paired against the default; positive is better.

| Filter alpha | ΔPSNR | Δproxy MS-SSIM | ΔAUC | ΔLPIPS gain | Decision |
|---:|---:|---:|---:|---:|---|
| `9*pi` | -2.3587 dB | -0.01087 | -1.5616 | -0.1033 | reject |
| `18*pi` | -1.0799 dB | -0.00371 | -0.7372 | -0.0406 | reject |
| `36*pi` | -0.4362 dB | -0.00106 | -0.2883 | -0.0127 | reject |

All three lost every quality/convergence aggregate; weakening the variance only approached the
unfiltered baseline monotonically. This suggests the persistent additive-renderer covariance floor
does not transfer directly to StructSplat's normalized compositor and already strong WSE/feature-
cap coverage. Do not spend a larger confirmation budget on this exact mechanism.

## Follow-up boundary

Keep the implementation as an opt-in research axis and codec-compatibility primitive. A future
coarse-to-fine experiment may release/materialize the floor before the terminal phase, but it must
first beat the unfiltered proxy on AUC and final quality; the persistent native rule is closed.

## Interfaces

`src/structsplat/config.py`, `src/structsplat/gaussians.py`, `src/structsplat/fit.py`,
`src/structsplat/codec.py`, `src/structsplat/cli.py`,
`benchmarks/fair_density_control_compare.py`, `tests/test_covariance_filter.py`.

## Depends on

FIT-004, CORE-002, COMP-002, BENCH-002.
