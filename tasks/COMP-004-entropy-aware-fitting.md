# COMP-004: QAT + entropy-aware fitting

**Status: partial.** First implementation slice landed 2026-07-07: fit-time QAT modes,
`lambda_rate`, a differentiable rate proxy, returned frozen codec config, CLI flags, tests, and a
small encode/RD smoke. Decision-grade RD improvements and lambda sweeps remain open.

## Context
The existing codec and COMP-003 ladder cover post-fit quantization plus QAT rungs. The stronger
compression direction is to make quantization and entropy part of fitting: STE/noise during
optimization, learnable scalar quantizers, and a differentiable rate estimate in the objective.

## Goal
Add an entropy-aware QAT phase that optimizes `distortion + lambda_rate * estimated_bits` before
final encoding.

## Approach
1. Reuse `CodecConfig` quantized views inside the fitter with straight-through gradients or
   additive quantization noise.
2. Add learnable per-attribute scalar quantizer ranges/steps where useful.
3. Estimate entropy from factorized or context-conditioned symbol probabilities during fitting.
4. Sweep `lambda_rate` to trace an actual RD curve with fitted-for-compression fields.

## Acceptance criteria
- [x] `FitConfig.qat_mode` supports at least `off`, `ste`, and `noise`; `lambda_rate` is logged.
- [x] Differentiable bit estimate covers positions, scales, rotations, colors, and opacity if
      enabled.
- [~] Final encoded file uses the same quantizer assumptions optimized during fitting. `fit()`
      returns `qat_codec_config`; automated final-file wiring through every codec benchmark remains
      open.
- [ ] Tests show QAT improves low-bit reconstruction over post-hoc quantization on a tiny fixture.
- [ ] `benchmarks/rate_distortion.py` can sweep lambda and write RD evidence for each point.
- [ ] COMP-003 rungs that remain codec-only are kept compatible with this fit-time mode.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/codec.py`, `src/structsplat/config.py`,
`benchmarks/rate_distortion.py`, `tests/test_codec.py`, `tests/test_fit*.py`.

## Current implementation notes

- `FitConfig.qat_mode` accepts `off`, `ste`, and `noise`.
- `lambda_rate` adds `lambda_rate * differentiable_rate_bpp(...)` during fitting.
- The in-loop rate proxy covers means, log-scales, rotations, colors, and opacity probabilities
  when opacity logits are present.
- `qat_mode="ste"` renders through the existing codec fake-quantized view.
- `qat_mode="noise"` renders through additive quantization noise using the same bit depths.
- Fit-time QAT currently fails closed for `color_basis="affine"` because codec v1 cannot encode
  affine color coefficients.
- Evidence: `ara/evidence/comp004-fit-time-qat-smoke-2026-07-07/run.md`.
- Next step (2026-07-07 benchmark review): the plumbing is validated; the missing piece is
  the lambda_rate sweep at 2-3 bit-depth ladders on the fair regime, reported as RD curves
  against post-hoc quantization of the same fitted fields (the honest control).

## Depends on
COMP-001, COMP-003, FIT-001.
