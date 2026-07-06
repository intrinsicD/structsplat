# COMP-004: QAT + entropy-aware fitting

**Status: todo.** Move compression pressure into optimization, not only post-fit coding.

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
- [ ] `FitConfig.qat_mode` supports at least `off`, `ste`, and `noise`; `lambda_rate` is logged.
- [ ] Differentiable bit estimate covers positions, scales, rotations, colors, and opacity if
      enabled.
- [ ] Final encoded file uses the same quantizer assumptions optimized during fitting.
- [ ] Tests show QAT improves low-bit reconstruction over post-hoc quantization on a tiny fixture.
- [ ] `benchmarks/rate_distortion.py` can sweep lambda and write RD evidence for each point.
- [ ] COMP-003 rungs that remain codec-only are kept compatible with this fit-time mode.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/codec.py`, `src/structsplat/config.py`,
`benchmarks/rate_distortion.py`, `tests/test_codec.py`, `tests/test_fit*.py`.

## Depends on
COMP-001, COMP-003, FIT-001.
