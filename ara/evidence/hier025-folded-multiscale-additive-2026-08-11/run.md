# HIER-025 folded multiscale additive diagnostic

## Evidence class

Frozen, dirty-source, producer-reviewed development diagnostic on four historically consumed
DIV2K training images. The source filenames, selection digests, file hashes, five arms, two seeds,
N=640 count, 500 attempted updates, metrics, and killing gates were frozen before Phase-B pixel
access. This can reject the tested basis; it cannot confirm a general method, semantic, default,
codec, rate, or novelty claim.

## Executed protocol

- Images: `0115.png`, `0457.png`, `0229.png`, and `0799.png`, selected by
  `SHA256("HIER-025-v1:" + filename)` after excluding all HIER-023/024 files.
- Raster/count/seeds: deterministic LANCZOS max-side 160, N=640, seeds 0/1.
- Controls: 500-step ordinary normalized, ordinary additive, and safeguarded projected-additive.
- Candidate: 16 additive grid Gaussians fitted for 100 L2 steps to factor-two low pass; 624
  signed-residual anisotropic WSE Gaussians fitted for 300 L2 steps; exact concatenation; 100
  full-target L1 + 0.3 SSIM additive steps with coarse geometry frozen; training mask removed.
- Candidate projection: the unchanged HIER-024 all-row PCG and target-known safety transaction.
- Endpoint: one opacity-free, denominator-free, mass-free, level-free direct additive
  `GaussianField`, rendered in one pass.

Command:

```bash
PYTHONPATH=src TORCH_CUDA_ARCH_LIST=8.9 \
python scripts/experiments/hier025_folded_multiscale_additive.py \
  tests/test_images/DIV2K_train_HR \
  results/hier025_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11
```

## Result

| arm | mean PSNR | mean MS-SSIM | mean LPIPS | pixel max | 7x7 max | PSNR AUC |
|---|---:|---:|---:|---:|---:|---:|
| normalized plain | 33.3654 | 0.991338 | 0.039086 | 0.23879 | 0.10368 | 31.7847 |
| additive plain | 32.4276 | 0.990802 | 0.057506 | 0.25040 | 0.10061 | 29.7343 |
| additive projected safe | 32.6322 | 0.991151 | 0.055478 | 0.24572 | 0.09666 | 29.7343 |
| folded multiscale additive | 30.8734 | 0.976632 | 0.087696 | 0.28189 | 0.10881 | 26.0709 |
| folded multiscale projected safe | 31.2239 | 0.982899 | 0.083812 | 0.26650 | 0.10325 | 26.0709 |

The unprojected folded basis is `-1.55421 dB` versus ordinary additive. With the identical
coefficient solve, it is `-1.40831 dB` versus projected additive and remains `-2.14150 dB` below
normalized. Projection selects seven of eight folded proposals and gains `+0.35054 dB` on mean,
so missing fixed-geometry RGB optimization is not the cause.

All integrity clauses pass: exact 16/624 accounting; all 500 updates completed; 20 charged
full-target observers; exact coarse-geometry freeze; training mask removed; maximum fold/endpoint
parity `4.17e-7/2.38e-7`; maximum candidate cold parity `1.79e-6`; maximum candidate coefficient
`1.5373`; and no opacity, mass, denominator, optimizer, target, auxiliary RGB, scaler, residual,
or level payload in a cold field. Every projection either satisfies all transaction clauses or
returns its incoming field exactly.

The candidate fails every declared quality gate: both PSNR floors, half-gap closure, mean
MS-SSIM/LPIPS/pixel/7x7 noninferiority, per-cell LPIPS/local guards, and full-target AUC. Its
trajectory reaches only 30.334 dB on average at the end of the proxy stages, drops when the joint
mixed loss begins, and recovers to 30.873 dB by step 500 versus 32.428 dB for direct additive.

Native full-frame/error/worst-crop review finds no new black hole, checker, gross lattice,
ringing, wash, or isolated color lobe. It does find material diffuse fine-detail blur, especially
on the `0457` skyline/window structure, `0115` insect contours, and `0799` aircraft edges. That is
the predeclared visual failure mode and agrees with the perceptual/structural losses.

## Decision

Reject `folded_grid16_residual` and do not tune its carrier count, scale, stage lengths, loss, or
residual construction on these pixels. The result does not prove normalization is necessary. It
shows that spending the equal 500-update horizon on separately optimized low-pass and residual
levels creates a worse N=640 additive span than full-target additive fitting. The next test must
change topology/capacity under a new output and data binding. The official validation filenames
remain unopened.

## Receipts

- Report: `results/hier025_div2k4_s160_n640_i500_s01_diagnostic_2026-08-11/index.html`
- Report checker: pass without override.
- Manifest SHA-256: `36d255c78ae39c9cfc70e8615df1a9821b58bcdd78a35288e8e0cd4816608dcb`
- Metrics SHA-256: `6b94638db4b1684baddf5754623f168d2735b31d1d27dc07f50b3a565748107a`
- Decision SHA-256: `7b35f4524c91d71a52e3582d34782af4b098e98e45fb92974517c6eda8b2652f`
- Bundle inventory: 669 manifest-bound files.

## Limitations

Four downscaled, historically consumed training images; two seeds; one RTX 4090; dirty executed
sources; producer-only review; unequal renderer equations; no codec/rate/downstream result; and no
distinct prospective or outcome reviewer. The first-call CUDA compilation outlier also makes the
aggregate wall-time columns unsuitable for a normalized-versus-additive speed claim.
