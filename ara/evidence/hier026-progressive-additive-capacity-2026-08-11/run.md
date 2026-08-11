# HIER-026 progressive pure-additive capacity confirmation

## Evidence class

Prospectively frozen, dirty-source, producer-reviewed confirmation diagnostic on four previously
unopened official DIV2K validation images. The archive identity, filenames, selection digests,
member hashes, seven arms, counts, schedules, two seeds, metrics, work units, and killing gates
were bound before Python decoded a selected pixel. This is stronger than the consumed development
screens, but absent distinct protocol/outcome review it remains provisional and cannot authorize a
semantic, default, codec, rate, full-resolution, or novelty claim.

## Executed protocol

- Sources: official `DIV2K_valid_HR.zip`, archive SHA-256
  `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`; prospectively ranked
  `0895.png`, `0860.png`, `0898.png`, and `0847.png`.
- Raster/seeds: deterministic LANCZOS max-side 160, seeds 0/1, required LPIPS.
- N=640 controls: ordinary normalized, ordinary additive, and safely projected additive, each
  with 500 L1 + 0.3 SSIM updates.
- Capacity controls: cold full-target projected additive at N=896 and N=960, 500 updates.
- Candidate: the exact shared N=640 additive base plus 256 signed residual births and 200 joint
  full-target updates at N=896, before/after the unchanged safe RGB projection.
- Endpoint: one four-array, opacity/mass/denominator/level-free additive `GaussianField`, rendered
  in one pass. Training-only scale caps are stripped before persistence.

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier026_progressive_additive_capacity.py \
  /tmp/structsplat-hier026-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier026_div2kvalid4_s160_capacity_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Result

| arm | N | mean PSNR | mean MS-SSIM | mean LPIPS | pixel max | 7x7 max |
|---|---:|---:|---:|---:|---:|---:|
| normalized plain | 640 | 26.7509 | 0.972047 | 0.144592 | 0.47811 | 0.14781 |
| additive plain | 640 | 25.8258 | 0.970966 | 0.210264 | 0.36985 | 0.14545 |
| additive projected safe | 640 | 25.9090 | 0.971181 | 0.209507 | 0.36268 | 0.14323 |
| cold additive projected | 896 | 27.4206 | 0.979240 | 0.149207 | 0.33409 | 0.12396 |
| progressive residual | 896 | 27.2743 | 0.981632 | 0.134367 | 0.30420 | 0.10818 |
| progressive residual projected | 896 | 27.5048 | 0.982141 | 0.133531 | 0.30046 | 0.10597 |
| cold additive projected | 960 | 27.6959 | 0.980720 | 0.145961 | 0.33001 | 0.12256 |

All 56 cells and all integrity gates pass. Every additive endpoint contains exactly `means`,
`log_scales`, `rotations`, and `colors`; counts, shared-base hashes, pre-projection branch hashes,
500/200 stage boundaries, `499,200` progressive row updates, cold work, projection rollback,
bounded coefficients, and maximum `2e-5` internal/cold parity are checker-validated. Projection
selects 4/8 N=640, 4/8 cold-N=896, 6/8 progressive-N=896, and 3/8 cold-N=960 proposals.

The representation signal is strong but the frozen composite gate is negative. Projected
progressive N=896 beats normalized by `+0.75388 dB` on mean and in every paired cell (minimum
`+0.04411 dB`); cold N=960 is `+0.94493 dB` on mean with minimum `+0.35273 dB`. Both improve mean
MS-SSIM and both local maxima. Progressive also improves mean LPIPS by `0.01106`. Nevertheless,
its `0860` seed-0 LPIPS is `+0.05447` worse than normalized, beyond the frozen `+0.01` cell guard.
Cold N=960 has LPIPS regressions of `+0.02910` on `0860` seed 0 and `+0.01618` on `0895` seed 1,
plus a `+0.02743` pixel-maximum regression on `0847` seed 1. Thus no rung is quality-capable under
the predeclared rule. Same-count projected additive remains `-0.84193 dB` on mean.

Native review finds that the higher-count pure fields remove much of normalized rendering's
polygonal/coverage breakup and reduce error on the alligator, blossom, and vehicle scenes. On the
dense forest scene they replace some blocky normalized error with diffuse directional foliage
smear. That material new blur agrees with the LPIPS counterexample, so visual review does not
override the gate.

## Consumed diagnostic successor signal

Only after the immutable HIER-026 decision, the four now-consumed images were used as killing
fixtures for ordinary cold additive counts. N=1024 still fails on `0860` seed 0 (`+0.03653` LPIPS
versus normalized). Projected N=1088 passes every HIER-026 numeric clause on all eight consumed
cells, with mean/minimum PSNR deltas `+1.68200/+0.98761 dB` and worst LPIPS delta `+0.00335`.
N=1152 also passes, at `+1.97971/+1.04845 dB`. These unbundled post-hoc probes select the next
killing counts; they are not confirmation evidence and must not be folded into this outcome.

## Decision

Retain HIER-026 as a near-miss that rejects N<=960 under the frozen perceptual/local gate. It does
not show that normalization is necessary: PSNR, MS-SSIM, and mean/local error already favor the
single-pass pure field. It shows that the tested N=896 progressive topology and N=960 cold capacity
do not robustly preserve perceptual structure. The next task should test the simpler ordinary
additive N=1088 candidate and N=1152 robust control on a newly selected untouched bank, keeping
the exact renderer, fit, projection, endpoint, and quality clauses.

## Receipts

- Report: `results/hier026_div2kvalid4_s160_capacity_s01_confirmation_2026-08-11/index.html`
- Report checker: pass without override.
- Manifest SHA-256: `079976d3425ffbc4660a3d89336a82b997a44023b1126189a045b55b570f61d8`
- Metrics SHA-256: `03dc78b034f964fe43ad9cae3fb628c013ff941bab135fd004f1e3da80d1eabe`
- Decision SHA-256: `8089b173e1c20c9da10acaa166eedff4c529efc1b24e772917315e0d529edef7`
- Bundle inventory: 1,022 manifest-bound files, 42 MiB.

## Limitations

Four downscaled validation images; two seeds; one RTX 4090; dirty executed sources; producer-only
review; no distinct prospective reviewer; unequal counts/equations/work; no complete bytes,
full-resolution, downstream, or rate result. The post-hoc count probe is deliberately segregated
from the immutable confirmation.
