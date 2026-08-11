# HIER-028 residual-pursuit pure-additive confirmation

## Evidence class

Prospectively frozen, dirty-source, producer-reviewed confirmation diagnostic on eight previously
unopened official DIV2K validation images. Archive identity, exclusions, selected names/member
hashes, four arms, the N=960+64 method, seeds, schedules, metrics, work units, and killing gates
were bound before Python decoded a selected pixel. This is positive bounded evidence, but remains
provisional without a distinct protocol/outcome reviewer and cannot change a maintained default.

## Executed protocol

- Sources: official `DIV2K_valid_HR.zip`, archive SHA-256
  `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`; prospectively ranked
  `0804.png`, `0830.png`, `0822.png`, `0812.png`, `0810.png`, `0862.png`, `0803.png`, and
  `0826.png` after excluding all HIER-026/HIER-027 names.
- Raster/seeds: deterministic LANCZOS max-side 160, seeds 0/1, required LPIPS.
- Baselines: normalized N=640, projected cold additive N=960, and projected cold additive N=1024.
- Candidate: the exact projected N=960 base plus 64 deterministic rows. Each row goes at the
  row-major highest raw-RGB-MSE pixel, has fixed `0.35 px` isotropic scale and zero rotation, and
  takes that pixel's current signed residual as RGB. The analytic additive reconstruction is
  updated before the next selection; the tail receives no optimizer steps.
- Endpoint: the bit-exact N=960 prefix plus 64 rows, persisted as only means, log-scales,
  rotations, and signed RGB, then cold-rendered as one ordinary additive sum.

This allocation principle is consistent with GaussianImage++'s published distortion-driven
densification, which adds primitives at high reconstruction distortion, while this task freezes a
narrower deterministic one-pass residual pursuit and makes no novelty claim. GaussianImage itself
establishes the additive-sum 2D Gaussian endpoint. Primary sources:
[GaussianImage](https://arxiv.org/abs/2403.08551),
[GaussianImage++](https://arxiv.org/abs/2512.19108), and
[AbsGS](https://arxiv.org/abs/2404.10484).

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier028_residual_pursuit_additive.py \
  /tmp/structsplat-hier028-div2k-valid-20260811/DIV2K_valid_HR \
  results/hier028_div2kvalid8_s160_residual_pursuit_s01_confirmation_2026-08-11 \
  --max-side 160 --seeds 0 1 --device cuda --lpips
```

## Result

| arm | N | mean PSNR | mean MS-SSIM | mean LPIPS | pixel max | 7x7 max |
|---|---:|---:|---:|---:|---:|---:|
| normalized plain | 640 | 29.4614 | 0.983110 | 0.079350 | 0.34432 | 0.12038 |
| projected cold additive base | 960 | 30.5081 | 0.988348 | 0.072225 | 0.29116 | 0.10004 |
| residual-pursuit additive | 1024 | 31.0818 | 0.989308 | 0.057393 | 0.14257 | 0.07428 |
| projected cold additive control | 1024 | 30.8585 | 0.989245 | 0.064812 | 0.27879 | 0.09851 |

All 64 cells and all integrity gates pass. Pursuit beats normalized N=640 by `+1.62037 dB` mean
PSNR, with paired gains from `+1.14979` to `+2.23016 dB`, while improving mean MS-SSIM, LPIPS,
pixel maximum, and 7x7 maximum. Every frozen aggregate and per-cell clause passes. Its displayed
pixel and 7x7 maxima also improve over the exact N=960 base in all 16 cells. The same-count cold
N=1024 control gains `+1.39705 dB` but fails the per-cell local clause, so allocation rather than
count alone explains the robust pass on this bank.

The endpoint has exactly 1,024 rows (`1.60x` normalized count), charges 480,000 base Gaussian-row
updates (`1.50x` normalized work proxy), 64 complete residual scans, and mean tail construction
time `0.0201 s`. No denominator, opacity, mass, source image, residual raster, pursuit coordinates,
optimizer state, or auxiliary payload persists. The N=960 base prefix is bit-exact in every cell;
maximum analytic/direct pursuit parity is `3.58e-7`.

## Native visual audit

Producer review covered both-seed native full sheets and error sheets for all eight sources, then
fixed-coordinate source/normalized/base/pursuit/cold comparisons around the initially suspicious
basket (`0822/s1`), squirrel (`0810/s1`), and fox (`0862/s1`) corrections. At native resolution
there is no material new lattice, checker, ringing, hole, wash, color lobe, blur, or tail speckle.
The magnified points align with real basket weave, eye/fur, and facial texture rather than arbitrary
pepper noise. At representative corrected coordinates, base-to-pursuit 3x3/7x7 RMSE falls from
`0.10944/0.09218` to `0.07530/0.08156`, `0.07141/0.05089` to `0.03113/0.02804`, and
`0.04606/0.03562` to `0.02699/0.02833`; their displayed center pixels match the source exactly.
The external native audit therefore closes the report's intentionally pending visual clause. The
immutable report itself is not rewritten after inspection.

## Results audit

The maintained report checker passes with the dirty-source provenance override. A separate
read-only recomputation over all persisted `analysis.npz` arrays found every reconstruction and
error finite; recomputed raw MSE differs from the ledger by at most `1.97e-11` and recomputed PSNR
by at most `2.51e-8 dB`. The manifest binds 1,334 files. Exact counts/work, shared-base hashes,
four-array payloads, projection rollback, pursuit contract, local non-regression, coefficient
bounds, and internal/cold/repeated render parity all pass; maximum maintained-render parity is
`4.30e-6`, below `2e-5`.

Repository verification passes Ruff, all 83 focused HIER-022--028 tests, and every structural
checker. The complete portable suite has `1,952 passed, 26 skipped, 9 failed`; those nine are the
unchanged inherited affine-condition, external-package subprocess-import, Torch-2.7 CUDA-property,
and descriptor-race failures, with no HIER-027/HIER-028 failure.

## Decision

Accept `residual_pursuit_additive_n1024` as a bounded, default-off pure-Gaussian solution for this
max-side-160 fidelity target. Normalization is not required to meet the frozen quality rule when
paying the measured 1.60x rows plus target-known residual allocation. It remains more row-efficient
and is still the maintained default: same-count N=640 additive is worse, the comparison is not
equal-byte, and neither full-resolution nor general-corpus/downstream behavior is established.

## Receipts

- Report: `results/hier028_div2kvalid8_s160_residual_pursuit_s01_confirmation_2026-08-11/index.html`
- Report checker: pass with `--allow-dirty`.
- Manifest SHA-256: `e9d36d18147bc46072aadf712b63ca63f041e8cd53fc038ea79726eade4b7c5e`
- Metrics SHA-256: `231f22f8c71f2d3d9c8e999fa7d4f4b1de7bb35e036616d36e75ce9ba6e67c63`
- Decision SHA-256: `368d99ce466acdbd8793f7d11eea13a0dc30e91f9fef1ecad33a6050a9c5f677`
- Bundle inventory: 1,334 manifest-bound files, 52 MiB.

## Limitations

Eight downscaled validation images; two seeds; one RTX 4090; dirty executed sources; producer-only
review; no distinct prospective reviewer; target-known encoder-side pursuit; unequal rows/work;
no complete bytes, adaptive stopping, full-resolution, downstream, actual-rate, or broad-corpus
result. The recipe is a research method, not a production pipeline or codec.
