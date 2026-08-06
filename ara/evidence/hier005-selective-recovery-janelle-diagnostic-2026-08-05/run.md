# HIER-005 selective-recovery Janelle diagnostic

## Scope and verdict boundary

This is a dirty-worktree, one-exposed-image, downscaled implementation diagnostic requested after
the first HIER-005 visualization showed square/quadtree artifacts. It tests whether interleaving
short optimizer blocks over only contraction-touched rows can repair those artifacts while leaving
never-touched pixel rows fixed. It is not preregistered, independently reviewed, held out,
equal-wall-time, a maintained benchmark bundle, a semantic/default decision, or a compression
result.

The implementation, driver, focused test, and synchronized architecture documents are bound in
path order by source-set SHA-256
`2f4f42bd139be98c27444e598b1ab7fdea17b1771438be7588104d4546e2d778`. HIER-005 remains
`in-review` with a distinct numerical/scientific reviewer required.

## Source and evaluation raster

- RGB source:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg`
- RGB SHA-256: `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`
- Supplied RGB bytes: `14,268,226`
- Native dimensions: `5,328x4,608`
- Mask:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png`
- Mask SHA-256: `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`
- Evaluation dimensions: `512x443`, produced with logged Pillow LANCZOS RGB and nearest mask
  resampling
- Evaluation active pixels: `15,929`
- Same-raster evaluation PNG bytes: `29,263`
- Metric domains: PSNR/MSE use the thresholded foreground mask; SSIM, MS-SSIM, and LPIPS use the
  complete black-matted evaluation raster.

The native JPEG numerator is resolution-mismatched to the resized field. Only the same-raster PNG
comparison is relevant to this diagnostic's storage sanity check, and neither denominator converts
the uncoded field payload into an actual codec rate.

## Selected method

The recovery lane runs after 1/16 increments of requested row reduction. At each checkpoint it
optimizes all active rows ever produced or retained by a contraction for up to 50 Adam steps using
the maintained direct additive CUDA renderer. Means/scales/rotations/coefficients use learning
rates `0.005/0.003/0.001/0.003`, with per-checkpoint trust limits of `1.5 px`, `0.35` log-scale, and
`0.35 rad`. Never-touched active pixel rows are a detached fixed base. The best masked-SSE step is
accepted only on a strict improvement; an accepted geometry step rebuilds all ready proposals.

The progress schedule replaced an action-count cadence after a preliminary sweep exposed a work
confound: lower target counts triggered more checkpoints, making the rate--distortion curve
non-comparable. The selected four rows each attempt exactly `16 x 50 = 800` optimizer steps.

## Command

```bash
PYTHONPATH=src python scripts/experiments/hier005_pixel_contraction.py \
  --images /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  --mask /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  --out results/hier005_janelle_c0001_selective_recovery_progress16_2026-08-05_v2 \
  --target-gaussians 2048 4096 8192 12000 \
  --max-side 512 --device cuda --renderer cuda_additive --lpips \
  --recovery-steps 50
```

The no-recovery comparator is
`results/hier005_janelle_c0001_diagnostic_2026-08-05_v3`. It was executed before the recovery code
was added, so this is not a same-clean-revision formal control. The default-off topology path is
covered by regression tests, and its repeated pre-recovery field hashes remained stable across the
diagnostic regenerations, but the source-set difference remains an evidence limitation.

## Diagnostic outcomes

| N | baseline PSNR | recovered PSNR | delta dB | baseline LPIPS | recovered LPIPS | SSE reduction | total s baseline -> recovery | touched / untouched |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 22.4088 | 29.1801 | +6.7713 | 0.036475 | 0.016161 | 78.969% | 9.990 -> 15.654 | 1,987 / 61 |
| 4,096 | 25.0461 | 30.5225 | +5.4764 | 0.045503 | 0.023314 | 71.662% | 8.600 -> 12.373 | 3,581 / 515 |
| 8,192 | 34.0755 | 52.3390 | +18.2634 | 0.015007 | 0.00001648 | 98.508% | 5.444 -> 9.140 | 2,723 / 5,469 |
| 12,000 | 47.1617 | 57.2446 | +10.0829 | 0.0002956 | 0.00000570 | 90.189% | 2.870 -> 7.126 | 1,315 / 10,685 |

Every row reached its exact count; all 16 checkpoints were accepted. Maintained-render maximum
absolute parity error ranged from `4.02e-7` to `1.31e-6`, below the existing `2e-6` diagnostic
tolerance. At N=8,192, the visible square/tree holes in the baseline reconstruction disappear and
the displayed error becomes nearly uniform-dark. N=2,048 and N=4,096 retain conspicuous
blob/cell structure, so recovery does not eliminate the low-capacity limitation.

The uncoded eight-float row estimate is `93,888`, `159,424`, `290,496`, and `412,352` bytes. Thus
the same-raster 29,263-byte PNG is respectively `3.21x`, `5.45x`, `9.93x`, and `14.09x` smaller.
Selective recovery improves distortion at a fixed float-row count; it does not solve compression.

## Repeatability and timing limits

CUDA recovery is numerically non-bit-reproducible because atomic-gradient accumulation can change
the optimizer trajectory. A second complete progress-16 run changed all four canonical field
hashes. Its PSNR difference relative to the preserved result was `0.2051 dB` at N=2,048,
`0.0228 dB` at N=4,096, `0.00257 dB` at N=8,192, and `0.00451 dB` at N=12,000. CPU focused tests
are bit-deterministic after elapsed time is removed from telemetry. Reported wall times are one
live RTX-3050 workstation observation and support no speed or convergence-performance claim.

## Artifact receipts

- Baseline metrics SHA-256:
  `9ef7beabce601db14ba081ea8f4900b6e5f5e0dfd1cc2ac08156c68f82a2c927`
- Baseline manifest SHA-256:
  `2493ccc1438819ea87e6b4748f48105ebff70da4f30353f8a222a6e60d387f69`
- Recovery metrics SHA-256:
  `fde416036a2a018030a0976f65c45917687ad942a92cff7a7e42cc8e5f637c66`
- Recovery manifest SHA-256:
  `18c509ee011b3afec654282d22998860f64917ecebf4b283805c071a2bf77c0f`
- Recovery HTML SHA-256:
  `d7124c70a854f081d484127d22908e99f1256afe8357e8682f49ef0007db39a1`
- Recovery manifest: 75/75 hashes verified; HTML: 56/56 local links resolved; curves: 39
  standalone SVG outcome plots.
- Executed driver/core snapshots inside the recovery report match SHA-256
  `07ce1ca11c0eb8856a8875a7fc9dac051a878af6cd98fae49bb294d63dbe2ee4` and
  `3532e26861dcc9bb9f004888f1ec670f2f55999f4975fcf45e315912ce2fd56c`.

## Verification

- Focused HIER-005 suite: `20 passed`.
- Pixel/Field-V2/render regression slice: `86 passed`.
- `./scripts/verify.sh`: `1,580 passed`, `4 skipped`, `514 deselected`; lint and every structural
  checker passed.

## Required next evidence

A distinct reviewer must reproduce the freeze invariant, SSE ledger, progress schedule, and CUDA
variance. A prospective multi-image protocol must compare fixed recovery work and wall time against
fixed-N/global/regional/current-pipeline controls. COMP-013/FIT-030 must measure a self-contained
quantized and entropy-coded stream before any compression or rate--distortion claim.
