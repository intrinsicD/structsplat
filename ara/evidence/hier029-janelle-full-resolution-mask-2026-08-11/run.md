# HIER-029 Janelle full-resolution mask diagnostic

## Evidence class

Frozen, exposed-source, dirty-worktree, producer-reviewed diagnostic on canonical Janelle C0001.
It tests the HIER-028 ladder at StructSplat's established max-side-1200 Janelle regime under paired
full-frame and masked-foreground objectives. It is not a confirmation, a native-5328 run, or
evidence for a renderer/default change.

## Bound source and protocol

- RGB: native 5328x4608 `C0001.jpg`, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`.
- Mask: native 5328x4608 `mask_C0001.png`, SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Raster: deterministic Pillow LANCZOS/nearest max-side 1200, yielding 1200x1038; the thresholded
  mask has 87,639 active pixels (`7.035886%`).
- Device/seed: RTX 3050 8 GiB, CUDA, seed 0, required LPIPS, 256-row render chunks.
- Arms per objective: normalized N=640; projected cold additive N=960; exact projected N=960 plus
  64 fixed `0.35 px` HIER-028 pursuit rows; projected cold additive N=1024.
- Fit: 500 attempted Adam updates, unchanged HIER-028 learning rates, L1 + 0.3 SSIM, feature cap
  12 px, best-PSNR/final-count checkpoints every 25, hard three-sigma support, no fade or AA.
- Masked mode: black-matted initialization target, `build_masked_field(..., contain=False)`,
  mask-weighted L1/SSIM, mask-restricted coefficient projection, and mask-restricted residual
  selection. The mask remains encoder-only and no exact containment claim is made.

Command:

```bash
PYTHONPATH=src python scripts/experiments/hier029_janelle_mask_diagnostic.py \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
  /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
  results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11 \
  --max-side 1200 --seed 0 --device cuda --lpips
```

## Metrics

The generic columns below use the mode's objective domain: complete RGB for `full_frame`, and the
black-matted foreground bounding-box crop for `masked_foreground`.

| mode | arm | N | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max |
|---|---|---:|---:|---:|---:|---:|---:|
| full | normalized | 640 | 16.50281 | 0.605899 | 0.618100 | 0.99869 | 0.98438 |
| full | additive base | 960 | 13.64208 | 0.307167 | 0.969812 | 0.94516 | 0.90712 |
| full | pursuit | 1024 | 13.64684 | 0.307204 | 0.969693 | 0.89027 | 0.84028 |
| full | cold additive | 1024 | 13.99777 | 0.327762 | 0.960037 | 0.98438 | 0.97716 |
| masked | normalized | 640 | 25.21509 | 0.939742 | 0.202788 | 0.64110 | 0.46469 |
| masked | additive base | 960 | 22.36101 | 0.909250 | 0.231356 | 0.83400 | 0.57274 |
| masked | pursuit | 1024 | 22.61468 | 0.911817 | 0.224539 | 0.61809 | 0.50567 |
| masked | cold additive | 1024 | 22.41523 | 0.910113 | 0.231128 | 0.83400 | 0.58251 |

Full-frame pursuit gains only `+0.00476 dB` over its exact N=960 base, loses `0.35093 dB` to the
same-count cold control, and loses `2.85597 dB` to normalized N=640. It improves the base's sparse
worst residuals but cannot repair its frame-scale support failure.

Masked pursuit is meaningfully stronger within the additive family: `+0.25367 dB` over its N=960
base and `+0.19945 dB` over cold N=1024, with lower LPIPS and local maxima. It still loses
`2.60041 dB` to masked normalized N=640 and has worse MS-SSIM/LPIPS. Its worst single pixel is
slightly better than normalized (`0.61809` versus `0.64110`), but its 7x7 maximum is worse
(`0.50567` versus `0.46469`).

Mask supervision improves foreground PSNR relative to the same full-frame-trained arm by
`+8.96455`, `+1.54501`, `+1.79868`, and `+0.85143 dB` for normalized, base, pursuit, and cold
N=1024. Full-canvas PSNR for masked arms is only `3.109--3.181 dB`, as expected: the background is
black and explicitly outside the modeled objective.

## Projection, endpoint, and work audit

All six projection uses roll back. On the full frame, PCG lowers raw MSE by roughly 2.3% but
worsens MS-SSIM and LPIPS. On the masked arms the solver selects step zero or produces only
roundoff-scale raw-MSE movement, and strict-MSE/LPIPS safety rejects it. No unsafe proposal becomes
an endpoint.

Every additive field persists exactly means, log-scales, rotations, and signed RGB. Both pursuit
rows share their mode's exact projected N=960 base, append exactly 64 rows, preserve that prefix
bit-exactly, and retain no mask. Selection scans all 1,245,600 raster positions for the full arm
and restricts argmax selection to 87,639 active pixels for the masked arm; work accounting still
records 79,718,400 computed pixel evaluations. Maximum analytic pursuit parity is `2.38e-7`,
maximum endpoint-internal parity is `9.54e-7`, and maximum cold/repeated parity is `4.77e-7`.

Best-checkpoint behavior is an important confound. Full and masked normalized fields select update
1; masked additive fits select update 26; full additive fits select update 500. All attempted 500
updates and their declared row-update work, but the selected endpoints do not receive equal useful
optimization horizons. Full normalized terminal PSNR falls to `10.2314 dB` from its selected
`16.5028 dB`. In masked mode, the optimization loss is mask-weighted but the inherited checkpoint
policy evaluates global PSNR against the black-matted raster, so out-of-mask spill can affect model
selection. This is evidence of resolution-sensitive optimizer/horizon behavior and a checkpoint-
domain limitation, not permission to tune this exposed diagnostic.

## Native visual audit

The full-frame additive base, pursuit, and cold N=1024 fields show pervasive periodic dot/hole
coverage across the walls, lights, fixtures, and subject. Pursuit changes isolated extremes but is
visually almost indistinguishable from its base. Normalized N=640 is coarse and polygonal but does
not show the same global hole lattice and is substantially closer on full-frame metrics.

Masked outputs remove the irrelevant background by objective construction. Normalized is visibly
blockier and blurrier on the person; additive arms retain more clothing/face detail but show a
regular stippled texture. Masked pursuit reduces several bright error extrema relative to both
additive controls, consistent with its local metrics, but does not erase the stipple or match the
normalized perceptual/structural scores. There is no evidence of an error-map/report mismatch.

## Execution and validation

The first driver execution completed all four full-frame cells, then failed before masked scoring
because a CUDA mask tensor was passed into the NumPy `MaskGeometry` boundary. Its immutable partial
bundle is retained with suffix `_failed_driver_mask_tensor`. A tiny masked CUDA smoke run verified
the corrected NumPy-mask boundary, and the unchanged frozen 2x4 protocol was rerun from scratch.
This is an implementation correction, not outcome-dependent method tuning.

The corrected report contains 242 manifest-bound files (225 MiB) and passes:

```bash
python scripts/check_report_bundle.py \
  results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11 \
  --allow-dirty
```

Focused contraction/projection/pursuit/driver tests pass 55/55 before the evidence run. The full
repository verification result is recorded in the task after final synchronization.

## Decision

Reject extrapolation of HIER-028's max-side-160 positive to this Janelle full-resolution regime.
The 1.60x-row pursuit field is not a normalized substitute here. Retain the two mask hooks as
default-off research infrastructure because they work as specified and improve the additive
foreground result, but do not change the maintained renderer, fitter, pipeline, or mask-
containment defaults. A successor requires fresh sources plus resolution-scaled count/work and a
prospectively frozen optimizer schedule.

## Receipts

- Report:
  `results/hier029_janelle_c0001_s1200_mask_factorial_s0_diagnostic_2026-08-11/index.html`
- Manifest SHA-256: `b5c772fd2ab972b6169f3e08f720d8369797bf8b4d63ed6ee593b21b023aa52d`
- Metrics SHA-256: `dc8fd6d121a9db7e071b1c35f65d99d2d6e347031c2be0863dcbe4fc23a0cbfb`
- Decision SHA-256: `f5385ca52ee702b18497ba4d42e4abb406e3586ad3a9408562745ef9d3bd0456`
- Index SHA-256: `54175569c9b69999c8f1f6f2680bcb4bf6508b3540971f86da8e9d7f46315818`

## Limitations

One exposed image; one seed; one RTX 3050; dirty sources; producer-only protocol/outcome review;
1200x1038 rather than native 5328x4608; unequal counts/work; checkpoint-horizon mismatch; target-
known pursuit; global black-matted checkpoint selection for the masked loss; no equal bytes, actual
rate, downstream behavior, broad-corpus result, or distinct review. Foreground metrics are black-
matted crop metrics, not alpha-composited perceptual quality.
