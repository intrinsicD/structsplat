# HIER-010 residual-anchor / appearance-projection Janelle diagnostic

## Evidence boundary

This is a dirty-source, two-exposed-view, source-snapshotted diagnostic of the prospectively
frozen HIER-010 four-arm screen. It can test whether a residual-selected exact-leaf reserve and a
safe touched-row RGB projection improve the existing HIER-005 trajectory on these two rasters. It
cannot establish general quality, artifact freedom, a production default, convergence speed,
actual compression, Field V2 semantics, or FIT-046's general variable-projection result.

- Sources: Janelle `frame_00008/C0001` and `C0004`, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` and
  `26eb4cf24a034eb830198df6e7a6ac409ccb7cf4814ff645c71d0b6966b7070e`; masks SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3` and
  `4702bfa9df354f38e35a63207a37d4ec1b753afc4d0668bd905f3cdab320f35d`.
- Both native sources are 5,328x4,608. Deterministic 512x443 evaluation contains 15,929 and
  10,980 active mask pixels. Both views belong to the same previously consumed capture group.
- Protocol authority: `tasks/HIER-010-residual-anchor-projection.md`; all eight frozen cells ran
  once without parameter changes.
- Device/runtime: NVIDIA GeForce RTX 4090, Python 3.11.15, NumPy 2.2.4, torch 2.7.0+cu126,
  exact CUDA additive renderer. CUDA reductions are numerically, not bit, reproducible.

## Frozen pipeline

The control is HIER-005's exact-7,000 near-delta (`sigma=0.18 px`), hard-3-sigma,
progress-normalized touched-only contraction with 16x50 attempted recovery steps. The first pass
is also used to remeasure residuals. Pixel RGB MSE and mask-aware 7x7 mean MSE are divided by their
active q99, combined pointwise by a maximum, ranked stably, thinned by radius-one NMS, and filled
to exactly 350 source leaves. The anchor arm reruns the same contraction while preserving those
leaves at the same final row count.

The optional finish projects only topology-touched, non-protected RGB coefficients. Sparse-tile
forward and transpose products avoid a dense pixel-by-row matrix; means, scales, rotations,
untouched/protected RGB, alpha, topology, and count remain fixed. Matrix-free PCG uses a `1e-8`
ridge, `1e-6` tolerance, and at most 48 iterations. Step zero is retained, and no later checkpoint
may increase raw masked SSE, displayed normalized worst-pixel/7x7 violation, or coefficient
magnitude beyond 16.

## Results

All metrics below were independently recomputed from the persisted lossless fields. PSNR/MSE are
raw foreground metrics; MS-SSIM and LPIPS use the complete black-matted raster; pixel and 7x7
maxima use the exact displayed 8-bit raster.

| image | arm | N | PSNR | masked MSE | MS-SSIM | LPIPS | pixel max | 7x7 max | pipeline s | gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| C0001 | HIER-005 control | 7,000 | 50.097060 | 9.77899e-6 | 0.999976397 | 0.000028193 | 0.026404 | 0.009518 | 8.723 | fail |
| C0001 | control + projection | 7,000 | **50.108004** | **9.75438e-6** | 0.999976754 | 0.000027994 | 0.026404 | **0.009469** | 9.495 | fail |
| C0001 | residual anchors | 7,000 | 49.897467 | 1.02389e-5 | 0.999976993 | 0.000027088 | 0.034187 | 0.012141 | 16.760 | fail |
| C0001 | anchors + projection | 7,000 | 49.908700 | 1.02125e-5 | **0.999977171** | **0.000026622** | 0.034187 | 0.012089 | 17.480 | fail |
| C0004 | HIER-005 control | 7,000 | 54.374098 | 3.65250e-6 | 0.999991179 | 0.000007455 | 0.014847 | 0.004597 | 5.529 | pass |
| C0004 | control + projection | 7,000 | **54.378459** | **3.64883e-6** | 0.999991119 | 0.000007338 | 0.014847 | 0.004620 | 6.279 | pass |
| C0004 | residual anchors | 7,000 | 54.183591 | 3.81629e-6 | **0.999991477** | **0.000005643** | 0.013202 | 0.004387 | 10.662 | pass |
| C0004 | anchors + projection | 7,000 | 54.189306 | 3.81127e-6 | **0.999991477** | 0.000005682 | **0.013202** | **0.004375** | 11.354 | pass |

The coefficient projection is safe under the frozen SSE/maximum-normalized-violation transaction
but small. Applied directly to HIER-005, it improves C0001 by
`+0.010944 dB` (`-0.252%` MSE) and C0004 by `+0.004361 dB` (`-0.100%` MSE). It selects PCG
iterations 22 and 16, leaves the displayed pixel maxima unchanged, and changes the 7x7 maximum by
`-0.0000496` and `+0.0000227`. The latter remains below the local gate but demonstrates why the
stage-zero normalized-violation guard, rather than every raw submetric independently, is a narrow
transactional safety rule.

Hard residual anchoring is not robust. The full composition loses `0.188360 dB` / raises MSE
`4.433%` on C0001 and loses `0.184792 dB` / raises MSE `4.347%` on C0004. On C0004 it improves
pixel and 7x7 maxima by `0.001645` and `0.000222` and improves LPIPS, but on C0001 it worsens both
local maxima by `0.007783` and `0.002571`. Therefore neither image has the required strict MSE win,
and C0001 also fails both local non-regression clauses. The frozen full-mechanism gate fails and
HIER-005 remains unchanged.

## Mechanism, visual, and integrity audit

- Every arm reaches exactly 7,000 rows. Control/anchor touched counts are 3,334/3,467 on C0001 and
  1,366/1,597 on C0004; both anchor fields retain exactly 350 protected rows.
- Projection selects iterations 22/28 on C0001 and 16/10 on C0004 for control/anchor fields. Each
  projected cell records 99 forward and 51 transpose applications. Adjoint relative error is at
  most `6.42e-7`; cold maintained-render and repeat parity are at most `4.92e-7` and `1.19e-7`.
- Independent cold replay recomputed all 8x12 primary/perceptual/local metrics. Maximum PSNR drift
  was `2.82e-7 dB`, maximum masked-MSE drift `6.32e-13`, maximum MS-SSIM drift `5.97e-8`, maximum
  LPIPS drift `1.68e-8`, and all displayed local metrics matched exactly. Counts, canonical/file
  hashes, and touched/protected analysis masks also match all eight report rows.
- Full reconstruction review finds no gross new artifact visible at ordinary scale. Amplified
  errors remain concentrated along garment texture, silhouette, hair, and facial/hand structure.
  The 350 red anchor sites densely follow those same residual regions. Per-arm worst crops move to
  different locations, so they corroborate subtle residual redistribution but are not a registered
  pixelwise comparison. The frozen numeric local gate controls the disposition.
- The 252,352-byte canonical fields are 8.62x and 11.68x larger than the corresponding 29,263- and
  21,597-byte same-raster PNGs. These are storage references, not complete codec rates.

## Report and disposition

- Portable report: `results/hier010_residual_anchor_projection_janelle_2026-08-10/index.html`;
  170 files, about 15 MB, with all fields, histories, analysis arrays, source/reconstruction/error/
  anchor/center visuals, worst crops, metric curves, and executed-source snapshots.
- The bundle passes
  `python scripts/check_report_bundle.py results/hier010_residual_anchor_projection_janelle_2026-08-10 --allow-dirty`.
- Manifest SHA-256:
  `80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba`.
- Executed driver/refinement/contraction source SHA-256:
  `6c5544929e4f139b0062e0550def5d3634a3605f1b64b3c8c67d3253a173c3a5` /
  `dd209ecac9f48b4c4ece1bd9dfa311e87da4edc00d488a8ea7bd2354c9072d6b` /
  `755c67175b8480749871ade065b0d33bc100a3e15347edd2c77af2a8016b5801`.
- After sealing the report, self-review tightened the reusable driver's full-argument validation
  and parity clauses and made an already-over-limit stage-zero projection return its unchanged
  field instead of raising. The executed command satisfies the tightened validator, every recorded
  stage zero is within the bound, and all added parity clauses pass, so this hardening changes no
  recorded cell or decision. The bundled executed-source snapshots remain the run authority.

Retain the projection as a default-off, bounded coefficient cleanup and as useful FIT-046
implementation evidence; its measured gain is too small to motivate a default by itself. Reject
the fixed 5% hard residual-anchor composition on these consumed views. A successor should change
the local topology transaction itself—preserving or restoring a leaf only when its measured
replacement contribution clears pixel, patch, and global SSE guards—rather than reserve a fixed
global fraction and force the rest of the second contraction to absorb the displaced row budget.
Any promotion requires independently approved, unexposed capture groups and a distinct review.
