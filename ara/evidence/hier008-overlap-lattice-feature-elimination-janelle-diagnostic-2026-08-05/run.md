# HIER-008 overlap-lattice / feature-elimination Janelle diagnostic

## Evidence boundary

This is a dirty-source, single-exposed-image, single-seed diagnostic of the frozen HIER-008 2x2.
It can reject exact mechanisms on this raster and verify that the optimizer and overlap prefit are
active. It cannot establish general quality, artifact freedom, competitive encoder speed, actual
compression, novelty, semantic selection, or a default.

- Source/mask: C0001 JPEG and exact mask, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` /
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`
- Native/evaluation raster: 5,328x4,608 JPEG, 14,268,226 bytes; deterministic 512x443 evaluation,
  15,929 foreground pixels and a 29,263-byte black-matted PNG.
- Protocol authority: `tasks/HIER-008-overlap-lattice-feature-elimination.md`; all eight frozen
  cells were run once and retained without parameter changes.
- Device/renderer: RTX 3050, exact CUDA additive renderer; topology and CPU solvers deterministic,
  CUDA optimizer accumulation not promised bit-exact.

## Frozen method

The support factor is a near-delta `sigma=0.18 px` pixel field versus a meaningfully overlapping
`sigma=0.50 px` field (axis-neighbour peak weight `0.1353`, diagonal `0.0183`). Before any removal,
matrix-free PCG solves signed RGB on the complete lattice under the actual masked finite-support
renderer. The topology factor is HIER-005 quadtree contraction versus a fixed-lattice survivor set
whose WSE crowding is divided by a local Schur removal price and whose radii/price protect structure-
tensor features and same-side RGB neighbourhoods.

Every reduced field receives the same 80-step all-row optimizer over RGB, centre, and log-scale,
with smoothed-error and feature weights plus a top-1% pixel tail. Means/log scales are limited to
`0.35 px` / `0.15`; coefficients are limited to absolute 16. A later checkpoint is selectable only
if SSE improves and neither raw worst-pixel nor raw worst-7x7 error exceeds step zero. The displayed
gate remains pixel maximum `<=0.02` and 7x7 maximum `<=0.01`.

## Results

| arm | N | reduction | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | opt step / gain | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| near-delta / quadtree | 8,192 | 1.94x | 35.129 | 0.998319 | 0.012203 | 0.10476 | 0.05734 | 80 / +0.634 dB | fail |
| overlap / quadtree | 8,192 | 1.94x | **45.953** | **0.999883** | **0.000543** | 0.10768 | **0.02534** | 50 / +2.144 dB | fail |
| near-delta / WSE-Schur | 8,192 | 1.94x | 16.471 | 0.959807 | 0.056154 | 0.63841 | 0.33216 | 0 / 0.000 dB | fail |
| overlap / WSE-Schur | 8,192 | 1.94x | 22.878 | 0.991541 | 0.052952 | 0.22032 | 0.13572 | 80 / +3.540 dB | fail |
| near-delta / quadtree | 4,096 | 3.89x | 27.714 | 0.994613 | 0.035185 | 0.20784 | 0.08888 | 80 / +2.662 dB | fail |
| overlap / quadtree | 4,096 | 3.89x | **31.096** | **0.996361** | **0.018931** | **0.10999** | **0.04678** | 80 / +3.725 dB | fail |
| near-delta / WSE-Schur | 4,096 | 3.89x | 13.222 | 0.913475 | 0.082159 | 0.77240 | 0.55546 | 30 / +0.000002 dB | fail |
| overlap / WSE-Schur | 4,096 | 3.89x | 17.607 | 0.973707 | 0.083198 | 0.55700 | 0.32042 | 80 / +2.502 dB | fail |

Meaningful overlap is a real positive factor inside both schedulers. It adds `10.824 dB` to the
8,192-row quadtree cell and `3.382 dB` at 4,096; it also sharply reduces distributed error (8,192
quadtree displayed q99 `0.02353` versus `0.07581`, and 7x7 maximum `0.02534` versus `0.05734`). The
worst isolated pixel does not improve, however, so the cell still fails closed.

The common optimizer is not inert. Six cells select a material later checkpoint, the overlap
quadtree gains `2.144/3.725 dB`, and selected raw pixel/patch maxima are below step zero. Mean shifts
stay `<=0.335 px`; log-scale changes stay `<=0.15`; the largest final coefficient is `3.164`, far
below the stability veto. The exact overlap prefit itself converges in 22 PCG iterations and 1.29 s,
with `2.22e-8` maximum pixel error, coefficient maximum `0.756`, and only `8.37e-5` negative RGB
fraction. There is no prefilter ringing or coefficient blow-up.

## Mechanism audit

The static fixed-scale WSE/Schur formulation is decisively rejected. It retains all top-decile
feature pixels within 1.5 px at 8,192 rows, so its failure is not explained by simply deleting the
edge set. It leaves q99/max nearest-centre distances `1.44/2.08 px`. At `sigma=0.5`, the nearest
single-Gaussian response at those distances is only about `0.016/0.00017`; even the allowed
`exp(0.15)` scale increase cannot close those flat-region holes. The saved image therefore contains
the predicted dot/lattice pattern. Static initial Schur price plus feature coverage is not a
substitute for dynamically expanding or merging the survivor covariance and re-evaluating actual
post-removal distortion.

The overlap quadtree avoids that catastrophic hole mechanism because moment contraction expands
support while replacing children. At 8,192 its q99/max centre distances are `0.775/1.555 px` and the
full image looks close, but residual hotspots remain on thin fabric/hair/boundary structure. At
4,096, square/ring impressions across the garment are unambiguously visible. Thus overlap solves
distributed coverage, not the protected-leaf/outlier problem.

The contextual strongest HIER-005 touched-recovery row remains better at 8,192: 52.356 dB and a
passing `0.01485/0.00532` local gate versus HIER-008 overlap/quadtree's 45.953 dB and failing
`0.10768/0.02534`. At 4,096 the new overlap/quadtree cell improves the contextual HIER-005 global
PSNR (31.096 versus 30.481) and local maxima (0.10999/0.04678 versus 0.20584/0.07074), but still has
visible periodic structure. This context is not an equal-work factorial: HIER-005 interleaved
16x50 touched-recovery attempts, while HIER-008 uses one common terminal 80-step block.

## Rate and performance boundary

- At 8,192 rows the canonical/estimated field is 290,496 bytes and the lossless NPZ is 293,596
  bytes: about `9.93x` larger than the same-raster 29,263-byte PNG. At 4,096 they are 159,424 and
  162,524 bytes, still about `5.45x` larger than that PNG.
- The recorded native-JPEG/estimated ratios `49.12x` and `89.50x` are deliberately visible but not
  valid compression ratios: the 14.27 MB JPEG is 5,328x4,608 while the field is 512x443.
- Cell totals are 0.63--9.77 s after shared setup; full overlap prefit is 1.29 s and each feature
  elimination pass about 1.56 s. This is useful reference latency, not a competitive encoder-speed
  claim or matched-work comparison.

## Integrity, reports, and disposition

- Portable report: `results/hier008_janelle_overlap_elimination_2026-08-05/index.html`; 242 files,
  20 MB, complete fields/histories/visuals and snapshot/optimizer curves. It passes
  `python scripts/check_report_bundle.py ... --allow-dirty`.
- Manifest SHA-256: `755d51a4332be452fe089b3e0c98e689c6a5052c1d302ca71d824650897e813f`.
- Executed module/driver SHA-256:
  `e5ed8d43a4a59e0f2ac39a5edbfc3c689bc2d32a34ce47bd4b676d0df367b02b` /
  `7afd8be580541c312865b4908d511712a8f5a5c44e73a488890c8f6a22482022`.
- Current focused-test/checker SHA-256:
  `8efad0fa52026a509fe507b5820b29276c5de4e7cb1c89579d81a8a94769f5c4` /
  `35f2a2c6cb15b42aee623eedcd7f97f12876d145d5b33239ae79fe02676ef364`.
- Focused overlap plus HIER-005 regression slice: 56 passed. Full repository verification is
  recorded in the task handoff after documentation closure.

Retain exact overlap prefitting and expanding quadtree contraction as useful components. Reject
fixed-lattice, fixed-scale WSE/Schur elimination on C0001. A bounded successor should add overlap
prefitting to HIER-005's interleaved touched recovery, preserve thin/high-feature leaves with a hard
veto or reserved detail rows, and only revisit WSE if every removal dynamically re-optimizes
covariance plus local appearance and is accepted on actual patch distortion. No fresh production
pipeline or default change is justified.
