# HIER-009 dynamic overlap / neighborhood-recovery Janelle diagnostic

## Evidence boundary

This is a dirty-source, single-exposed-image, single-seed diagnostic of the prospectively frozen
HIER-009 four-arm screen. It can test the specified mechanisms on this raster and verify that the
dynamic optimizer, direct-neighbor scope, and protected-leaf invariants are active. It cannot
establish general artifact freedom, a production default, competitive encoding speed, actual
compression, novelty, or a selected Field V2 semantic.

- Source/mask: C0001 JPEG and exact mask, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` /
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Native/evaluation raster: 5,328x4,608 JPEG, 14,268,226 bytes; deterministic 512x443 evaluation,
  15,929 foreground pixels and a 29,263-byte black-matted PNG.
- Protocol authority: `tasks/HIER-009-dynamic-overlap-neighborhood-recovery.md`; all eight frozen
  cells were run once and retained without parameter changes.
- Device/renderer: RTX 3050, exact CUDA additive renderer; topology and CPU solvers deterministic,
  CUDA optimizer accumulation not promised bit-exact.

## Frozen method

The control starts from the historical near-delta `sigma=0.18 px` source-RGB lattice. The three
overlap arms start from an exactly PCG-prefiltered `sigma=0.50 px` pixel lattice. All arms then use
HIER-005's live transaction: propose hard, parent-plus-detail, and exact-count pair actions from the
current field; exactly refit and score local finite-support distortion; commit a support-disjoint
batch; run one bounded recovery checkpoint; rebuild geometry-dependent proposals; and repeat.

`overlap_touched` optimizes only topology-touched active rows. `overlap_halo` additionally optimizes
active rows whose rounded centers lie in the direct Chebyshev-radius-one 3x3 neighborhood of each
newly touched row. Only accepted changed neighbors remain eligible later. The protected arm reserves
exactly 5% of the target rows using deterministic radius-one NMS over the maximum of normalized
structure-tensor energy and normalized RGB high-pass magnitude. Protected RGB may be refit, while
protected means and covariances are restored exactly after every step. Every arm receives 16x50
attempted Adam steps with identical learning rates and trust regions.

## Results

| arm | N | reduction | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | persistent halo | protected | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| delta / touched | 8,192 | 1.94x | **52.338** | **0.999981** | **0.000016** | **0.01485** | **0.00530** | 0 | 0 | pass |
| overlap / touched | 8,192 | 1.94x | 47.395 | 0.999916 | 0.000192 | 0.06815 | 0.02294 | 0 | 0 | fail |
| overlap / 3x3 halo | 8,192 | 1.94x | 45.963 | 0.999872 | 0.001123 | 0.05513 | 0.02736 | 1,946 | 0 | fail |
| overlap / halo + protect | 8,192 | 1.94x | 46.991 | 0.999905 | 0.000487 | 0.05168 | 0.02049 | 1,951 | 410 | fail |
| delta / touched | 4,096 | 3.89x | 30.547 | 0.996966 | 0.022970 | 0.20729 | 0.07086 | 0 | 0 | fail |
| overlap / touched | 4,096 | 3.89x | 39.802 | 0.999421 | 0.002833 | 0.09003 | 0.03335 | 0 | 0 | fail |
| overlap / 3x3 halo | 4,096 | 3.89x | 40.801 | 0.999569 | 0.002633 | **0.07989** | 0.02781 | 836 | 0 | fail |
| overlap / halo + protect | 4,096 | 3.89x | **41.115** | **0.999576** | **0.002327** | 0.08583 | **0.02511** | 835 | 205 | fail |

The direct-neighbor scope is useful at the aggressive 4,096-row endpoint. Relative to
overlap/touched, it gains `+0.999 dB`, lowers the displayed pixel maximum from `0.09003` to
`0.07989`, lowers the 7x7 maximum from `0.03335` to `0.02781`, and visually removes the obvious
block/quadtree lattice. The remaining error is lower-amplitude but distributed over garment
texture and boundaries, so this is not artifact-free.

At 8,192 rows the same halo is a mixed or negative factor: pixel maximum improves
`0.06815 -> 0.05513`, but PSNR drops `1.433 dB`, q99 worsens, and the 7x7 maximum rises
`0.02294 -> 0.02736`. The optimizer is spreading local error rather than eliminating it. Feature
protection repairs part of that trade-off: versus the unprotected halo it gains `+1.028 dB` and
reduces the 7x7 maximum to `0.02049` at 8,192; at 4,096 it gains `+0.314 dB` and lowers the 7x7
maximum to `0.02511`, while worsening the isolated pixel maximum. Only delta/touched at 8,192
passes the frozen `0.02/0.01` displayed pixel/7x7 gate.

## Mechanism and invariant audit

- Every one of the 16 recovery checkpoints was accepted in every cell. Halo checkpoints optimized
  as many as 4,514 active rows, included up to 1,951 direct neighbors, and accepted 5,725--6,970
  newly changed neighbor events across a trajectory. Recovery is therefore active, not a no-op.
  The reported sum of local checkpoint PSNR gains is attribution along changing topologies and
  must not be read as a terminal optimizer delta.
- The 410/205 protected leaves survive exactly at 8,192/4,096 rows. Maximum protected-geometry
  error is `0.0 px`; 69/38 locally overfull protected regions fail closed while other regions still
  reach exact target count.
- The overlap PCG prefit converges in 22 iterations with coefficient absolute maximum `0.7563`.
  Maintained-render parity is at most `7.16e-7`; repeated-render parity is at most `1.79e-7`.
- Top-feature coverage within 1.5 px remains 1.0 at 8,192 and at least 0.9937 at 4,096. The residual
  failure is not simple feature deletion; it is the interaction of contraction support, local
  coefficient/geometry freedom, and hard finite-support error redistribution.
- Visual inspection confirms the metric diagnosis. Delta/touched at 4,096 has a conspicuous square
  lattice. Overlap removes most of it, and the halo/protected arms further soften it, but visible
  fine texture/ring residuals remain in amplified errors and worst crops.

## Rate and performance boundary

- The reduction counts only active mask pixels: `15,929 -> 8,192` is `1.94x`, and
  `15,929 -> 4,096` is `3.89x`.
- Canonical/estimated fields are 290,496 and 159,424 bytes; lossless NPZ files are 293,596 and
  162,524 bytes. The canonical fields are therefore about `9.93x` and `5.45x` *larger* than the
  same-raster 29,263-byte PNG. The native-JPEG ratios are resolution-mismatched and invalid as
  compression evidence.
- Per-cell total time is 10.55--14.50 seconds after shared setup. Attempted optimizer steps are
  matched, but halo arms optimize more rows, so this is not a matched-FLOP convergence-speed claim.

## Integrity, reports, and disposition

- Portable report: `results/hier009_janelle_dynamic_overlap_recovery_2026-08-06/index.html`; 243
  filesystem entries including the manifest, about 24 MB, with all eight fields, histories,
  source/prefit/reconstruction/error/feature/protected/center/worst-crop visuals, and snapshot plus
  recovery curves. It passes `python scripts/check_report_bundle.py ... --allow-dirty`.
- Manifest SHA-256:
  `c1d8c488c8200edd8b4d68c103b41f9347c8844a5c4c41607431fbd4be67f60d`.
- Executed driver/contraction/overlap source SHA-256:
  `3ed96bbed503288c0d705c0a2327b8814c5e69ca8d07356476674f6b2144fed5` /
  `558127380e3b05c9e763fc17c824289973464c8ed204e948242632c85f8477ea` /
  `a7667c023669e99e4cc26f95806399abcc0898b08f1ed40abe7e30be10c89c30`.
- The combined pixel-contraction, overlap, and HIER-009 focused slice passes 64 tests. Full
  repository verification passes 1,662 tests with 4 skips before the final ARA-only epilogue; all
  ARA/docs/task/workflow structural checks pass again after that epilogue.

Retain the 3x3 halo as a useful aggressive-reduction mechanism and the hard feature reserve as a
promising local-artifact control. Do not replace the 8,192-row delta/touched fallback, promote a
default, or claim compression. A successor should make recovery acceptance explicitly local-
artifact/Pareto-aware and adapt halo freedom to contraction severity rather than applying it
uniformly.
