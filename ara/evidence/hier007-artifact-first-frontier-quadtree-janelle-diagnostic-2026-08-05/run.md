# HIER-007 artifact-first frontier-quadtree Janelle diagnostic

## Evidence boundary

This is a dirty-source, single-exposed-image, single-seed diagnostic of one frozen HIER-007
mechanism factorial. It can reject the exact combined mechanism on this raster and verify the
reference implementation. It cannot establish general image quality, perceptual artifact freedom,
convergence speed, production performance, compression, semantic selection, novelty, or a default.
HIER-005 and HIER-006 rows are retained context, not jointly rerun equal-work controls.

- Source: `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg`
- Mask: `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png`
- Source/mask SHA-256: `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` /
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`
- Native source: 5,328x4,608 JPEG, 14,268,226 bytes
- Evaluation raster: 512x443, 15,929 active-mask pixels; exact evaluation PNG 29,263 bytes
- Seed/device: deterministic topology seed 0; RTX 3050, CUDA renderer
- Protocol authority: `tasks/HIER-007-artifact-first-frontier-quadtree.md`

## Frozen method

All arms clone one hash-identical level-6 mask-moment base. A split removes its active parent and
adds all mask-present children, so active nodes remain a mask-partition antichain. Geometry is
fixed. The two axes are smoothed residual energy versus worst pixel/7x7-patch priority, and
new-child-only versus finite-support-overlap RGB reconciliation. Optimized rows receive bounded
post-Adam weights from sigma-1.5 mask-normalized smoothed residual exposure. A cold full render
accepts only by the float32-tolerant raw pixel/patch-violation-then-SSE key; rejected batches back
off by deterministic priority-prefix halving and rejected singletons block. The displayed gate is
pixel RGB-RMSE max <=0.02 and maximum complete black-matted 7x7 RMSE <=0.01.

The exact command is preserved in both manifests and in the task protocol. No parameter was
changed after outcomes were opened.

## Terminal results

| arm | active / stored | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | gate | arm time | event proxy |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| energy / new-only | 8,192 / 11,333 | 40.0348 | 0.999330 | 0.001023 | 0.047168 | 0.021467 | fail | 357.0 s | 165,765 B |
| artifact-first / new-only | 8,192 / 11,278 | 34.5693 | 0.998277 | 0.003716 | 0.312440 | 0.119657 | fail | 369.8 s | 165,098 B |
| energy / overlap | 8,192 / 11,303 | 38.8300 | 0.999633 | 0.001753 | 0.180037 | 0.094417 | fail | 536.4 s | 994,841 B |
| artifact-first / overlap | 8,192 / 11,241 | 26.0347 | 0.990103 | 0.022401 | 0.348476 | 0.172236 | fail | 1,132.8 s | 1,206,214 B |

At the approximately 4,096-row snapshots, the four HIER-007 PSNRs are 26.439, 25.456, 28.745,
and 24.483 dB in table order; none passes. Contextual HIER-006 reaches 32.882 dB and
0.107301/0.037518 at 8,192 and also fails. Contextual HIER-005 hard3/touched reaches 52.356 dB and
0.014847/0.005315 at 8,192 and passes. Parent replacement therefore removes a real retained-
ancestor penalty—the energy/new-only arm gains 7.153 dB over HIER-006—but is not sufficient.

## Mechanism audit

The combined arm is decisively negative. It makes 1,773 replacement attempts for 279 accepted
stages, including 1,382 batch backoffs and 112 blocked singletons. Of the accepted stages, 215
select an optimizer checkpoint above step zero, so overlap fitting is active rather than a no-op.
Artifact-first selection concentrates capacity on the current global maximum; overlap updates can
move that maximum into another still-coarse cell, and complete-child growth leaves no positive-net
capacity to repair the late hotspot. Its terminal worst pixel lies in a level-3 cell despite 6,579
level-0 active rows. The first shared replacement also demonstrates objective mismatch: raw
normalized violation falls 46.270 to 45.208 while SSE rises about 1,310.39 to 1,690.94.

Visual inspection agrees with the metrics. Energy/new-only is comparatively clean but retains a
fine boundary/hair defect. Artifact-first/overlap contains strong square and ring imprints across
clothing and the face. This rejects the frozen hard artifact-first plus overlap-local policy on
C0001; it does not reject quadtree scheduling or parent replacement generally.

## Integrity and rate boundary

- Independent terminal cold replay differs by at most `6.2e-8 dB` PSNR and `3.9e-8` displayed
  maximum; all saved rollback, antichain/partition, untouched-row, and shared-base checks pass.
- Shared canonical base hash:
  `5018da29360ee3a9645761e2def894ff831887251e1af1d33d759694eb676102`.
- Every terminal canonical active field is 290,496 bytes; lossless NPZ is about 293,596 bytes.
  The field is therefore about 9.93x larger than the 29,263-byte same-raster PNG.
- Final-frontier structural proxies are about 128 kB and still exceed PNG. Event proxies charge
  coefficient revisions and rise to 0.995--1.206 MB for overlap arms. These omit a complete
  decoder/entropy model and are explicitly not codec rates.
- The apparent roughly 49.1x source-JPEG/field ratio is invalid for codec comparison because the
  native JPEG and 512x443 evaluation raster have different resolutions.

## Reports and source receipts

- Frozen executed bundle: `results/hier007_janelle_artifact_first_quadtree_2026-08-05/index.html`
- Authoritative visual handoff: `results/hier007_janelle_artifact_first_quadtree_contextfix_2026-08-05/index.html`
- The second bundle is packaging-only: it restores HIER-005 rows whose status is `diagnostic`,
  preserves all original fields/metrics/curves/trajectories, retains the executed source snapshot,
  and records the correction in its manifest. Both pass `check_report_bundle.py --allow-dirty`.
- Executed/current module SHA-256:
  `cfa628f6da5649c1c9272e3cf3a2361b213b4ae84ef9ccb9a3fc50c40fb28599`
- Executed driver SHA-256:
  `4a62b7896e6b01b0cf97db6b8a3d388d345a6a252f872284142faaf29adc9e43`
- Current packaging-fix driver SHA-256:
  `60d7848bd1840778838a3f7e4cfab52059b12cb6fd51ed175b0c1bf753a2b0ce`
- Current tests SHA-256:
  `31dd42b9e7ce3fc73ade8602d91c0b0f99999963b2acf2986f48fca4270d9193`

## Verification and disposition

- `python -m pytest -q tests/test_artifact_first_quadtree.py`: 18 passed.
- Relevant field/render regression slice: 142 passed.
- `./scripts/verify.sh`: 1,636 passed, 4 skipped, 514 deselected; Ruff and every structural
  checker passed.

Retain energy/new-only parent replacement as the structural HIER-007 control. Reject and do not
retune the combined artifact-first/overlap policy on C0001. Any successor should prospectively
test a smooth commit-aligned pixel/patch-tail objective, regional no-new-hotspot and material-SSE
trust regions, reserved late repair capacity, and continuous parent-to-child transfer before
spending effort on caching or fused kernels.
