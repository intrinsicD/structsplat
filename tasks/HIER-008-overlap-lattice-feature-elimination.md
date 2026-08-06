# HIER-008 — Overlap lattice and feature-safe elimination

## Context

HIER-005 can preserve the exposed C0001 image at 8,192 rows, but its `0.18 px` pixel leaves are
numerically delta functions: an axial neighbour receives only about `2e-7` of the peak. HIER-006
and HIER-007 show that adding retained or replacement quadtree parents does not remove the visible
grid failure. The next bounded question is whether real local overlap plus feature-aware sample
elimination gives appearance coefficients somewhere useful to move, while retaining an explicit
artifact veto.

## Goal

Implement and expose a deterministic 2x2 diagnostic that factors lattice support
(`near_delta=0.18 px`, `overlap=0.50 px`) against simplification scheduler (`quadtree`,
`feature_wse_schur`), performs exact/matrix-free appearance prefitting before reduction, applies
the same bounded post-reduction optimizer to every arm, and reports quality, local artifacts,
optimizer contribution, stability, time, and byte proxies at 8,192 and 4,096 rows.

## Method contract

- Pixel centres use integer `xy`, peak-one direct-additive Gaussians, signed RGB coefficients, a
  hard axis-aligned `3 sigma` support box, and the source mask as exact output alpha.
- Every support width first receives a matrix-free least-squares appearance solve on the full
  active lattice. Source RGB is never reused unchanged for the overlap arm. The report records
  residual, iteration, coefficient amplification, and negative-coefficient diagnostics.
- `quadtree` uses HIER-005's deterministic local contraction and exact discrete local coefficient
  resolve, initialized from the solved lattice coefficients.
- `feature_wse_schur` keeps lattice geometry while eliminating points. Its dynamic WSE crowding
  score uses structure-tensor-adaptive radii. A removal is protected by the source feature energy
  and priced by the squared RGB coefficient times the residual energy of projecting that basis
  onto same-side immediate neighbours (a local Schur-complement proxy). Thus crowded flat samples
  disappear before isolated or feature-bearing samples. Survivor coefficients are globally
  re-solved on the actual masked finite-support renderer.
- All four arms use the same post-reduction optimizer: all surviving RGB, centre, and log-scale
  variables may move, with centre and log-scale trust regions. The pixel objective combines
  masked RGB error, smoothed-error weighting, structure-tensor weighting, and a top-tail term.
  Checkpoint zero is always retained. A later checkpoint is selectable only when masked SSE is
  lower and neither raw maximum pixel RMSE nor raw maximum 7x7 patch RMSE exceeds checkpoint zero
  beyond float32 comparison tolerance. This optimizer is diagnostic and does not alter defaults.
- The estimated 32-byte row payload, canonical lossless NPZ, and native/evaluation image ratios
  remain separate. None is a complete coded-stream or compression claim.

## Non-goals

- Selecting a production default, claiming novelty, or replacing the normalized maintained path.
- Claiming actual rate without COMP-013's complete grammar, quantization, headers, and entropy
  coding.
- Retuning after C0001 outcomes, accessing held-out data, or treating a dirty single-image run as
  confirmatory evidence.

## Acceptance criteria

- [x] Typed NumPy-first APIs validate overlap prefitting, feature/Schur elimination, exact counts,
      deterministic survivor nesting, and coefficient/geometry stability.
- [x] The common optimizer records step-zero and later checkpoints, selected-step parameter
      movement, objective change, exact cold-render metrics, and rollback behavior.
- [x] Synthetic flat, step-edge, diagonal-edge, gradient, and checkerboard tests demonstrate
      exact initialization, feature retention, count reduction, and artifact-safe optimization.
- [x] The frozen C0001 2x2 diagnostic below is executed without outcome-driven retuning and emits
      source/reconstruction/error/feature/survivor visuals, all requested curves, raw tables,
      configs, histories, field payloads, source snapshots, and a browsable `index.html`.
- [x] `python scripts/check_report_bundle.py RESULTS_DIR`, focused tests, self-review, and
      `./scripts/verify.sh` pass before handoff.

## Interfaces touched

`src/structsplat/overlap_elimination.py`, `src/structsplat/pixel_contraction.py`,
`scripts/experiments/hier008_overlap_elimination.py`, `tests/test_overlap_elimination.py`,
`docs/architecture.md`, `tasks/INDEX.md`, and the HIER-008 diagnostic/evidence bundle.

## Depends on

HIER-005/006/007, CORE-013, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This exposed dirty-tree
diagnostic has no prospective distinct protocol review and cannot support a promoted claim.

## Frozen exposed-image diagnostic protocol (2026-08-05, before HIER-008 outcomes)

- Question: at identical exact row counts, does meaningful pixel-neighbour overlap and/or a
  feature-protected WSE/Schur elimination order improve the quality--artifact--reduction tradeoff
  over the near-delta quadtree reference, and does the common optimizer produce a measurable safe
  gain?
- Data: C0001 JPEG and binary mask from
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`, resized by the existing
  HIER driver contract to `512x443` (`15,929` foreground pixels). The driver records native and
  evaluation hashes. C0001 is exposed development data; no validation or confirmation data may be
  accessed.
- Arms: Cartesian product of support `{near_delta=0.18, overlap=0.50}` and scheduler
  `{quadtree, feature_wse_schur}`. Seed `0`; signed coefficients; cutoff `3.0`; fade `0.0`;
  targets `{8192,4096}`. No arm may be dropped after execution.
- Prefit: normal-equation PCG tolerance `1e-8`, maximum `200` iterations, diagonal ridge `1e-8`.
  WSE uses alpha `8`, structure density floor `0.20`, density power `0.50`, radius clip
  `[0.65,2.25]`, RGB barrier `0.10`, feature protection `4.0`, and Schur ridge `1e-6`. Its
  un-clipped base radius is `sqrt(active_pixels / (pi * terminal_target_count))`; local density
  divides that radius by the square root of its foreground-mean-normalized value.
- Common optimizer: `80` attempted Adam steps, checkpoints every `10`; learning rates RGB `0.01`,
  means `0.003`, log scales `0.002`; maximum centre shift `0.35 px`, maximum absolute log-scale
  shift `0.15`; smoothed-error sigma `1.5 px`, error weight `2.0`, feature weight `2.0`, top `1%`
  tail weight `2.0`. Fixed seed `0`, CUDA additive renderer, chunk `256`.
- Metrics: foreground raw MSE/PSNR/SSE; full black-matted SSIM/MS-SSIM/LPIPS; displayed pixel
  RMSE q99/q99.9/max and maximum 3/7/15/31-pixel patch RMSE; provisional displayed C0001 gate
  pixel max `<=0.02` and 7x7 max `<=0.01`; raw maxima; optimizer pre/post gain, selected step,
  parameter displacement, and weighted objective; feature-retention and hole-radius diagnostics;
  PCG convergence and coefficient stability; wall time; canonical/lossless/estimated byte ledgers
  and source ratios. Counts, quality metrics, optimizer checkpoints, and timing are plotted.
- Killing rules: reject an arm as unsafe if it misses the exact count, is non-finite, fails cold
  render parity, exceeds absolute coefficient `16`, exceeds the declared geometry trust region,
  or produces a new optimizer-selected raw local-artifact maximum. Interpret any displayed-gate
  failure as a negative cell even if global PSNR improves. Do not retune this run.
- Exact command:
  `python scripts/experiments/hier008_overlap_elimination.py --images /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg --mask /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png --out results/hier008_janelle_overlap_elimination_2026-08-05 --max-side 512 --target-gaussians 4096 8192 --support-arms near_delta=0.18 overlap=0.50 --schedulers quadtree feature_wse_schur --cg-tolerance 1e-8 --cg-max-iterations 200 --cg-ridge 1e-8 --wse-alpha 8 --density-base 0.20 --density-power 0.50 --radius-min 0.65 --radius-max 2.25 --rgb-barrier 0.10 --feature-protection 4.0 --schur-ridge 1e-6 --optimizer-steps 80 --checkpoint-every 10 --lr-rgb 0.01 --lr-means 0.003 --lr-log-scales 0.002 --max-mean-shift 0.35 --max-log-scale-shift 0.15 --error-smoothing-sigma 1.5 --error-weight 2.0 --feature-weight 2.0 --tail-fraction 0.01 --tail-weight 2.0 --sigma-cutoff 3.0 --support-fade-alpha 0.0 --estimated-row-bytes 32 --seed 0 --device cuda --renderer cuda_additive --render-chunk 256 --lpips --error-scale 4.0`
- Evidence class: dirty-source, single-image, single-seed diagnostic. Snapshot every changed source
  and the dirty repository identity into the result bundle; preserve all negative/error cells.

## Notes

The reversible fallback is HIER-005 unchanged. HIER-008 is default-off and should be removed or
left as a benchmark reference if overlap prefitting rings, WSE loses edge samples, the optimizer
cannot earn a safe gain, or 4,096 rows remains visibly structured.

## Diagnostic outcome (2026-08-05)

The eight frozen cells completed at exact counts without retuning. The full overlap lattice is a
stable endpoint: PCG converges in 22 iterations/1.29 seconds with maximum pixel error `2.22e-8`,
coefficient maximum `0.756`, and negligible negative-coefficient fraction. Overlap is also a large
positive factor for expanding quadtree contraction: at 8,192 rows it raises matched PSNR from
35.129 to 45.953 dB and the common optimizer contributes a measured +2.144 dB. The optimizer is
active in six cells, stays inside all trust regions, and every selected checkpoint preserves or
improves its step-zero raw pixel and patch maxima.

No new cell passes the displayed artifact gate. Overlap/quadtree reaches pixel/7x7 maxima
`0.10768/0.02534` at 8,192 and `0.10999/0.04678` at 4,096; the latter contains visible periodic
rings. Fixed-scale WSE/Schur is rejected more strongly. Its top-feature coverage succeeds, but the
8,192-row overlap survivor set leaves q99/max centre gaps `1.44/2.08 px`, far outside meaningful
support for a `0.50 px` kernel, and reaches only 22.878 dB with visible dot holes. Static feature
and Schur prices do not replace dynamic covariance expansion plus actual post-removal distortion.

The portable 242-file report is
`results/hier008_janelle_overlap_elimination_2026-08-05/index.html`; its manifest SHA-256 is
`755d51a4332be452fe089b3e0c98e689c6a5052c1d302ca71d824650897e813f` and it passes the dirty-
diagnostic report checker. The executed module/driver hashes are preserved inside the bundle as
`e5ed8d43...` / `7afd8be5...`. A post-run fail-closed API-validation patch does not change valid
HIER-008 computation; current module/driver/test/checker hashes are `b9d4b7e5...`, `7afd8be5...`,
`8724cefc...`, and `35f2a2c6...`. Focused overlap plus HIER-005 tests pass 57/57. Full verification
passes with 1,655 tests, 4 skips, and 514 deselections; Ruff and every structural checker pass.

### Self-review

The matrix-free forward/transpose, PCG residual, nested-count logic, Field V2 semantics, trust-
region clamping, step-zero rollback, cold-render parity, report projection, and visual mechanism
diagnosis were checked against tests and persisted receipts. The strongest limitations are
scientific rather than hidden implementation failures: one exposed downscaled image, one CUDA
trajectory, a static rather than dynamically recomputed Schur price, fixed survivor covariance,
terminal 80-step optimization rather than HIER-005's interleaved 16x50 recovery, proxy bytes, and
no distinct review. The result supports a component-level design direction, not a method claim.

### Handoff

#### Objective

Implement and visibly test meaningful direct-neighbour Gaussian overlap and feature-protected
WSE/Schur simplification with an optimizer that can measurably improve all surviving rows.

#### Changes

Added exact fixed-lattice appearance solves, feature/radius/Schur analysis, nested WSE elimination,
Field V2 materialization, a common bounded artifact-safe optimizer, prefiltered HIER-005 input, 19
focused tests including an eight-cell end-to-end report test, a 2x2 diagnostic driver, report-
checker schema support, documentation, evidence, and complete visual/curve output.

#### Evidence

The report gate, 57-test focused slice, visual inspection, coefficient/geometry stability audit,
and full 1,655-test repository gate pass. Exact metrics, mechanism diagnosis, hashes, byte caveats,
and contextual HIER-005 comparison are in
`ara/evidence/hier008-overlap-lattice-feature-elimination-janelle-diagnostic-2026-08-05/run.md`.

#### Assumptions

The 0.02/0.01 displayed thresholds are provisional C0001 guardrails. Thirty-two bytes per row and
canonical/lossless containers are not complete codec rates. Native-source ratios are invalid for
compression comparison because native and evaluation dimensions differ.

#### Uncertainties

Generalization, deterministic CUDA small effects, matched convergence work, actual compressed
bytes, dynamic local elimination, and independent numerical/scientific review remain unresolved.

#### Review focus

Check normal-operator/diagonal correctness, local Schur correlation signs, feature barrier and WSE
heap updates, exact survivor nesting, optimizer artifact non-regression, source-snapshot identity,
and the conclusion that support-spacing—not missing feature centres—kills the WSE arm.

#### Protected actions not taken

Did not retune or overwrite C0001, drop negative cells, weaken the gate, access confirmation data,
call byte proxies compression, promote the method/default/semantics, commit, push, or claim novelty.

#### Recommended next action

Retain exact overlap prefitting and expanding quadtree contraction. If another bounded diagnostic
is authorized, combine overlap prefitting with HIER-005's interleaved touched recovery and a hard
protected-thin-feature leaf/reserved-detail rule. Revisit WSE only with dynamic covariance/local
appearance variable projection and actual patch-distortion commits. A fresh production pipeline is
not justified.
