# HIER-010 — Residual-anchored contraction with safe appearance projection

## Context

HIER-005's near-delta, hard-3-sigma, touched-only recovery is the only tested contraction arm
that passes the exposed 8,192-row local artifact gate.  Its 4,096-row endpoint fails visibly, and
terminal all-active recovery improves average error while moving geometry and turning the 8k pass
into a failure.  HIER-009 shows that feature protection can improve patch error but that overlap
and broad neighborhood recovery redistribute error.  FIT-033/038/040 show that exact partial
appearance solves and residual remeasurement are useful components, while FIT-043 rejects an
unmodified sequential controller.  The next HIER-005 action explicitly calls for local
uncontraction/preserved pixel leaves under an artifact-first gate.

This task tests a count-neutral two-pass successor at the user's requested exact 7k operating
point.  It preserves the proven HIER-005 compositor, support, and touched-only trajectory; uses the
first pass only to identify residual sites whose original pixel leaves must survive the second
pass; then conditionally projects only topology-touched, non-protected RGB coefficients with all
geometry and untouched leaves frozen.

## Goal

Implement and run a source-bound diagnostic that determines whether residual-guided leaf
preservation plus fail-closed matrix-free RGB projection materially lowers the remaining 7k error
without worsening HIER-005's displayed pixel or 7x7-patch artifact maxima.

## Non-goals

- Do not select additive Field V2 semantics, implement FIT-046's general production solver, or
  change the normalized/default pipeline.
- Do not call row counts, canonical arrays, or reference NPZ sizes compressed rate.
- Do not retune HIER-005's support, recovery learning rates, or the local artifact thresholds.
- Do not treat C0001 or C0004 as held-out or independent confirmation; both are exposed Janelle
  development views from one capture group.
- Do not claim equal work: the two-pass arms intentionally spend an additional contraction.

## Acceptance criteria

- [x] Residual-anchor selection is deterministic, selects an exact bounded subset of active source
  pixels, and combines isolated-pixel and 7x7 patch residual evidence without outcome-dependent
  thresholds.
- [x] `PixelContractionResult` exposes immutable active-row touched/protected provenance aligned
  with its returned field.
- [x] The appearance projection is matrix-free, freezes geometry and all untouched/protected rows,
  includes a step-zero checkpoint, and can only select a checkpoint that does not worsen the
  stage-zero displayed normalized artifact violation or raw masked SSE.
- [x] Focused synthetic tests cover forward/transpose parity, coefficient recovery, provenance,
  deterministic anchor selection, and fail-closed checkpointing.
- [x] The exact frozen eight-cell diagnostic completes with raw tables, per-arm fields, histories,
  source/config hashes, full/worst-crop visuals, metric curves, and a portable `index.html`.
- [x] `python scripts/check_report_bundle.py RESULTS_DIR --allow-dirty` and focused metric/provenance
  recomputation pass.
- [ ] Documentation and task state are synchronized and `./scripts/verify.sh` passes.

## Interfaces touched

- `src/structsplat/pixel_contraction.py`
- `src/structsplat/contraction_refinement.py`
- `scripts/experiments/hier010_residual_anchor_projection.py`
- `tests/test_pixel_contraction.py`
- `tests/test_contraction_refinement.py`
- `scripts/check_report_bundle.py` only if a new report schema requires a narrow registration
- `docs/architecture.md`, `docs/additive_field_v2.md`
- `tasks/INDEX.md`, `tasks/SESSION-BRIEF.md`

## Depends on

HIER-005/009, CORE-013, BENCH-002, FIT-033/038/040/043, ADR-0006

## Frozen diagnostic protocol

### Evidence class and question

This is a dirty-source-capable, source-snapshotted diagnostic with no prospective independent
review.  It asks whether the full composition improves two exposed images under the rule below.
It may kill or motivate the mechanism but cannot promote a semantic, default, rate, or general
quality claim.

### Sources and roles

- Method-development view: native `frame_00008/rgb/C0001.jpg`, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b`; mask
  `mask/mask_C0001.png`, SHA-256
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`.
- Correlated transfer diagnostic: native `frame_00008/rgb/C0004.jpg`, SHA-256
  `26eb4cf24a034eb830198df6e7a6ac409ccb7cf4814ff645c71d0b6966b7070e`; mask
  `mask/mask_C0004.png`, SHA-256
  `4702bfa9df354f38e35a63207a37d4ec1b753afc4d0668bd905f3cdab320f35d`.
- Both native RGBs are 5,328x4,608.  Deterministic evaluation uses Pillow LANCZOS RGB and nearest
  mask resize to 512x443 at threshold 0.5, producing 15,929 and 10,980 active pixels respectively.
- Both views were used by earlier development work; there is no held-out or confirmation role.

### Common field and recovery protocol

- Exact target: 7,000 signed direct-additive rows; no random seed.
- Near-delta leaves: isotropic sigma 0.18 px; hard AABB support at 3 sigma; no tail subtraction.
- HIER-005 exact-count topology: proposal batch 64, merge batch 8, pair shortlist 3, exact-option
  shortlist 2, `pair_policy=exact_count`.
- Selective recovery: touched rows only, progress schedule, 16 checkpoints x 50 attempted Adam
  steps; CUDA additive renderer; chunk 256; learning rates means/scales/rotation/RGB
  `0.005/0.003/0.001/0.003`; trust regions `1.5 px/0.35/0.35 rad`.
- Device is the available RTX 4090 through `device=cuda`, `renderer=cuda_additive`.  CUDA atomic
  accumulation is numerically, not bit, reproducible.

### Arms

Each image retains all four rows at exact N=7,000:

1. `h005_control`: one HIER-005 hard3/touched pass.
2. `control_projection`: arm 1 plus the projection below.
3. `residual_anchor`: rerun the identical contraction while protecting exactly 350 source leaves
   (5% of target N) selected from arm 1's residual.
4. `anchor_projection`: arm 3 plus the identical projection.

The anchor score is fixed before execution.  On active pixels compute raw per-pixel RGB MSE and a
mask-aware 7x7 mean-MSE map; divide each map by its active-pixel q99 (using a positive machine-
epsilon floor), take their pointwise maximum, and rank by descending score with stable row-major
ties.  Select with Chebyshev-radius-1 NMS, then fill any shortfall from the same stable ranking.

Projection trains exactly the active rows marked topology-touched and not protected by the
contraction result.  It freezes means, scales, rotations, untouched rows, protected rows, alpha,
and topology.  Preconditioned CG solves the masked finite-support additive normal equations with a
`1e-8` Tikhonov pull to stage-zero coefficients, tolerance `1e-6`, and at most 48 iterations.  The
dense pixel-by-row matrix is forbidden.  Step zero and every iteration are retained.  A checkpoint
is selectable only if coefficients are finite with absolute maximum <=16, raw masked SSE is no
higher than step zero within `1e-8 * max(SSE0,1)`, and displayed normalized artifact violation is
no higher than step zero within `1e-9`.  Among selectable checkpoints choose lowest raw SSE, then
lowest displayed violation, then earliest iteration.  Otherwise return step zero exactly.

### Metrics and accounting

- Primary fidelity: raw foreground masked MSE/PSNR.
- Perceptual: full black-matted SSIM, MS-SSIM, and LPIPS.
- Local safety: exact displayed 8-bit foreground pixel RMSE q99/q99.9/max and maximum complete
  black-matted patch RMSE at 3/7/15/31; unchanged provisional gate pixel max <=0.02 and 7x7 max
  <=0.01.
- Integrity: exact count, field/source/mask hashes, active-row provenance counts, maintained versus
  solver render parity <=2e-6, repeated render parity, frozen-array identity, CG residual and
  forward/transpose adjoint checks.
- Work: exclusive/cumulative wall time, contraction actions/checkpoints/attempted recovery steps,
  CG iterations/forward/transpose applications, and peak CUDA allocation.  No equal-work or speed
  conclusion is permitted.
- Byte columns are canonical raw and lossless reference storage only, explicitly not codec rate.

### Decision and killing rule

The full `anchor_projection` mechanism advances only if, on both images, it reaches exact count,
passes parity/integrity checks, strictly lowers masked MSE versus `h005_control`, and does not
worsen either displayed pixel maximum or displayed 7x7 maximum.  A local-gate failure remains a
failure even if PSNR improves.  Any source/hash mismatch, nonfinite value, projection parity
failure, missing arm, or overwritten negative row invalidates the diagnostic.  Failure leaves
HIER-005 unchanged and does not authorize retuning on these consumed views.

### Exact command

```bash
PYTHONPATH=src python scripts/experiments/hier010_residual_anchor_projection.py \
  --images \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0004.jpg \
  --masks \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0004.png \
  --out results/hier010_residual_anchor_projection_janelle_2026-08-10 \
  --target-gaussians 7000 --max-side 512 --mask-threshold 0.5 \
  --anchor-count 350 --anchor-patch-side 7 --anchor-nms-radius 1 \
  --projection-ridge 1e-8 --projection-tolerance 1e-6 \
  --projection-max-iterations 48 --projection-coefficient-limit 16 \
  --leaf-scale 0.18 --sigma-cutoff 3.0 --support-fade-alpha 0.0 \
  --recovery-steps 50 --recovery-progress-checkpoints 16 \
  --device cuda --renderer cuda_additive --render-chunk 256 --lpips
```

## Notes

The reversible fallback is HIER-005 unchanged.  General additive variable projection remains
FIT-046's decision and is still blocked on BENCH-020; this task implements only the contraction-
provenance subset needed for the bounded diagnostic.

## Diagnostic outcome (2026-08-10)

All eight frozen cells reached exactly 7,000 rows.  The report retains the lossless field, complete
contraction/recovery/projection histories, analysis masks, full and worst-crop visuals, curves, raw
tables, configuration, and executed-source snapshots for every cell.

| image | arm | PSNR | masked MSE | MS-SSIM | LPIPS | pixel max | 7x7 max | cumulative s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C0001 | HIER-005 control | 50.097060 | 9.77899e-6 | 0.999976397 | 0.000028193 | 0.026404 | 0.009518 | 8.723 |
| C0001 | control + projection | **50.108004** | **9.75438e-6** | 0.999976754 | 0.000027994 | 0.026404 | **0.009469** | 9.495 |
| C0001 | residual anchors | 49.897467 | 1.02389e-5 | 0.999976993 | 0.000027088 | 0.034187 | 0.012141 | 16.760 |
| C0001 | anchors + projection | 49.908700 | 1.02125e-5 | **0.999977171** | **0.000026622** | 0.034187 | 0.012089 | 17.480 |
| C0004 | HIER-005 control | 54.374098 | 3.65250e-6 | 0.999991179 | 0.000007455 | 0.014847 | 0.004597 | 5.529 |
| C0004 | control + projection | **54.378459** | **3.64883e-6** | 0.999991119 | 0.000007338 | 0.014847 | 0.004620 | 6.279 |
| C0004 | residual anchors | 54.183591 | 3.81629e-6 | **0.999991477** | **0.000005643** | 0.013202 | 0.004387 | 10.662 |
| C0004 | anchors + projection | 54.189306 | 3.81127e-6 | **0.999991477** | 0.000005682 | **0.013202** | **0.004375** | 11.354 |

Projection alone is safe under the frozen SSE/maximum-normalized-violation transaction but
negligible: `+0.010944/+0.004361 dB`, with `-0.252/-0.100%`
masked MSE on C0001/C0004.  It selects PCG iteration 22/16 and leaves the isolated displayed
maximum unchanged.  Hard residual anchoring is not robust.  The full composition loses
`0.188360/0.184792 dB` and raises MSE `4.433/4.347%`; C0004's local maxima improve, while C0001's
worsen.  Neither image has the required strict MSE win and C0001 also violates both local
non-regression clauses.  The frozen mechanism gate therefore fails, HIER-005 remains unchanged,
and these consumed views must not be used to tune another reserve fraction.

All count/hash/provenance checks pass.  Cold replay independently recomputes all primary,
perceptual, and local metrics for all eight fields: maximum PSNR drift is `2.82e-7 dB`, MSE drift
`6.32e-13`, MS-SSIM drift `5.97e-8`, LPIPS drift `1.68e-8`, and displayed local metrics match
exactly.  Maintained/repeated renderer parity is at most `4.92e-7`/`1.19e-7`; projection adjoint
error is at most `6.42e-7`.  The report bundle passes its structural checker.  Visual review finds
subtle residual redistribution concentrated on garment texture, silhouettes, hair, face, and
hands, with no gross new ordinary-scale artifact; per-arm worst crops move spatially and therefore
remain qualitative rather than registered comparisons.

Evidence:
`ara/evidence/hier010-residual-anchor-projection-janelle-diagnostic-2026-08-10/run.md`.
Portable report: `results/hier010_residual_anchor_projection_janelle_2026-08-10/index.html`.
Manifest SHA-256:
`80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba`.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba`

### Handoff log

This is an exposed dirty-source diagnostic without a prospective distinct protocol reviewer.  The
following self-review is provisional; it cannot promote a scientific claim or default, and the
repository-wide portable gate retains two untouched baseline failures after environment isolation.

### Handoff

#### Objective

Build and test a substantially stronger exact-7k successor to HIER-005 by combining residual-guided
leaf preservation with a matrix-free, fail-closed appearance projection, while retaining exact count
and the displayed local-artifact guard.

#### Changes

Added immutable touched/protected row provenance to pixel contraction; implemented deterministic
residual-anchor selection and sparse-tile PCG coefficient projection in
`structsplat.contraction_refinement`; added focused invariant tests and the complete HIER-010
experiment/report driver; registered its diagnostic report schema; and synchronized architecture,
Field V2, task, ARA trace/staging, and evidence documentation.  Post-run self-review also tightened
the reusable driver's exact-argument/parity enforcement and made an incoming over-limit stage zero
return unchanged; neither branch changes any sealed row, and the bundled snapshots remain the run
authority.

#### Evidence

The portable report is
`results/hier010_residual_anchor_projection_janelle_2026-08-10/index.html`; manifest SHA-256
`80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba`.  Its 170-file bundle
passes `check_report_bundle --allow-dirty`; all eight persisted fields independently reproduce
counts, hashes, provenance, metrics, parity, and the failed frozen decision.  The focused slice
passes 45 tests, targeted and repository-wide Ruff pass, and every docs/ARA/task/script/workflow
structural checker passes.  The complete portable pytest gate reaches 1,725 passes, 25 skips, and
514 deselections but has three failures in untouched subsystems: finite rank-deficient affine
condition-number reporting, CUDA property availability in SSP2E environment capture, and SSP2V's
opened-descriptor path-swap race.  With CUDA hidden, the SSP2E case passes and the two unchanged
baseline failures remain.  The final full-gate acceptance criterion is intentionally unchecked.

#### Assumptions

An exact 350-leaf reserve is the frozen 5% intervention; residual maps use only the first-pass cold
error; the two-pass arms intentionally receive more work; and the projection's transactional safety
uses the maximum normalized pixel/7x7 violation, matching the pre-run protocol rather than requiring
every nonbinding raw local metric to be monotone.

#### Uncertainties

Both views are exposed and correlated, each cell has one CUDA trajectory, and no distinct reviewer
approved the protocol.  The projection gain is far below the repository's later measured CUDA
nondeterminism envelope on another pipeline.  Worst crops are selected separately per arm and are
not registered comparisons.  No held-out capture group, matched-work study, complete codec, or
general FIT-046 alternation test is present.

#### Review focus

Audit sparse forward/transpose support parity and adjoint error, active-row provenance alignment,
the immutable frozen/protected coefficient path, stage-zero rollback, the difference between
normalized-violation safety and individual raw local metrics, and the independently recomputed
failure of strict MSE plus pixel/patch non-regression on both views.

#### Protected actions not taken

No maintained pipeline/default, renderer, Field V2 semantic, codec, artifact threshold, FIT-046
decision, external repository, commit, push, or unrelated affine/SSP2 benchmark code was changed.
Existing user and dirty-worktree changes were preserved.

#### Recommended next action

Obtain distinct numerical/scientific review and separately resolve or explicitly waive the two
untouched portable-gate baseline failures before terminal task acceptance.  Keep projection as a
bounded default-off cleanup, reject fixed-fraction hard anchoring on these consumed views, and test
any successor only as transaction-local preservation/uncontraction on independently approved new
capture groups.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Focused selection, provenance, forward/transpose, recovery, freeze, bound, and rollback tests pass.
Every saved field reaches exact count, hashes and provenance masks agree with its row, maintained
and repeated render parity remain sub-micro-unit, and an independent cold replay reproduces all
reported metric domains and the frozen rejection.

#### Evidence quality

The protocol was frozen before outcomes, all positive and negative cells are retained, source
snapshots bind the dirty execution, raw fields/histories/analysis/visuals are present, and the
portable bundle validates.  The evidence is diagnostic only because both views were exposed, are
from one capture group, use one CUDA trajectory, and lack prospective distinct review.

#### Simplicity

The method reuses HIER-005 unchanged, adds one deterministic reserve and one geometry-frozen linear
finish, and fails closed to the incoming field.  It does not introduce a parallel production path
or fold the task-local projection into the still-blocked general FIT-046 decision.

#### Missing cases

Distinct code/scientific review, independently approved unexposed capture groups, replicated CUDA
trajectories, matched work, registered fixed crops, complete-byte rate, general variable projection,
and a completely green repository-wide portable gate remain missing.

#### Required changes

None for retaining this diagnostic and its negative result.  Distinct review plus resolution or an
explicit maintainer disposition of the two untouched baseline gate failures is required before
terminal acceptance; no scientific claim or default may be promoted from this run.

#### Optional improvements

Add registered cross-arm crop coordinates and evaluate transaction-local leaf restoration whose
commit checks actual global SSE, displayed worst pixel, and displayed worst patch rather than
reserving a global fraction before the second contraction.
