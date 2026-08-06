# HIER-009 — Dynamic overlap contraction with neighborhood recovery

## Context

HIER-008 establishes that an exactly prefiltered `0.50 px` pixel lattice is stable and materially
helps HIER-005 quadtree contraction, but its fixed-lattice WSE arm removes centers from a static
ordering and leaves visible holes. HIER-005 already provides the missing dynamic transaction:
propose from the current field, locally refit, measure exact discrete distortion, commit a
non-conflicting batch, optimize, rebuild the frontier, and repeat. Its `touched` recovery freezes
all untouched pixel leaves, however, even though a meaningful-overlap replacement is coupled to
its direct 3x3 neighborhood. The bounded successor is therefore exact overlap plus dynamic
HIER-005 contraction, a direct-neighbor recovery halo, and an optional hard feature-detail reserve.

## Goal

Implement and expose a deterministic default-off dynamic contraction path that starts from an
exactly prefiltered overlap lattice, repeatedly contracts and re-scores the changed field,
optimizes topology-touched rows plus an explicitly bounded 3x3 neighbor halo, preserves selected
thin/high-feature leaf geometry, and compares the resulting quality/artifact/reduction tradeoff
against the strongest touched-only controls at 8,192 and 4,096 rows.

## Method contract

- The initial `overlap` lattice uses integer pixel centers, isotropic `0.50 px` scales, signed RGB,
  peak-one direct-additive kernels, hard `3 sigma` AABBs, exact source alpha, and the HIER-008
  mask-aware PCG appearance prefilter. `delta` retains the historical `0.18 px` source-RGB
  endpoint.
- Topology is HIER-005's current dynamic engine. Each iteration proposes local hard,
  parent-plus-detail, or exact-count pair actions from the current field; solves candidate
  coefficients over the actual finite-support discrete renderer; ranks by exact distortion per
  estimated byte; commits a support-disjoint batch; and invalidates every overlapping cached
  proposal. Recovery acceptance rebuilds and re-scores the frontier before contraction continues.
- `touched_neighborhood` contains all active topology-touched rows plus active rows whose rounded
  centers lie within Chebyshev distance one of a newly touched row at a recovery checkpoint.
  A neighbor whose accepted optimizer update changes the field remains recovery-eligible at later
  checkpoints. Rows outside this accumulated halo remain a detached fixed base.
- Feature protection selects exactly `round(0.05 * target_count)` mask-present pixel leaves by a
  deterministic priority equal to the maximum of normalized structure-tensor energy and
  normalized one-pixel RGB high-pass magnitude, using stable Chebyshev-radius-one NMS. A protected
  leaf cannot be removed: contraction may carry its exact mean/covariance as a detail basis and
  refit its RGB. Protected means/covariances are restored after every optimizer step. If the
  retained protected multiplicity cannot be represented by the existing two-atom quadtree state,
  that region fails closed while other regions remain eligible.
- Recovery uses the existing progress-normalized 16 checkpoints x 50 attempted Adam steps. Every
  arm shares learning rates, trust regions, renderer, pair policy, target counts, and metric
  domains. Work is matched in attempted steps, not optimized-row FLOPs; optimized counts and time
  remain explicit.
- Requested count is the terminal criterion. A cell that protection makes unreachable reports its
  achieved count and stop reason rather than dropping protection or weakening the gate. Estimated
  row bytes and lossless NPZ remain non-codec references.

## Non-goals

- Reintroducing HIER-008's fixed-scale WSE/static-Schur survivor path.
- A global all-pairs merge search, full support-overlap optimizer closure, or unrestricted
  all-active fitting.
- Retuning protection fraction, NMS radius, recovery work, or the provisional artifact gate after
  accessing C0001 outcomes.
- Selecting Field V2 semantics, a production pipeline/default, novelty, convergence-speed, or
  actual compression before the owning CORE/BENCH/COMP tasks.

## Acceptance criteria

- [x] Typed NumPy-first feature selection and protected-topology APIs validate shapes, masks,
      deterministic ties, exact reserve counts, sparse/odd inputs, and fail-closed multiplicity.
- [x] Default behavior is bitwise/regression-compatible when no protection mask is supplied.
- [x] Protected feature means/covariances survive every accepted topology and recovery action;
      RGB remains locally refittable and the final protected count is exact.
- [x] `touched_neighborhood` includes the direct 3x3 active halo, persists only accepted changed
      neighbors, freezes all rows outside the scope, and rebuilds current-field proposals after an
      accepted optimizer checkpoint.
- [x] Synthetic flat, step, diagonal, thin-line, checker, sparse-mask, and exact-count tests cover
      selection, halo membership, local optimization impact, protection, and cold field parity.
- [x] The frozen C0001 four-arm x two-count diagnostic below emits source/prefit/reconstruction/
      error/feature/protected/center/worst-crop visuals, topology and recovery histories, every
      snapshot and optimizer curve, raw metric ledgers, fields, configs, source snapshots, and a
      browsable portable `index.html`.
- [x] Focused tests, report-bundle validation, adversarial result audit, self-review, docs/task/ARA
      synchronization, and `./scripts/verify.sh` pass before handoff.

## Interfaces touched

`src/structsplat/pixel_contraction.py`, `src/structsplat/overlap_elimination.py`,
`scripts/experiments/hier009_dynamic_overlap_recovery.py`, `scripts/check_report_bundle.py`,
focused tests, `docs/architecture.md`, `docs/additive_field_v2.md`, the core skill, this task,
the Index/session brief, and the diagnostic evidence bundle.

## Depends on

HIER-005/008, CORE-013, BENCH-002, ADR-0006

## Frozen exposed-image diagnostic protocol (2026-08-06, before outcomes)

- Question: does replacing HIER-008's one-shot elimination with current-field dynamic contraction,
  and giving each touched Gaussian its direct 3x3 optimization neighborhood, preserve the positive
  overlap factor while removing visible artifacts; does a 5% protected feature reserve help or
  merely block useful reduction?
- Data: the previously exposed C0001 JPEG and binary mask under
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`, resized exactly to
  `512x443` with the established source/mask hashes and `15,929` active pixels. No held-out or
  confirmation image may be accessed.
- Arms, in fixed order:
  1. `delta_touched`: `0.18 px`, source RGB, HIER-005 `touched` recovery;
  2. `overlap_touched`: `0.50 px`, exact prefilter, HIER-005 `touched` recovery;
  3. `overlap_halo`: same overlap endpoint with `touched_neighborhood` recovery;
  4. `overlap_halo_protected`: same halo plus the frozen 5% feature-leaf reserve.
- Counts: exact N=`8,192` and `4,096`; signed coefficients; `exact_count` pair policy;
  proposal/merge/pair/exact shortlists `64/8/3/2`; cutoff `3.0`, fade `0.0`; estimated row price
  `32` bytes. No outcome cell may be dropped.
- Prefit: PCG tolerance `1e-8`, maximum `200` iterations, diagonal ridge `1e-8`. Protection uses
  reserve fraction `0.05`, high-pass sigma `1.0`, Chebyshev NMS radius `1`, stable score/y/x ties,
  and no outcome-driven threshold.
- Recovery: `16x50` attempted steps; RGB/means/log-scales/rotation learning rates
  `0.003/0.005/0.003/0.001`; mean/log-scale/rotation trust
  `1.5 px/0.35/0.35 rad`; RTX-3050 `cuda_additive`, chunk `256`. The halo radius is exactly one
  rounded pixel and is not scale-tuned.
- Metrics: foreground raw MSE/PSNR/SSE; black-matted SSIM/MS-SSIM/LPIPS; displayed pixel-RMSE
  q99/q99.9/max and maximum 3/7/15/31-pixel patch RMSE; provisional C0001 gate pixel max
  `<=0.02` and 7x7 max `<=0.01`; prefit diagnostics; topology/recovery SSE and PSNR trajectories;
  optimizer scope/change counts; protected coverage/geometry/count; cold/repeated parity; time;
  canonical/lossless/estimated byte ledgers and separately labeled source ratios.
- Killing rules: a cell is negative if it misses exact count, loses a protected leaf, moves
  protected geometry, violates cold parity, becomes non-finite, or fails the displayed gate.
  A mean improvement cannot rescue a local-gate failure. Do not retune or overwrite this run.
- Exact command:
  `python scripts/experiments/hier009_dynamic_overlap_recovery.py --images /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg --mask /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png --out results/hier009_janelle_dynamic_overlap_recovery_2026-08-06 --max-side 512 --target-gaussians 4096 8192 --arms delta_touched overlap_touched overlap_halo overlap_halo_protected --delta-scale 0.18 --overlap-scale 0.50 --cg-tolerance 1e-8 --cg-max-iterations 200 --cg-ridge 1e-8 --protected-fraction 0.05 --protected-highpass-sigma 1.0 --protected-nms-radius 1 --proposal-batch-size 64 --merge-batch-size 8 --pair-shortlist 3 --exact-option-shortlist 2 --recovery-checkpoints 16 --recovery-steps 50 --lr-coefficients 0.003 --lr-means 0.005 --lr-scales 0.003 --lr-rotations 0.001 --max-mean-shift 1.5 --max-log-scale-shift 0.35 --max-rotation-shift 0.35 --sigma-cutoff 3.0 --support-fade-alpha 0.0 --estimated-row-bytes 32 --device cuda --renderer cuda_additive --render-chunk 256 --lpips --error-scale 4.0`
- Evidence class: dirty-source, single-image, single-seed diagnostic. Snapshot every executed
  task-local source and retain every negative/error cell.

## Notes

The reversible fallback is unchanged HIER-005 hard3/touched. Static WSE elimination remains
rejected. A later dynamic support-overlap graph may replace the rounded-center halo only after this
bounded direct-neighbor test establishes whether local optimizer freedom is beneficial.

## Diagnostic outcome (2026-08-06)

All eight frozen cells reached exact count with maintained-render parity at most `7.16e-7`.
Every one of 16 recovery checkpoints was accepted per cell, proving that the interleaved optimizer
was active. Halo arms optimized up to 4,514 rows at a checkpoint, included up to 1,951 direct
neighbors, and accepted 5,725--6,970 newly changed neighbor events over the trajectory. Protected
arms retained exactly 410/205 leaves at 8,192/4,096 with zero mean/covariance error; 69/38 locally
overfull regions failed closed without preventing exact global targets.

| arm | N | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| delta_touched | 8,192 | 52.338 | 0.999981 | 0.000016 | 0.01485 | 0.00530 | pass |
| overlap_touched | 8,192 | 47.395 | 0.999916 | 0.000192 | 0.06815 | 0.02294 | fail |
| overlap_halo | 8,192 | 45.963 | 0.999872 | 0.001123 | 0.05513 | 0.02736 | fail |
| overlap_halo_protected | 8,192 | 46.991 | 0.999905 | 0.000487 | 0.05168 | 0.02049 | fail |
| delta_touched | 4,096 | 30.547 | 0.996966 | 0.022970 | 0.20729 | 0.07086 | fail |
| overlap_touched | 4,096 | 39.802 | 0.999421 | 0.002833 | 0.09003 | 0.03335 | fail |
| overlap_halo | 4,096 | 40.801 | 0.999569 | 0.002633 | 0.07989 | 0.02781 | fail |
| overlap_halo_protected | 4,096 | 41.115 | 0.999576 | 0.002327 | 0.08583 | 0.02511 | fail |

The 3x3 halo is useful at aggressive contraction: at 4,096 rows it gains `+0.999 dB`, lowers both
local maxima, and visibly removes the block/quadtree lattice relative to overlap/touched. It is a
mixed factor at 8,192: the isolated maximum improves but PSNR loses `1.433 dB` and 7x7 error
worsens, so unconstrained neighborhood freedom redistributes error. Protection recovers
`+1.028/+0.314 dB` at 8,192/4,096 and improves patch error, but no overlap cell passes. The
unchanged delta/touched 8,192 fallback remains strongest and is the only artifact-gate pass.

Canonical 8,192/4,096 fields are 290,496/159,424 bytes: 9.93x/5.45x larger than the same-raster
PNG. The 1.94x/3.89x row reductions are against 15,929 active mask pixels, not the full raster.
No actual compression, convergence-speed, production, or general-quality claim follows.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `c1d8c488c8200edd8b4d68c103b41f9347c8844a5c4c41607431fbd4be67f60d`

### Handoff log

This is an exposed dirty-source diagnostic without a prospective distinct protocol reviewer; it
cannot promote a scientific claim or default. The following self-review is provisional, and a
distinct reviewer remains required for scientific acceptance.

### Handoff

#### Objective

Implement the missing current-field merge/refit/re-score loop with the user's clarified recovery
scope: optimize every topology-touched Gaussian plus its direct 3x3 Gaussian-neighbor halo, then
test whether exact initial overlap and protected feature leaves improve visual quality at 8k/4k.

#### Changes

Added deterministic exact-count feature-leaf selection to `overlap_elimination`, extended
`pixel_contraction` with persistent accepted `touched_neighborhood` recovery and hard protected-
geometry propagation, and added the complete HIER-009 four-arm experiment/report driver. Ordinary
regions still use live HIER-005 hard/detail/pair contractions; protected regions carry exact
protected leaves and fail closed when the two-atom state cannot represent their multiplicity.
Added seven HIER-009 tests and report-schema validation, synchronized the architecture/Field V2
design, core skill, task graph, evidence note, and generated session brief.

#### Evidence

The frozen 8-cell report is
`results/hier009_janelle_dynamic_overlap_recovery_2026-08-06/index.html`; its manifest SHA-256 is
`c1d8c488c8200edd8b4d68c103b41f9347c8844a5c4c41607431fbd4be67f60d`. Executed source snapshots
match the pre-run driver/contraction/overlap files byte-for-byte. The report checker passes with
`--allow-dirty`; the combined pixel/overlap/HIER-009 focused slice passes 64 tests; targeted Ruff
passes; `./scripts/verify.sh` passes with 1,662 tests and 4 skips before the final ARA-only
epilogue, after which the ARA/docs/task/workflow structural checks pass again.

#### Assumptions

The direct neighborhood is defined by rounded active centers at Chebyshev radius one, rather than
transitive support overlap. Accepted numerical change, not mere optimizer eligibility, makes a new
neighbor persistent. The displayed C0001 pixel/7x7 gate is a diagnostic local-artifact criterion,
not a universal visibility threshold. Attempted steps are matched while optimized-row FLOPs are
explicitly unmatched.

#### Uncertainties

This exposed single image and CUDA trajectory cannot establish general behavior. The halo improves
4k but regresses some 8k metrics, and protection trades patch improvement for a worse isolated 4k
pixel. There is no held-out confirmation, matched-FLOP convergence study, learned/adaptive scope,
complete codec, entropy model, or independent numerical/scientific review.

#### Review focus

Audit protected multiplicity across hard/detail/pair actions, exact geometry restoration after
Adam, fixed-base detachment outside the accumulated halo, persistence only after accepted changes,
cache/frontier rebuild after recovery, and the distinction between summed local checkpoint gains
and terminal optimizer attribution. Reproduce the 4k visual block removal and challenge whether
the remaining distributed texture error is perceptually preferable.

#### Protected actions not taken

No maintained renderer, pipeline default, semantic decision, codec, artifact threshold, frozen
result, external repository, commit, push, or unrelated IDE/user file was changed. HIER-008 static
WSE remains rejected and existing dirty HIER-005--008 work was preserved.

#### Recommended next action

Obtain a distinct numerical/scientific review. If continued, test an adaptive recovery scope whose
checkpoint acceptance is Pareto-safe in SSE, worst pixel, and worst local patch, enabling the 3x3
halo only where contraction severity predicts a net benefit; keep delta/touched 8k as the fallback.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Focused invariants pass: zero-protection behavior is exact, direct-neighbor membership is exactly
3x3, protected geometry/count survives, accepted neighbor changes occur, and cold render parity is
sub-micro-unit. The implementation remains default-off.

#### Evidence quality

The protocol was frozen before outcome access, every planned cell and negative result is retained,
executed sources are snapshotted, raw fields/histories/curves are present, and the portable bundle
validates. Evidence remains diagnostic because the source was exposed and dirty and there was no
prospective distinct reviewer.

#### Simplicity

The change reuses the existing contraction/recovery engine and adds one bounded scope plus one
protected tag, rather than introducing a second scheduler. The rounded-center halo is intentionally
the smallest direct interpretation of the user's 3x3 request.

#### Missing cases

Independent code/scientific review, held-out images, multiple seeds/devices, a matched-FLOP arm,
adaptive halo selection, local-artifact-aware optimizer acceptance, and complete-byte coding remain
missing.

#### Required changes

None for this diagnostic handoff. Distinct review is required before any scientific acceptance or
default/claim promotion.

#### Optional improvements

Record per-parameter movement by touched versus neighbor rows, compare a support-overlap graph to
the rounded halo, and use a no-new-hotspot/Pareto checkpoint gate before enlarging the optimizer
scope.
