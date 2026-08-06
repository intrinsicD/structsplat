# HIER-007 — Artifact-first frontier quadtree reconciliation

## Context

HIER-006 established a useful negative control: a literal progressive Gaussian prefix spends most
of an 8,192-row budget on retained ancestors, leaves isolated boundary defects under-refined when
splits are ranked by summed smoothed error, and cannot reconcile overlapping coefficients because
accepted rows are immutable. The user's approved successor keeps the quadtree only as a scheduler
and address structure. A split deactivates its active parent, activates all mask-present children,
and jointly refits the new children with spatially overlapping active neighbors while every other
row stays frozen.

## Goal

Implement a default-off NumPy-first parent-replacing frontier quadtree, isolate artifact-first
selection and overlap-local coefficient reconciliation in a frozen 2x2 diagnostic, and emit a
cold-rendered visual report with full quality, local-artifact, convergence, work, topology, and
honestly labeled byte-proxy curves.

## Method contract

- The active node keys are a deterministic quadtree antichain whose cells partition the active
  mask. Splitting a frontier parent removes it from the active render and inserts every
  mask-present child, so the net active-row change is `children - 1`; inactive ancestors remain
  scheduler/history records, not active Gaussian rows.
- Geometry is the HIER-006 mask-moment rule and stays fixed. Child RGB starts from the mean target
  residual after removing the selected parent contribution. No geometry parameter is optimized.
- The selection axis is either mask-aware Gaussian-smoothed residual energy per net added row or
  artifact-first lexicographic priority: the largest normalized raw foreground pixel/centered
  complete-7x7-patch violation inside a parent, then that smoothed energy score. Ties use stable
  node-key order.
- The reconciliation axis optimizes either new children only or new children plus every surviving
  active row whose finite-support AABB intersects a selected parent support AABB expanded by the
  7x7 gate radius. All other rows form a detached frozen base and must remain bit-exact.
- Every optimized row receives a fixed post-Adam update multiplier derived from mask-normalized
  sigma-smoothed residual MSE averaged under its finite-support Gaussian. This preserves Adam
  preconditioning while giving high-error supports larger updates.
- Candidate checkpoints minimize masked MSE plus a weighted worst-pixel tail, but commit only when
  a cold full-field render lexicographically improves normalized raw worst-pixel/7x7-patch
  violation and then foreground SSE, using the HIER-006 float32 equivalence band. Rejected trials
  roll back topology and coefficients exactly. A rejected multi-parent batch is retried as its
  deterministic higher-priority half; only a rejected singleton parent is blocked, preventing one
  bad split from discarding an otherwise useful batch.
- Displayed 8-bit output is the artifact-gate authority. Active full-field/reference bytes are
  reported separately from final-frontier and progressive-event structural proxies. Inactive
  nodes, coefficient revisions, mask/tree side information, and their omissions remain explicit;
  no proxy is called an implemented codec rate.
- The method remains absent from the maintained pipeline and does not select Field V2 semantics.

## Non-goals

- Claiming novelty, actual compression, artifact freedom, production readiness, or superiority
  from one exposed resized image and one seed.
- Adding a zero-moment/lifting primitive, optimizing geometry, training a split predictor, or
  changing the normalized renderer/default pipeline. Parent replacement is the chosen branch of
  the HIER-006 successor recommendation; lifting details need a separate semantic design.
- Consuming held-out/confirmation data or using this diagnostic to rescue a failed formal gate.

## Acceptance criteria

- [x] Typed deterministic APIs validate all inputs and expose active/stored node counts, exact
      split/reconciliation membership, rollback, stages/checkpoints, work, snapshot fields, and
      complete byte/proxy ledgers without importing torch at module import time.
- [x] Tests cover mask-partition/antichain invariants, parent replacement/count arithmetic,
      artifact-first isolated-defect priority, support-overlap membership, frozen-row bit
      exactness, exact rollback, CPU determinism, cap behavior, cold save/load parity, and a small
      four-arm report.
- [x] The four frozen arms share one identical base state and differ only in selection priority
      and reconciliation scope. Negative, rejected, missing, or failed cells remain visible.
- [x] The task-local report exposes full images and worst-error crops, active-depth maps, raw
      JSON/JSONL/CSV, fields/config/history, and curves over active count and attempted optimizer
      work for quality, displayed/raw artifacts, topology, bytes/proxies, timings, and update
      weights; `scripts/check_report_bundle.py --allow-dirty` passes.
- [x] The exact C0001 diagnostic below is executed once without outcome-driven retuning, and the
      combined arm is compared only at its measured scope with the three factorial controls plus
      contextual HIER-005/HIER-006 rows.
- [x] Architecture, Field V2 design, core skill, task Index/session brief, and ARA records stay
      synchronized; focused tests and `./scripts/verify.sh` pass.

## Interfaces touched

`src/structsplat/artifact_first_quadtree.py`, a task driver under `scripts/experiments/`, focused
tests, `docs/architecture.md`, `docs/additive_field_v2.md`, the core skill, this task, the Index,
generated session brief, and diagnostic result/ARA evidence.

## Depends on

HIER-005, HIER-006, CORE-013, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. This exposed dirty-tree
diagnostic can guide the next design only; independent numerical/scientific review is still
required and no formal prospective protocol approval is implied.

## Notes

### Frozen exposed-image diagnostic protocol (2026-08-05, before HIER-007 outcomes)

- Data: exact HIER-005/HIER-006 C0001 source and mask, SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` and
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, loaded through the existing
  deterministic max-side-512 LANCZOS/nearest path to the same 512x443 raster and 15,929-pixel mask.
  C0001 is exposed development/diagnostic data; there is no validation or confirmation access.
- Shared construction: seed 0/no stochastic topology; level-6 base; mask-moment geometry;
  0.18-pixel leaf variance; peak-one hard 3-sigma support; no support fade; parent replacement;
  at most 256 proposed child rows per stage; active cap 8,192; milestone 4,096; exact complete-child
  groups. Rejected batches use deterministic priority-prefix halving until acceptance or a rejected
  singleton is blocked; there is no post-outcome parameter change.
- Shared fitting: one base field fitted once and cloned into all arms; 400 base Adam steps. Each
  split trial gets 50 RGB-only Adam steps at LR 0.05, checkpoints every 5 steps, masked MSE plus
  four times the worst 1% foreground pixel-MSE mean. Error exposure is mask-normalized Gaussian
  smoothing sigma 1.5 px; row-weight power/floor/ceiling are 0.5/0.05/4.0. Overlap scope is hard
  finite-support AABB intersection with the union of selected parent support AABBs expanded by 3
  pixels.
- Frozen 2x2 arms: `energy__new_only`, `artifact_first__new_only`, `energy__overlap`, and
  `artifact_first__overlap`. Selection priority and reconciliation membership are the only axes.
  The combined proposal is the last arm; the other three are mechanism controls, not upstream
  methods. Arm order is fixed as written.
- Commit/stop: compare each cold trial to its unchanged active field by normalized raw pixel/7x7
  violation then foreground SSE with the 32-float32-epsilon tie band. Stop on the exact displayed
  gate (`pixel RGB-RMSE max <= 0.02`, maximum complete black-matted 7x7 RMSE `<= 0.01`), otherwise
  fail closed at active cap, exhausted frontier, or explicit no-progress condition. Do not weaken
  the gate or rescue an arm after opening outcomes.
- Metrics/artifacts: masked PSNR/MSE; full black-matted SSIM/MS-SSIM/LPIPS; displayed pixel tails
  and 3/7/15/31 patch maxima; raw gate metrics; active/inactive/stored nodes; attempted/accepted
  splits, optimized/frozen rows, optimizer steps, update-weight summaries, and phase/wall times;
  canonical/lossless active bytes; native source/evaluation PNG ratios; final-frontier and full
  progressive-event byte proxies with omissions stated; cold fields, full/error/depth images,
  worst crops, histories, configs, source snapshots, and all curves.
- Context only: corrected HIER-006 level/count rows and HIER-005 hard-3-sigma touched-only
  N=4,096/8,192 rows are loaded from their retained reports, not rerun or called equal-work arms.
- Exact intended command:

  `python scripts/experiments/hier007_artifact_first_quadtree.py --images /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg --mask /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png --out results/hier007_janelle_artifact_first_quadtree_2026-08-05 --max-side 512 --start-level 6 --max-gaussians 8192 --leaf-scale 0.18 --sigma-cutoff 3.0 --support-fade-alpha 0.0 --error-smoothing-sigma 1.5 --error-weight-power 0.5 --error-weight-floor 0.05 --error-weight-ceiling 4.0 --overlap-margin 3 --max-child-rows-per-stage 256 --base-steps 400 --layer-steps 50 --learning-rate 0.05 --tail-fraction 0.01 --tail-weight 4.0 --checkpoint-every 5 --pixel-threshold 0.02 --patch7-threshold 0.01 --estimated-row-bytes 32 --device cuda --renderer cuda_additive --render-chunk 256 --milestone-counts 4096 --arms energy__new_only artifact_first__new_only energy__overlap artifact_first__overlap --lpips --error-scale 4.0 --hier005-report results/hier005_janelle_artifact_hard3_touched_2026-08-05 --hier006-report results/hier006_janelle_progressive_residual_quadtree_corrected_2026-08-05`
- Evidence class: dirty-source, single-image, single-seed diagnostic. Source files are snapshotted
  into the immutable output directory. The run may falsify this exact mechanism locally but may
  not support an actual-rate, default, production, general-quality, or formal comparative claim.

### Diagnostic outcome and audit (2026-08-05)

The frozen run is retained unchanged at
`results/hier007_janelle_artifact_first_quadtree_2026-08-05`. Its executed module SHA-256 is
`cfa628f6da5649c1c9272e3cf3a2361b213b4ae84ef9ccb9a3fc50c40fb28599`; the executed driver is
`4a62b7896e6b01b0cf97db6b8a3d388d345a6a252f872284142faaf29adc9e43`. The original HTML omitted
HIER-005 context because its loader admitted `status=ok` but those retained rows are explicitly
`status=diagnostic`. No field, metric, curve, or trajectory was rerun or changed. The corrected
packaging-only copy at
`results/hier007_janelle_artifact_first_quadtree_contextfix_2026-08-05` preserves the executed
source snapshot, records the packaging correction and source hash in its manifest, restores both
HIER-005 rows, and passes `check_report_bundle.py --allow-dirty`. The original bundle also passes.

| arm at terminal | active / stored nodes | PSNR | MS-SSIM | LPIPS | pixel max | 7x7 max | gate | arm seconds | progressive-event proxy |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| energy / new-only | 8,192 / 11,333 | 40.035 | 0.999330 | 0.001023 | 0.047168 | 0.021467 | fail | 357.0 | 165,765 B |
| artifact-first / new-only | 8,192 / 11,278 | 34.569 | 0.998277 | 0.003716 | 0.312440 | 0.119657 | fail | 369.8 | 165,098 B |
| energy / overlap | 8,192 / 11,303 | 38.830 | 0.999633 | 0.001753 | 0.180037 | 0.094417 | fail | 536.4 | 994,841 B |
| artifact-first / overlap | 8,192 / 11,241 | 26.035 | 0.990103 | 0.022401 | 0.348476 | 0.172236 | fail | 1,132.8 | 1,206,214 B |

All arms fail the frozen 0.02/0.01 displayed artifact gate. The strongest HIER-007 arm is the
structural control `energy__new_only`: removing inactive parents improves by 7.153 dB and reduces
both local maxima relative to HIER-006's 8,192-row retained-parent row, but it still trails the
contextual HIER-005 8,192-row result (52.356 dB, 0.014847/0.005315, pass). At approximately 4,096
rows, no HIER-007 arm passes; their PSNR range is 24.483--28.745 dB versus HIER-005's 30.481 dB.

The proposed combined interaction is rejected on C0001. It uses 1,773 replacement trials for 279
accepted stages, with 1,382 batch backoffs and 112 rejected singleton blocks. Although 6,579
terminal active rows reach level 0, its worst displayed pixel is inside an active level-3 cell.
Artifact-first allocation concentrates budget on current maxima, overlap updates can move the
maximum into a still-coarse region, and complete-child expansion leaves no reserve to refine that
late hotspot at the cap. The lexicographic maximum-first transaction can also accept SSE damage
when the global maximum falls: the common first replacement lowers raw normalized violation from
46.270 to 45.208 while increasing SSE from 1,310.39 to about 1,690.94. The optimizer does activate
(215/279 accepted combined stages select a step above zero), so this is not a no-op scope failure.

Cold independent terminal replay agrees with reported PSNR to `<=6.2e-8` dB and displayed maxima
to `<=3.9e-8`; all saved fields share the same base canonical hash, every rollback/partition and
nonlocal-bit-exact flag is true, and the generic report gate finds no structural problem. Native
visual inspection confirms the quantitative result: the combined reconstruction contains strong
quadtree-aligned square/ring artifacts across the clothing and face, whereas energy/new-only is
visually much cleaner but retains a fine boundary/hair error.

None of the byte ledgers is compression evidence. Every terminal canonical active field is
290,496 bytes versus the exact 29,263-byte 512x443 evaluation PNG (0.101x, i.e. about 9.93 times
larger). Even the non-self-contained final-frontier proxies are about 128 kB (0.228x). The overlap
progressive-event proxies are 0.995--1.206 MB because revised neighbor coefficients are charged.
The apparent 49.1x canonical-field ratio against the 14,268,226-byte JPEG is invalid as a codec
comparison because that source is 5,328x4,608.

Verdict: retain parent replacement as a useful structural control, but reject the frozen artifact-
first/overlap policy and current reference runtime. A successor must align its differentiable
objective with pixel/patch commit metrics, forbid new regional hotspots and material SSE damage,
reserve late repair capacity, and transfer parent mass to children continuously (cross-fade or a
lifting/zero-moment constraint) before hard deactivation. Geometry/support caching and fused local
transactions are performance follow-ups only after that quality gate passes.

### Handoff

#### Objective

Implement and visibly test the user's scheduler-only hierarchy with parent replacement,
artifact-first allocation, smoothed-error-weighted overlap-local refitting, and fail-closed
transactions without changing maintained StructSplat behavior.

#### Changes

Added the NumPy-first `artifact_first_quadtree` reference, shared-base 2x2 driver, and 18 focused
tests. Added active-frontier partitions, deterministic artifact/energy ranks, support-local RGB
blocks, post-Adam row weights, batch backoff, exact rollback, snapshot fields, complete work and
proxy ledgers, 79 report curves, full/crop/depth visuals, and source snapshots. Synchronized the
core/architecture/Field V2/task surfaces. The current module/driver/test SHA-256 values are
`cfa628f6da5649c1c9272e3cf3a2361b213b4ae84ef9ccb9a3fc50c40fb28599`,
`60d7848bd1840778838a3f7e4cfab52059b12cb6fd51ed175b0c1bf753a2b0ce`, and
`31dd42b9e7ce3fc73ade8602d91c0b0f99999963b2acf2986f48fca4270d9193`.

#### Evidence

Both original and context-corrected HIER-007 bundles pass the diagnostic report gate. The relevant
field regression slice passes 142/142 before documentation closure. Independent cold replay and
visual inspection support the negative disposition summarized above; no arm passes the declared
gate and the combined arm is worst on PSNR and local artifacts. The full repository gate passes
with 1,636 tests, 4 skips, and 514 deselections.

#### Assumptions

Mask-derived geometry and tree decisions are shared only inside explicitly labeled proxies. The
displayed 8-bit thresholds are diagnostic C0001 guardrails, not a universal perception model.
HIER-005/HIER-006 rows are contextual, not jointly rerun equal-work controls.

#### Uncertainties

One exposed resized image and one CUDA execution cannot establish general hierarchy behavior.
Atomic accumulation is not bit-exact, although cold replay differences are negligible here. No
complete codec, quantizer, entropy model, cold structured decoder, disjoint data, matched search
budget, or distinct prospective/result review exists.

#### Review focus

Check active-frontier partition/count invariants, support-overlap membership, post-Adam weighting,
batch-backoff rollback, cold lexicographic acceptance, shared-base identity, event-byte accounting,
the context-only packaging correction, and the conclusion that objective/commit mismatch plus late
hotspot migration—not retained ancestors—kills the combined arm.

#### Protected actions not taken

Did not retune C0001, weaken the gate, rerun or overwrite fields after outcomes, drop negative
arms, call proxies codec rates, promote HIER-007, change renderer/Field V2/default semantics,
consume confirmation data, commit, push, or claim novelty/general superiority.

#### Recommended next action

Have a distinct reviewer reproduce the invariants and audit the corrected bundle. If hierarchy
work continues, open a separate frozen task for parent-to-child continuation with a smooth
pixel/patch-tail objective, regional hotspot/SSE trust region, and reserved final repair budget;
screen it first against `energy__new_only` and HIER-005, then optimize implementation speed only if
the artifact gate passes.
