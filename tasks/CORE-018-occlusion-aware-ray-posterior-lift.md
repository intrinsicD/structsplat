# CORE-018 — Occlusion-aware ray-posterior surface lift

## Context

CORE-016's codec-native appearance plane preserves useful radiance at substantially lower teacher
input bytes, but its CompactCarve lift leaves halos, blur, and floaters.  CORE-017 proves that
frontmost alpha-shell placement improves aggregate quality and silhouette localization, yet its
directional trailing doubles fail native visual review.  Masks constrain a visual hull; they do not
identify the visible surface inside it.  Existing realtime-gs controls also rule out the two
obvious shortcuts: component-level Splat-SfM leaves conspicuous floaters on Janelle, while a raw
patch/epipolar graph failed its strict semantic-precision gate.

The materially different observable is therefore a **depth posterior along each source ray**.
Decoded packet appearance supplies shared dense features and local radiance; calibrated projection
supplies each finite depth hypothesis; a robust best-view likelihood with an explicit dustbin
handles occlusion; and reciprocal source-ray agreement rejects isolated modes before any Gaussian
is emitted.  This is a controlled composition of plane-sweep stereo, robust multi-view aggregation,
and consistency filtering, not a novelty claim.  The candidate selection record is
`docs/research/2026-08-06-core018-geometry-portfolio.md`.

## Goal

Implement and kill-test a default-off, packet-derived, occlusion-aware ray-posterior initializer
that emits a finite realtime-gs `Gaussians3D` field, improves disjoint-view convergence and visible
geometry without double silhouettes, and preserves complete byte/time accounting from image files
through the final 3D model.

## Non-goals

- Do not change realtime-gs source, CompactCarve defaults, StructSplat's maintained conversion
  pipeline, renderer equations, packet grammar, or any production default.
- Do not tune or rerun the consumed `frame_00008`/`frame_00009` visual outcomes.
- Do not call DINO, plane sweep, reciprocal filtering, or their combination novel; shared feature
  weights are an amortized dependency and are not per-scene payload bytes.
- Do not claim physical depth, general compression, production speed, or artifact freedom from one
  disjoint development scene.  Persisted packet/model bytes, not Gaussian count, define rate.
- Do not force low-confidence rays into geometry merely to fill the exact budget; a declared,
  separately counted coverage fallback is allowed only when its provenance is visible.

## Acceptance criteria

- [x] A lazy optional module builds deterministic packet-derived feature pyramids without opening
      original RGB files, exposes model/weight provenance, and keeps metadata/cameras on CPU while
      bounded feature and scoring batches may run on CUDA.
- [x] The initializer reuses the existing structural measure only for source-ray proposals, scores
      bounded depths using source-excluded calibrated views, aggregates a fixed number of best
      visible likelihoods plus a dustbin, refines the selected mode locally, and records posterior
      entropy, margin, view support, reciprocal support, fallback lineage, work, memory, and time.
- [x] Reciprocal agreement is computed from independently selected source-ray modes and cannot use
      reporting cameras, source RGB, training renders, or post-optimization state.  Exact-N output
      uses deterministic balanced selection and fails closed when neither accepted nor explicitly
      allowed fallback rows can fill the budget.
- [x] Synthetic tests cover a textured front surface, a repeated-texture distractor, one adversarial
      occluded view, a depth tie, a reciprocal inconsistency, exact count, finite covariance, CPU
      determinism, device return semantics, and malformed packets/cameras/configuration.
- [x] A diagnostic-only disjoint `karate/frame_00060` comparison reuses one immutable packet set and
      compares ordinary interior consensus, posterior without reciprocity, and the complete
      posterior.  All arms use the same training/reporting cameras, 10,000 initial rows, staged
      500-step fixed-topology plus 1,000-step shared density refinement, and a 30,000-row cap.
- [x] The report includes complete original JPEG, packet, shared-model, final-model, and index bytes;
      encode/decode/feature/lift/train/render times; peak host/CUDA memory; Gaussian-count and all
      quality/convergence curves; native target/init/final/RGB-error/alpha/depth-support panels; and
      machine-readable receipts.  It reports both `original_files / packets` and
      `original_files / (packets + final_model)` without treating model weights as per-scene bytes.
- [x] Fail the complete candidate if it retains directional double silhouettes/floaters on any
      reporting view, loses more than 0.1 dB terminal reporting PSNR or 0.01 MS-SSIM to the strongest
      control, worsens LPIPS or gradient MAE, reaches the strongest control's terminal PSNR later,
      exceeds twice the control's complete pre-training wall-clock, or fails to reduce the complete
      scene representation below the summed original JPEG bytes.  A pass authorizes only a
      prospectively reviewed multiscene/multiseed confirmation.
- [x] Task, Index, generated brief, architecture/ADR boundary, tests, evidence disposition, exact
      command, and visual decision are synchronized; focused checks and `./scripts/verify.sh` pass.

## Diagnostic protocol

This is a one-scene, one-seed, disjoint development killing test, not a promotion run.

- **Data:** `/home/alex/Dropbox/Work/Janelle/karate/frame_00060` with its sibling
  `calibration_dome.json`.  Use the loader's undistorted downscale-4 tensors for packet creation and
  downscale-8 tensors for training/reporting.  Reporting cameras are prospectively frozen as
  `C0004`, `C0025`, `C1004`, and `C1005`; they must not enter packet-derived geometry scoring or
  training.
- **Packets:** one full-canvas WebP-quality-80 dual-plane packet per construction view, exact full
  alpha, 1,024 structural proposals, lattice sigma `0.45`, radius `3`, eight Jacobi steps, seed by
  construction-view index.  Cold-reload the exact same packet hashes for every arm.
- **Features:** Apache-2.0 DINOv2-S/14 checkpoint plus a packet-derived local radiance/gradient
  descriptor.  Record the exact model identifier, weight hash, preprocessing, tensor shapes, and
  shared inference time.  No source image may be opened after packet construction.
- **Geometry:** 10,000 output rows from two proposal candidates per requested row, 24 coarse depth
  samples inside the shared explicit bounds, four source-excluded construction neighbors, robust
  best-two-view likelihood plus dustbin, nine-point local depth refinement, and deterministic
  reciprocal consistency against independently solved candidate rays.  The no-reciprocal arm is
  identical except for that gate/score.  Surface-cover reconciliation may change only covariance
  and opacity after means/radiance/count are fixed.
- **Optimization:** identical realtime-gs `gsplat` training for every arm: 500 fixed-topology steps
  followed by 1,000 ordinary shared density-controller steps, evaluation every 100 attempted steps,
  final checkpoint, SH degree 3, seed 0, no masks, and at most 30,000 Gaussians.
- **Decision:** apply every scalar/resource gate above, then inspect every reporting PNG at stored
  pixels.  Scalar success cannot override the artifact gate.  Do not retune this scene after the
  first complete outcome.

## Interfaces touched

`src/structsplat/realtime_gs_ray_posterior.py`, the codec-native adapter only if a bounded query seam
is missing, focused tests, one bounded driver under `scripts/experiments/`, architecture/research
documentation, an ADR amendment or successor if ownership changes, ARA evidence after execution,
this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

CORE-016/017, CORE-013, BENCH-019/020, BENCH-002, ADR-0006/0032

## Agent workflow

- Driver: codex-root
- Reviewer: codex-root
- Turn: reviewer
- Reviewed revision: commit `6525f82`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.  Any formal
confirmation requires the distinct prospective protocol review that this diagnostic deliberately
does not claim.

### Handoff

#### Objective

Review the default-off packet-derived ray-posterior initializer and the exposed disjoint-view
diagnostic, preserving the fail-closed reciprocal result and the negative visual disposition.

#### Changes

Added lazy packet-only semantic/detail feature construction, robust source-excluded depth scoring
with a dustbin, reciprocal mode support, deterministic exact-budget selection, optional surface
cover, and a narrow continuous-appearance query seam.  Added the frozen three-arm driver, synthetic
and realtime-gs integration tests, immutable result/audit records, and synchronized architecture,
ADR, README, task/index/brief, and core-skill boundaries.

#### Evidence

The immutable partial bundle manifest is
`e11c4a73e8a94afbf149e52b9a1acc889bf22a7beb2ec4bd89c27ac36f8d0610`.
All 246 recorded path/byte/SHA descriptors replay, and the reporting split is disjoint.  Packet
decode averages 38.581 dB while 1,360,834 packet bytes replace 15,741,328 original JPEG bytes.  The
no-reciprocal arm improves initialization by 1.8463 dB but is 0.8459 dB behind interior consensus at
step 500, reaches that control's terminal PSNR only at step 1,200, worsens gradient MAE, and ends at
30,000 versus 29,422 rows.  The complete reciprocal arm fails before training at its frozen 75%
primary-support floor.  Native inspection finds translucent smeared volumes in both completed arms.
The optional realtime-gs focused suite passes 40 tests; before commit `6525f82`,
`./scripts/verify.sh` passed with 1,686 tests, 19 skips, 514 deselections, and all structural gates
green.

#### Assumptions

The downscale-8 calibrated tensors are the diagnostic target.  Shared learned weights are an
amortized dependency and not per-scene payload, while their provenance remains explicit.  A
fail-closed arm is a valid protocol outcome, not missing evidence to be rescued by lowering its
threshold.

#### Uncertainties

This is one consumed development scene and one seed, with dirty executed source, a task-local report
schema, approximate timing, reduced resolution, no physical geometry truth, and no distinct review.
Its numerical deltas cannot establish general compression, runtime, or reconstruction quality.

#### Review focus

Audit packet-only feature provenance, reporting-view exclusion, dustbin/evidence semantics,
reciprocal independence, deterministic balanced selection, exact-count and primary-fraction
failure behavior, covariance/radiance immutability, byte/time boundaries, convergence arithmetic,
and the mandatory visual rejection.

#### Protected actions not taken

No realtime-gs source, maintained StructSplat renderer/default, packet grammar, prior immutable
result, reporting target, public claim row, or unrelated IntelliJ file was changed.  The consumed
scene was not rerun or retuned after outcome inspection.

#### Recommended next action

Obtain distinct review.  Retire independent per-ray matching as the geometry unit; test a spatially
coherent multiview geometry prior with calibrated alignment on new disjoint data, keeping the
interior and no-reciprocal arms as causal controls.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Commit `6525f82` keeps torch imports lazy, leaves training/render equations untouched, derives
features only through packet appearance queries, excludes the source view from depth likelihoods,
and computes reciprocal support from independently selected candidate modes.  Invalid evidence
enters an explicit dustbin, ties remain deterministic, exact-N selection is balanced, and the
primary-support floor fails before optimization.  Surface-cover checks forbid changes to means or
SH.  Focused tests exercise coherent/occluded/repeated-texture depths, ties, no-evidence states,
reciprocal floaters, exact counts, determinism, finite covariance, device behavior, index isolation,
malformed ownership, frozen split/configuration, and decision gates.

#### Evidence quality

Question, split, packet set, arms, budgets, seed, telemetry, and killing rule were frozen before the
first outcome.  Every surviving arm and the full-arm error receipt remain preserved; descriptor
receipts and decision arithmetic replay.  Evidence remains diagnostic because the bundle is
partial and dirty-source, the custom checker reports its 42 documented contract mismatches, and
one exposed scene/seed plus self-review cannot support promotion.

#### Simplicity

The implementation is isolated behind one optional module and one ten-line adapter seam, reusing
CompactCarve proposals, realtime-gs cameras/Gaussians/surface cover, and the existing packet
appearance plane.  It changes no format or default.  The negative result prevents further
threshold or optimizer complexity from accumulating around an ambiguous independent-ray unit.

#### Missing cases

Distinct review, clean-source execution, multiple fresh scenes/seeds, geometry ground truth,
occlusion/thin-structure strata, full-resolution evidence, a maintained report schema, production
packet creation, coded final-model rate, render FPS, and end-to-end latency remain absent.

#### Required changes

None for retaining the implementation and negative result as default-off diagnostic evidence.
Distinct review is required before treating the workflow record as accepted, and a successor must
change the geometry model rather than tune thresholds on `frame_00060`.

#### Optional improvements

If this family is reused as a control, port its bundle to the maintained report schema and expose a
small portable receipt checker.  Future geometry tests should measure calibrated camera/depth
alignment before Gaussian training and reject visually incoherent geometry at step zero.

## Notes

The causal prediction is specific: if CORE-017's trail is caused mainly by unresolved depth rather
than packet appearance or the optimizer, a source-excluded posterior should become narrow at
repeatable surfaces, reciprocal agreement should delete isolated modes, and quality should improve
at step zero and early fixed-topology checkpoints.  If posterior geometry is not visually cleaner
before density growth, retire this route rather than hiding it behind longer optimization.

## Diagnostic outcome — 2026-08-06

The immutable local v1 bundle is
`results/core018_ray_posterior_karate_frame00060_2026-08-06_v1/`, manifest SHA-256
`e11c4a73e8a94afbf149e52b9a1acc889bf22a7beb2ec4bd89c27ac36f8d0610`.  It is a partial,
dirty-source, task-local diagnostic.  Interior consensus and the no-reciprocal posterior complete;
the full candidate fails closed before optimization because its selected reciprocal-primary
fraction is below the frozen `0.75` floor.  The scene was not retuned or rerun.

The no-reciprocal arm improves initial reporting PSNR by 1.8463 dB and has lower complete
pretraining time, but by the fixed-topology step-500 checkpoint it is 0.8459 dB worse than interior
consensus.  It ends only 0.0928 dB higher, with worse gradient MAE, 30,000 versus 29,422 final rows,
and a slightly larger final model.  Candidate median entropy is 0.9596, mean selected confidence is
0.04149, and median reciprocal support is zero.  Both completed arms are visibly unusable smeared
volumes on every reporting view.  The full method and the independent-ray geometry unit are
rejected; lowering the reciprocal threshold on this consumed scene is prohibited.  Exact protocol,
metrics, hashes, visuals, checker limitations, and causal disposition are in
`docs/research/2026-08-06-core018-ray-posterior-results-audit.md`.

## Ledger disposition — 2026-08-08

The outstanding ARA disposition is now closed. Every number above was independently recomputed from
the raw bundle rather than copied from prose, and `manifest.json` re-hashes to `e11c4a73…`. The
rejection is bound to `ara/evidence/core018-ray-posterior-karate-2026-08-06/run.md`, trace nodes
`N258`/`N259`/`N262`, staging `O135`, and refuted claim `C62`. No method, threshold, budget, or
result was changed. The distinct-review requirement is unchanged and still open; the ledger records
a rejection, not an acceptance.
