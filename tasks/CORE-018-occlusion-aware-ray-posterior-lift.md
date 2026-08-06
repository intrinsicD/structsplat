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
- [ ] A diagnostic-only disjoint `karate/frame_00060` comparison reuses one immutable packet set and
      compares ordinary interior consensus, posterior without reciprocity, and the complete
      posterior.  All arms use the same training/reporting cameras, 10,000 initial rows, staged
      500-step fixed-topology plus 1,000-step shared density refinement, and a 30,000-row cap.
- [ ] The report includes complete original JPEG, packet, shared-model, final-model, and index bytes;
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
- [ ] Task, Index, generated brief, architecture/ADR boundary, tests, evidence disposition, exact
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
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.  Any formal
confirmation requires the distinct prospective protocol review that this diagnostic deliberately
does not claim.

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
