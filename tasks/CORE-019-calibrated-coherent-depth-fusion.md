# CORE-019 — Calibrated coherent-depth fusion

## Context

CORE-016 preserves useful packet appearance at lower teacher-input rate, but its CompactCarve
geometry is volumetric. CORE-017 moves centers to a visual-hull shell yet retains directional
double silhouettes. CORE-018 replaces mask support with independently scored ray-depth modes; its
no-reciprocal arm remains a translucent smeared volume and its reciprocal arm fails the frozen
support floor. The unit that must change is therefore the geometry field before ray selection, not
another merge threshold, covariance repair, touched-row optimizer, or longer fit.

A construction-only probe on already-consumed CORE-018 packet decodes establishes that a pinned
VGGT model fits the available RTX 3050 and produces spatially coherent four-view depth that aligns
to the known cameras. The probe is not method evidence: it opened no fresh reporting view and did
not emit or optimize Gaussians. The candidate selection and exact probe boundary are recorded in
`docs/research/2026-08-07-core019-coherent-depth-portfolio.md`.

## Goal

Implement and kill-test a lazy, default-off packet-to-realtime-gs initializer that predicts
overlapping coherent multiview depth groups, aligns each group to known calibration, fuses depth
and uncertainty per construction view, accepts only occlusion-consistent continuous surface
proposals, and contracts/eliminates feature-redundant proposals to an exact 10,000-Gaussian field
with overlap-safe surfel covariance and complete rate/time/quality accounting.

## Non-goals

- Do not change realtime-gs source, StructSplat's maintained conversion pipeline, renderer or
  optimizer equations, the `.sgdp` grammar, or any default.
- Do not use reporting cameras, source RGB after packet construction, renders, optimized geometry,
  or target metrics in grouping, depth alignment/fusion, proposal selection, contraction, WSE, or
  stopping.
- Do not call VGGT, Sim(3) alignment, projective fusion, surfels, or weighted sample elimination
  novel. This is a systems composition and empirical question.
- Do not claim physical depth, generalization, commercial usability, artifact freedom, or general
  compression from one exposed development scene. The public checkpoint is CC-BY-NC-4.0; a
  production adapter must also accept separately supplied commercial weights.
- Do not hide the 5.03 GB encoder checkpoint, packet bytes, or final scene bytes. Shared encoder
  weights are excluded from per-scene payload but are reported separately with license and cold
  load cost.

## Acceptance criteria

- [ ] A lazy optional module imports without torch, realtime-gs, VGGT, or CUDA and exposes pinned
      source/checkpoint receipts, injected-predictor tests, mixed-precision inference, bounded
      four-view grouping, and the single OOM fallback frozen below.
- [ ] Every predicted group is aligned by one Sim(3) from predicted to known camera centers; only
      the group scale converts its depths, while known calibrated rays own back-projection. Views
      appearing in multiple groups use robust confidence-weighted depth fusion and MAD uncertainty.
- [ ] Candidate acceptance distinguishes projective support, compatible occlusion, free-space
      contradiction, invalid evidence, and reporting-view exclusion. Output means are continuous
      back-projections/contractions and are never snapped to a pixel, voxel, KD-tree, or quadtree.
- [ ] Structural and bounded flat-cover proposals carry explicit lineage. Consistency-gated local
      contraction and dynamic feature/normal/color-aware WSE produce an exact count, preserve a
      declared per-view cover floor, and report displacement, cluster, crowding, rejection, and
      feature-retention tails.
- [ ] Degree-0 packet appearance, local normals, tangent coverage, bounded normal thickness, finite
      SPD covariance, opacity, lineage, depth, uncertainty, and score form a valid realtime-gs
      `CompactInitializationResult`; optional surface-cover reconciliation cannot change means,
      radiance, lineage, or count.
- [ ] Synthetic tests cover exact calibrated planes/surfaces, group-scale recovery, overlap fusion,
      half-pixel conventions, occlusion versus contradiction, depth/color/normal discontinuities,
      deterministic exact-N WSE, continuous non-snapped centers, finite SPD covariance, missing
      evidence, malformed receipts, lazy import, and report-camera isolation.
- [ ] A development-only four-arm run on the frozen scene/split below cold-reloads identical packet
      hashes, uses the common realtime-gs optimizer/schedule, and stores every requested metric,
      curve, native visual, model, byte/time/memory receipt, and a replayable decision.
- [ ] The complete candidate fails closed on any reporting-view smear, duplicate shell, trail,
      floater/sheet, grid imprint, boundary hole, or thin-feature deletion; scalar averages cannot
      override the native visual gate. No threshold may be rescued on this scene.
- [ ] Task, Index, generated brief, architecture/ADR boundary, README/core skill where relevant,
      tests, evidence disposition, exact command, ARA records, and visual decision are synchronized;
      focused checks and `./scripts/verify.sh` pass.

## Frozen development protocol

This is one exposed scene and one seed. Without a distinct prospective protocol reviewer, its
outcome is diagnostic and cannot promote a default or public claim.

- **Data:** `/home/alex/Dropbox/Work/Janelle/karate/frame_00005` and sibling
  `calibration_dome.json`. Reporting cameras were selected from calibration geometry before the
  frame was opened: `C0024`, `C0010`, `C1004`, `C0022`. All other available calibrated cameras are
  construction cameras. Packet creation uses undistorted downscale-4 tensors; training/reporting
  use undistorted downscale-8 tensors.
- **Common budgets:** seed 0; 10,000 initial rows; SH degree 3; 500 fixed-topology steps followed by
  the unchanged shared density controller through step 1,500; density begins no earlier than step
  600; maximum 30,000 rows; checkpoints at step 0 and every 100 attempted steps.
- **Packets:** identical cold-reloaded codec-native WebP-quality-80 dual-plane packets for every
  arm, full canvas/alpha, 1,024 structural proposals per construction view, lattice sigma `0.45`,
  radius `3`, eight Jacobi steps, seed by construction-view index. No original RGB may be opened
  after packet construction.
- **VGGT dependency:** source revision `a288dd0f14786c93483e45524328726ab7b1b4ce`;
  `facebook/VGGT-1B` revision `860abec7937da0a4c03c41d3c269c366e82abdf9`;
  `model.safetensors` byte count `5,026,367,224`, SHA-256
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`,
  CC-BY-NC-4.0. Aggregator is bfloat16, camera/depth heads float32, and the outer CUDA autocast is
  bfloat16. Whole-model bfloat16 is prohibited by the pinned head boundary.
- **Inference:** calibration-only deterministic groups of exactly four construction views, max
  side 392 and patch multiple 14. One max-side-336 retry is permitted only after CUDA OOM. Reject
  nonfinite predictions or a group with nonpositive/degenerate Sim(3). Predicted poses are
  scale/diagnostic state, never deployment cameras.
- **Fusion/support:** confidence-weighted robust median/MAD across overlapping group estimates.
  For candidate depth `z` projected into target fused depth `d`, `|z-d|` inside the uncertainty-
  scaled tolerance is support; `d < z-tolerance` is compatible occlusion; `d > z+tolerance` is a
  free-space contradiction; invalid/outside/low-confidence samples provide no evidence. Complete
  candidates need at least one independent support view and bounded contradiction.
- **Selection:** structural mass proposals plus a bounded low-density confidence cover; continuous
  local contraction only across compatible depth/normal/color/support state; dynamic weighted
  sample elimination to exactly 10,000; overlap-safe surfel covariance; all output rows trainable.

## Causal arms

1. `interior`: ordinary CompactCarve interior consensus plus surface cover.
2. `posterior_no_reciprocal`: CORE-018's completed independent-ray negative control.
3. `vggt_raw_known_ray`: calibrated VGGT depths and deterministic exact-budget selection, without
   projective support, contraction, or weighted elimination.
4. `vggt_coherent_wse`: complete coherent fusion/support/contraction/WSE candidate.

All arms consume the same packet files and training/reporting split. The raw/full pair is the
mechanism ablation. A failed candidate remains a persisted error arm; controls are not skipped.

## Metrics and artifacts

- Construction/reporting PSNR, SSIM, MS-SSIM, LPIPS, MAE, gradient MAE, p95/p99 absolute error,
  alpha mean/p05, and Gaussian count at every checkpoint; steps/time to each control terminal
  target and quality-vs-time area through step 500.
- Camera alignment, fused depth/MAD/confidence, projective support/occlusion/contradiction, normal
  agreement, rejection counts, contraction size/displacement/covariance inflation, WSE crowding,
  feature retention, screen-space holes, and nearest-neighbor regularity.
- Packet encode/decode, model hash/load/inference, fusion/selection/lift/train/render times; peak
  host RSS and CUDA memory; group and query work.
- Exact original JPEG bytes, complete packet bytes, initial/final compressed NPZ, PLY and raw model
  bytes; `original/model`, `original/packets`, and conservative
  `original/(packets+final_model)`. Encoder bytes are separate and never scene compression.
- Immutable plan/manifest, JSON/JSONL/CSV curves, target/init/checkpoint/final/RGB-error/alpha/depth/
  support panels, NPZ/PLY models, hashes, HTML index, portable checker output, and results audit.

## Frozen advancement gate

The construction-only predecessor probe already passed finite output, four-view `<=120 s`,
projective valid fraction `>=0.25`, relative-depth median/p90 `<=0.12/0.35`, and RGB-L1 median/p90
`<=0.15/0.30`. The method advances beyond this diagnostic only when all following conditions hold:

- Every native reporting image passes the artifact gate above.
- At step 0, the complete arm improves reporting PSNR by at least 2 dB over `interior` and does not
  worsen gradient MAE or LPIPS.
- At step 500 it is within 0.1 dB PSNR and 0.01 MS-SSIM of the strongest control, has no LPIPS or
  gradient-MAE regression, and reaches that control's terminal reporting PSNR no later in steps or
  wall time.
- Terminal reporting PSNR/MS-SSIM/LPIPS/gradient MAE is Pareto-nondominated, no protected metric
  regresses, full beats raw on at least one prespecified geometry-tail and one quality/convergence
  measure without more final rows, and conservative `original/(packets+final_model) > 1`.

## Exact command

```bash
PYTHONPATH=/home/alex/Documents/vggt:/home/alex/Documents/realtime-gs/src \
  /home/alex/Documents/realtime-gs/.venv/bin/python \
  scripts/experiments/core019_coherent_depth_downstream.py \
  --frame /home/alex/Dropbox/Work/Janelle/karate/frame_00005 \
  --weights /home/alex/.cache/huggingface/hub/models--facebook--VGGT-1B/blobs/f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e.repairing-20260807 \
  --out results/core019_coherent_depth_karate_frame00005_2026-08-07_v4
```

## Interfaces touched

`src/structsplat/realtime_gs_coherent_depth.py`, focused tests, one bounded driver under
`scripts/experiments/`, architecture/research documentation, ADR-0032's successor boundary, ARA
evidence after execution, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

CORE-016/017/018, CORE-013, BENCH-019/020, BENCH-002, ADR-0006/0032

## Agent workflow

- Driver: codex-root
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`. A formal
confirmation requires a distinct prospective protocol review; self-review can retain only a
diagnostic result.

## Notes

The first immutable execution attempt (`..._v1`) failed before packet construction because the
driver inherited C0000/C0001 from CORE-018 although frame_00005 contains neither image. The v2
driver implements the already-frozen rule “all available cameras except the four reporting
cameras”: 26 construction cameras plus the unchanged four reporting cameras. No image, metric,
depth prediction, Gaussian, or optimizer state existed when this correction was made.

The immutable v2 execution then failed two construction-count contracts before either coherent
arm emitted a Gaussian: the inherited interior control's 2x proposal pool could not supply 10,000
supported rows, and the coherent structural sampler exhausted eight rounds only 163 rows short of
its 32,000-row quota. The v3 correction changes no support, depth, appearance, optimizer, or
reporting rule: the interior control alone receives a 4x candidate pool and coherent proposal
generation receives twelve bounded rounds. The successful v2 posterior arm remains negative
diagnostic evidence, not a source of threshold tuning.

The causal prediction is intentionally strict: coherent depth must produce visibly surface-like
step-zero geometry. If it requires the optimizer to erase a volume, if WSE exposes regular spacing
or holes, or if the raw and full arms are visually indistinguishable, retire the composition rather
than adding threshold or optimizer complexity on the consumed frame.

The immutable v3 execution completes all four arms. The full support/WSE candidate improves over
raw known-ray depth at the terminal checkpoint (+0.3707 dB PSNR, +0.02838 MS-SSIM, -0.02210 LPIPS,
-0.000142 gradient MAE, and 1,223 fewer final rows), while preserving all 1,500 hard feature anchors
and contracting only 96 eliminated cross-view proposals. It nevertheless starts 0.9690 dB below
interior consensus, misses the +2 dB and LPIPS step-zero gates, fails every step-500 comparison,
never reaches the interior terminal PSNR, and ends 0.6437 dB below it. Native review rejects every
arm: the coherent variants progress from black holes/floaters to broad gray sheets and radial
streaks with erased detail under the shared optimizer. The result is negative; v4 is only the
committed-source/schema-clean diagnostic replay required for handoff, not a rescue. See
`docs/research/2026-08-07-core019-coherent-depth-results-audit.md`.
