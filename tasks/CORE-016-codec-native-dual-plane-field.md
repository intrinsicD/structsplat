# CORE-016 — Codec-native dual-plane Gaussian observation field

## Context

HIER-005--009 show that reducing an explicit pixel-Gaussian field by local contraction can retain
average quality while still producing unacceptable lattice, hole, or redistributed local-error
artifacts.  The same explicit rows currently carry appearance, renderer weight, structural meaning,
and most of the encoded geometry cost.  Realtime-gs already exposes a pluggable point-query backend,
so a Stage-1 representation does not need to materialize every appearance sample as an independently
stored and lifted Gaussian.

## Goal

Implement and kill-test a default-off, self-contained reference packet with two coupled but
semantically distinct planes:

1. an implicit pixel-lattice normalized-Gaussian appearance field whose decoded coefficients are
   carried by a conventional image codec; and
2. a sparse nonnegative anisotropic structural Gaussian measure used for proposals and 2D-to-3D
   lifting.

The packet must support continuous point queries, complete-byte accounting, exact alpha gating,
deterministic cold decode, and a paired realtime-gs `ObservationQueryBackend` adapter without
changing the maintained StructSplat pipeline or realtime-gs checkout.

## Non-goals

- Do not claim a new Laplacian pyramid, RBF method, image codec, structured Gaussian grammar, or
  global compression result; those mechanisms have direct prior art.
- Do not promote Field V2 semantics, replace COMP-013, change `GaussianField`, alter the conversion
  CLI/defaults, or write into the currently dirty realtime-gs worktree.
- Do not treat an exposed Janelle diagnostic as held-out or confirmation evidence.
- Do not describe a sparse structural field without its paired appearance backend as a faithful
  teacher.

## Acceptance criteria

- [x] A NumPy/Pillow reference packet has a strict versioned grammar, canonical metadata and
      checksums, bounded decoding, exact complete-byte accounting, and deterministic repeated
      encode/decode for supported codecs.
- [x] The appearance plane implements a finite, numerically continuous normalized Gaussian-lattice
      query at arbitrary crop/canvas coordinates; pixel-center replay, alpha gating, constant-field
      reproduction, boundary behavior, and malformed inputs have focused tests.
- [x] A deterministic exact-count structural allocator reuses the repository structure tensor,
      keeps nonnegative mass separate from appearance, records the seed/config, and exports the
      structure plane through `ObservationField2D`.
- [x] A lazy optional adapter produces a paired realtime-gs structural
      `GaussianObservationField` plus `ObservationQueryBackend`; tests or a local compatibility
      check verify query/weight/coordinate parity without importing realtime-gs or torch at base
      module import time.
- [x] A bounded diagnostic driver reports source and canonical-PNG bytes, packet/component bytes,
      original-file ratio, PSNR/SSIM/MS-SSIM when available, gradient/high-pass error, worst-pixel
      and multiscale-patch error, encode/cold-decode/query/render time, structural count, and a
      quality--bytes--time curve with visual originals/reconstructions/errors.
- [x] The first Janelle C0001 diagnostic compares fixed codec/quality and structural-count ladders
      against the extant `.rtgsv`/HIER-009 evidence without relabelling incompatible metrics.  Kill
      the architecture if no packet is artifact-safe, smaller than the exact source, materially
      faster to encode than the extant iterative fit, and query-compatible with realtime-gs.
- [x] The research portfolio, prior-art threats, architecture boundary, task state, and generated
      session brief are synchronized; focused tests and `./scripts/verify.sh` pass.
- [x] A source-grounded exposed multiview assay propagates the paired CUDA query backend through
      real CompactCarve lifting and common 3DGS refinement, retains reporting-only cameras outside
      packet construction/training, charges complete input bytes, and reports full convergence,
      quality, alpha, count, memory, model, and native-pixel visual artifacts.

## Interfaces touched

`src/structsplat/codec_native_field.py`, an optional realtime-gs adapter module, focused tests,
`scripts/experiments/`, `docs/adr/`, `docs/research/`, `docs/architecture.md`, ARA staging/evidence
only if a diagnostic is retained, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Exposed real-multiview development protocol

Frozen before the downstream arm outcomes are accessed. This is a single-frame, single-seed,
reduced-resolution diagnostic on already exposed Janelle data. It may reject or refine CORE-016,
but cannot promote a default, satisfy BENCH-019, or support a general compression/reconstruction
claim.

- **Question:** can the required dual-plane packet/backend pair replace the extant compact 2D
  teacher at materially lower complete input bytes without materially degrading source-grounded
  3D initialization, convergence, or held-out view quality?
- **Source and split:** calibrated `frame_00008`; training cameras `C0001`, `C0006`, `C0012`,
  `C0019`, `C0022`, `C0028`, `C0031`, `C0039`; reporting-only cameras `C0004`, `C0025`,
  `C1004`. Targets are loaded directly from the common calibrated RGB/masks, never rendered from
  either teacher. Packet construction uses the undistorted `downscale=4` training tensors;
  optimization and reporting use independently loaded undistorted `downscale=8` source tensors.
- **Arms:** existing full-resolution `gaussians2d/*.rtgsv` control; WebP quality-92 dual-plane
  packets with 512 structural rows/view; and the same packets with 2,048 structural rows/view.
  Candidate packets use sigma `0.45`, radius `3`, eight Jacobi prefilter steps, exact alpha, a
  16-pixel packet-resolution mask-crop margin, and per-view seeds `0..7`.
- **Rate:** sum complete physical bytes for the eight reconstruction inputs, plus per-member packet
  ledgers, control container bytes, canonical undistorted crop-PNG bytes, raw RGB bytes, and final
  NPZ/PLY bytes. Ratios with unlike source/crop extents remain explicitly contextual.
- **Lift:** the same compact bounds, `CompactCarveConfig(n_init_3d=835,
  candidate_multiplier=4, anchor_mode="mass_random", samples_per_ray=48,
  query_batch_size=4096, seed=0, bounds_scale=0.5, min_views=2, hull_fraction=0.85,
  coverage_scale=1.0, coverage_threshold=0.40, color_std_sigma=0.20, min_score=0.05)`.
  Structural metadata and cameras remain on CPU; all indexed teacher queries use the paired CUDA
  backend. No structural-color fallback is permitted.
- **Refinement:** identical fixed-topology 835-Gaussian gsplat runs for 1,000 attempted steps,
  evaluation every 50 steps, final checkpoint, SH degree 3, masks enabled, antialiasing enabled,
  standard random background, seed 0, and no densification. Fixed topology deliberately isolates
  input/initialization quality before any capacity-changing follow-up.
- **Outputs:** init/final train and held-out PSNR variants, SSIM, MS-SSIM, LPIPS, alpha metrics,
  elapsed time, loss, Gaussian count, lift diagnostics, checkpoint curves, PLY/NPZ models, and
  native-resolution target/render/absolute-error panels for all three held-out views. Input hashes,
  exact StructSplat/realtime-gs revisions, dirty status, source snapshots, environment, and every
  failed cell remain visible.
- **Development gate:** advance the architecture to a separately reviewed variable-topology assay
  only if the 2,048-row arm is at least 3x smaller than the control input bundle, finishes without
  non-finite values or visible grid/hole artifacts, and ends within 1.0 dB held-out foreground PSNR,
  0.02 held-out MS-SSIM, and 0.03 held-out LPIPS of the control. Prefer 512 rows only if it is within
  0.25 dB of 2,048 rows. Otherwise diagnose or kill the representation; do not rescue it by looking
  at the reporting cameras while retuning.
- **Execution:** `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --out
  results/core016_multiview_downstream_janelle_2026-08-06_v1`. The output directory is immutable.

### Post-v1 variable-topology recovery protocol

Frozen after auditing v1 and before accessing any variable-topology outcome. The arm selection is
explicitly post-hoc: v1 selected the 512-row packet as its observed rate/quality Pareto point. This
follow-up can test whether standard 3D density control removes v1's unacceptable blur, rays, bright
silhouette halos, and missing thin structure; it cannot convert the original frozen v1 decision or
its already viewed reporting cameras into confirmation evidence.

- Reuse the exact v1 source tensors, eight-train/three-reporting split, packet construction,
  complete-byte accounting, CompactCarve configuration, 835-row initialization, renderer, loss,
  masks, SH schedule, seed, and final-checkpoint rule.
- Compare only the extant RTGSV control and post-hoc-selected WebP-q92/512-row dual-plane arm.
- Run 2,000 attempted gsplat steps with `gsplat-default` density control: start 100, stop 1,000,
  interval 100, absolute-gradient threshold `8e-4`, split scale fraction `0.01`, split factor `1.6`,
  prune opacity `0.005`, prune scale fraction `0.1`, revised opacity, opacity reset 1,000/value
  `0.011`, and maximum 20,000 Gaussians. Evaluate every 100 steps and retain full curves.
- The scalar gate requires at least 3x lower complete input bytes; candidate deltas no worse than
  -0.5 dB foreground PSNR, -0.01 MS-SSIM, +0.02 LPIPS, and -0.02 alpha IoU; absolute candidate
  held-out PSNR at least 22.5 dB and alpha IoU at least 0.90; both final counts at most 20,000; and
  candidate final count at most 1.10x control. Native-pixel visual review is independently
  mandatory, so the driver cannot auto-advance this profile even when every scalar passes.
- Execute with `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile density --out
  results/core016_multiview_density_janelle_2026-08-06_v2`. The output directory is immutable.

### Full-capture geometry protocol

Frozen after the v2 scalar/visual audit and before any 23-view outcome. V2 showed that standard
density control removes most fixed-topology blur but leaves streaks/floaters and absolute silhouette
quality below gate. The causal hypothesis for this follow-up is sparse angular support: eight input
cameras force large 3D splats to bridge unobserved regions. This is again exposed post-hoc
development evidence, not a new held-out split.

- Retain `C0004`, `C0025`, and `C1004` as reporting-only. Use all other 23 calibrated cameras in
  `frame_00008` for packet construction, lifting, and training. No reporting-camera packet may
  exist. Load packet inputs at downscale 4 and common source targets at downscale 8 exactly as v1/v2.
- Compare the existing RTGSV control against only the WebP-q92/512-row dual-plane packet. Use the
  exact paired CUDA query backend, complete physical input bytes, common compact bounds, and the
  v1/v2 masks/renderer/loss/seed conventions.
- Increase the matched CompactCarve output from 835 to 5,000 Gaussians. Then run the exact v2
  2,000-step `gsplat-default` density schedule with a 20,000-Gaussian cap and 100-step telemetry.
- Require candidate/control input ratio at least 3x; candidate deltas no worse than -0.5 dB PSNR,
  -0.01 MS-SSIM, +0.02 LPIPS, and -0.02 alpha IoU; absolute candidate reporting PSNR at least 24.0
  dB and alpha IoU at least 0.93; both final counts at most 20,000; and candidate final count at most
  1.10x control. Native-pixel review must separately reject visible grids, rays, streaks, floaters,
  bright halos, holes, or missing thin anatomy; the driver therefore cannot auto-advance.
- Execute with `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile full --out
  results/core016_multiview_full23_janelle_2026-08-06_v3`. The output directory is immutable.

### Post-v3 matched-topology protocol

Frozen after the v3 scalar and native-pixel audits and before accessing any matched-cap outcome.
V3 cleared the complete-input-rate and absolute/relative quality gates, but its 11,689-Gaussian
candidate exceeded the 10,022-Gaussian control by 16.6%, failing the predeclared 1.10x count gate.
The candidate was visibly much better than the control but still exhibited thin silhouette halos
and isolated floaters, so this experiment tests count efficiency only and cannot establish an
artifact-free endpoint. It is exposed post-hoc development evidence on the same reporting views.

- Reuse the exact v3 23-train/three-reporting split, source tensors, packets, byte accounting,
  5,000-Gaussian CompactCarve initialization, renderer, loss, SH schedule, seed, 2,000 steps, and
  100-step telemetry. Compare only the existing RTGSV control and WebP-q92/512-row candidate.
- Change one downstream variable: set the `gsplat-default` density controller's maximum from
  20,000 to 10,000 Gaussians for both arms. Retain every other density-control parameter.
- Retain v3's rate and absolute/relative quality gates. Require both final models at or below
  10,000 Gaussians and retain the candidate-at-most-1.10x-control count gate. Native-pixel review
  remains mandatory and the driver cannot auto-advance even if every scalar gate passes.
- Execute with `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile matched10k --out
  results/core016_multiview_full23_matched10k_janelle_2026-08-06_v4`. The output directory is
  immutable.

### Post-v4 silhouette-supervision protocol

Frozen after the v4 scalar and native-pixel audits and before accessing any stronger-mask outcome.
V4 put both arms at exactly 10,000 Gaussians and cleared every scalar gate, but its candidate still
showed soft silhouette halos and fine-detail blur. Its final sampled-view alpha-loss contribution
was smaller than the color and D-SSIM terms, motivating one direct mask-supervision assay. This is
again exposed post-hoc development evidence on already viewed reporting cameras.

- Reuse the exact v4 inputs, 23/three split, initialization, 10,000-Gaussian density schedule,
  renderer, SH schedule, random backgrounds, seed, 2,000 steps, telemetry, and final checkpoint.
- Change only the two coefficients of the existing exact-mask objective: raise
  `mask_alpha_lambda` from `0.05` to `0.20` and `outside_alpha_lambda` from `0.01` to `0.05` for
  both arms. This is one silhouette-supervision factor, not a packet or geometry change.
- Retain v4's scalar gates, tighten absolute candidate reporting alpha IoU from 0.93 to 0.95, and
  require native-pixel review. Relative to v4, prefer the change only if it visibly reduces halos
  and floaters without losing more than 0.2 dB candidate PSNR or worsening gradient MAE. The driver
  cannot auto-advance because visual disposition remains mandatory.
- Execute with `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile silhouette --out
  results/core016_multiview_full23_silhouette_janelle_2026-08-06_v5`. The output directory is
  immutable.

### Post-v5 late silhouette-polish protocol

Frozen after v5's scalar and visual audits and before accessing any polish outcome. Strong exact-
mask weights throughout training raised candidate alpha IoU from 0.9497 to 0.9604 but lost 0.446
dB against v4 and worsened gradient MAE, failing the predeclared retention preference. The causal
hypothesis is that strong alpha pressure during density/appearance formation sacrifices texture,
whereas a short fixed-topology cleanup after normal training may tighten silhouettes while
preserving v4's representation. This remains exposed post-hoc development evidence.

- Reproduce the exact v4 2,000-step training under the 10,000-Gaussian cap. Then freeze topology
  and run 250 additional steps: no density events, no added/removed Gaussians, SH degree 3, exact
  masks and random backgrounds, evaluation every 50 steps, and final-checkpoint selection.
- In only the polish phase, use v5's `mask_alpha_lambda=0.20` and
  `outside_alpha_lambda=0.05`. Reset Adam for the explicit new phase, reduce every parameter-group
  base learning rate to 0.20x, preserve the global means-LR coordinate at steps 2,000--2,250, and
  use deterministic seed 1 for polish view sampling.
- Retain v4's complete-input, relative-quality, 10,000-count, and absolute 24 dB gates; require
  candidate alpha IoU at least 0.95. Prefer polish only if it visibly reduces halos/floaters while
  keeping candidate PSNR within 0.2 dB of v4 and gradient MAE no worse than v4. Native-pixel review
  remains mandatory and the driver cannot auto-advance.
- Execute with `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile latepolish --out
  results/core016_multiview_full23_latepolish_janelle_2026-08-06_v6`. The output directory is
  immutable.

## Multiview diagnostic outcome (2026-08-06)

All outcomes are exposed, single-frame, single-seed, reduced-resolution development evidence. The
three reporting cameras were never packet inputs or training views, but they were inspected and
reused for post-hoc follow-ups; they are not confirmation data. The custom bundles are internally
hash/byte replayable but are not accepted by `scripts/check_report_bundle.py`.

| run | train views | topology | candidate reporting PSNR / alpha IoU | count control / candidate | disposition |
|---|---:|---|---:|---:|---|
| v1 fixed | 8 | fixed 835 | 21.112 / 0.6573 | 835 / 835 | safety arm misses 3x rate; severe blur/rays/halos |
| v2 density | 8 | cap 20k | 22.403 / 0.8752 | 5,631 / 7,688 | misses absolute PSNR, alpha, and count gates |
| v3 full capture | 23 | cap 20k | 25.203 / 0.9483 | 10,022 / 11,689 | quality/rate pass; candidate exceeds 1.10x count gate |
| **v4 matched cap** | **23** | **cap 10k** | **25.188 / 0.9497** | **10,000 / 10,000** | **retained Pareto point; scalar pass, visual fail** |
| v5 strong mask | 23 | cap 10k | 24.742 / 0.9604 | 10,000 / 10,000 | reject: -0.446 dB and worse gradient MAE vs v4 |
| v6 late polish | 23 | cap 10k + fixed 250 | 24.872 / 0.9548 | 9,989 / 10,000 | reject: -0.316 dB and worse gradient MAE vs v4 |

For v4, the complete 23-packet candidate input is 956,301 bytes versus 3,850,647 bytes for the
23 extant RTGSV containers (`4.0266x`). Both final models contain exactly 10,000 Gaussians. Against
the same reporting targets, the candidate is `+1.1761 dB` foreground PSNR, `+0.00650` MS-SSIM,
`-0.01944` LPIPS, and `+0.02097` alpha IoU. It first reaches the control's final reporting PSNR at
step 500 / 5.148 native optimization seconds; the control first reaches that level at step 1,400 /
13.266 seconds. This is a time-to-lower-target observation, not an end-to-end speed win: candidate
lift is 9.467 versus 4.894 seconds, full native training is 20.740 versus 18.521 seconds, and peak
VRAM is 0.366 versus 0.310 GiB. Candidate packet construction is measured, while production of the
pre-existing control containers is unavailable.

Native-pixel review finds no periodic grid/KD-tree imprint and substantially better anatomy than
the control, but still finds soft silhouette halos, fine-detail blur, and sparse floaters. V5 and
V6 tighten alpha but do not clear those artifacts without violating their frozen v4-retention
guards. Consequently v4 is the simplest retained development configuration; CORE-016 remains
default-off and does not establish artifact freedom, full-resolution quality, generality,
end-to-end speed, final-3D storage compression, or BENCH-019 confirmation.

## Depends on

CORE-013, BENCH-019, BENCH-020, COMP-013, BENCH-025, HIER-005/009, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex-root
- Reviewer: codex-root
- Turn: reviewer
- Reviewed revision: commit `30f62c9`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.  Any
formal result beyond the exposed diagnostic requires a distinct prospective protocol review before
execution.

### Handoff

#### Objective

Review the default-off v2 packet, continuous Gaussian-lattice decoder, independent structural
measure, realtime-gs adapter, and the narrow exposed C0001 killing-test interpretation.

#### Changes

Added the strict `.sgdp` producer/decoder and exact byte ledger, cardinal Gaussian prefilter,
structure-tensor/Halton exact-count allocator, lazy paired realtime-gs adapter, task-local diagnostic,
focused tests, ADR-0032, research portfolio, results audit, and ARA evidence note. No maintained
pipeline or format dispatch changed.

#### Evidence

`PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src pytest -q
tests/test_codec_native_field.py` passes 13 tests. `./scripts/verify.sh` passes with 1,673 passed, 6
skipped, and all five structural checkers green. The selected ignored bundle's 25-file custom
manifest, packet component sum, and cold byte-identical resave were independently replayed.

#### Assumptions

The C0001 source/mask are exposed development data. Bilinear decoded-raster interpolation is only an
off-grid control. The historical `.rtgsv`, iterative fit, and HIER-009 rows are contextual rather
than rate/work/preprocessing-matched baselines.

#### Uncertainties

Cardinal ringing remains measurable, the custom diagnostic schema is not accepted by
`check_report_bundle.py`, and neither 512-row structural sufficiency nor real multiview quality has
been established. The selected sigma/step setting is post-hoc.

#### Review focus

Audit finite-kernel coordinate/boundary parity, signed-coefficient conditioning, packet decode
bounds and canonicality, exact byte accounting, crop/full-frame rate wording, and whether the paired
backend is propagated through a real multiview lift without falling back to structural colors.

#### Protected actions not taken

No realtime-gs file, maintained StructSplat default, renderer equation, Field V2 semantic selection,
claim-ledger row, held-out split, or existing evidence bundle was changed or consumed.

#### Recommended next action

Obtain a distinct code/results review, then preregister the analytic/supersampled ringing assay and
a matched full-frame real-multiview downstream experiment before any promotion.

### Handoff

#### Objective

Propagate the codec-native pair through a real source-grounded multiview realtime-gs path, determine
whether it remains competitive at lower complete teacher-input bytes and equal final topology, and
test direct artifact-cleanup variants without modifying the realtime-gs worktree or maintained
StructSplat pipeline.

#### Changes

Added a CPU-structural/CUDA-query device split and telemetry passthrough to the lazy adapter, plus a
CUDA integration test. Added the bounded `core016_multiview_downstream.py` driver with common source
targets, explicit train/reporting separation, complete packet/control ledgers, exact repository and
source snapshots, CompactCarve/3DGS execution, convergence/quality/alpha/count/memory telemetry,
models, native visuals, and task-local HTML. Executed frozen v1--v6 profiles, retained matched-10k
v4, rejected strong-mask v5 and late-polish v6, and corrected future cross-run retention gates and
profile-aware plot labels without rewriting immutable results. Synchronized README, architecture,
ADR-0032, core skill, task/index, results audit, and ARA evidence/logic/trace.

#### Evidence

V4 manifest SHA-256 is `9c546d5c7b65f483326e525d8080b0a1c0928ff0805f50255b1191c4ff2d651c`.
At equal 10,000 final Gaussians, candidate/control inputs are 956,301/3,850,647 bytes and reporting
PSNR is 25.188/24.012 dB; all within-run scalar gates pass but native visual review fails. An
independent v2--v6 replay validates every receipt, 100 byte-identical packet resaves, source tensor
hashes, split membership, curve counts, finite models, and exact decision arithmetic. The maintained
report checker rejects the documented custom schema. The focused adapter slice passes 14 tests and
targeted Ruff/compile/docs/ARA checks pass. `./scripts/verify.sh` passes with 1,673 tests, 7 skips,
514 deselections, and all five structural checkers green before commit `30f62c9`.

#### Assumptions

Complete input bytes price only the reconstruction teachers; final model bytes are reported
separately. Existing RTGSV production time is unavailable. The common downscale-8 source tensors
are ground truth for this diagnostic, while candidate packets use downscale-4 calibrated crops.
Reporting-only means excluded from packet construction and training, not sealed or unbiased after
the first inspection.

#### Uncertainties

One exposed frame/nominal seed and three reused reporting views do not establish generality. CUDA
density control is not bitwise deterministic. The packet/control preprocessing and extents differ,
v5 timing has external CPU contention, custom reports are not portable-schema bundles, and native
halos/blur/floaters remain. Full-resolution quality, final-3D storage rate, render FPS, end-to-end
production time, multiscene behavior, and distinct scientific review remain open.

#### Review focus

Audit CPU/CUDA query device and return-device semantics, counter/payload accounting, train/reporting
leakage, complete-byte comparability, source-target independence, curve timing, count enforcement,
manual artifact disposition, custom-report limitations, the corrected v5/v6 cross-run gates, and
every claim that might overstate time-to-control-target as complete-pipeline convergence.

#### Protected actions not taken

No realtime-gs source, maintained StructSplat conversion path/default/format/renderer, reporting
target, existing evidence bundle, ignored immutable result, unrelated IntelliJ file, or public claim
ledger row was modified. No general compression, SOTA, artifact-free, held-out, or BENCH-019 claim
was made.

#### Recommended next action

Obtain distinct code/scientific review. If CORE-016 continues, stop tuning the exposed reporting
views and preregister a clean full-resolution multiscene assay with matched preprocessing, multiple
seeds, production input-generation timing, final 3D byte/FPS accounting, and an explicit
geometry/visibility artifact objective rather than another global alpha-weight sweep.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Commit `30f62c9` preserves the NumPy/torch import boundary and the required structural/query pair.
The adapter keeps CPU metadata valid while moving indexed structure and appearance work to CUDA,
returns query values to the caller's device, and mirrors realtime-gs telemetry. The new CUDA parity
test and six executed profiles exercise the real consumer. Review found and corrected the missing
v5/v6 cross-run gates and the profile-dependent plot label before commit. No differentiable
StructSplat loss/render path or maintained format/default is changed.

#### Evidence quality

Every protocol was written before its corresponding outcome, negatives were retained, reporting
views were excluded from packet construction/training, and the independent replay validates hashes,
bytes, cold packets, source tensors, splits, curves, finite models, and decision arithmetic. Native
visuals are inspected at stored resolution. Evidence remains diagnostic: one exposed frame/nominal
seed, reused reporting cameras, unmatched preprocessing/input-production time, non-bitwise CUDA
density, and an unrecognized custom report schema prohibit confirmation or promotion.

#### Simplicity

The production-facing change is one optional `query_device` seam plus telemetry propagation in the
existing lazy adapter. All orchestration, profiles, report generation, and post-hoc variants remain
in one bounded `scripts/experiments/` driver. The retained method is the simpler normal-loss 10k-cap
v4; v5/v6 machinery remains reproducible negative evidence rather than a default stage.

#### Missing cases

Distinct code/scientific review, clean portable report packaging, multiple scenes/seeds/devices,
matched full-resolution preprocessing and input production, sealed held-out/confirmation views,
render FPS, final 3D codec bytes, end-to-end wall time, and a geometry/visibility-specific artifact
objective remain absent. Thin structures, hair, disocclusions, and non-black backgrounds are not
separately stratified.

#### Required changes

None for retaining the implementation as a default-off development pilot. Distinct review and a
new preregistered dataset/protocol are required before any scientific acceptance, default change,
or broader compression/convergence/artifact claim.

#### Optional improvements

Add a maintained checker/schema if this diagnostic family is reused; isolate RTGSV production time
with a clean matched producer; make CUDA deterministic or run paired seeds; report final model
codec/FPS; and test an explicit visibility/geometry regularizer on disjoint full-resolution data
instead of further alpha-weight tuning on the exposed cameras.

## Notes

The key falsifiable systems claim is narrower than novelty: separating the query-quality plane from
the lift-structure plane should remove explicit per-row geometry from the appearance byte budget and
avoid nonlinear per-image fitting, while still giving realtime-gs continuous colors and a bounded
set of physical lifting proposals.  The conventional codec is a charged component, not a free
baseline or hidden source image.  The reversal path is deletion of this default-off module and task
lineage; all maintained formats and defaults remain unchanged.

The exposed development pilot survives the frozen systems killing rule but does not select a
default. The post-hoc selected v2 packet is 3,896,344 complete bytes, gives below-display error at
decoded pixel centers, and has query parity plus a synthetic two-view CompactCarve smoke. Its
3.662x source ratio compares a full frame with a crop packet; crop-local canonical PNG ratio is
1.139x. Off-grid bilinear-control sampling retains 3.784% local-envelope and 0.0244% global-range
escape. The custom diagnostic manifest passes independent size/hash replay but is not accepted by
the maintained `check_report_bundle.py` schema. The exposed multiview extension retains v4 only as
a development Pareto point: equal 10k topology and better rate/reporting metrics, but worse complete
resources and a failed native visual gate; v5/v6 artifact-cleanup variants are rejected. See
ADR-0032, the
`docs/research/2026-08-06-codec-native-dual-plane-portfolio.md` portfolio, the paired results audit,
and `ara/evidence/core016-codec-native-dual-plane-janelle-2026-08-06/run.md`. Distinct scientific
review, held-out full-frame rate evidence, clean multiscene confirmation, and artifact-free
downstream quality remain open.
