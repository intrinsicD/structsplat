# CORE-017 — Visibility-ordered alpha-shell surface lift

## Context

CORE-016 decouples a codec-backed continuous appearance query from sparse structural proposals and
survives its exposed multiview byte/quality screen, but its retained 3D model still has soft halos,
fine-detail blur, and floaters.  Reinspection isolates a depth/geometry mismatch before appearance
optimization: CompactCarve chooses the strongest consensus point anywhere inside a ray's supported
visual-hull interval, then interprets the broad depth-score peak as render covariance.  A disposable
cover-only probe improves foreground reconstruction but moves substantial alpha outside the mask,
showing that covariance repair cannot turn volumetric centers into a surface.

The packet already contains exact alpha and a sparse proposal measure as separately owned signals.
The next cheapest causal test is therefore to use proposal mass only to choose rays, alpha support
only to order depth along those rays, and appearance only to initialize radiance.  This is a
shape-from-silhouette/surface-cover systems recombination, not a novelty claim.

## Goal

Implement and kill-test a default-off realtime-gs integration that selects the first depth sample
on each source ray attaining maximal multiview alpha support, then optionally replaces localization
covariance with a cover-consistent local surface covariance, without adding packet bytes or changing
the maintained StructSplat pipeline.

## Non-goals

- Do not change realtime-gs source, CompactCarve defaults, the `.sgdp` grammar, StructSplat's
  maintained conversion path, renderer equations, or any production default.
- Do not claim physical surface recovery, concavity recovery, general compression, artifact
  freedom, or held-out quality from a masked single-scene diagnostic.
- Do not tune CORE-016's consumed `frame_00008` reporting views or rewrite its immutable bundles.
- Do not treat alpha masks as depth ground truth; the candidate reconstructs the visual-hull shell.

## Acceptance criteria

- [x] The optional adapter exposes an alpha-support query backend whose calibrated inside weight
      produces a declared CompactCarve soft-coverage value, preserves continuous appearance color,
      is exactly zero outside alpha, works across CPU metadata/CUDA query devices, and never invokes
      the sparse structural index for placement support.
- [x] A bounded lift helper composes unchanged CompactCarve anchor/ray sampling with a first-maximum
      alpha-support depth rule and optional realtime-gs surface-cover reconciliation.  It records
      the independent proposal/support/appearance ownership, all resolved configs, timings, and
      pre/post geometry diagnostics while remaining deterministic on CPU.
- [x] Synthetic tests distinguish interior-consensus and alpha-shell depths, verify first-maximum
      tie order, exact count, finite parameters, surface-cover planarity/scale behavior, counter
      isolation, malformed-input rejection, and no effect on the existing paired backend.
- [x] A diagnostic-only `frame_00009` 2x2 factorial compares `{interior consensus, alpha shell}` x
      `{inherited localization covariance, cover-consistent surface covariance}` with one shared
      WebP-q92/512-structure packet set, the same 23 training and three reporting cameras, 5,000
      Gaussians, 1,000 fixed-topology gsplat steps, seed 0, and 100-step telemetry.
- [x] The diagnostic reports complete packet bytes, lift/index/query work, surface geometry,
      convergence, PSNR/SSIM/MS-SSIM/LPIPS, gradient and tail errors, alpha inside/outside/IoU,
      final model bytes/count, peak VRAM, native target/init/final/error panels, and all curves.
      Preserve every arm and fail the candidate if the combined arm gains less than 0.5 dB reporting
      foreground PSNR over interior/inherited, worsens alpha IoU by more than 0.01, worsens alpha
      outside by more than 0.01, worsens gradient MAE, or retains visible halos/floaters/double
      silhouettes.  Passing only authorizes a separately reviewed variable-topology test.
- [x] The task, Index, generated session brief, architecture/research boundary, ARA trace/evidence
      (if the diagnostic is retained), and exact reproduction command are synchronized; focused
      tests and `./scripts/verify.sh` pass.

## Diagnostic protocol

This is an exposed, single-frame, single-seed mechanism diagnostic.  It is not a formal promotion
run and does not consume or create confirmation evidence.

- **Data:** calibrated `frame_00009`; training cameras are the same 23 non-reporting IDs used by
  CORE-016 v4, while `C0004`, `C0025`, and `C1004` are reporting-only.  Packet construction uses
  calibrated undistorted downscale-4 crops; training/reporting use common calibrated undistorted
  downscale-8 source RGB and masks.
- **Input:** construct one immutable quality-92 WebP dual-plane packet per training view with 512
  structural proposals, sigma `0.45`, radius `3`, eight Jacobi steps, exact alpha, 16-pixel crop
  margin, and per-view seeds `0..22`.  Every arm reuses those exact packet bytes and decoded tensors.
- **Placement:** use the CORE-016 matched-10k CompactCarve configuration except that all arms stop
  at 5,000 initial rows and topology remains fixed.  Interior arms use the ordinary paired backend
  and color-consistency score.  Alpha-shell arms replace placement support with a calibrated
  alpha-only backend and neutralize color variance; PyTorch's first-index `argmax` selects the first
  depth attaining the maximal support plateau.  Appearance colors still come from the codec-native
  continuous query.
- **Covariance:** inherited arms retain CompactCarve's lifted localization covariance and opacity.
  Cover arms apply realtime-gs `SurfelInitConfig` defaults without a contributor-resolution floor;
  means, colors/SH, count, packet bytes, and selected source rays stay unchanged.
- **Refinement:** identical 1,000-step fixed-topology gsplat training, evaluation every 100 steps,
  final checkpoint, SH degree 3, masks and antialiasing enabled, normal CORE-016 loss weights,
  deterministic seed 0, and no density events or polish.
- **Decision:** apply the numerical and mandatory native-visual killing rule in the acceptance
  criteria.  Do not retune this frame after viewing outcomes; any successful combined arm advances
  unchanged to a new scene/multiseed/variable-topology protocol with a distinct prospective review.
- **Execution:** from the repository root, run
  `PYTHONPATH=src:/home/alex/Documents/realtime-gs/src
  /home/alex/Documents/realtime-gs/.venv/bin/python
  scripts/experiments/core016_multiview_downstream.py --profile surface2x2
  --frame /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00009
  --out results/core017_visibility_surface_janelle_frame00009_2026-08-06_v1`.
  The output directory is immutable after the first outcome is exposed.

## Interfaces touched

`src/structsplat/realtime_gs_adapter.py`, an optional lift-composition module if needed, focused
tests, one bounded driver under `scripts/experiments/`, research/architecture documentation, ARA
staging/evidence if retained, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

CORE-016, CORE-013, BENCH-019/020, BENCH-002, ADR-0006/0032

## Agent workflow

- Driver: codex-root
- Reviewer: codex-root
- Turn: reviewer
- Reviewed revision: commit `12fecb8`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.  A formal
result beyond this diagnostic requires a distinct prospective protocol review with no outcome
access.

### Handoff

#### Objective

Review the default-off visibility-ordered alpha-shell placement backend, optional surface-cover
composition, and the exposed fixed-5k causal diagnostic without promoting the visually failed route.

#### Changes

Added a calibrated no-extra-payload alpha-support backend and a lazy surface-lift helper that keeps
sparse ray proposals, alpha placement support, codec radiance, and render extent independently
owned. Extended the existing CORE-016 driver with one shared-packet 2x2 placement/covariance
factorial, alpha visuals/curves, fail-closed gates, and exact source/receipt capture. Added synthetic
CPU/CUDA compatibility and driver-decision tests, executed and audited `frame_00009`, and
synchronized README, architecture, ADR-0032, task/index/brief, research audit, core skill, and ARA
evidence.

#### Evidence

The immutable manifest SHA-256 is
`bf14b8e8d08609bdf89dd3c4474422a7ea8c0281c45cb81de8ba50e17252be2e`. Every arm uses 970,310
input bytes and ends with 5,000 Gaussians. Shell/inherited gains 1.5873 dB; shell/cover gains 1.3809
dB and 0.14037 alpha IoU, lowers gradient MAE by 0.001347 and outside alpha by 0.01602, and passes
all scalar gates. Native review fails for trailing smear/double silhouettes and blur. Independent
replay validates 222 receipts and exact decision arithmetic; the documented custom report checker
rejection has four schema errors. The optional realtime-gs slice passes 22 tests. `./scripts/verify.sh`
passes with 1,673 tests, 15 skips, 514 deselections, and all five structural gates green before
commit `12fecb8`.

#### Assumptions

The downscale-8 source tensors are the diagnostic target; “native” means unrescaled stored
evaluation pixels, not full camera resolution. Packet bytes price the shared reconstruction inputs,
while final model bytes are separate. Reporting-only cameras are excluded from construction and
training but are exposed development views after inspection.

#### Uncertainties

One frame/seed, reused reporting views, dirty executed source, custom schema, approximate timing,
masked visual-hull geometry, 48 depth samples, and fixed topology prohibit generalization. The
method does not recover concavities or physical surfaces, and residual directionally smeared
geometry remains visually unacceptable despite favorable aggregate metrics.

#### Review focus

Audit the analytic soft-coverage calibration, CPU metadata/CUDA return-device behavior, first-tie
ordering, structural-counter isolation, surface-cover immutability, exact packet reuse/count gates,
train/reporting separation, timing boundaries, and every statement distinguishing numerical
improvement from artifact freedom or production readiness.

#### Protected actions not taken

No realtime-gs source, `.sgdp` grammar, maintained StructSplat conversion path/default/renderer,
existing CORE-016 result, reporting target, public claim-ledger row, or unrelated IntelliJ file was
modified. The consumed frame was not retuned and the failed visual result was not overwritten.

#### Recommended next action

Obtain distinct code/scientific review. Do not advance this exact alpha-shell route. A successor
must change the visibility/geometry model on disjoint data—rather than add alpha weights or retune
the consumed views—and must retain the shell placement arm as a causal control.

### Review

#### Verdict

Provisionally accepted (self-reviewed)

#### Self-reviewed

Yes

#### Correctness

Commit `12fecb8` preserves the NumPy/torch import split and modifies no differentiable training or
render equation. The alpha weight is the algebraic inverse of CompactCarve's declared soft-coverage
mapping, outside-alpha support is zero, wrapper outputs return to the query caller's device, and
runtime counter checks fail if the structural index is touched. CompactCarve supplies the tested
first-index `argmax`; cover reconciliation is exact-checked not to change means or SH. Synthetic
tests cover ties, distinct shell/interior depths, deterministic geometry, finite covariance, exact
count, malformed pairing, CPU import, and CUDA query plumbing.

#### Evidence quality

The factorial, command, split, budget, seed, metrics, and killing rule were frozen before execution;
all arms and negatives are preserved. The same packet hashes and bytes are reused, 222 receipts and
gate arithmetic replay cleanly, and stored-resolution RGB/alpha panels were inspected at enlarged
pixel scale. Evidence is diagnostic only: dirty source snapshots, one exposed frame/seed, reused
reporting cameras, reduced resolution, task-local schema, and no independent prospective/results
review prevent scientific promotion.

#### Simplicity

The implementation adds one narrow wrapper and one composition helper around existing realtime-gs
interfaces. It reuses CompactCarve for ray proposals/depth sampling and realtime-gs for covariance
cover instead of duplicating either. The experiment extends the existing bounded driver and adds no
packet member, production option, format, or default.

#### Missing cases

Distinct review, disjoint multiscene/multiseed/full-resolution evidence, physical-depth or geometry
truth, concavity/thin-structure/disocclusion strata, non-black backgrounds, production packet
generation, final-model codec/FPS, and end-to-end latency remain absent. Visual smear survives even
though the combined scalar gate passes.

#### Required changes

None for retaining the implementation and negative visual disposition as default-off diagnostic
evidence. Distinct review and a new prospectively frozen geometry mechanism are required before any
scientific acceptance, variable-topology advancement, or maintained/default integration.

#### Optional improvements

Add a portable checker/schema only if this diagnostic family is reused; provide explicit depth or
multi-layer visibility evidence on disjoint data; stratify artifacts by silhouette direction and
occlusion; and account for production packet generation, final 3D coding, and render throughput.

## Notes

The causal prediction is directional: if arbitrary interior depth is the main source of CORE-016's
halos and floaters, enforcing a first-visible shell should improve alpha localization before or
early in optimization, and cover covariance should help only after centers lie on a coherent shell.
If alpha-shell placement fails while ordinary interior placement remains stronger, retire this
masked visual-hull route rather than adding mask weights or tuning the consumed views.

## Diagnostic outcome — 2026-08-06

The immutable v1 bundle is
`results/core017_visibility_surface_janelle_frame00009_2026-08-06_v1/`, manifest SHA-256
`bf14b8e8d08609bdf89dd3c4474422a7ea8c0281c45cb81de8ba50e17252be2e`. All arms reuse the same
970,310 packet bytes and finish at exactly 5,000 Gaussians. Relative to interior/inherited,
shell/inherited gains 1.5873 dB reporting PSNR, while shell/cover gains 1.3809 dB, 0.14037 alpha
IoU, and 0.001347 gradient MAE and lowers outside alpha by 0.01602. Shell depth scoring evaluates
zero sparse-index pairs versus 212,517,051 and lifts in 5.81--6.25 versus 8.95--9.42 seconds.
Cover-only loses 0.7035 dB and 0.0161 alpha IoU.

The combined arm passes all frozen scalar gates, but mandatory native review fails: directional
trailing smear/double silhouettes remain on all three reporting views and fine detail is still soft.
The route is therefore not advanced or retuned on this frame. The bundle is diagnostic because it
uses one exposed frame/seed, dirty source snapshots, and a task-local schema; independent replay
validates 222 receipts, while the maintained report checker returns its four expected schema errors.
