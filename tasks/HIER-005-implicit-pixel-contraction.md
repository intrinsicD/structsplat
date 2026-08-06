# HIER-005 — Implicit pixel-field contraction

## Context

A pixel-centered Gaussian per source sample is a useful exact-or-near-exact endpoint, but
materializing and optimizing that field is the worst-rate representation. The useful algorithmic
question is whether the endpoint can seed a deterministic coarse-to-fine contraction path: propose
local parents cheaply, measure the distortion caused by eliminating children, and accept the best
distortion-per-byte actions without allocating one trainable row per pixel. The existing triage
merge is a lifecycle heuristic for the normalized field and is not a rate--distortion contraction
reference for additive Field V2.

## Goal

Implement a default-off reference method that treats pixel leaves implicitly, contracts them
through a deterministic quadtree-local hierarchy, exactly re-solves accepted local appearance,
exports a valid direct-additive `ObservationField2D`, and emits comparison-compatible diagnostic
artifacts at requested Gaussian counts.

## Method contract

- Source samples are integer-`xy`, HWC `[0,1]` RGB observations with a declared isotropic leaf
  scale. Pixel leaves are generated procedurally and never become torch parameters.
- A hard contraction replaces two to four currently active siblings by one moment-matched Gaussian.
  An optional parent-plus-detail contraction retains the single child basis that minimizes the
  analytic continuous-domain residual while still reducing row count.
- The image-sized quadtree frontier is scheduled by a cheap deterministic within-region RGB proxy.
  Once a region is shortlisted, its contraction options are ordered by the exact continuous inner
  product of two unnormalized peak-one 2D Gaussians. That score is estimated RGB squared
  distortion divided by estimated bytes removed; it is proposal telemetry, not an actual
  compressed-rate claim.
- Before acceptance, coefficients are re-solved against the true image residual over the union of
  changed finite supports and the exact discrete additive renderer computes the local SSE change.
  Only support-disjoint actions may share a batch.
- Recovery is optional and count-progress normalized. The preserved `touched` scope keeps
  never-touched pixel leaves bitwise fixed. The comparison-only `all_error_weighted` scope
  optimizes every active row, using mask-aware Gaussian-smoothed residual MSE averaged under each
  Gaussian by one matrix-free additive-renderer VJP as a post-Adam row-update multiplier. Both
  scopes retain only strict masked-SSE improvements and rebuild geometry-dependent proposals.
- Terminal local rescue is separate from fixed-count recovery. It requires a signed direct field,
  freezes the complete base prefix and all rescue geometry, selects stable high-residual foreground
  centers with local NMS, optimizes only appended rescue RGB coefficients, and keeps the base or a
  candidate by lexicographic raw local-violation/SSE order. It reports its explicit row/payload
  overhead and cannot turn a failing displayed-image gate into an exact-count success.
- The hierarchy stops at the requested count when an exact-count action exists. If the available
  contraction arity cannot hit the count, it stops above it and reports the reason; it must not
  silently overshoot.
- The output uses direct additive, peak-one, AABB-support Field V2 semantics. Current normalized
  defaults, `scripts/convert.py`, and compressed-stream policy remain unchanged.

## Non-goals

- Selecting additive semantics, changing any production default, or replacing the maintained
  conversion pipeline.
- Claiming actual compression from estimated row bytes or the lossless reference NPZ size;
  COMP-013/FIT-030 own complete coded bytes and byte-priced stopping.
- Claiming that the hierarchy, merge rule, or parent-plus-detail basis is novel.
- Running a formal result-bearing benchmark before CORE-013/BENCH-020 review and a prospective
  protocol review; this task may produce implementation diagnostics only.
- Replacing the maintained fitter, making all-active recovery a default before matched evidence,
  or implementing a neural/amortized encoder.

## Acceptance criteria

- [x] A typed, deterministic API implements implicit leaves, closed-form Gaussian overlaps,
      moment parent proposals, hard and parent-plus-detail contraction options, stale-safe local
      candidate scheduling with a bounded exact shortlist, exact local coefficient re-solving,
      and support-disjoint batching.
- [x] Requested-count stopping, odd image sizes, masks, signed and nonnegative coefficient domains,
      numerical degeneracies, and deterministic ties fail closed or have explicit behavior.
- [x] The final active set exports through the CORE-013 lossless field contract and renders through
      the maintained additive renderer without a dense pixel-by-Gaussian allocation.
- [x] A task-specific experiment driver records source identity/bytes, canonical estimated field
      bytes, lossless reference-container bytes, count/quality/runtime metrics, contraction history,
      renderings, config, and a self-contained diagnostic HTML report.
- [x] The diagnostic driver can deterministically resize a native source/mask pair while retaining
      original-file provenance, records both original-file and same-raster PNG byte ratios, and
      emits a labeled budget curve for every recorded outcome metric.
- [x] An opt-in interleaved recovery schedule optimizes only active rows ever touched by a
      contraction, leaves untouched pixels bitwise fixed, preserves exact requested counts, rolls
      back non-improving checkpoints, rebuilds geometry-dependent proposals after acceptance, and
      records bounded-work checkpoint telemetry.
- [x] A separate opt-in recovery arm can optimize every active Gaussian with row-wise Adam-update
      weights derived matrix-free from a mask-aware smoothed residual-energy field, records the
      attribution/weight distribution, preserves the touched-only arm as a matched control, and
      avoids a dense pixel-by-Gaussian matrix.
- [x] The diagnostic report records localized artifact outcomes—foreground pixel-RMSE tails and
      maximum multiscale patch RMSE—using explicit metric domains and a curve for every scalar.
- [x] A bounded exposed-image factorial compares hard versus faded support at 3.0 and 4.5 sigma
      under touched interleaving and topology-frozen terminal all-active recovery, preserving exact
      commands, source snapshots, visuals, raw rows, and negative cells.
- [x] If the provisional local-error gate fails, a separately identified bounded repair arm adds
      only locally selected rescue rows, records the row/payload overhead, and fails closed rather
      than claiming an artifact-free exact-count result.
- [x] Focused tests cover the overlap formula, exact local solve, count monotonicity, determinism,
      odd dimensions/masks, save/load, maintained-renderer parity, and an end-to-end driver smoke.
- [x] Architecture/task documentation explicitly labels the method default-off and diagnostic;
      current renderer, pipeline, field, and codec defaults are unchanged.
- [x] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/pixel_contraction.py`, a task-specific driver under `scripts/experiments/`, focused
tests, `docs/architecture.md`, `docs/additive_field_v2.md`, this task, the Index, and generated
session brief.

## Depends on

CORE-013, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. Any later formal
comparison requires the prospective `### Protocol review` block and a clean result-bearing tree.

### Handoff

#### Objective

Implement the pixel-endpoint contraction idea as a default-off, exact-count direct-additive
reference that can be cold-rendered and compared later without changing the current pipeline or
pretending estimated row bytes are compression.

#### Changes

Added a NumPy-first `pixel_contraction` API with procedural pixel leaves, reusable float32 atom
slots, a quadtree frontier, moment-matched parents, parent-plus-detail and pair options, exact
continuous Gaussian-product scoring inside shortlisted regions, exact discrete finite-support
coefficient solves, signed/tiny exact NNLS domains, support-disjoint commits, overlap-aware cache
invalidation, exact target stopping, packed-mask export, and lazy maintained-renderer bridging.
Added the HIER-005 driver with cold lossless load, source/raw/reference byte ledgers,
PSNR/SSIM/MS-SSIM/optional LPIPS, timing/actions/parity, images/history/config, tidy JSON/JSONL/CSV,
manifest, and diagnostic HTML. Added focused mathematical, adversarial, deterministic, masked,
serialization, no-torch-import, and subprocess smoke tests; synchronized the Field V2 design,
architecture module map, task graph, Index, and session brief.

#### Evidence

The focused HIER-005 suite passes 14/14 tests. The field/render regression slice passes 80/80.
`./scripts/verify.sh` passes with 1,574 tests, 4 skips, 514 deselections, and all docs, ARA, task,
script-layout, and agent-workflow checks clean. On the existing 64x48 diagnostic source at N=1,024,
the throughput policy recorded 23.523 dB in 1.982 contraction seconds and the `always` pair policy
recorded 24.828 dB in 2.883 seconds; maintained-render parity for the former was `7.75e-7` max abs.
Those reports live under a temporary directory and are smoke diagnostics, not evidence bundles.
The reviewed source-set digest covers the implementation, driver, tests, and two synchronized
architecture documents in path order under `sha256sum`.

#### Assumptions

The default 0.18-pixel leaf is a numerically near-delta endpoint under the declared AABB renderer,
not a mathematical Dirac basis. The default 32-byte row price is the eight-float32 uncoded payload
and orders proposals only. A cheap RGB proxy schedules the image-sized region frontier; exact
Gaussian products rank the bounded options only after a region is shortlisted. Exact acceptance
uses the float32 geometry/coefficient state that is exported. Packed alpha gates output but does
not assert hard support containment.

#### Uncertainties

There is no full-resolution scaling study, accelerator, optimizer continuation, selected Field V2
semantic, actual codec, equal-work incumbent control, or held-out result. The smoke source's 32 KiB
estimated field is larger than its 7.8 KiB PNG, which reinforces rather than resolves the rate
question. CORE-013 and BENCH-020 remain under review/open, and critical numerical code has no
distinct reviewer in this turn.

#### Review focus

Reproduce the Gaussian-product formula against quadrature; audit covariance moment matching and
RS export; verify signed and two-column NNLS solves; challenge root/exact-count viability on sparse
odd masks; inspect stale heap entries, support-box independence, and dirty-box cache invalidation;
and confirm every byte label/report warning prevents an estimated-rate claim.

#### Protected actions not taken

No current renderer/default, maintained pipeline, `scripts/convert.py`, codec, BENCH-020 outcome,
formal result, ARA claim, external repository, commit, or push was changed. Unrelated IDE files and
the pre-existing DOCS-007/literature-review worktree changes were preserved.

#### Recommended next action

A distinct numerical/scientific reviewer should reproduce the focused suite and inspect the bound
source set. After CORE-013/BENCH-020 select a usable semantic contract, FIT-045 can preregister this
as a direct contraction control; only COMP-013/FIT-030 may replace its estimated slopes with
complete cold-decoded byte deltas.

### Handoff

#### Objective

Test the hypothesis that quadtree/grid artifacts can be repaired by interleaving short optimizer
blocks while keeping every never-touched pixel Gaussian fixed, then expose the result visually and
as complete diagnostic budget curves on Janelle frame `C0001`.

#### Changes

Added default-off selective recovery to HIER-005. Every active slot ever produced or retained by a
contraction remains optimizer-eligible; never-touched active leaves are subtracted into a detached
fixed base and their arrays never mutate. A checkpoint jointly fits touched means, scales,
rotations, and RGB coefficients with parameter-specific learning rates and bounded trust regions,
keeps the best masked-SSE step, rolls back any non-improvement, and rebuilds geometry-dependent
proposal state after acceptance. The default progress schedule spaces work by fractions of rows
removed instead of action count, preventing target-dependent optimizer-work confounding. Added
checkpoint telemetry, recovery curves/artifacts, explicit CPU/CUDA determinism labels, native and
evaluation-raster provenance, all-metric curve export, focused freeze/telescoping/determinism/work
tests, and synchronized architecture/design documentation.

#### Evidence

The source set is bound by SHA-256
`2f4f42bd139be98c27444e598b1ab7fdea17b1771438be7588104d4546e2d778` over the implementation,
driver, focused test, `docs/architecture.md`, and `docs/additive_field_v2.md` in that path order.
The focused suite passes 20/20; the pixel/Field-V2/render regression slice passes 86/86; and
`./scripts/verify.sh` passes with 1,580 tests, 4 skips, 514 deselections, and every structural gate
clean.

The dirty single-image diagnostic resized the exact 5,328x4,608 source/mask pair to 512x443 with
15,929 active pixels. All four recovery rows ran 16 checkpoints x 50 steps and hit the exact
requested count. Against the no-recovery diagnostic, masked PSNR changed by +6.771 dB at N=2,048,
+5.476 dB at N=4,096, +18.263 dB at N=8,192, and +10.083 dB at N=12,000. At N=8,192, PSNR changed
from 34.076 to 52.339 dB and LPIPS from 0.015007 to 0.00001648; the visible square/tree holes were
removed. Total diagnostic wall time changed from 5.444 to 9.140 seconds there. Recovery optimized
2,723 active rows and preserved 5,469 untouched rows. The complete recovery report is
`results/hier005_janelle_c0001_selective_recovery_progress16_2026-08-05_v2/index.html`; its
`metrics.json` SHA-256 is `fde416036a2a018030a0976f65c45917687ad942a92cff7a7e42cc8e5f637c66`
and its manifest SHA-256 is
`18c509ee011b3afec654282d22998860f64917ecebf4b283805c071a2bf77c0f`.

#### Assumptions

Recovery uses the same direct additive AABB renderer and masked-SSE objective as contraction
acceptance. “Untouched” means an active source-pixel slot that has never been an output of a merge,
replacement, or retained-detail action. Sixteen progress checkpoints are an equal attempted-work
control across these targets, not an optimized schedule.

#### Uncertainties

This is one exposed, resized, dirty-worktree diagnostic without a prospective protocol, held-out
data, matched wall-time control, uncertainty interval, or independent review. PSNR/MSE use the
foreground mask, whereas SSIM/MS-SSIM/LPIPS use the black-matted full raster. CUDA atomic-gradient
order is not bit-deterministic: a repeat changed all canonical field hashes and changed PSNR by up
to 0.205 dB at N=2,048 (0.0045 dB or less at N=8,192/12,000). Timings are single live-workstation
observations. The 29,263-byte same-raster PNG is still smaller than every uncoded field payload:
the N=2,048 field is 93,888 bytes and the N=8,192 field is 290,496 bytes. Native-JPEG ratios are
resolution-mismatched and cannot support compression claims.

#### Review focus

Audit the detached-base algebra and bitwise freeze invariant; confirm touched-slot lineage across
slot reuse; reproduce CPU determinism and SSE telescoping; challenge checkpoint crossing and
frontier rebuilding; repeat CUDA rows to estimate outcome variance; and verify that all rate labels
remain payload proxies rather than codec claims.

#### Protected actions not taken

No current renderer, maintained fitter, pipeline/default, converter, semantic choice, codec,
formal benchmark, ARA claim, external repository, commit, or push was changed. Existing unrelated
IDE and DOCS-007/ARA worktree changes were preserved.

#### Recommended next action

Have a distinct numerical/scientific reviewer reproduce this source set. If it survives, run a
prospectively frozen multi-image comparison with deterministic CPU or repeated CUDA recovery and a
matched-work fixed-N fitter; pursue an actual quantized/entropy-coded stream separately, because
selective recovery solves the visible topology artifact but not compression.

### Handoff

#### Objective

Test the user's proposal to optimize every active Gaussian rather than only contraction-touched
rows, while weighting each row by a spatially smoothed estimate of the error under its support.

#### Changes

Preserved `recovery_scope=touched` and added default-off `all_error_weighted`. At every recovery
checkpoint, the new scope forms foreground RGB residual MSE, applies mask-normalized Gaussian
smoothing, and uses one additive-renderer color VJP to compute each active Gaussian's
support-averaged error exposure without a dense pixel-by-Gaussian matrix. Scores use a configurable
power/floor/ceiling and approximately mean-one normalization. Because fixed scalar gradient weights
are largely canceled by Adam's second moment, the implementation multiplies the actual row update
after Adam preconditioning. Every active row is materialized; trust regions, best-SSE rollback,
exact counts, and frontier rebuilding remain unchanged. Added scope/attribution/weight telemetry,
five additional report curves, CLI/config provenance, mathematical/update/determinism/driver tests,
and synchronized architecture, Field V2, task, and core-skill documentation.

#### Evidence

The source set is bound by SHA-256
`60eff11c57e3975966e209743529e1e633c4b7eec1c2e02866ed93adef6cdf6b` over implementation,
driver, focused test, both architecture documents, and the core skill in declared path order.
Focused HIER-005 tests pass 27/27, the pixel/Field-V2/render slice passes 93/93, and
`./scripts/verify.sh` passes with 1,587 tests, 4 skips, 514 deselections, and every structural gate
clean.

The 256-side N=1,024 probe separated touched-only (31.915 dB), all-active uniform updates
(35.839 dB), and all-active error weighting. Raw-error repeats recorded 40.962/41.645 dB;
sigma-1.5 smoothed repeats recorded 41.161/40.256 dB. Thus all-active and weighting have large
single-image effects, while the incremental smoothing ordering is unresolved under CUDA variation.
Sigma 3.0 and power 1.0 were lower single runs (39.689/40.234 dB), so the requested sigma-1.5,
power-0.5 formulation was retained without claiming it is selected.

On the exact same 512x443 Janelle source/mask, source snapshot, 16x50 attempted-step schedule,
renderer, and metric domains, all-active versus touched-only masked PSNR changed by +5.015 dB at
N=2,048, +10.456 dB at N=4,096, -6.310 dB at N=8,192, and +2.042 dB at N=12,000. Total wall-time
ratios were 1.150x, 1.231x, 1.404x, and 1.556x. The N=4,096 cell pattern largely disappeared; at
N=8,192 a few square defects returned. An independent N=8,192 all-active repeat reached 47.970 dB,
still 4.367 dB below the touched-only row, so that loss is not explained by one CUDA trajectory.
All checkpoints individually reduced current masked SSE, but accepted all-active geometry changed
later topology proposals; the terminal path therefore need not dominate touched-only.

The all-active report is
`results/hier005_janelle_c0001_all_error_weighted_progress16_2026-08-05/index.html` with metrics
SHA-256 `d9fe98c9fe9d7a0f59697bd248c823666b9eec4b2f547f1cdb266685a24a1b47` and manifest SHA-256
`b66a3bd4cb99267b78e4df64fa03215e0d32f39983643eadf666201b92cd6306`.
The matched touched report is
`results/hier005_janelle_c0001_touched_control_progress16_current_2026-08-05/index.html` with
metrics SHA-256 `cda33647a131ca1c15339dd91e7a4be4f0b8db02696aca645e6ca1f80cc51326` and manifest SHA-256
`eaf9be4bbff2ba356e8dd2f6a8b9216bcb593b6ceef3e6cdb357d45baefe7566`.
Both manifests verify 80/80 files, expose 44 SVG curves, resolve 61/61 local HTML links, and
snapshot the executed core/driver exactly.

#### Assumptions

“Error produced” is operationalized as mask-smoothed residual energy averaged under a Gaussian's
peak-one support. It is an exposure/priority score, not causal error attribution: the ordinary
renderer gradient still determines update direction. The weight is frozen within each checkpoint
and recomputed after further topology actions. Attempted optimizer steps are matched, but
all-active checkpoints process more rows and are not equal-FLOP controls.

#### Uncertainties

This is post-hoc tuning and diagnosis on one exposed downscaled image, not a prospectively reviewed
or held-out experiment. CUDA recovery is non-bit-reproducible and the low-count repeat spread is
large. There are no optimizer-step quality curves, equal-FLOP/equal-wall-time controls,
multi-image uncertainty, or complete coded bytes. SSIM/MS-SSIM/LPIPS include the black-matted
background while PSNR/MSE use foreground pixels. The uncoded payload is unchanged and remains
larger than the same-raster PNG at every count. No convergence-speed, compression, general method,
semantic, or default claim is authorized.

#### Review focus

Verify the one-VJP numerator/denominator attribution identity, mask-normalized blur at boundaries,
post-Adam row scaling, all-active slot mapping, trust-region/rollback behavior, and proposal-cache
rebuild after geometry changes. Recompute all four matched deltas from raw rows, inspect the 4k/8k
visual reversal, and challenge whether a future terminal-only all-active polish can preserve the
touched topology without adding unmatched work.

#### Protected actions not taken

No current renderer, maintained fitter/pipeline, converter, default recovery scope, semantic
choice, codec, formal protocol, held-out split, ARA claim, external repository, commit, or push was
changed. Existing unrelated IDE and DOCS-007/ARA worktree changes were preserved.

#### Recommended next action

A distinct numerical/scientific reviewer should reproduce source-set `60eff11c`. The next bounded
mechanism test should preserve touched-only interleaving and apply smoothed all-active weighting
only as a terminal, rollback-safe polish under matched total optimizer work; the current mixed
curve does not justify replacing touched-only recovery globally.

### Handoff

#### Objective

Execute the user-authorized artifact-safety diagnostic: isolate support truncation from recovery
schedule with a complete hard/faded 3.0/4.5-sigma factorial at 4k/8k, gate localized displayed
errors rather than average fidelity alone, and run the frozen bounded local repair only after the
fixed-count 4k gate fails.

#### Changes

Extended the diagnostic metric contract with exact displayed-8-bit foreground pixel-RMSE tails,
fractions, and maximum 3/7/15/31-pixel black-matted patch RMSE plus gate fields and one SVG curve
per scalar. Added a terminal signed-residual rescue API that requires direct signed semantics,
keeps every base row bit-exact, chooses stable residual peaks with Chebyshev-radius NMS, freezes
rescue means/scales/rotations, optimizes rescue RGB only, and checkpoints lexicographically by raw
normalized pixel/patch violation then SSE with the unchanged base eligible. Added a task-specific
repair driver with independent base forks, cold load and repeated render, full/error/worst-crop and
center visuals, raw/display metrics, payload overhead, 36 curves, source snapshots, manifest, and
internal verification. Extended the repository report-bundle gate for both explicit HIER-005
diagnostic schemas, including exact manifest, ledger, gate-arithmetic, parity, curve, and artifact
link checks. Added validation, freeze, alpha, determinism, parity, and end-to-end report tests;
synchronized architecture, Field V2 design, task, Index, session brief, and ARA evidence.

#### Evidence

The ordered implementation/driver/repair-driver/report-checker/test/architecture/design source
ledger has SHA-256
`53c3b32baf0d6bb3bddf3ff376a6662002416d89423ed851f0b2d931106eab32`.
Focused HIER-005 tests pass 38/38; the pixel/Field-V2/render regression slice passes 103/103;
`./scripts/verify.sh` passes with 1,598 tests, 4 skips, 514 deselections, and all structural gates.

All 16 fixed-count cells use the bound 512x443/15,929-pixel C0001 raster and exact counts. Every
4k arm fails the provisional pixel-max `0.02` plus 7x7-max `0.01` gate. At 8k, hard3 touched,
hard4.5 touched, and fade4.5 touched pass; hard3 touched records 52.356 dB, pixel max 0.0148, and
7x7 max 0.0053. Fade3 and every terminal all-active 8k arm fail. Larger support does not improve
the local result; full 3-sigma fade changes and worsens it. The selection rule chooses hard3
terminal 4k as the least-violating failing repair base.

The independent 0/102/205/410 repair ladder reaches N=4,096/4,198/4,301/4,506. The authoritative
v2 CUDA row selects 102 rescue Gaussians at step 206, improving PSNR 36.695 to 36.909 dB, display
pixel max 0.0736 to 0.0642, 7x7 max 0.0335 to 0.0283, and raw normalized violation 3.688 to 3.212;
it still fails. The 410-row arm reaches 37.506 dB and LPIPS 0.006864 but retains pixel max 0.0655,
so all repair budgets fail closed. Every base prefix is bit-exact and all parity maxima are below
`2.4e-7`. Three CUDA executions preserve the verdict and selected 102-row minimum; the 205-row
checkpoint moves by atomic-order noise, as explicitly documented.

Each matrix report verifies 75/75 manifest files, 73/73 contained links, two finite rows, and 53
curves through the repository-wide report checker. Repair v2 verifies 86 manifest entries, 83
links, four rows, prefix identity, and render parity through both its internal verifier and the
repository-wide checker. The complete tables, visual audit, hashes, limitations, and supersession record are in
`ara/evidence/hier005-artifact-safety-janelle-diagnostic-2026-08-05/run.md`; the authoritative
repair report is
`results/hier005_janelle_artifact_local_repair_v2_2026-08-05/index.html`.

#### Assumptions

The displayed-PNG local gate operationalizes this user's artifact priority only for the exposed
evaluation raster. Hard/faded rows refit under their declared support semantics. Terminal and
interleaved schedules match 800 attempted Adam steps but not optimized-row work, FLOPs, or wall
time. Rescue row price is the existing 32-byte uncoded float estimate plus the shared alpha
payload; it is not a complete stream.

#### Uncertainties

This remains one exposed, resized, dirty-source image without a prospective distinct reviewer,
held-out data, observer study, native-resolution fit, equal-compute control, uncertainty interval,
or complete codec. CUDA optimizer trajectories are non-bit-reproducible. The 4k infeasibility
verdict applies to the tested mechanism and provisional gate, not all Gaussian representations.
The same-raster PNG is 5.45x smaller than the base uncoded field and 5.90x smaller than the 10%
repair field, so no compression or convergence-speed claim is supported.

#### Review focus

Audit raw-versus-displayed metric separation, complete-window black-matted patch pooling, support
factorial identity, the terminal `1x800` checkpoint boundary, fixed-base and fixed-rescue-geometry
invariants, stable residual/NMS selection, tail objective, lexicographic rollback, declared-alpha
matting, cold joint-render parity, repair selection arithmetic, CUDA repeat interpretation, and
the explicit failure to weaken the artifact gate.

#### Protected actions not taken

No current renderer, maintained fitter/pipeline, converter, semantic/default choice, actual codec,
formal benchmark, held-out split, scientific claim, external repository, commit, or push changed.
The original pre-alpha-correction repair bundle remains preserved and labeled superseded rather
than overwritten as evidence. Existing unrelated IDE and DOCS-007/ARA worktree changes remain
untouched.

#### Recommended next action

A distinct numerical/scientific reviewer should reproduce source-set `53c3b32b`. Do not tune the
exposed support or rescue ladder further. A fresh prospective disjoint-image study should retain
hard3 touched as the simplest 8k passing control and test a stronger fail-closed low-budget
fallback—local uncontraction/preserved pixel leaves or a complete-codec residual exception
channel—under local feasibility constraints, matched work, repeated/deterministic recovery, and
complete cold bytes.

## Notes

### Artifact-factorial diagnostic protocol (2026-08-05)

This is a user-authorized, post-hoc, exposed-image diagnostic. It is not a formal BENCH-020/021
result and has no prospective distinct protocol reviewer. Preserve every run in a new output
directory and interpret only the measured Janelle C0001 raster.

- Source/mask: the previously bound native C0001 JPEG and mask with SHA-256
  `ae24fe99d3f8edbd04cd2c85ebc4fe9bfd95abe878c22abb7691cadcfc5c411b` and
  `94dcbf7005dbeb1d183e259a569d783aa5df900255e763385bed91f02d3b80c3`, resized exactly as before
  to `512x443` with `15,929` active pixels.
- Counts: exact N=`4,096` and `8,192`; signed direct-additive coefficients, `exact_count` pair
  policy, RTX-3050 `cuda_additive`, LPIPS enabled.
- Support axis: cutoff `{3.0, 4.5}` × fade alpha `{0.0, 1.0}`. The complete 2x2 isolates larger
  negligible-tail support from subtractive continuous fade; every arm refits under its declared
  semantic.
- Recovery axis: touched-only progress interleaving `16x50` versus all-active error-weighted
  topology-frozen terminal polish `1x800`. This matches attempted Adam steps, not optimized-row
  work, FLOPs, or wall time; all resource counters remain visible.
- Primary artifact diagnostics: foreground pixel RGB-RMSE q99/q99.9/max and fractions above
  `0.05/0.10`; maximum black-matted RGB patch RMSE at `3/7/15/31` pixels. Global masked PSNR/MSE,
  SSIM, MS-SSIM, LPIPS, exact count, parity, and timings remain secondary/context outcomes.
- Provisional C0001 development gate: pixel max `<=0.02` and maximum 7x7 patch RMSE `<=0.01`,
  followed by native-pixel visual inspection. These post-hoc thresholds operationalize the user's
  artifact priority for this raster only; they are not human-observer or general visibility claims.
- Support hypothesis dies if neither cutoff nor fade improves the localized metrics and visible
  cell/hole morphology against hard-3 support. Terminal recovery is not selected if it reintroduces
  a localized violation relative to its touched control.
- Only if the best fixed-count arm fails, run one labeled repair ladder with at most 10% additional
  locally selected rescue rows. Report achieved count and every payload denominator; if the gate
  still fails, label that requested count infeasible rather than weakening the threshold.

The fixed-count matrix is now exposed. For the conditional repair stage, select the failing arm
that minimizes `max(pixel_max/0.02, patch7_max/0.01)`, breaking ties by higher masked PSNR. This
selects hard-3 terminal at N=4,096. Fork its exact persisted field into independent rescue rows
`{102,205,410}` (2.5/5/10% rounded). Select raw-residual peaks by stable descending RGB MSE with
Chebyshev-radius-1 NMS; add fixed `0.75 px`, zero-rotation signed Gaussians; freeze the complete
base field and rescue geometry; optimize rescue RGB only for 400 Adam steps at LR 0.05 on global
masked MSE plus `4x` the worst 1% pixel-MSE mean. Checkpoint lexicographically by normalized raw
pixel-max/7x7-patch violation then SSE. The display-PNG metrics above remain final authority.

No retuning after this section is written may become selection evidence. Any implementation bug
requires a new directory and explicit supersession record.

FIT-045 may consume this implementation as a direct contraction control after BENCH-020 selects
semantics. FIT-030 may replace estimated row-byte slopes with complete cold-decoded codec deltas.
Those later tasks own promotion, not HIER-005.
