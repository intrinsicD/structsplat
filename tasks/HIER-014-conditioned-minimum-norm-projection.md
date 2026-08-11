# HIER-014 — Conditioned minimum-norm appearance projection

## Context

HIER-013 rejects direct promotion of HIER-012 on the complete requested COCO/DIV2K development
bank.  Forty-two of forty-eight all-row cells stop before PCG because the incoming HIER-005 RGB
coefficients exceed the frozen absolute limit 16; their median/max absolute coefficients are
91.797/2010.808.  The six admissible cells still hit the 48-iteration cap, and large cancelling
coefficients amplify CUDA accumulation drift enough to fail 141/192 cold-render parity checks.

The existing projection is not a minimum-norm re-solve.  It applies Tikhonov regularization around
the incoming coefficient vector, starts PCG at that vector, and forms the frozen image by
subtracting its trainable contribution from a separately accumulated full render.  Near-null
coefficient components are therefore preserved by construction, while two large cancelling
renders are subtracted before the solve.  Established sparse least-squares practice suggests a
smaller causal test before changing topology again: restart the fixed-geometry solve at the
origin, regularize toward zero, use diagonal column equilibration, and construct the frozen
contribution directly.

## Goal

Determine whether an origin-restarted, zero-centered, explicitly based matrix-free RGB solve can
turn unsafe exact-7k HIER-005 fields into bounded, cold-stable, lower-error fields across diverse
images without changing geometry, topology, support, row count, or Field V2 semantics.

## Non-goals

- Do not raise the coefficient limit, retune the sixteen consumed HIER-013 images, or call them
  held-out/confirmation data.
- Do not change HIER-005 contraction/recovery, the maintained normalized pipeline, Field V2
  semantic selection, a codec, or any production default.
- Do not claim novelty for zero-centered regularization, Jacobi/column equilibration, PCG, LSQR,
  variable projection, or stable radial-basis approximation.
- Do not proceed to geometry alternation, bounded NNLS/BVLS, a changed basis, or local
  uncontraction unless this cheaper coefficient-only killing test fails and a new task records the
  resulting hypothesis.

## Diagnostic protocol

This is an engineering/development diagnostic because no distinct outcome-unseen reviewer is
available.  Every run uses a new output directory, records the exact source tree/environment, and
keeps failed or unsafe rows.  It cannot produce a default or claim-ready result.

### Phase A — correctness and mechanism

- Preserve the current input-centered/subtractive behavior as the default and bit-compatible
  control.
- Add an opt-in zero-centered solve that starts from zero even when stage zero is outside the
  coefficient bound.  Stage zero remains the exact fallback.
- Add an opt-in explicit frozen-base construction; an all-row solve has an exact zero frozen base,
  while a partial solve renders the frozen rows directly instead of subtracting two accumulated
  images.
- Verify the matrix-free forward/transpose pair, diagonal preconditioner, objective center,
  unsafe-stage-zero recovery, exact fallback, coefficient bound, maintained-render transaction,
  and deterministic CPU behavior on closed-form/tiny dense least-squares oracles.

### Phase B — disjoint Kodak selection

Development inputs are exactly `kodim01`, `kodim07`, `kodim13`, and `kodim19` from
`/home/alex/Documents/datasets/kodak24`, with file hashes recorded before execution.  Each image is
resized deterministically with Pillow LANCZOS to maximum side 512, uses a full-frame mask, exact
N=7,000, the unchanged HIER-013 HIER-005 configuration, seed 0, `cuda_additive`, chunk 256, and
required LPIPS.

The frozen source SHA-256 values are `a56e27cb...23bd` (kodim01),
`b77d3f00...54a4` (kodim07), `bc34a3ce...9b08` (kodim13), and
`b7450b26...bbdc` (kodim19); the driver binds and reports the full digests.

Compare the shared HIER-005 control with:

1. `legacy_input_subtract` — HIER-013's input-centered, subtractive-base projection;
2. `origin_subtract` — zero-centered/origin-restarted PCG with the legacy subtractive base;
3. `origin_explicit` — the same solve with the frozen contribution built directly.

All projections use ridge `1e-8`, relative normal tolerance `1e-6`, at most 96 iterations, and
coefficient absolute limit 16.  The diagonal of the normal operator remains the matrix-free Jacobi
preconditioner.  This 2x2 mechanism decomposition is fixed before Kodak outcomes; do not search a
ridge/cap/iteration ladder on these four images.

Select `origin_explicit` only if all four fields run past the fallback, every selected coefficient
is bounded, every maintained/cold transaction and parity check passes, geometric-mean MSE ratio
versus HIER-005 is `<=0.90`, no image has higher MSE beyond relative tolerance `1e-8`, aggregate
MS-SSIM and LPIPS do not regress, neither displayed worst-pixel nor 7x7 maximum worsens on any
image, and median projection overhead is `<=25%` of contraction time.  Attribute the effect to
restart versus explicit-base construction from the two intermediate arms.  If this gate fails,
retain the negative result and stop this candidate rather than tuning Kodak.

### Phase C — consumed-bank robustness replay

Only after the Phase-B recipe is frozen, run HIER-005, `legacy_input_subtract`, and the selected
recipe once on the exact 16 SHA-bound sources already registered by HIER-013.  Use the same
maximum-side-512 raster, full-frame mask, exact N=7,000, seed 0, renderer, chunk, metrics, and
coefficient limit.  These images are reporting-only and cannot select or rescue the method.

The engineering target corresponding to “work well everywhere” is explicit and bounded to this
bank: all 16 selected cells must execute a nonzero solve, preserve exact count/non-RGB arrays,
stay within the coefficient and cold-parity bounds, never regress MSE or either displayed local
maximum, and achieve geometric-mean MSE ratio `<=0.90`.  Full-frame and worst-crop visual review
must show no new ringing/checker/lattice failure.  A miss is a mapped counterexample, not grounds
to change the frozen recipe in place.

## Acceptance criteria

- [x] Backward-compatible typed configuration exposes zero versus input regularization center,
      solver start, and subtractive versus explicit frozen-base construction without changing the
      default HIER-010/012 behavior.
- [x] Closed-form/dense-oracle tests cover the minimum-norm solution, an unsafe unbounded input,
      partial/all-row frozen bases, coefficient-bound fallback, transaction revalidation, adjoint
      parity, and CPU determinism.
- [x] A bounded task driver records Phase-B/Phase-C row ledgers, solver trajectories, source and
      field hashes, exact counts, coefficient distributions, condition/precondition diagnostics,
      full/perceptual/local metrics, work/memory, cold parity, and portable visuals.
- [x] The Kodak killing gate is executed before the consumed-bank replay and its result is retained
      whether positive or negative; no executed bundle is overwritten or selectively repaired.
- [x] Any quantitative disposition receives a results-audit and an appropriately scoped ARA
      observation/evidence record; no production/default claim is made.
- [x] Task/Index/session/docs are synchronized, focused tests pass, report bundles receive their
      applicable structural check, and `./scripts/verify.sh` is run with unrelated failures
      disclosed.

## Interfaces touched

`src/structsplat/contraction_refinement.py`, focused projection tests, a bounded driver under
`scripts/experiments/`, narrow report-schema registration if needed, `docs/additive_field_v2.md`,
`docs/architecture.md`, ARA evidence/trace records, this task, the Index, and the generated session
brief.

## Depends on

HIER-013, CORE-013, BENCH-002, ADR-0006

## Reversible fallback

The existing input-centered mode remains the default.  Every opt-in solve retains the exact
incoming field as stage-zero fallback, and no maintained pipeline consumes HIER-014.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `c547e987a6fd72e5ebf65ae1401fbf2a74cd0750a25615a3946b2ce81927592b`

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.  No formal
`### Protocol review` is claimed for this diagnostic; a later result-bearing promotion requires a
clean tree and distinct prospective reviewer.

### Handoff

#### Objective

Test whether HIER-013 failed primarily because its fixed-geometry RGB solve inherited large
near-null coefficient components, before spending another development bank on topology or geometry.

#### Changes

Preserved the legacy input-centered/subtractive solve as the default and added opt-in origin
restart, zero-centered regularization, explicit frozen-base construction, and unsafe-stage-zero
reconditioning.  Added dense-oracle and compatibility tests plus a source-bound four-image Kodak
driver with complete solver, field, metric, work, parity, and visual artifacts.

#### Evidence

The 16-cell Kodak bundle has manifest SHA-256
`c547e987a6fd72e5ebf65ae1401fbf2a74cd0750a25615a3946b2ce81927592b`.
Independent manifest replay verifies 189/189 file hashes and the JSON/JSONL row ledger.  The
origin/explicit arm reconditions three of four fields and reduces geometric-mean MSE by only
`0.7212%` (`+0.03144 dB`); mean LPIPS worsens `0.001035`, one image retains the unsafe fallback,
and one accepted field worsens the 7x7 maximum.  The frozen gate fails and Phase C was not run.

#### Assumptions

The HIER-005 geometry, additive peak-one equation, exact 7,000-row count, coefficient limit 16,
and displayed local transaction remain fixed.  A zero-centered Krylov solve is treated as a
numerical conditioning control, not a new image representation.

#### Uncertainties

This is a dirty-source, one-seed development diagnostic on four Kodak images without distinct
prospective review.  The task-specific bundle predates registration in the repository-wide bundle
checker, so its manifest and ledgers were replayed independently rather than accepted as a
claim-ready report.

#### Review focus

Audit legacy compatibility, the zero-centered normal equations, exact stage-zero fallback, the
three successful coefficient-range collapses, the kodim13 local-guard rejection, and the refusal
to run the consumed-bank replay after the Kodak gate failed.

#### Protected actions not taken

No coefficient cap, consumed HIER-013 source, maintained pipeline/default, renderer, Field V2
semantic, codec, external repository, commit, or push changed.  The failed Kodak bundle was not
overwritten or selectively rerun.

#### Recommended next action

Treat fixed geometry as the remaining bottleneck.  On prospectively selected images, test bounded
appearance reconditioning followed by trust-region all-row geometry relaxation and reprojection,
with a direct fixed-count normalized fit as the representation-quality control.

### Review

#### Verdict

Provisionally accepted as a negative mechanism result

#### Self-reviewed

Yes

#### Correctness

The focused suite covers exact minimum-norm recovery, partial/all-row base construction, unsafe
stage zero, fallback, bounds, validation, and legacy defaults.  All expected cells, hashes, counts,
and non-RGB identity checks replay, and the aggregate was independently recomputed from raw rows.

#### Evidence quality

The source bindings and failure gate are adequate for killing this numerical explanation, but the
dirty tree, one CUDA trajectory, unsupported report schema, and absent distinct reviewer prohibit
promotion or a general claim.

#### Simplicity

The default behavior is unchanged.  The opt-in switches isolate regularization center, start
vector, and base construction without replacing PCG or materializing a dense design matrix.

#### Missing cases

Clean execution, distinct review, repeated CUDA trajectories, a geometry-changing successor,
matched direct-fit control, selected Field V2 semantics, and complete-rate evidence remain open.

#### Required changes

None for retaining the negative diagnostic.  Do not run Phase C or tune this Kodak quartet.

#### Optional improvements

Register the task schema with `check_report_bundle.py` when another maintained HIER report consumes
the format; the independent 189-file replay is sufficient for this non-claim diagnostic.

## Result

Phase B completed on 2026-08-10 and failed.  Compared with HIER-005, origin restart plus explicit
base construction recorded geometric-mean MSE ratio `0.9927876691`, mean PSNR delta
`+0.031436 dB`, mean MS-SSIM delta `+0.0015530`, and mean LPIPS delta `+0.0010347` (worse).  It
selected nonzero iterations on `3/4` images.  `kodim13` safely returned stage zero because later
bounded iterates slightly worsened the displayed local violation; `kodim19` reconditioned an
incoming coefficient maximum of `183.56` to `7.81` but regressed the 7x7 maximum and LPIPS.

Origin/subtractive and origin/explicit aggregates are numerically indistinguishable, so frozen-base
subtraction was not the causal problem.  The coefficient-range hypothesis is only partly true:
origin restart makes unsafe fields numerically usable, but fixed HIER-005 geometry still leaves the
lattice and yields negligible quality movement.  The protocol therefore forbids Phase C, which was
not executed.  Evidence:
`ara/evidence/hier014-conditioned-projection-kodak-diagnostic-2026-08-10/run.md`.

## Notes

The numerical foundation is established sparse least squares and stable radial-basis computation,
not a novelty claim.  The recipient-specific causal prediction is that coefficient range and cold
parity improve specifically when the inherited near-null component is discarded; explicit-base
construction should matter most where subtractive accumulation drift is large.  If bounded
minimum-norm coefficients still leave the visible lattice, the next hypothesis is geometric/basis
inadequacy rather than another coefficient cap.
