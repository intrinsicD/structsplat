# HIER-012 — Global safeguarded appearance projection

## Status

In review.  Exposed successor/attribution diagnostic only; no independent confirmation or
default authorization.

## Context

HIER-011 replaced low-value rows with residual atoms at exact N=7,000.  Its frozen run materially
improves HIER-005 on both exposed Janelle views and repairs all local gates, but the full
touched-row finish gains `+0.5416/+0.0799 dB`; C0004 narrowly misses HIER-011's prospective
`+0.10 dB` clause.  The exchange search genuinely saturates after 68/5 safe pivots, so a larger
pivot cap is not the answer.

A post-HIER-011 successor feasibility check removed HIER-010's inherited touched-row restriction
while retaining its matrix-free PCG and fail-closed checkpoint transaction.  Projecting all 7,000
RGB columns on fixed HIER-005 geometry reaches about `52.33/56.47 dB`, slightly better than first
exchanging rows and then projecting all columns.  This task packages and audits that simpler
winner with the necessary controls.  Both views informed the choice, so the outcome is descriptive
development evidence rather than a prospective test.

## Goal

Produce a portable exact-7k attribution report for the strongest pipeline found: persisted
HIER-005 geometry/topology plus global safeguarded matrix-free RGB projection.

## Non-goals

- Do not change geometry, topology, row count, alpha, support, filtering, field semantics, or any
  default pipeline.
- Do not call reference field bytes compression rate or make an equal-work/speed claim.
- Do not call C0001/C0004 held-out, transfer, confirmation, or independent evidence.
- Do not reinterpret this exposed feasibility-driven run as FIT-046/BENCH-020 authorization.

## Acceptance criteria

- [x] All five arms retain exactly 7,000 rows and preserve all non-RGB arrays bit-for-bit from
  their bound geometry source.
- [x] Global projection remains matrix-free and trains exactly all 7,000 RGB rows; every PCG
  checkpoint is finite and the selected checkpoint satisfies the stage-zero SSE/displayed-
  violation transaction.
- [x] The selected `global_projection` arm gains at least 1.5 dB over HIER-005 on both views,
  strictly lowers masked MSE, passes pixel-max `<=0.02` and 7x7-max `<=0.01`, and does not worsen
  either local maximum.
- [x] Maintained/repeated renderer parity and the projection adjoint check are each `<=2e-6`.
- [x] Cold replay reproduces all persisted primary/perceptual/local metrics within declared
  numerical tolerances.
- [x] The portable report retains lossless fields, PCG histories, source/config/input hashes,
  full/worst-crop visuals, curves, raw tables, and executed-source snapshots.
- [x] Focused tests, report-bundle validation, structural checks, and the repository gate are run;
  unrelated baseline failures are disclosed.

## Arms and selection

All rows are exact N=7,000:

1. `h005_control`: sealed HIER-010 HIER-005 field.
2. `touched_projection`: sealed HIER-010 touched-only projection.
3. `guarded_exchange`: sealed HIER-011 exchange field.
4. `exchange_global_projection`: arm 3 plus all-row projection.
5. `global_projection`: arm 1 plus all-row projection; selected pipeline.

The global solve uses HIER-010's finite-support matrix-free operator with all row-mask entries
true, zero protected rows, Tikhonov pull `1e-8`, relative tolerance `1e-6`, at most 48 PCG
iterations, coefficient absolute limit 16, and the same raw-SSE/displayed-normalized-violation
fail-closed checkpoint rule.  Geometry and all non-RGB arrays are immutable.  The selected
pipeline must meet every acceptance clause above.  Between the two global-projection arms, lower
masked MSE wins per view; if they split, use lower geometric mean MSE.  This rule was written after
both feasibility probes and therefore supports pipeline selection only, not confirmation.

## Frozen input bindings

- HIER-010 manifest SHA-256:
  `80b84bce9b5ec72e9369fd61474d761c8ecd3f2a9f6ed9495f7cb67f14dd81ba`.
- HIER-011 manifest SHA-256:
  `c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9`.
- HIER-010 HIER-005 field file/canonical hashes:
  C0001 `cfa05c3c...1dac`/`9bfbe941...fb5b`; C0004
  `6743d914...1f5b`/`8c440605...c5e8` (full values enforced by the driver).
- HIER-010 touched-projection file/canonical hashes:
  C0001 `b15cadb6...7595`/`21a93aef...a59c`; C0004
  `7404c470...da3c`/`177987a8...ac6`.
- HIER-011 guarded-exchange file/canonical hashes:
  C0001 `7a0260beed61742dad65beb7fe951a3dbe17be356bee73909e5ebf907819433c`/
  `8bc2b7b1896912335164ca19331e9885f7dd873dbe4ea53b3ca9fd995e9909f3`; C0004
  `69bfd813a4edd4fe0d547413d7355f5af02d4178db7733436f20619c9d7fd0da`/
  `385961330433cf4d3175bfccc12429a527ce236efc0a9a06e825e7eb062794ca`.
- Native image/mask hashes and deterministic 512-pixel rasterization are exactly HIER-010's
  complete bindings and are rechecked by the driver.

## Exact command

```bash
PYTHONPATH=src python scripts/experiments/hier012_global_appearance_projection.py \
  --images \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0004.jpg \
  --masks \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0004.png \
  --hier010-root results/hier010_residual_anchor_projection_janelle_2026-08-10 \
  --hier011-root results/hier011_guarded_residual_column_exchange_janelle_2026-08-10 \
  --out results/hier012_global_appearance_projection_janelle_2026-08-10 \
  --projection-ridge 1e-8 --projection-tolerance 1e-6 \
  --projection-max-iterations 48 --projection-coefficient-limit 16 \
  --max-side 512 --mask-threshold 0.5 --device cuda \
  --renderer cuda_additive --render-chunk 256 --lpips
```

## Interfaces touched

- `scripts/experiments/hier012_global_appearance_projection.py`
- `scripts/check_report_bundle.py` for narrow schema registration
- `docs/architecture.md`, `docs/additive_field_v2.md`
- `tasks/INDEX.md`, `tasks/SESSION-BRIEF.md`
- `ara/trace/exploration_tree.yaml`, `ara/staging/observations.yaml`, `ara/evidence/README.md`

## Depends on

HIER-005/010/011, FIT-005/033/046, CORE-013, BENCH-002, ADR-0006

## Reversible fallback

HIER-005 remains unchanged.  Projection stage zero is the exact fallback.

## Diagnostic outcome (2026-08-10)

All ten cells retain exactly 7,000 rows.  The selected HIER-005-plus-global-projection arm reaches
`52.334526/56.470211 dB`, gains `+2.237466/+2.096112 dB`, and reduces masked MSE by
`40.262/38.285%` on C0001/C0004.  Displayed pixel/7x7 maxima are
`0.016010/0.004586` and `0.008163/0.003136`, so both local gates pass.  Every non-RGB array is
bit-exact; the projection itself adds `0.820/0.683 s` to the persisted HIER-005 cumulative work.

Exchange plus the same global solve reaches `52.258653/56.455593 dB`, so direct global projection
has lower MSE on both views and 1.04% lower geometric-mean MSE.  Exchange retains better C0001
LPIPS and isolated-pixel max, which remains an explicit objective tradeoff.  The selected arm
clears the descriptive `+1.5 dB`/integrity gate, but both views informed the method choice: this is
the strongest observed exact-7k development pipeline, not independent confirmation or a default.

Independent cold replay verifies all field hashes/counts, non-RGB identity, PCG transactions, and
selection.  Maximum PSNR/MSE/LPIPS drift is `5.16e-7 dB`/`4.34e-13`/`1.26e-8`; local metrics match
exactly.  Internal/cold and repeated renderer parity are at most `1.79e-7` and `1.19e-7`.

Evidence:
`ara/evidence/hier012-global-appearance-projection-janelle-diagnostic-2026-08-10/run.md`.
Portable report:
`results/hier012_global_appearance_projection_janelle_2026-08-10/index.html`.
Manifest SHA-256:
`3abc28551be8c5a58bf4fc3a2ab4dc4acb11b731e11055e6b63f0673d2ea834b`.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `3abc28551be8c5a58bf4fc3a2ab4dc4acb11b731e11055e6b63f0673d2ea834b`

### Handoff log

Provisional self-review only.  Both views were exposed during successor selection, the tree was
dirty, and no distinct prospective reviewer participated.  The result may select a development
pipeline and motivate a fresh study; it cannot promote FIT-046, Field V2 semantics, or defaults.

### Handoff

#### Objective

Package and audit the strongest exact-7k successor found after HIER-011, while retaining the direct
controls needed to decide whether topology exchange still earns its place.

#### Changes

Added the five-arm HIER-012 source-bound driver and diagnostic schema, exercised HIER-010's existing
matrix-free projection with all 7,000 rows trainable, persisted all solver/field/metric/visual
artifacts, and synchronized architecture, Field V2, task, ARA, and evidence documentation.  The
selected method changes RGB coefficients only.

#### Evidence

The ten-cell report manifest is
`3abc28551be8c5a58bf4fc3a2ab4dc4acb11b731e11055e6b63f0673d2ea834b`.
The bundle and independent replay pass all exact-count, file/canonical hash, non-RGB identity,
checkpoint, renderer, metric, and selection checks.  The selected arm gains +2.2375/+2.0961 dB
and reduces MSE 40.26/38.29% with both local gates passing.  Repository-wide Ruff/structural
checks pass; portable pytest reaches 1,736 passes with the same three untouched baseline failures
as HIER-010.  The `verify.sh` wrapper cannot import Ruff in this shell, so its equivalent commands
were run directly.

#### Assumptions

Raw masked MSE is the primary remaining-error objective; the same HIER-010 ridge/tolerance/bound
and fail-closed transaction are retained; lower MSE selects between direct and exchange geometry;
and unequal projection work is reported rather than matched.

#### Uncertainties

Both correlated views informed the successor choice, one CUDA trajectory is retained, later safe
PCG iterates can be rejected by the display guard, and C0001 exposes an MSE versus LPIPS/isolated-
pixel tradeoff between the two global arms.  No actual-rate or unseen-image result exists.

#### Review focus

Audit the all-true trainable mask, exact freeze of every non-RGB array, step-zero/selectable
checkpoint ordering, cold-render and bilinear-adjoint parity, metric domains, and the explicit
MSE/LPIPS/local tradeoff in the simplicity decision.

#### Protected actions not taken

No maintained/default pipeline, Field V2 semantic, general FIT-046 interface, codec, renderer,
external repository, commit, push, or artifact threshold was changed.  The report does not call
reference bytes rate or exposed views confirmation.

#### Recommended next action

Run a clean, preregistered, distinct-reviewed screen on independently approved images with
HIER-005, touched projection, global projection, and exchange-plus-global controls.  Include
complete work/actual-byte accounting and only then feed the component into FIT-046/BENCH-021.

### Review

#### Verdict

Provisionally accepted as the next development pipeline

#### Self-reviewed

Yes

#### Correctness

All ten fields have exact count and bound hashes, all global arms train 7,000 rows, every non-RGB
array is bit-exact, selected checkpoints satisfy the fail-closed transaction, renderer drift is
sub-micro-unit, and independent cold metrics reproduce.

#### Evidence quality

Artifacts and replay are unusually complete for a diagnostic, but the method was selected after
both views were inspected.  This supports engineering selection and a new hypothesis, not a
publishable/general/default claim.

#### Simplicity

The selected pipeline is HIER-005 plus one fixed-geometry coefficient solve.  It removes HIER-011
topology work because direct geometry has lower MSE on both views; stage zero is an exact fallback.

#### Missing cases

Prospective independent images, distinct reviewer, clean source, multiple CUDA replicates, matched
work, complete codec bytes, selected Field V2 semantics, and repository-wide baseline fixes remain
missing.

#### Required changes

None for retaining this diagnostic and using the arm as the next development candidate.  Any
maintained/default/FIT-046 promotion requires the missing prospective gates.

#### Optional improvements

Add registered cross-arm crops and an explicitly multi-objective selection report so a later study
can quantify the C0001 MSE versus LPIPS/worst-pixel tradeoff rather than choosing one silently.
