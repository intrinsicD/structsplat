# HIER-011 — Guarded residual column exchange

## Status

In review.  This is an exposed, source-snapshotted diagnostic.  It cannot promote a default,
semantic, rate, or general quality claim.

## Context

HIER-005's exact-count hard-3-sigma/touched recovery remains the strongest contraction result.
At exactly 7,000 rows it reaches 50.097060/54.374098 dB on exposed C0001/C0004, but C0001's
displayed worst-pixel RMSE is 0.026404 and fails the provisional 0.02 gate.  HIER-010 showed that
fixed residual-leaf reservation spends capacity before its value is known: the 350-row reserve
loses about 0.19 dB on both views.  Its fixed-geometry RGB projection is safe but gains only
0.010944/0.004361 dB.  FIT-033/038/040 show that residual remeasurement and exact partial color
solves are useful; FIT-043 shows that unconstrained residual appends do not preserve an exact
prefix/count contract.

The missing operation is count-neutral capacity reallocation.  This task treats the active
7,000-row field as an active set.  A residual atom may enter only by paying the measured deletion
cost of a leaving row, and an exchange commits only after the maintained cold renderer verifies a
strict global SSE improvement with non-regression of both displayed local maxima.

## Goal

Build and run the strongest bounded exact-7k successor supported by the existing evidence:
matrix-free residual column generation, exact removal pricing, guarded one-for-one exchange, and
the already-safe HIER-010 touched/new-row appearance projection.

## Non-goals

- Do not change Observation Field V2 semantics, the normalized/default pipeline, or HIER-005.
- Do not append rows, call row count or reference NPZ bytes compressed rate, or claim equal work.
- Do not describe C0001/C0004 as held-out or independent; both are exposed correlated Janelle
  development views and HIER-010 already consumed them.
- Do not tune on C0004 after the development-bank choice is frozen.
- Do not promote a scientific claim without a clean run and distinct prospective review.

## Acceptance criteria

- [x] Exact removal price and entering-atom gain agree with direct finite-support reconstruction.
- [x] Every committed pivot preserves exactly 7,000 rows, changes exactly one row, strictly lowers
  raw masked SSE, and does not worsen displayed pixel-max or 7x7-max RMSE.
- [x] Candidate ranking, NMS, donor ties, locking, rollback, and stopping are deterministic.
- [x] The implementation is matrix-free, default-off, NumPy-safe at import, and maintains cold
  renderer parity below `2e-6`.
- [x] Focused tests cover analytical pricing, exact-count/frozen-row invariants, deterministic
  replay, unsafe rollback, and maintained-render parity.
- [x] A frozen two-view diagnostic retains four exact-count arms, raw trajectories, lossless
  fields, source/config hashes, full/worst-crop visuals, metric curves, and portable `index.html`.
- [ ] The full `exchange_projection` arm improves PSNR by at least 0.10 dB versus the HIER-005
  control on both views, passes pixel-max `<=0.02` and 7x7-max `<=0.01`, and strictly improves
  masked SSE without worsening either local maximum on either view.
- [x] The report bundle, cold metric replay, focused tests, and repository verification are run;
  any unrelated baseline failure is recorded rather than hidden.

## Interfaces touched

- `src/structsplat/residual_exchange.py`
- `scripts/experiments/hier011_guarded_residual_column_exchange.py`
- `tests/test_residual_exchange.py`
- `scripts/check_report_bundle.py` for the narrow diagnostic-schema registration
- `docs/architecture.md`, `docs/additive_field_v2.md`
- `tasks/INDEX.md`, `tasks/SESSION-BRIEF.md`
- `ara/trace/exploration_tree.yaml`, `ara/staging/observations.yaml`, `ara/evidence/README.md`

## Depends on

HIER-005/009/010, FIT-033/038/040/043, CORE-013, BENCH-002, ADR-0006

## Development-bank preflight

C0001 is the method-development view.  Before opening C0004, compare exactly three fixed atom
banks for at most 32 accepted pivots from the persisted HIER-010 `h005_control` field:

1. `compact`: isotropic sigma `(0.18, 0.30, 0.45)` px.
2. `multiscale`: isotropic sigma `(0.30, 0.45, 0.60, 0.75)` px.
3. `oriented`: the multiscale bank plus sigma `(0.75, 0.30)` px at rotations
   `(0, pi/4, pi/2, 3pi/4)`.

All banks use residual RGB-energy ranking, the top 96 Chebyshev-radius-1 NMS sites, the 64
lowest-price eligible donors, and a 24-pair cold-render frontier.  A new row is locked against
later deletion.  The selected bank is the safe arm with lowest raw SSE after 32 pivots; ties use
lower displayed pixel max, lower 7x7 max, fewer tested proposals, then bank order.  This preflight
is development evidence only and must be retained separately from the frozen report.

## Frozen diagnostic protocol

This section is completed, source-hashed, and snapshotted after the C0001 bank preflight and
before C0004 is opened.  The fixed elements already decided are:

- Exact target: 7,000 signed direct-additive peak-one rows under the inherited hard AABB
  3-sigma support, packed alpha, and no filtering/dilation.
- Bases: the persisted HIER-010 `h005_control` and `control_projection` fields, verified by file
  and canonical hashes before use.  This removes contraction rerun nondeterminism.
- Arms: `h005_control`, `control_projection`, `guarded_exchange`, and
  `exchange_projection`, all at exactly 7,000 rows.
- Exchange: at most 128 accepted one-for-one pivots; residual/site/donor lists are recomputed
  after every commit.  The leaving-row deletion price and entering coefficient/gain use the exact
  masked finite-support basis.  Candidate and donor support boxes must be disjoint, making their
  paired predicted SSE delta exact.  Nonfinite/over-limit coefficients are ineligible.
- Transaction: rank negative reduced-cost pairs deterministically; cold-render at most the first
  24.  Commit the first pair whose raw masked SSE strictly improves beyond numerical tolerance
  and whose displayed pixel and 7x7 maxima are individually nonworse.  Otherwise stop with the
  last committed field.  Replaced rows are locked and the count never changes.
- Projection: run HIER-010's `1e-8`-ridge, tolerance-`1e-6`, 48-iteration fail-closed PCG on the
  union of the inherited touched rows and newly exchanged rows.  Geometry, untouched rows,
  topology, alpha, and semantics remain frozen.  The selected checkpoint must not worsen its
  stage-zero SSE or displayed normalized artifact violation.
- Primary/local/perceptual/integrity/work metrics are inherited from HIER-010.  Full mechanism
  success is exactly the final unchecked acceptance criterion above.

### Frozen bank and input bindings (sealed before C0004 execution)

The C0001 preflight completed all 32 safe pivots for all three banks.  Compact ended at
SSE `0.4413312593`, PSNR `50.345451`, pixel max `0.0144974`, and 7x7 max `0.0075162`;
multiscale ended at `0.4375248764`, `50.383070`, `0.0144974`, and `0.0071011`; oriented ended at
`0.4316094323`, `50.442188`, `0.0144974`, and `0.0072757`.  Oriented has the lowest SSE and is
therefore selected.  Its exact ordered bank is:

```text
(0.30,0.30,0), (0.45,0.45,0), (0.60,0.60,0), (0.75,0.75,0),
(0.75,0.30,0), (0.75,0.30,pi/4), (0.75,0.30,pi/2), (0.75,0.30,3pi/4)
```

All persisted inputs live below
`results/hier010_residual_anchor_projection_janelle_2026-08-10/artifacts`:

| image | input | file SHA-256 | canonical field SHA-256 |
|---|---|---|---|
| C0001 | `C0001__h005_control__n7000/field.observation.npz` | `cfa05c3cc5bfe5f747e14bae2cfe254283593123a4eb92177f73b95a071d1dac` | `9bfbe941b90bac66a7c6ce3166fffcd76224520f40edc0800a9de4a1ea9cfb5b` |
| C0001 | `C0001__control_projection__n7000/field.observation.npz` | `b15cadb6b5211dcd0cb70cf074534a8f3e8d7650b47cb3ac5f98d656c07a7595` | `21a93aef6d249eb788a29a54d246d118031929d41bd4557e7059e78aa5c9a59c` |
| C0004 | `C0004__h005_control__n7000/field.observation.npz` | `6743d9141791c8932532708085815157e7c918ee398c1ab5fbcbc0342b111f5b` | `8c4406059c5bc68254fd2b16740019ae550230aa6778369f99ac9d606e9635e8` |
| C0004 | `C0004__control_projection__n7000/field.observation.npz` | `7404c470ba90b806ebff5e9cd20eb623ab62f30dcff12e5baca3c3ca72f9da3c` | `177987a89c891d278ed84d5458fe350abbd1e6343a0c91fcf695842b0c952ac6` |

The H005 analysis ledgers that carry inherited touched-row provenance are bound by SHA-256
`bb03c256a99959689f15b11b1122ccf671b91653cebcd26f584537f1bc0b48a5` (C0001) and
`b6dda6626c0bdd1a0a52a7465a7bbfdf4e10da4debc7c0d96a4bde1864791a90` (C0004).
Native RGB/mask hashes remain HIER-010's frozen C0001
`ae24fe99...c411b`/`94dcbf70...0c3` and C0004
`26eb4cf2...070e`/`4702bfa9...35d` bindings; the driver verifies the complete values.

Exact command:

```bash
PYTHONPATH=src python scripts/experiments/hier011_guarded_residual_column_exchange.py \
  --images \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0001.jpg \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/rgb/C0004.jpg \
  --masks \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0001.png \
    /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0004.png \
  --hier010-root results/hier010_residual_anchor_projection_janelle_2026-08-10 \
  --out results/hier011_guarded_residual_column_exchange_janelle_2026-08-10 \
  --max-exchanges 128 --site-count 96 --site-nms-radius 1 \
  --donor-count 64 --proposal-frontier 24 --coefficient-limit 16 \
  --projection-ridge 1e-8 --projection-tolerance 1e-6 \
  --projection-max-iterations 48 --projection-coefficient-limit 16 \
  --max-side 512 --mask-threshold 0.5 --device cuda \
  --renderer cuda_additive --render-chunk 256 --lpips
```

## Reversible fallback

HIER-005 remains unchanged.  A failed exchange returns its previous exact field; a failed
projection returns exchange stage zero.

## Diagnostic outcome (2026-08-10)

The oriented bank won the declared C0001 preflight and was frozen before C0004.  The exact-count
exchange commits 68/5 pivots and then stops with `no_improving_pair`; the 128-pivot cap never
binds.  Exchange plus the inherited touched/new-row projection improves HIER-005 by
`+0.541598/+0.079863 dB`, reduces masked MSE by `11.725/1.822%`, and changes displayed pixel/7x7
maxima from `0.026404/0.009518` to `0.013585/0.005999` on C0001 and from
`0.014847/0.004597` to `0.009335/0.003988` on C0004.  Both local gates pass.

The frozen mechanism decision is nevertheless negative because C0004 misses the declared
`+0.10 dB` material floor.  HIER-005 stays unchanged, these views are not retuned, and HIER-012
tests the coefficient-scope bottleneck exposed by the result.  The eight-cell report and cold
replay pass; maximum PSNR/MSE/LPIPS drift is `2.26e-7 dB`/`2.16e-13`/`9.65e-9`, local metrics
match exactly, and repeated renderer drift is `1.19e-7`.

Evidence:
`ara/evidence/hier011-guarded-residual-column-exchange-janelle-diagnostic-2026-08-10/run.md`.
Portable report:
`results/hier011_guarded_residual_column_exchange_janelle_2026-08-10/index.html`.
Manifest SHA-256:
`c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9`.

## Agent workflow

- Driver: codex
- Reviewer: codex
- Turn: reviewer
- Reviewed revision: report manifest `c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9`

### Handoff log

Provisional self-review only.  The report is source-snapshotted and structurally validated, but
the dirty run and absent distinct prospective reviewer prohibit claim/default promotion.

### Handoff

#### Objective

Replace HIER-010's harmful up-front residual reserve with a count-neutral, value-priced active-set
transaction that can materially lower exact-7k error without moving either displayed local maximum
in the wrong direction.

#### Changes

Added the default-off sparse `structsplat.residual_exchange` reference, analytical removal/candidate
pricing, cold-render rollback, immutable trajectory records, focused invariant tests, the frozen
HIER-011 driver/report schema, and synchronized architecture, task, ARA, and evidence records.

#### Evidence

The eight-cell report manifest is
`c15d18ee3b1eca4782c4400e2f94ffe35dca6a0b383ba960c6260d396c849bf9`.
Bundle validation, independent cold replay, exact-count/hash checks, trajectory monotonicity, and
56 focused HIER tests pass.  Repository-wide Ruff/structural checks pass; portable pytest reaches
1,736 passes with the same three untouched baseline failures as HIER-010.  The full frozen
mechanism gate fails only its C0004 `+0.10 dB` floor.

#### Assumptions

Support-disjoint enter/leave boxes make the paired analytical SSE delta exact; one locked row per
commit prevents cycling; C0001 alone selects the oriented bank; and the maintained CUDA render,
not the pricing surrogate, is the acceptance authority.

#### Uncertainties

Both views were already exposed by HIER-010, C0001 chose the bank, each cell has one numerical CUDA
trajectory, work is unequal, and no complete codec or independent capture group is present.

#### Review focus

Audit residual sign in deletion pricing, AABB/rounding parity, support-disjointness, deterministic
tie order, exact one-row replacement and locking, per-step local monotonicity, and the negative
C0004 material-floor decision.

#### Protected actions not taken

No maintained default, semantic, renderer, artifact threshold, codec, external repository, commit,
push, or unrelated baseline code was changed.  The frozen result was not retuned after C0004.

#### Recommended next action

Use HIER-011 as a local-tail attribution control.  HIER-012 shows global coefficient scope is the
larger error bottleneck; any claim-ready continuation needs new images and distinct review.

### Review

#### Verdict

Provisionally accepted as a retained negative-gate diagnostic

#### Self-reviewed

Yes

#### Correctness

Synthetic removal prices match direct reconstruction; rollback, exact count, frozen rows, locking,
deterministic CPU replay, lazy torch import, cold renderer parity, and all saved trajectory
invariants pass.

#### Evidence quality

The bank choice precedes C0004 and every cell/history/field/visual is retained, but prior exposure,
dirty execution, correlation, and absent distinct review cap the evidence at diagnostic status.

#### Simplicity

The method adds one exact one-for-one transaction around the existing field and renderer.  It has
no maintained dispatch and returns the incoming field when no pair is safe.

#### Missing cases

Unseen images, multiple capture groups, CUDA replicates, matched work, actual bytes, distinct
review, and a fully green repository-wide gate remain missing.

#### Required changes

None for retaining the diagnostic.  Do not promote the failed full mechanism or retune its frozen
view; obtain independent evidence before reuse as a primary pipeline stage.

#### Optional improvements

For a future allocation study, price overlapping multi-row exchanges or use block reoptimization,
but only after global projection is included as the direct stronger control.
