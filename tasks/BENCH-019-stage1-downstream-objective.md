# BENCH-019 — Stage-1 downstream-objective validity

## Context

The proposed Field V2 has two consumers: image reconstruction and realtime-gs lifting. Existing
Stage-1 comparisons rank fields primarily by foreground image metrics, while realtime-gs also
consumes component geometry, density/coverage, and point-query behavior. The current additive
weight/color factorization is source-render non-identifiable, and bounded realtime-gs diagnostics
motivate testing whether the ambiguity matters downstream. Before selecting a new representation
or loss, the project must determine which Stage-1 diagnostics, if any, predict fixed-protocol
downstream quality.

## Goal

A reuse-first, cross-repository experiment that ranks existing Stage-1 field families under one
pinned realtime-gs lift/train/evaluation protocol and determines whether image fidelity,
alpha/boundary, coverage/query, or correspondence diagnostics are valid downstream selection
surrogates.

## Non-goals

- Implementing Field V2, changing either repository's default, or tuning realtime-gs per field.
- Claiming population-level correlation from correlated views or fewer than three independent
  capture groups.
- Selecting a structural-mass target before the downstream response is measured.
- Repairing incomplete historical fields or treating a converted normalized field as an exact
  additive teacher.

## Protocol requirements

- Freeze scene/frame bundles, training/held-out cameras, field-family inclusion, source and field
  hashes, StructSplat/realtime-gs commits, environments, seeds, downstream schedule, and missing
  policy before execution.
- Reuse complete existing additive and normalized bundles first. Any newly fitted family receives
  the same source pixels, masks, camera records, count/byte target, and seed policy.
- Use at least two stage frames and three downstream seeds; cluster uncertainty by frame/capture,
  never by view. If fewer than three independent capture groups exist, label any correlation
  workload-specific and prohibit a general surrogate claim.
- Candidate Stage-1 predictors: foreground PSNR/MS-SSIM/LPIPS, boundary error, alpha agreement,
  structural coverage, cold point-query error, track/correspondence yield, field conditioning,
  rows, and complete bytes.
- Downstream responses: held-out PSNR/MS-SSIM/LPIPS, alpha/geometry diagnostics, convergence curves,
  terminal rows, fit time, and failure rate under one unchanged realtime-gs protocol.

## Acceptance criteria

- [ ] A committed protocol manifest and adapter bind all fields, pixels, masks, cameras, commits,
      environments, seeds, schedules, metrics, and split roles before outcome access.
- [ ] A/A replay proves the adapter cannot change source field semantics or downstream config;
      normalized and additive inputs are labelled and queried through their exact equations.
- [ ] The report contains paired field-family rankings, rank correlations, leave-one-frame-out
      diagnostics, frame-cluster uncertainty, missing/error cells, and representative downstream
      visuals; no view is treated as an independent replicate.
- [ ] The decision is explicit: select one validated Stage-1 objective/surrogate, require downstream
      evaluation in every later gate, or record the question unavailable. No post-hoc metric blend.
- [ ] A portable report bundle passes `scripts/check_report_bundle.py`; a distinct results audit
      recomputes the rankings and decision from raw rows.
- [ ] Outcome recorded in ARA, `docs/additive_field_v2.md`, and the Index with exact scope.
- [x] `./scripts/verify.sh` passes (1,520 passed, 4 skipped on 2026-08-03).

## Interfaces touched

Pinned external-run adapter under `benchmarks/native_runners/` or a bounded driver under
`scripts/experiments/`; report/metric adapters; tests; `ara/evidence/`; this task, the Index, and
`docs/additive_field_v2.md`. Realtime-gs changes require its own repository authority and are not
silently made here.

## Implementation progress (2026-08-03)

- `benchmarks.stage1_downstream_objective` now owns a passive protocol lifecycle: draft, clean
  source/artifact binding, review digest, distinct outcome-unseen approval, final freeze, stable
  cell plan, result-row validation, and report analysis. It records but never executes the pinned
  realtime-gs command.
- Additive and normalized families must declare their exact equation/blend pair and semantic
  digest. Every result cell must match the frozen field manifest and a downstream-factor digest
  that is constant across field families. The selected A/A cell checks both identities plus every
  frozen predictor/response tolerance.
- Analysis averages seeds/initializers within a frame-family unit, ranks within frame, clusters the
  bootstrap by capture, emits per-frame and leave-one-frame-out diagnostics, retains error/missing
  cells, and implements only the frozen single-predictor priority. The report schema is accepted by
  `scripts/check_report_bundle.py` without pretending downstream fields are StructSplat NPZ jobs.
- Focused CPU tests cover semantic relabelling rejection, split integrity, clean-source and
  distinct-review lifecycle, matched config enforcement, A/A failure, surrogate selection,
  portable report validation, and artifact tamper detection.
- Formal evidence is intentionally not started. The live frame-00008 mask-contained production is
  incomplete; frame 00009 has no matched three-provider 11k bundle; and both frames are one capture
  group. The general claim therefore still needs complete matched fields on at least three
  independently approved capture groups; the new source-only portfolio does not satisfy that gate.
  A supplied-frame-only run must remain workload-specific.
- Isolated realtime-gs driver checkpoint `d3e76fe` now implements the exact
  `structsplat.bench019.cell.v1` exporter, verified family/A-A downstream factor, raw JSON-pointer
  extraction, six-artifact binding, explicit error cells, and receipt-required stable assembly.
  Synthetic-success and calibrated Stage intentional-error diagnostics both pass this repository's
  row validator with zero problems. This external checkpoint is a driver handoff pending distinct
  implementation review; it is not yet an accepted executor or authority to freeze a protocol.
- The realtime-gs source portfolio at that checkpoint binds 88 files across three development and
  three disjoint confirmation acquisition groups. Its exact descriptor is 39,474 bytes with
  SHA-256 `cc9918b835c4c507a7959af0c60270e12b673007cf3605b459e1acc0549f6305`.
  The portfolio explicitly keeps formal protocol, confirmation outcome, and complete-family gates
  false: five groups have no matched fields, TUM adapters/keyframes are unfrozen, Karate has no
  frozen mask policy, and the Stage mask-contained family was only 13/26 at the snapshot.
- Do not add a competing executor here. After distinct review accepts the realtime-gs checkpoint,
  implement/freeze the remaining Stage-1 predictor collector and source adapters under their
  owning repository, produce matched development fields, then bind the accepted commits,
  environments, portfolio, and exact artifacts into a prospective BENCH-019 protocol.

## Depends on

BENCH-001/002, CORE-012

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending (external driver checkpoint `d3e76fe` is not yet accepted)

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. A distinct reviewer must approve the frozen protocol digest before any formal
downstream run.

Implementation began on 2026-08-03 on branch
`bench/019-stage1-downstream-objective`. The adapter, validation, and analysis surfaces may be
developed and exercised with synthetic or dirty-source diagnostic inputs during the driver turn.
No claim-bearing downstream execution is authorized until a distinct reviewer approves the exact
frozen protocol digest; the user's requested same-data production comparison remains the final
promotion gate.

## Notes

This is the first gate. A positive image-metric correlation permits a cheap later objective; a
negative result is equally actionable because it prevents optimizing an attractive but irrelevant
Stage-1 score.
