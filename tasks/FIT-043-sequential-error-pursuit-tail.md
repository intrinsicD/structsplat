# FIT-043: Sequential error-then-pursuit tail

## Status

Completed negative under the frozen exact-prefix rule. The sequence reaches the cumulative detail
target in `51/51` cells and passes the other three decision rules, but one of 44 executed pursuit
cells refreshes one inherited `scale_max` certificate element at schedule entry. Exact equality
against the persisted error field is therefore only `43/44`; reject this controller without
retuning. This is not FIT-042 confirmation and cannot change a default.

## Context

FIT-031's natural error-only tail and FIT-040's orthogonal pursuit tail solve different objectives.
Across the 51 non-reference Janelle cells in
`ara/evidence/janelle-cross-view-tail-diagnostic-2026-07-28/`, error-only wins foreground-PSNR
gain in every cell but reaches the declared `25%/20%` high-pass/Laplacian target in only `7/51`;
pursuit reaches it in `51/51` with far fewer rows. FIT-042 decision rule 5 explicitly says this
objective split is not evidence for combining the tails and requires a new task.

The plausible composition is ordered, not fused: run the global error-only stage first, freeze all
base plus error rows, and run pursuit only for the cumulative fine-detail deficit that remains.
Running pursuit first would let FIT-031's global optimizer overwrite the pursuit solution. Naively
running FIT-040 after FIT-031 with its ordinary stage-entry `25%/20%` rule would instead demand a
second full relative reduction and over-treat the image.

## Goal

Determine whether a separate `error_only -> orthogonal_pursuit` sequence preserves FIT-031's
global gain while recovering FIT-040's cumulative fine-detail target, and whether the pursuit
stage can stop against the original pre-tail baseline without changing either shipped single-stage
method.

## Frozen exposed-data protocol

### Inputs and reuse

1. Use exactly the 51 eligible cells in
   `runs/janelle_cross_view_tail_diagnostic_20260728/manifest.json`, excluding the original
   `frame_00008/C0001` reference as that manifest does. The frozen input manifest SHA-256 is
   `f8958a583c238cf649b9662b84130d75d2bf3afad7760f43ce22da9245ee6976`.
2. Reuse each persisted base, natural FIT-031 field, FIT-040 field, target, mask, metrics, crop,
   and source binding. Verify every referenced file against the SHA-256 recorded in its cell
   result before loading it. The input summary and comparison-table SHA-256 values are
   `c1cafe794fc73e8e32212d00b4edc9e8e48fe863d71c97195f8bd9db64bcc638`
   and `bdde1bd87adba10bd9ee1e33bc189c963d597a4fc81ae4311739d85c4a24c564`.
3. Do not rerun `base`, `error_only`, or `pursuit_only`. Execute only the missing
   `error_then_pursuit` arm. This is a correlated, one-seed, same-capture diagnostic on exposed
   archived bases, not an independent-image, equal-row, actual-rate, or population result.
4. Use the same exact-CUDA renderer, max-side-1200 materialized target/mask bytes, FIT config,
   mask constraint, Pillow environment, seed 0, and 128-row/2,048-row FIT-040 pursuit limits as
   the input run. A renderer, target, mask, or environment mismatch fails before a result cell.

### Sequential controller

For original-base detail error `b`, post-error detail error `e`, and cumulative reduction target
`T`, the pursuit stage uses the exact equivalent stage-entry target

`max(0, 1 - b * (1 - T) / e)`.

Apply it independently with `T=0.25` for sigma-1.5 high-pass MSE and `T=0.20` for Laplacian MSE.
The zero clamp conservatively forbids regression when error-only already cleared one metric.

1. If error-only already clears both cumulative targets and is protected-safe, reuse it as the
   combined result and record `already_satisfied`; add no pursuit rows.
2. Otherwise start from the exact persisted error-only field. Run the unchanged FIT-040 pursuit
   mechanism with only the two adjusted stopping targets changed. All base and error rows are the
   inherited frozen prefix. The stage-entry protected gate remains relative to the accepted
   error-only state.
3. Recompute the final cumulative detail reductions and protected gate against the original base.
   A stage can be locally accepted but the combined result is a failure if this outer gate, exact
   prefix equality, containment, or outside-zero check fails.
4. Do not enable both production flags, change their mutual-exclusion validation, merge their
   scores, globally optimize after pursuit, retune thresholds, or consume any FIT-042 source.

### Measurements and report

Persist per-cell bindings, adjusted targets, skip/run disposition, stage and cumulative detail
reductions, protected decisions relative to both stage entry and original base, inherited-prefix
tensor checks, added and total tail rows, foreground/boundary quality, termination, timing, full
renders, fixed prior detail crops, and residual crops. Reuse single-arm values only from the
audited source payloads and clearly label their timing as historical.

Write resumable per-cell JSON, a tidy CSV, aggregate JSON, an exact executed-source snapshot,
independent audit JSON, reproduction commands, and a portable
`ara/evidence/fit043-sequential-error-pursuit-janelle-2026-07-28/index.html`.

## One-shot decision rules

The sequence is a **viable dual-objective exposed-data option** only if:

1. all 51 combined cells clear the cumulative `25%/20%` target, the original-base protected gate,
   containment/outside-zero checks, and all non-skipped error-prefix equality checks;
2. the combined foreground-PSNR gain retains at least `95%` of error-only's gain in every cell and
   its median loss relative to error-only is no worse than `0.05 dB`;
3. the non-skipped pursuit stage improves both detail metrics relative to error-only in every cell;
4. the median incremental pursuit rows do not exceed pursuit-only's median rows.

If any rule fails, reject this controller without retuning. If all pass, the result authorizes
only a later explicit opt-in pipeline-interface task/ADR amendment. It does not establish row,
byte, work, or general efficiency: the combined arm inherits FIT-031's much larger natural count.
The default remains neither tail, and objective-specific single-stage dispatch remains valid.

## Acceptance criteria

- [x] The frozen input hashes and every per-cell source/field binding pass before GPU work.
- [x] Focused tests cover the cumulative target transform, already-satisfied skip, stale binding
      rejection, and prefix equality.
- [x] Only `error_then_pursuit` is newly executed; every reused value retains its source path/hash.
- [x] All 51 cells finish or remain visible as failures; the one-shot rules are evaluated once.
- [x] An independent audit recomputes bindings, metrics, prefix checks, aggregates, and the
      decision from persisted fields and source rows.
- [x] The evidence bundle contains a portable visual `index.html` and exact rerun commands.
- [x] Record a scoped ARA staging observation, update the evidence/task indexes, run the
      results-audit and review skills, and execute the repository verification gate.

## Interfaces touched

`scripts/experiments/fit043_*`, focused tests, `ara/evidence/fit043-*`,
`ara/staging/observations.yaml`, and `tasks/INDEX.md`. Production scheduling, renderer, primitive,
codec, flags, mutual-exclusion validation, and defaults remain unchanged.

## Depends on

FIT-031, FIT-040, FIT-041, FIT-042, CORE-012, BENCH-002, ADR-0029, ADR-0030.

## Notes

This task tests separate ordered stages, not a combined score or optimizer. Its sources were
already exposed during method development and the cross-view diagnostic, so even a unanimous
result is mechanism-transfer evidence within one capture only. FIT-042 remains the authority for
independent quality confirmation of pursuit itself.

## Result

The source-bound RTX-4090 run completed all 51 cells with no execution failure. FIT-031 already
met both cumulative detail targets in seven cells, so those pursuit stages correctly became
no-ops. The remaining 44 cells ran the unchanged pursuit mechanism with only the algebraically
adjusted stopping targets:

- cumulative target, original-base protected gate, and exact outside zero: `51/51`;
- stage-entry protected gate: `44/44` executed cells;
- median new pursuit rows: 256 over all cells and over executed cells, versus 384 for the reused
  pursuit-only arm;
- median combined-tail rows: 6,272, so the sequence is not a row-efficiency result;
- median cumulative high-pass/Laplacian reductions: `27.297%/28.269%`;
- median foreground-PSNR gain: `+3.619718 dB`, `+0.048888 dB` above error-only; every cell retains
  at least 95% of its error-only gain;
- every executed pursuit stage improves both declared detail metrics.

Rules 2--4 pass. Rule 1 fails: `frame_00008/C0021` changes one `scale_max` element on one of 11,184
inherited rows by `0.8666666` during the schedule-entry containment refresh. Means, log scales,
rotations, colors, opacities, and every other inherited tensor remain exact, and the pursuit wave
freezes the post-refresh prefix; nevertheless the preregistered comparison is against the exact
persisted error field, so prefix equality is `43/44` and the frozen verdict is negative.

The independent cold audit passes all ten audit categories and reproduces the failed decision.
Evidence:
`ara/evidence/fit043-sequential-error-pursuit-janelle-2026-07-28/`.

Focused verification passes: 45 FIT-043/pursuit/schedule tests, ruff, docs sync, ARA, task policy,
script layout, and diff whitespace. The full portable gate completes with `1,492 passed`,
`4 skipped`, and four failures that reproduce the pre-existing unrelated worktree baseline:
rank-deficient affine-carrier condition-number finiteness, no-mask PSNR bit-identity, missing
PyTorch-2.7 CUDA PCI properties, and a filesystem race expectation in the SSP2V decode worker.
