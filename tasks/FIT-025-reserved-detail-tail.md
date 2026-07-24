# FIT-025: Reserved error-adaptive detail tail

**Status: implemented and development-screened; specialized tail not promoted.**

## Context

FIT-024 separates fixed-capacity storage from topology policy, but
`SafeScheduleConfig.capacity` still serves two roles: physical tensor capacity and the active-row
limit consumed by boundary/redistribution growth. Raising it therefore exposes every spare row to
the ordinary schedule instead of reserving rows for a late fine-detail pass.

## Goal

Separate physical capacity from the normal active-row limit and add an opt-in, post-color-solve
detail tail that activates reserved rows in bounded error-driven transactions.

## Acceptance criteria

- [x] Fixed storage may preallocate more rows than the pre-tail active limit without exposing the
      reserve to coverage, boundary, or redistribution phases.
- [x] The historical behavior is unchanged when the pre-tail limit equals capacity and the detail
      tail is disabled.
- [x] Tail candidates are restricted to well-covered interior high-frequency residuals and use
      only detail birth or moment-preserving split.
- [x] Tail rows activate in bounded batches, retain the full safe commit gate, and stop after the
      first transaction with no safe/effective winner.
- [x] Config, history, storage telemetry, runner CLI, and tests expose physical capacity,
      pre-tail limit, reserve, batch size, and realized activation.
- [x] A source-bound three-arm Janelle development run compares identical fixed physical storage:
      11k baseline, generic +512 active budget, and an adaptive post-color detail tail up to +512.

## Interfaces touched

`src/structsplat/safe_schedule.py`, `deprecated_scripts/fit_janelle_safe_commit_schedule.py`, a bounded
three-arm runner, focused tests, documentation, and evidence.

## Depends on

FIT-023, FIT-024, ADR-0021, CORE-010/011.

## Development result

All arms use a 12,024-row fixed physical pool and the byte-identical 5,000-row initialization.
Generic +512 reaches `27.219252/11.582541 dB` foreground/boundary versus
`27.065330/11.412448 dB` for baseline and `27.106930/11.432527 dB` for the equal-count adaptive
tail. It is nonworse on every protected metric, strictly better on every nontrivial quality and
coverage metric, and fastest in the observed execution.

The tail safely accepts all four 128-row birth waves, but gain per row falls from `3.3301e-5` to
`2.7866e-6`. Keep it opt-in and default-off. A nonzero threshold is not selected post hoc.

Evidence:
`ara/evidence/fit025-reserved-detail-tail-janelle-2026-07-24/run.md`,
`runs/janelle_C0001_detail_tail_ablation_20260724/index.html`, and ADR-0022.
