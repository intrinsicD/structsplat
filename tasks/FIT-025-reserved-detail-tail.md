# FIT-025: Reserved error-adaptive detail tail

**Status: in progress.**

## Context

FIT-024 separates fixed-capacity storage from topology policy, but
`SafeScheduleConfig.capacity` still serves two roles: physical tensor capacity and the active-row
limit consumed by boundary/redistribution growth. Raising it therefore exposes every spare row to
the ordinary schedule instead of reserving rows for a late fine-detail pass.

## Goal

Separate physical capacity from the normal active-row limit and add an opt-in, post-color-solve
detail tail that activates reserved rows in bounded error-driven transactions.

## Acceptance criteria

- [ ] Fixed storage may preallocate more rows than the pre-tail active limit without exposing the
      reserve to coverage, boundary, or redistribution phases.
- [ ] The historical behavior is unchanged when the pre-tail limit equals capacity and the detail
      tail is disabled.
- [ ] Tail candidates are restricted to well-covered interior high-frequency residuals and use
      only detail birth or moment-preserving split.
- [ ] Tail rows activate in bounded batches, retain the full safe commit gate, and stop after the
      first transaction with no safe/effective winner.
- [ ] Config, history, storage telemetry, runner CLI, and tests expose physical capacity,
      pre-tail limit, reserve, batch size, and realized activation.
- [ ] A source-bound three-arm Janelle development run compares identical fixed physical storage:
      11k baseline, generic +512 active budget, and an adaptive post-color detail tail up to +512.

## Interfaces touched

`src/structsplat/safe_schedule.py`, `scripts/fit_janelle_safe_commit_schedule.py`, a bounded
three-arm runner, focused tests, documentation, and evidence.

## Depends on

FIT-023, FIT-024, ADR-0021, CORE-010/011.
