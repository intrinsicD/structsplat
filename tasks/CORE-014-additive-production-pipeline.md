# CORE-014 — Additive production pipeline (default-off)

## Context

The semantic, convergence, rate, and acceleration tasks deliberately produce separable parts. A
single production integration task is needed to bind their frozen outputs into one callable
pipeline without repeating the live diagnostic's mixed, per-block policies or changing the
maintained normalized default prematurely.

## Goal

A default-off, end-to-end Observation Field V2 profile under the sole conversion entry point,
using exactly the recipe, stopping rule, rate controller, codec, and accelerated path selected by
the preceding gates.

## Non-goals

- Re-running scientific selection inside integration or retaining every experimental knob.
- Changing the default profile; CORE-015 owns any promotion after BENCH-022.
- Per-block trial/rollback, mixed semantic containers, or implicit normalized/additive conversion.

## Pipeline contract

- One public configuration/profile owns preparation, selected initialization, bounded fitting,
  topology/rate allocation, field validation, the BENCH-025-selected direct or structured codec,
  encode/decode verification, and telemetry.
- Use stage-boundary finite checkpoints and one rare global safety fallback. Do not port the
  diagnostic schedule's repeated transactional block gate unless BENCH-021 explicitly selected it.
- Stop through the frozen convergence controller: full-evaluation distortion/downstream surrogate,
  marginal improvement per measured work/byte, patience, hard time/work limit, and best-checkpoint
  restoration.
- Emit a versioned Field V2 stream plus manifest; measure cold load/query using decoded state.

## Acceptance criteria

- [ ] `scripts/convert.py` exposes one explicit default-off profile and no second conversion CLI;
      programmatic API and CLI resolve to the same validated config digest.
- [ ] Every selected component is referenced to its deciding report/ADR. Rejected alternatives
      are absent from the production profile rather than hidden behind undocumented branches.
- [ ] Seeded end-to-end fixtures cover masks, alpha policy, crop/canvas transforms, empty/tiny
      inputs, interruption/resume at declared stage boundaries, codec verification, and fallback.
- [ ] Telemetry partitions preparation, initialization, fitting, allocation, evaluation,
      checkpointing, encode/decode, rejected work, renderer calls/pixels, complete bytes, memory,
      and downstream adapter time.
- [ ] Legacy normalized and native-additive profiles, output schemas, and defaults are unchanged;
      explicit compatibility adapters retain their semantic-exactness labels.
- [ ] A new ADR records the integrated but default-off profile and the exact evidence still needed
      for promotion.
- [ ] Focused integration tests, docs/task synchronization, and `./scripts/verify.sh` pass.

## Interfaces touched

`scripts/convert.py`, public pipeline/config API, selected fit/rate/codec modules, checkpoint and
telemetry schemas, integration tests, ADR/docs, `docs/additive_field_v2.md`, this task, and Index.

## Depends on

CORE-013, BENCH-021/025, COMP-013/014, FIT-030, PORT-006, ADR-0006

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`. A full self-review and
distinct correctness/performance review are required before BENCH-022.

## Notes

This task begins only if BENCH-020 selects compatible semantics and BENCH-021 freezes a recipe.
Otherwise it is marked abandoned with the deciding evidence rather than implemented speculatively.
COMP-014 satisfies this dependency with a reviewed terminal no-code disposition when BENCH-025
selects the direct codec; integration never waits for an unauthorized structured implementation.
