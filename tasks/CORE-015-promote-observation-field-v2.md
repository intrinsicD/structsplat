# CORE-015 — Promote Observation Field V2 conditionally

## Context

Changing a production default is a migration and compatibility decision, not another experiment.
It is justified only if BENCH-022 issues a positive verdict for one exact integrated profile.

## Goal

Promote the BENCH-022-approved Observation Field V2 profile through explicit defaults, migration
documentation, deprecation boundaries, and regression coverage—or close this task without code if
the promotion gate fails.

## Non-goals

- Altering the approved algorithm, retraining/tuning it, or weakening BENCH-022 guardrails.
- Deleting legacy readers/profiles or rewriting historical ADRs and report artifacts.
- Promoting optional learned, temporal, or rich-primitive branches.

## Acceptance criteria

- [ ] BENCH-022 contains an audited positive verdict and exact profile/codec/config digest; absent
      that verdict, this task is marked abandoned with no default mutation.
- [ ] One reviewed ADR records the default decision, evidence scope, supported hardware/runtime,
      fallback, legacy profile name, migration window, and reversal criteria.
- [ ] CLI/API defaults, examples, tests, package metadata, and user documentation change together;
      explicit legacy selection reproduces the prior default on seeded fixtures.
- [ ] Existing Field/SSPL1 readers remain compatible, Field V2 streams remain versioned, and no
      adapter upgrades its semantic-exactness claim during migration.
- [ ] Release notes state quality/rate/speed boundaries and known non-generalized evidence without
      turning task targets into measured claims.
- [ ] Full regression/performance gate, docs/task/ARA synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Default profile/config resolution, CLI/API docs and examples, compatibility tests, ADR/release
notes, `docs/architecture.md`, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

BENCH-022, CORE-014

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.

## Notes

This terminal task intentionally has no “try anyway” path. A failed or unavailable BENCH-022 gate
leaves the current default intact while preserving the opt-in research profile if useful.

