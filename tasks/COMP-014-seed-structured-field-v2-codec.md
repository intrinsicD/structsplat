# COMP-014 — Seed-structured Field V2 codec (conditional)

## Context

If BENCH-025 shows that direct-row coding misses a material, usable compression opportunity, the
decoder needs an explicit way to regenerate locally correlated Gaussian attributes from compact
seed state. SGI is the direct prior-art control. The representation must remain self-contained,
versioned, queryable after bounded cold expansion, and semantically identical to the BENCH-020
Field V2 equation.

## Goal

Conditional on BENCH-025 authorization, implement and confirm one frozen seed-local generator
grammar that beats COMP-013 at complete actual bytes without violating quality, downstream,
decode/query, memory, determinism, or compatibility guards. If the gate is negative, close this
task without implementation.

## Proposed grammar boundary

- spatial seeds/regions with deterministic coverage and ordering;
- a small, explicitly versioned generator that emits geometry and Field V2 attributes for a
  bounded number of local rows;
- quantized seed state, generator parameters or shared-model identity, residual attributes,
  alpha, indexes, entropy tables, and checksums counted in the stream/deployment ledger;
- decode to an ordinary validated `ObservationField2D` before the existing renderer/query API.

The first self-contained arm stores all required generator state per stream. A shared pretrained
generator is a separate deployment arm that counts model distribution, versioning, training, and
resident memory rather than treating the prior as free.

## Non-goals

- Changing additive/normalized semantics, hiding a neural network in every point query, or
  modifying COMP-013 streams in place.
- Implementing multiple generator families after outcomes are visible.
- Claiming novelty for seed-local generation, multiscale fitting, or context entropy coding.

## Acceptance criteria

- [ ] Work begins only from a positive BENCH-025 verdict naming one grammar and frozen boundaries;
      a negative/unavailable verdict yields a terminal no-code disposition.
- [ ] A versioned spec and fail-closed decoder define seeds, regions, generator, quantizers,
      residuals, ordering, entropy/index/alpha/header/checksum payloads, resource limits, and exact
      expanded Field V2 semantics.
- [ ] Tiny fixtures and differential tests prove decoded fields match the reference generator and
      Field V2 renderer/query equations; corrupt, unknown, oversized, nonfinite, and inconsistent
      streams fail before trusted output.
- [ ] Encode/decode and seed ordering replay deterministically. Cold expansion time, expanded
      resident memory, random-tile access, and steady query/render costs are measured separately.
- [ ] A preregistered RD confirmation compares COMP-014, direct COMP-013, native SGI where
      available, and conventional-codec context at equal complete bytes and declared work/training
      budgets on development and sealed data.
- [ ] Promotion requires the exact BENCH-025 byte advantage and all quality/downstream/decode/query/
      memory guardrails; otherwise CORE-014 uses direct COMP-013.
- [ ] Spec, security/correctness tests, portable report, independent audit, ARA disposition,
      docs/task synchronization, and `./scripts/verify.sh` pass.

## Interfaces touched

New conditional Field V2 stream/decoder modules, seed generator and fit/export path, COMP-013
dispatch/query API, tests/fuzz fixtures, benchmark/report tooling, ADR/docs,
`docs/additive_field_v2.md`, this task, CORE-014, and the Index.

## Depends on

BENCH-025, COMP-013, CORE-013, BENCH-020, COMP-008/009

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Method, codec/security, and results reviews are required before integration.

## Notes

The decoded object, not the seed syntax, remains the realtime-gs contract. This keeps downstream
semantics stable while allowing a structured cold-storage representation to earn its complexity.

