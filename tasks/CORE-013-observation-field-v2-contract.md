# CORE-013 — Observation Field V2 semantic contract

## Context

ADR-0006 already authorizes an opt-in additive renderer, but the repository has no typed boundary
that distinguishes authoritative additive RGB coefficients from optional structural mass and
packed alpha. The current `GaussianField` and SSPL1 interfaces are normalized-history objects and
must not be silently reinterpreted. BENCH-019 decides which structural quantities need to survive
the Stage-1 boundary.

## Goal

A default-off, versioned, lossless `ObservationField2D` reference contract implementing the schema,
equations, validation, coordinate transforms, alpha binding, and explicit adapters described in
`docs/additive_field_v2.md`, without changing current pipeline behavior or compressed-codec policy.

## Non-goals

- Choosing additive versus normalized production semantics; BENCH-020 owns that decision.
- Optimizing the field, quantizing it, adding an entropy coder, or changing `scripts/convert.py`.
- Deriving `color = rgb_coeff / mass` as an authoritative value.
- Treating legacy normalized-to-additive array conversion as semantic equivalence.

## Acceptance criteria

- [ ] A typed field stores geometry, authoritative `rgb_coeff`, optional independently named
      `structural_mass`, optional counted `background_rgb`, explicit coefficient domain, packed
      alpha, canvas/crop transform, optional camera metadata, renderer equation, support/filter/
      matting semantics, and a schema version with strict shape/range validation.
- [ ] CPU reference appearance and density queries implement the equations in the architecture
      document; tiny closed-form fixtures cover no rows, one row, overlaps, off-canvas support,
      masks, and zero/near-zero structural density.
- [ ] Signed/nonnegative coefficients, zero/DC background, alpha matting, and display clipping are
      distinct declared semantics with fixtures; no clamp or matte silently changes the oracle.
- [ ] Lossless save/load and canonical hashing preserve every tensor and semantic field exactly;
      malformed, ambiguous, unknown-version, nonfinite, and metadata-mismatched objects fail closed.
- [ ] Adapters for direct additive, current factorized additive, and normalized fields declare
      `pixel_exact`, `component_semantics_exact`, and any required assumptions. Tests prevent a
      normalized adapter from claiming additive semantic exactness.
- [ ] Current `GaussianField`, NPZ, SSPL1, renderer defaults, CLI outputs, and import-without-torch
      NumPy modules remain unchanged on seeded regression tests.
- [ ] The format is extensible without pre-allocating a mixed-atom or neural-decoder grammar.
- [ ] Documentation, task state, and any new ADR references are synchronized.
- [ ] `./scripts/verify.sh` passes.

## Interfaces touched

New typed module under `src/structsplat/`, additive/density CPU oracle hooks, lossless reference IO,
tests, `docs/additive_field_v2.md`, `docs/architecture.md`, this task, and the Index. COMP-013 owns
the compressed stream.

## Depends on

BENCH-019, CORE-001/002, COMP-002, ADR-0006

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using `tasks/README.md`.

## Notes

If BENCH-019 finds no downstream need for structural mass, the field keeps the slot optional and
the production candidate may select the eight-parameter direct appearance form. A scientific or
architectural reviewer distinct from the Driver is required before this contract is accepted.
