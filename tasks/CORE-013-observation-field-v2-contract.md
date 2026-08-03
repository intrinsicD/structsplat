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

- [x] A typed field stores geometry, authoritative `rgb_coeff`, optional independently named
      `structural_mass`, optional counted `background_rgb`, explicit coefficient domain, packed
      alpha, canvas/crop transform, optional camera metadata, renderer equation, support/filter/
      matting semantics, and a schema version with strict shape/range validation.
- [x] CPU reference appearance and density queries implement the equations in the architecture
      document; tiny closed-form fixtures cover no rows, one row, overlaps, off-canvas support,
      masks, and zero/near-zero structural density.
- [x] Signed/nonnegative coefficients, zero/DC background, alpha matting, and display clipping are
      distinct declared semantics with fixtures; no clamp or matte silently changes the oracle.
- [x] Lossless save/load and canonical hashing preserve every tensor and semantic field exactly;
      malformed, ambiguous, unknown-version, nonfinite, and metadata-mismatched objects fail closed.
- [x] Adapters for direct additive, current factorized additive, and normalized fields declare
      `pixel_exact`, `component_semantics_exact`, and any required assumptions. Tests prevent a
      normalized adapter from claiming additive semantic exactness.
- [x] Current `GaussianField`, NPZ, SSPL1, renderer defaults, CLI outputs, and import-without-torch
      NumPy modules remain unchanged on seeded regression tests.
- [x] The format is extensible without pre-allocating a mixed-atom or neural-decoder grammar.
- [x] Documentation, task state, and any new ADR references are synchronized.
- [x] `./scripts/verify.sh` passes.

## Interfaces touched

New typed module under `src/structsplat/`, additive/density CPU oracle hooks, lossless reference IO,
tests, `docs/additive_field_v2.md`, `docs/architecture.md`, this task, and the Index. COMP-013 owns
the compressed stream.

## Depends on

BENCH-019, CORE-001/002, COMP-002, ADR-0006

## Agent workflow

- Driver: codex
- Reviewer: pending-distinct
- Turn: reviewer
- Reviewed revision: tree `1ec324555b3779ee94e571637de65b1bbb7d224e`

### Handoff log

### Handoff

#### Objective
Implement the neutral Field V2 semantic boundary needed to compare additive and normalized
families without reinterpreting the incumbent field, selecting a production semantic, or changing
the maintained conversion path.

#### Changes
Added the NumPy-only schema-`2.0.0` `ObservationField2D`, explicit canvas/crop, RS geometry,
coefficient/mass/background/alpha/camera semantics, peak-one appearance and independent-density
oracles, explicit raw/matted/display operations, strict deterministic lossless reference NPZ and
canonical hashing, and direct/factorized/normalized adapter declarations. Synchronized the Field
V2 design and architecture module map; no CLI, legacy field, renderer, or codec code changed.

#### Evidence
Twenty-six new closed-form/adversarial tests pass. The 104-test field/render/codec/CLI regression
slice passes. A 100-seed randomized factorized-additive parity sweep covered off-canvas means,
opacity, covariance filtering, AA dilation, rounded support boundaries, and three fade settings;
its worst absolute NumPy-versus-torch difference was `1.0925e-6`, within a `1e-6`
absolute-plus-relative diagnostic comparison. `./scripts/verify.sh` passed with 1,546 tests, 4 skips, and all docs,
ARA, task, script-layout, and agent-workflow checks clean. The reviewed implementation tree is
`1ec324555b3779ee94e571637de65b1bbb7d224e`.

#### Assumptions
`pixel_exact` means the same bound mathematical pixel equation and support/filter conventions,
not bit-identical accumulation across NumPy float64 evaluation and torch float32 atomics. The
factorized adapter applies only to constant colors, folds sigmoid opacity into authoritative RGB,
and deliberately does not create structural mass. The reference NPZ is lossless interchange, not
the COMP-013 compressed stream or actual-rate evidence.

#### Uncertainties
No distinct reviewer is available in this driver turn. BENCH-019 still lacks a complete matched
multi-capture downstream result, so the mass slot remains optional and no semantic winner is
selected. `hard_contained` is a recorded producer boundary policy; this contract validates its
alpha binding but does not rerun the producer's geometry-containment certificate on load.

#### Review focus
Audit malformed-container fail-closed behavior, canonical-hash completeness, covariance-AABB
parity at floating-point radius boundaries, raw/matted/DC ordering, the normalized adapter's
structural inability to claim exactness, and whether producer-asserted hard containment needs a
separate certificate before BENCH-020.

#### Protected actions not taken
No realtime-gs files or active conversion outputs were modified. No semantic/default/codec choice,
formal experiment, external write, commit push, or independent-review claim was made. Unrelated
IDE changes remain untouched.

#### Recommended next action
A distinct scientific/architectural reviewer should reproduce the focused parity and malformed
fixtures against the recorded tree. After any required revision, BENCH-020 may freeze the semantic
factorial; it must not treat this neutral contract as evidence that additive rendering wins.

## Notes

If BENCH-019 finds no downstream need for structural mass, the field keeps the slot optional and
the production candidate may select the eight-parameter direct appearance form. A scientific or
architectural reviewer distinct from the Driver is required before this contract is accepted.
