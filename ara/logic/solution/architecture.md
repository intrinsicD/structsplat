# Architectures

## A01: Evidence-gated Observation Field V2 boundary

- **Design**: Keep the current normalized `GaussianField` authoritative while evaluating a new,
  default-off `ObservationField2D` boundary. The candidate stores geometry, authoritative additive
  RGB coefficients, exact/declared alpha semantics, explicit coefficient and DC/background domains,
  and optional independently supervised nonnegative structural mass. Direct additive, dual
  additive, and normalized semantics remain matched controls until BENCH-020 selects one.
- **Status**: proposed; no implementation, semantic selection, or default promotion yet.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`docs/additive_field_v2.md`, `tasks/CORE-013-observation-field-v2-contract.md`,
  `tasks/BENCH-019-stage1-downstream-objective.md`,
  `tasks/BENCH-020-field-semantics-factorial.md`]
- **Dependencies**: [N207, N208, N209, N210]
- **From staging**: O97

## A02: Direct codec before conditional decoded structure

- **Design**: Establish a complete-byte direct Field V2 codec first. Then compare it with native
  SGI and a bounded seed-local generator oracle. A separately versioned seed-structured grammar is
  implemented only if BENCH-025 demonstrates a complete-byte advantage within cold-decode,
  expanded-memory, random-access, query, quality, and downstream guards; both grammars decode to
  the same semantic field boundary.
- **Status**: proposed conditional branch; COMP-014 closes without code on a negative or unavailable
  BENCH-025 verdict.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`docs/additive_field_v2.md`, `tasks/COMP-013-observation-field-v2-codec.md`,
  `tasks/BENCH-025-structured-codec-necessity.md`,
  `tasks/COMP-014-seed-structured-field-v2-codec.md`, `tasks/COMP-008-mean-conditioned-entropy-oracle.md`,
  `tasks/COMP-009-ssp2e-actual-coder.md`]
- **Dependencies**: [N209, N210]
- **From staging**: O101

## A03: Appearance domain and DC are semantic axes

- **Design**: Field V2 declares coefficient domain, optional counted alpha-gated DC/background,
  authoritative pre-clamp rendering, matting, and evaluation clipping. BENCH-020 eliminates and
  confirms zero-DC/nonnegative versus DC-plus-bounded-or-signed-residual candidates before solver,
  loss, codec, or pipeline promotion; downstream components inherit the selected domain exactly.
- **Status**: proposed semantic sub-gate; no coefficient/DC winner is selected.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`docs/additive_field_v2.md`, `tasks/BENCH-020-field-semantics-factorial.md`,
  `tasks/CORE-013-observation-field-v2-contract.md`,
  `tasks/FIT-046-additive-variable-projection.md`,
  `tasks/COMP-013-observation-field-v2-codec.md`]
- **Dependencies**: [N209, N210]
- **From staging**: O102

