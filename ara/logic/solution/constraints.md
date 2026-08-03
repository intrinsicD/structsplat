# Constraints

## K01: Boundary controls must align loss, gate, and profile semantics

- **Constraint**: A boundary/no-boundary comparison is decision-valid only when its target/matting,
  pixel weights, containment, outside/coverage gates, manifest, and evaluation policy agree with
  the declared arm. The current custom no-boundary endpoint remains diagnostic and cannot select a
  renderer, boundary policy, loss, stage order, or default.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: BENCH-019/020/022 and any reused frame_00008 fields.
- **Evidence**: [`ara/evidence/frame00008-three-arm-audit-2026-08-03/run.md`,
  `tasks/BENCH-020-field-semantics-factorial.md`, `docs/additive_field_v2.md`]
- **From staging**: O95

## K02: Promotion requires downstream utility and complete cold rate

- **Constraint**: Stage-1 selection reports image fidelity and fixed-protocol realtime-gs utility.
  Compression selection uses complete cold-decoded bytes including alpha, headers, indexes,
  codebooks/generator state, and side streams, with active-crop and full-canvas bpp plus load,
  expansion, random-access, and query cost. Rows, analytical bits, or tensor-body bytes cannot
  substitute for the production rate.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: BENCH-019/020/022/023/024/025, COMP-013/014, FIT-030, CORE-014/015.
- **Evidence**: [`docs/additive_field_v2.md`, `tasks/BENCH-019-stage1-downstream-objective.md`,
  `tasks/COMP-013-observation-field-v2-codec.md`,
  `tasks/BENCH-022-additive-production-confirmation.md`]
- **From staging**: O98

## K03: Current donor mechanisms are controls, not StructSplat novelty

- **Constraint**: Treat regional densification/merge, seed-structured/context coding,
  Predict--Optimize--Distil, frequency-bearing atoms, and fused/per-Gaussian optimization as
  prior-art-controlled transfers associated respectively with LocoADC, SGI, AIR-family work,
  WIPES, and Faster-GS. Separate native evidence from local adaptations and make no generic novelty
  claim for those mechanisms.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: FIT-045, COMP-014, FF-002/003, CORE-008, PORT-006, and related reports/ADRs.
- **Evidence**: [`docs/additive_field_v2.md`,
  `tasks/FIT-045-residual-budgeted-densification.md`,
  `tasks/COMP-014-seed-structured-field-v2-codec.md`,
  `tasks/CORE-008-hybrid-edge-primitives.md`]
- **From staging**: O100

