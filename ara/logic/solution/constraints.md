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

## K04: Downstream rows require paired provenance receipts

- **Constraint**: BENCH-019 aggregation accepts a cell only with its paired exporter receipt and
  revalidates the exact protocol, family-invariant factor, source manifest, metric-source files,
  canonical row, and six cell artifacts. Row-schema validity alone cannot establish that metrics
  came from the frozen raw sources.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: BENCH-019 realtime-gs export, assembly, protocol review, and results audit.
- **Evidence**: [`realtime-gs@d3e76fe:src/rtgs/bench019.py`,
  `realtime-gs@d3e76fe:tests/test_bench019_exporter.py`,
  `tasks/BENCH-019-stage1-downstream-objective.md`]
- **From staging**: O103

## K05: Original-file compression requires both exact file operands

- **Constraint**: Report complete cold-package bpp as the primary rate. Compute original-file
  compression ratio only as exact supplied-source bytes divided by the complete self-contained
  package, with exact-source, canonical lossless PNG, and raw-RGB denominators kept separate. Count
  every per-image header, index, codebook, mask, quantizer parameter, context/generator state, and
  side stream; shared deployed models require a versioned installation-size and amortization ledger.
  Gaussian count, parameter BPP, theoretical entropy, or raw-RGB reduction cannot substitute for
  this measurement.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: BENCH-005/007/025, COMP-013/014, FIT-030, FF-003, and future original-file
  storage reports.
- **Evidence**: [`ara/evidence/2d-gaussian-image-fields-sota-review-2026-08-04.md`,
  `tasks/DOCS-007-2d-gaussian-sota-review.md`]
- **From staging**: O104

## K06: CUDA recovery outcomes require repeat-aware evidence

- **Constraint**: Do not treat a single CUDA additive-recovery field hash or small metric delta as
  deterministic. Atomic-gradient accumulation can change optimizer trajectories even when topology
  and inputs are fixed. Formal small-effect comparisons must use deterministic CPU recovery or a
  predeclared repeated-CUDA design that reports outcome dispersion; timing and exact replay claims
  remain separate.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: HIER-005 and any later FIT-045/FIT-030 comparison that reuses selective recovery.
- **Evidence**: [`ara/evidence/hier005-selective-recovery-janelle-diagnostic-2026-08-05/run.md`,
  `src/structsplat/pixel_contraction.py`,
  `scripts/experiments/hier005_pixel_contraction.py`,
  `tasks/HIER-005-implicit-pixel-contraction.md`]
- **From staging**: O110

## K07: Recovery rollback is local to the current topology path

- **Constraint**: A strict masked-SSE improvement at an interleaved optimizer checkpoint protects
  the current field only. When accepted means, scales, or rotations rebuild the contraction
  frontier, later topology proposals can differ, so checkpoint-local rollback cannot establish
  terminal dominance over touched-only, fixed-geometry, or no-recovery trajectories. Compare final
  paths explicitly and isolate terminal polish from topology-interleaved recovery when attribution
  matters.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: HIER-005 and later FIT-045/FIT-030 topology-plus-recovery comparisons.
- **Evidence**: [`ara/evidence/hier005-all-active-error-weighted-janelle-diagnostic-2026-08-05/run.md`,
  `src/structsplat/pixel_contraction.py`, `tasks/HIER-005-implicit-pixel-contraction.md`]
- **From staging**: O113

## K08: Cross-kernel checkpoint ties require one domain or a precision band

- **Constraint**: A lexicographic local-artifact/SSE acceptance key cannot treat last-bit outputs
  from different reduction precisions or kernels as strict mathematical order. Evaluate competing
  keys in one numeric domain and define a tolerance derived from the maintained renderer/reduction
  precision before applying the SSE tie-break. The tolerance must remain far below one displayed
  quantization step and cannot excuse a material gate regression.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: HIER-006 and future FIT-045/FIT-049 local-constraint or transactional topology
  controllers that compare cached, optimizer, and cold-rendered reductions.
- **Evidence**: [`src/structsplat/progressive_residual_quadtree.py`,
  `tests/test_progressive_residual_quadtree.py`,
  `ara/evidence/hier006-progressive-residual-quadtree-janelle-diagnostic-2026-08-05/run.md`,
  `tasks/HIER-006-progressive-residual-quadtree.md`]
- **From staging**: O118

## K09: Meaningful lattice overlap requires an exact guarded endpoint

- **Constraint**: Do not infer useful direct-neighborhood overlap from support enumeration alone.
  Before contracting a broader pixel-centered Gaussian lattice, solve mask-present RGB coefficients
  against the actual sampled renderer and require cold full-lattice parity, finite bounded
  coefficients, stable convergence, and explicit ringing, cutoff, and mask-boundary checks. Direct
  source RGB is not a valid endpoint when neighboring kernels have material response.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Applies to**: HIER-008 and future FIT-045/HIER overlap-prefilter or lattice-contraction paths.
- **Evidence**: [`src/structsplat/overlap_elimination.py`,
  `tests/test_overlap_elimination.py`,
  `ara/evidence/hier008-overlap-lattice-feature-elimination-janelle-diagnostic-2026-08-05/run.md`,
  `tasks/HIER-008-overlap-lattice-feature-elimination.md`]
- **From staging**: O122

## K10: Elimination support must follow survivor spacing and actual patch distortion

- **Constraint**: A method that deletes lattice centers must adapt effective covariance to the
  surviving local spacing or prove, through an exact coverage veto, that fixed support remains
  reconstructive. Feature retention, WSE crowding, and static local Schur prices are proposal
  signals only; acceptance must evaluate post-refit pixel/patch distortion, including holes between
  surviving centers, before committing a removal.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Applies to**: HIER-008 and future FIT-045/HIER elimination, merge, or structured-subsampling
  methods.
- **Evidence**: [`src/structsplat/overlap_elimination.py`,
  `tests/test_overlap_elimination.py`,
  `ara/evidence/hier008-overlap-lattice-feature-elimination-janelle-diagnostic-2026-08-05/run.md`,
  `tasks/HIER-008-overlap-lattice-feature-elimination.md`]
- **From staging**: O125
