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

## A04: Default-off implicit pixel contraction control

- **Design**: Keep active pixel leaves procedural and outside torch autograd. Resolve a quadtree
  frontier into reusable float32 atom slots using moment parents, an optional parent-plus-one-detail
  basis, and pair actions needed for exact count. Export only the terminal direct-additive field
  through `ObservationField2D`; do not route the method through the current normalized pipeline or
  conversion CLI.
- **Status**: implemented reference substrate, default off and pending distinct numerical review;
  no selected semantics, matched-method result, actual-rate result, or production promotion.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `scripts/experiments/hier005_pixel_contraction.py`,
  `tests/test_pixel_contraction.py`,
  `docs/additive_field_v2.md`,
  `tasks/HIER-005-implicit-pixel-contraction.md`]
- **Dependencies**: [N218, N219, N220]
- **From staging**: O106

## A05: Separate contraction proposals from acceptance authority

- **Design**: Schedule the image-sized quadtree frontier with a cheap deterministic regional proxy;
  rank a bounded region's hard/detail/pair options with the exact continuous peak-one Gaussian
  product and eliminated additive coefficients; then re-solve coefficients and measure distortion
  on the true discrete finite-support patch. Commit only support-disjoint batches and invalidate a
  cached solve after any overlapping commit. Treat the configured row price as proposal telemetry;
  preserve complete cold-stream replay and stopping for COMP-013/FIT-030.
- **Status**: local analytic/discrete controller implemented and diagnostic driver cold-loads final
  fields; neighbor/global continuation, complete codec replay, matched controls, and formal
  selection remain open.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `tests/test_pixel_contraction.py`,
  `ara/evidence/hier005-pixel-contraction-diagnostic-2026-08-05/run.md`,
  `tasks/HIER-005-implicit-pixel-contraction.md`,
  `tasks/FIT-030-rate-aware-continuous-allocation.md`,
  `tasks/FIT-045-residual-budgeted-densification.md`]
- **Dependencies**: [N218, N219, N220, N221]
- **From staging**: O107

## A06: Selective recovery freezes the untouched pixel basis

- **Design**: Interleave bounded differentiable-renderer recovery checkpoints during implicit
  contraction, but optimize only active atom slots that have already been outputs of a merge,
  retained-detail, or teleportation action. Represent never-touched active pixel leaves as a
  detached fixed render; keep previously touched active rows eligible for later interaction repair;
  commit only the best strict masked-SSE improvement; and rebuild geometry-dependent proposals
  after an accepted checkpoint. Normalize attempted optimizer work by fractions of requested row
  reduction rather than topology-action count.
- **Status**: implemented default-off HIER-005 diagnostic path, pending distinct numerical review
  and matched multi-image evidence; no production fitter, semantic, or default selection.
- **Provenance**: user
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `scripts/experiments/hier005_pixel_contraction.py`, `tests/test_pixel_contraction.py`,
  `docs/architecture.md`, `docs/additive_field_v2.md`,
  `ara/evidence/hier005-selective-recovery-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N222, N223, N224]
- **From staging**: O109

## A07: All-active recovery uses mask-smoothed error exposure

- **Design**: Preserve touched-only recovery as the fixed-basis control and expose a separate
  comparison scope that trains every active Gaussian. Build a mask-aware smoothed RGB residual-MSE
  field, estimate each Gaussian's support-averaged exposure with additive-renderer color VJPs, and
  normalize, power-transform, and bound those row scores without materializing a dense
  pixel-by-Gaussian matrix. Multiply actual post-Adam row updates so adaptive moment normalization
  does not cancel the intended allocation, then retain the existing trust regions, best-SSE
  checkpoint, rollback, and frontier-rebuild safeguards.
- **Status**: implemented default-off HIER-005 diagnostic scope, retained as experimental after a
  mixed one-image budget curve; pending distinct numerical review and matched multi-image evidence.
- **Provenance**: user
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `scripts/experiments/hier005_pixel_contraction.py`, `tests/test_pixel_contraction.py`,
  `docs/architecture.md`, `docs/additive_field_v2.md`,
  `ara/evidence/hier005-all-active-error-weighted-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N225, N226]
- **From staging**: O112

## A08: Artifact-safe contraction branches only after topology is stable

- **Design**: Treat localized artifact absence as a fail-closed feasibility boundary rather than
  selecting by average SSE alone. Compare refitted support semantics first; preserve touched-only
  interleaving as the topology-stable control; apply all-active geometry only as a terminal branch;
  evaluate exact displayed pixel and multiscale patch maxima; and, only after a fixed-count
  failure, fork the persisted field into bounded local repair arms. Current repair selects stable
  residual peaks, freezes the full base and appended geometry, optimizes signed rescue RGB only,
  and keeps the unchanged base or a candidate by worst-local-violation then SSE. If no branch
  passes, report the requested count as infeasible; local uncontraction/preserved pixel leaves
  remain the stronger unimplemented fallback.
- **Status**: the support/recovery factorial, localized gate, and RGB-only rescue are implemented as
  default-off HIER-005 diagnostics. On the exposed Janelle raster, three touched 8k rows pass while
  every 4k and repair row fails. Threshold transfer, stronger repair, held-out validation,
  complete bytes, and distinct numerical/scientific review remain open.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `scripts/experiments/hier005_pixel_contraction.py`,
  `scripts/experiments/hier005_artifact_repair.py`, `tests/test_pixel_contraction.py`,
  `docs/architecture.md`, `docs/additive_field_v2.md`,
  `tasks/HIER-005-implicit-pixel-contraction.md`,
  `ara/evidence/hier005-artifact-safety-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N227, N228, N229, N230, N231, N232, N234, N235]
- **From staging**: O115

## A09: Parent-preserving residual quadtree is a default-off negative control

- **Design**: Start with every mask-present cell at a declared coarse quadtree level. Derive each
  peak-one Gaussian from active-pixel moments plus fixed leaf variance. Rank frontier parents by
  mask-aware smoothed residual energy per appended row; retain a selected parent, append all
  mask-present signed residual children, and optimize only the new RGB block against a detached
  prefix. Compare the unchanged prefix and checkpoints by local-artifact violation then SSE, cold-
  render the joint field before commit, and preserve every earlier array bit-exact so each accepted
  count is independently renderable. Charge retained ancestors in full field counts/bytes and
  label tree/coefficients-only numbers as non-codec proxies.
- **Status**: implemented as default-off HIER-006 and retained as a negative control. The corrected
  exposed C0001 3,986/8,192 prefixes fail the 0.02/0.01 displayed gate at 0.2223/0.0860 and
  0.1073/0.0375; 5,106 terminal rows are ancestors. No current-pipeline, semantic, codec, or
  default change; distinct numerical review and any successor experiment remain open.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/progressive_residual_quadtree.py`,
  `scripts/experiments/hier006_progressive_residual_quadtree.py`,
  `tests/test_progressive_residual_quadtree.py`,
  `docs/architecture.md`, `docs/additive_field_v2.md`,
  `tasks/HIER-006-progressive-residual-quadtree.md`,
  `ara/evidence/hier006-progressive-residual-quadtree-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N228, N236, N237, N238]
- **From staging**: O117

## A10: Parent-replacing frontier reconciliation is an isolated negative diagnostic

- **Design**: Keep the quadtree as a scheduler and spatial address structure rather than an
  additive ancestor stack. Splitting deactivates the selected active parent and activates every
  mask-present child, preserving an antichain partition. Hold geometry fixed; compare smoothed-
  energy versus worst-pixel/patch-first selection and new-child-only versus finite-support-overlap
  RGB reconciliation from one hash-identical base. Weight post-Adam RGB updates by mask-aware
  smoothed residual exposure, but let a cold full render with the local-artifact/SSE key own the
  transaction, deterministic batch backoff, and exact rollback. Charge inactive nodes and revised
  coefficients in separate structural ledgers and never label them complete codec bytes.
- **Status**: implemented as default-off HIER-007. Parent replacement with energy/new-only is a
  useful structural control, but every exposed 8k arm fails the diagnostic artifact gate and the
  combined artifact-first/overlap arm is rejected for severe quadtree-aligned artifacts. No
  production, renderer, semantic, codec, or default promotion; distinct review and disjoint
  evidence remain pending.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/artifact_first_quadtree.py`,
  `scripts/experiments/hier007_artifact_first_quadtree.py`,
  `tests/test_artifact_first_quadtree.py`, `docs/architecture.md`,
  `docs/additive_field_v2.md`, `tasks/HIER-007-artifact-first-frontier-quadtree.md`,
  `ara/evidence/hier007-artifact-first-frontier-quadtree-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N228, N238, N239, N240, N241]
- **From staging**: O120

## A11: Fixed-lattice feature-aware elimination is a negative diagnostic architecture

- **Design**: Begin with a mask-aware exact color prefit on an implicit pixel-centered Gaussian
  lattice, then compare expanding quadtree contraction against an exactly nested survivor sequence.
  The tested survivor scheduler combines structure-tensor feature importance, density-aware WSE
  competition, bilateral feature barriers, and a same-side local static Schur price. Materialize
  common Field V2 outputs and apply one all-row optimizer whose smoothed residual, feature weighting,
  geometry/RGB trust regions, and raw-local-error rollback are identical across schedulers.
- **Status**: implemented as default-off HIER-008 and retained only as a negative diagnostic for
  fixed-scale/static-price WSE elimination. Meaningful overlap is a useful positive component for
  expanding contraction, but all exposed cells fail the artifact gate and fixed-scale WSE leaves
  visible coverage holes. Dynamic covariance/merge support, local appearance variable projection,
  and actual patch-error removal commits were not implemented and remain a distinct hypothesis.
- **Provenance**: user-revised
- **Crystallized via**: verbal-affirmation
- **Evidence**: [`src/structsplat/overlap_elimination.py`,
  `scripts/experiments/hier008_overlap_elimination.py`, `tests/test_overlap_elimination.py`,
  `tasks/HIER-008-overlap-lattice-feature-elimination.md`,
  `ara/evidence/hier008-overlap-lattice-feature-elimination-janelle-diagnostic-2026-08-05/run.md`]
- **Dependencies**: [N242, N243, N244, N245]
- **From staging**: O123

## A12: Dynamic overlap contraction exposes bounded direct-neighbor recovery

- **Design**: Start from either the near-delta pixel endpoint or an exactly guarded overlapping
  pixel lattice. Reuse HIER-005's current-field local proposal, discrete coefficient projection,
  support-disjoint commit, recovery, and proposal-rebuild transaction. A
  `touched_neighborhood` checkpoint optimizes topology-touched active rows plus active rounded
  centers in the direct Chebyshev-radius-one halo of newly touched rows; only accepted changed
  neighbors persist, and all other rows remain a detached fixed base. A deterministic feature
  reserve may carry up to two exact protected leaf geometries through a region, refit their RGB,
  and fail locally overfull regions closed.
- **Status**: implemented as default-off HIER-009 and retained as a mixed diagnostic architecture.
  The halo visibly removes low-count block structure and improves the 4k overlap arm, but
  redistributes 8k error; protection helps patch error, yet every overlap cell fails the local
  gate. Delta/touched 8k remains the fallback. No semantic, production, codec, convergence-speed,
  or general artifact-freedom promotion is authorized.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/structsplat/pixel_contraction.py`,
  `src/structsplat/overlap_elimination.py`,
  `scripts/experiments/hier009_dynamic_overlap_recovery.py`,
  `tests/test_dynamic_overlap_recovery.py`,
  `tasks/HIER-009-dynamic-overlap-neighborhood-recovery.md`,
  `ara/evidence/hier009-dynamic-overlap-neighborhood-recovery-janelle-diagnostic-2026-08-06/run.md`]
- **Dependencies**: [N224, N228, N244, N245, N246, N247, N248]
- **From staging**: O126
