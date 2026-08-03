# Heuristics

## H01: Matched Cross-Repo Evaluation Protocol
- **Rationale**: Compare image set, resolution, Gaussian budget, iteration or stopping mode, metrics, and seed under one protocol, and report representation quality separately from codec bpp when repos define storage differently.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/coco_fit_compare.py`, `results/coco_fit_compare/summary.md`, `results/coco_fit_compare/delta_after_update.md`]
- **From staging**: O03

## H02: Feature-Adaptive Scale Caps as Search Baseline
- **Rationale**: Use the feature-adaptive `feature12` scale cap as the stage-search baseline after the held-out capped benchmark improved mean PSNR, suppressed final max/p95 scale, won most images, and reduced runtime versus uncapped initialization.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`src/structsplat/config.py`, `src/structsplat/init.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`]
- **From staging**: O08

## H03: Tensor-Aware Residual Densification Candidate
- **Rationale**: Add residual Gaussians with local edge-tangent orientation, anisotropic scales, inherited scale caps, and renderer-dependent target/residual colors so capped initialization can be paired with searchable adaptive density control.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/structsplat/fit.py`, `benchmarks/stage_search.py`, `tests/test_fit_dynamics.py`]
- **From staging**: O10

## H04: Complete-Pair Familywise Dominance Audit
- **Rationale**: A strict multi-objective dominance statement must use identical paired cells for every core metric, orient all gains consistently, cluster repeated seeds/budgets within source image, exclude over-budget arms, and control familywise error across metrics. Marginal 95% intervals remain useful per-metric diagnostics but do not independently establish joint 95% dominance.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/fair_density_control_compare.py`, `benchmarks/native_reference_compare.py`, `tests/test_fair_density_control_compare.py`, `tests/test_native_reference_compare.py`]
- **From staging**: O34

## H05: Isolated Native-Reference Provenance Contract
- **Rationale**: Run incompatible external repositories in isolated environments; bind every row
  to repository, dependency, Python-source, binary, device, adapter, and target-pixel hashes;
  distinguish algorithm profiles from official-environment/native-authentic protocols; export
  float reconstructions for shared final metrics; and keep analytical rate, actual bitstreams,
  native trajectories, and non-comparable timing protocols separate. This prevents editable-package
  contamination, stale cache reuse, false pixel pairing, and paper/native claims from proxy rows.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/native_image_gs_compare.py`, `benchmarks/native_runners/image_gs.py`, `benchmarks/native_reference_compare.py`, `tests/test_native_image_gs_compare.py`, `tasks/BENCH-005-native-reference-pipelines.md`]
- **From staging**: O38

## H06: Preserve the Final-Count Invariant When Selecting Checkpoints
- **Rationale**: Score checkpoint candidates only after step/count/color transitions, retain the
  best state separately per Gaussian count, always include the terminal state, and restore only
  a state whose count equals terminal N. This prevents a pre-step history sample or an earlier
  lower-capacity state from masquerading as an equal-budget endpoint.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/structsplat/config.py`, `src/structsplat/fit.py`, `benchmarks/fair_density_control_compare.py`, `tasks/FIT-015-full-count-checkpoint-selection.md`]
- **From staging**: O40

## H07: Use Same-Trajectory Audits for Nondeterministic CUDA Policy Attribution
- **Rationale**: If independently rerun atomic-CUDA trajectories diverge before a candidate
  policy activates, their endpoint delta is not causal evidence for that policy. Compare valid
  states within one trajectory or branch both policies from an identical field and optimizer
  checkpoint before attributing the difference.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/structsplat/fit.py`, `benchmarks/fair_density_control_compare.py`, `tasks/FIT-015-full-count-checkpoint-selection.md`]
- **From staging**: O41

## H08: Separate Storage Semantics and Align Convergence Endpoints
- **Rationale**: Equal-storage comparisons are only auditable when analytical representation
  payload, actual codec stream, source/target/reconstruction containers, and decoded memory have
  distinct fields. A convergence curve must end at the reconstruction that is actually scored
  (including checkpoint restoration or final solves), use an explicit hold-last rule for early
  exits, and expose capacity mismatches and max-horizon censoring rather than silently ranking
  them as equal-rate converged cells.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/storage_budget.py`, `benchmarks/storage_budget_compare.py`, `benchmarks/fair_density_control_compare.py`, `benchmarks/results_index.py`, `tasks/BENCH-006-fixed-storage-convergence.md`]
- **From staging**: O48

## H09: Apply Representation-Preserving Orders as One Late Tuple Permutation
- **Rationale**: When ordering should change only prefix semantics, first finish any
  row-index-sensitive geometry transform such as alternating flanking, compute the ordering while
  the original sampling metric is still available, materialize colors/scales/caps/opacities in the
  legacy row association, and then apply one permutation to every row-aligned attribute. This
  preserves the complete represented tuple set and prevents per-leaf or pre-transform ordering
  from silently changing geometry.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/structsplat/init.py`, `src/structsplat/sampling.py`, `tests/test_init_stages.py`, `tests/test_sampling.py`]
- **From staging**: O51

## H10: Make Mechanism Guards Deterministic and Source-Bound

- **Rationale**: A CPU device label does not guarantee repeatable optimization trajectories;
  parallel reduction order produced small FIT-018 drift before the final guard. Pin the relevant
  thread count and deterministic-algorithm mode, hash every input plus tracked and untracked
  source file that defines the intervention, record the dirty snapshot and environment, and replay
  the frozen run before treating small deltas as evidence. Keep timing comparisons separate from
  exact non-timing replay checks.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/responsibility_split_compare.py`,
  `tests/test_responsibility_split_compare.py`,
  `ara/evidence/fit018-responsibility-guard-2026-07-15/config.json`,
  `ara/evidence/fit018-responsibility-guard-2026-07-15/rerun_aggregate.json`]
- **From staging**: O54

## H11: Certify Exact Equivalence Before Grouping Allocator State

- **Rationale**: A group-level allocator is representation-invariant only when every asserted
  member has identical geometry/color/filter attributes and compatible background/detail role,
  and the renderer-relevant weight differs solely by fractions that sum to the original. Aggregate
  additive sufficient statistics such as responsibility mass/error before applying a nonlinear
  exponent or top-k. Keep group IDs outside production state until approximate families show
  separate utility; ordinary split lineage changes geometry and is not an exact gauge class.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/gauge_equivalence_audit.py`,
  `tests/test_gauge_equivalence_audit.py`,
  `tasks/FIT-019-opacity-gauge-equivalence.md`,
  `docs/adr/0014-keep-opacity-gauge-groups-benchmark-only.md`]
- **From staging**: O59

## H12: Recover Output-Only Benchmark Failures From Frozen Measurements

- **Rationale**: If every expensive measurement is already present in an immutable journal/row
  table and failure occurs only while serializing a heterogeneous derived table, do not recompute
  or silently overwrite the run. Aggregate with the captured source snapshot, assert exact equality
  between frozen and repaired aggregation, repair only the serializer using a union schema, record
  the finalization hashes, then run a measurement-equivalent replay. Replay reports must name every
  excluded timing/provenance field and expected source difference rather than claim literal
  untouched-source identity.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/perturb_recover_spectroscopy.py`,
  `tests/test_perturb_recover_spectroscopy.py`,
  `results/fit020_response_spectroscopy_v1_2026-07-15/finalization.json`,
  `results/fit020_response_spectroscopy_v1_2026-07-15/replay_comparison.json`,
  `tasks/FIT-020-perturb-recover-spectroscopy.md`]
- **From staging**: O63

## H13: Select Context-Dependent Stream Edits by Integer Cap, Not Local Byte Ratio

- **Rationale**: A heterogeneous edit can change Morton order, dynamic ranges, headers, and the
  compression context of every attribute stream. Its independently encoded complete-container
  delta may be negative or non-monotone, so distortion divided by delta bytes is undefined or
  misleading and deltas cannot be added into a sequential budget. Persist and cold-validate every
  self-contained counterfactual, count all syntax/side streams, and select feasible candidates by
  integer cap plus stable Pareto/tie order. Treat the result as an operational oracle until a
  separately validated local price model reproduces it.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/marginal_cold_stream_rd.py`,
  `tests/test_marginal_cold_stream_rd.py`,
  `tasks/COMP-006-marginal-cold-stream-rd.md`,
  `docs/adr/0016-keep-marginal-cold-stream-birth-benchmark-only.md`]
- **From staging**: O66

## H14: Recompute Exact Worker Capabilities From Sealed Authorities

- **Rationale**: A valid sandbox attestation and a self-consistent persisted request prove only
  what capability was granted, not that the operation needed exactly that capability. For every
  protected-stage worker, derive the permitted command, config identity, source, environment, cwd,
  read-only/read-write inventories, launcher, denied probes, and timeout from sealed upstream and
  artifact authorities; require exact equality and reject coherently resealed extras. Keep this
  check alongside OS isolation, regenerated replay, immutable publication, and terminal-journal
  enforcement rather than relying on worker-reported no-access flags.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/ssp2v_actual_run.py`, `benchmarks/ssp2v_landlock.py`,
  `tests/test_ssp2v_actual_run_preflight_policy.py`]
- **From staging**: O80

## H15: Isolate Field Semantics Before Loss and Stage Sweeps

- **Rationale**: When compositor, parameterization, mask/matting, containment, initialization,
  topology, and commit policy change together, endpoint quality cannot identify the value of a loss
  or stage order. First compare native additive, direct/dual additive plain fit, normalized plain
  fit, and the maintained staged control under matched targets, geometry, rows/raw bytes, work, and
  boundary semantics; only then screen objective and scale/topology order on the selected contract.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`docs/additive_field_v2.md`, `tasks/BENCH-020-field-semantics-factorial.md`,
  `tasks/FIT-049-field-v2-objective-screen.md`, `tasks/FIT-048-additive-stage-order.md`]
- **From staging**: O96

## H16: Advance Field V2 Through Killing Gates, Not a Monolithic Pipeline Build

- **Rationale**: Resolve downstream objective and semantics first; separately screen initializer,
  loss, stage order, parameter schedule, allocation, conditional coefficient solve, and unbiased
  tile sampling; compose only preregistered interactions; then build direct/conditional structured
  codecs, byte-priced control, and end-to-end acceleration. Integrate one default-off profile and
  open sealed production confirmation before any default change. Learned, rich-atom, and temporal
  lanes stay optional after the base field is confirmed.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`docs/additive_field_v2.md`, `tasks/INDEX.md`,
  `tasks/BENCH-019-stage1-downstream-objective.md`,
  `tasks/BENCH-021-additive-convergence-portfolio.md`,
  `tasks/BENCH-022-additive-production-confirmation.md`]
- **From staging**: O99
