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
