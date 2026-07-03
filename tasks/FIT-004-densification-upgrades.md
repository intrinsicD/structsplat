# FIT-004: Densification & convergence upgrades (function-preserving growth, relocation, NMS)

**Status: partial; core modes implemented, stretch items open.** From the 2026-07-03 repo
review + SOTA survey. These are quality/convergence improvements; each lands behind a config
flag so ABL-002's stage axes stay comparable.

## Context
The trace's own conclusion (O09) is that in-loop density control, not first placement, is the
current blocker. Published tools, in ascending effort:

1. **Spatial suppression on residual adds.** `_add_from_residual` takes the top-k residual
   pixels with no minimum spacing, so every new Gaussian in one event clusters on the single
   worst region. (`src/structsplat/fit.py:274-281`)
2. **Function-preserving growth** (Bulò et al., ECCV 2024, arXiv 2404.06109). Adam moments are
   already carried across restructures, but the *rendered function* still jumps: new Gaussians
   inject target colors into a converged normalized sum. Correct parent/child weights so pixel
   sums are preserved at insertion; place children at ±σ along the major axis (long-axis
   split, arXiv 2411.10133). Removes the post-split PSNR dip (est. 10–30% fewer iterations
   after each wave).
3. **Residual-color option for normalized adds** (Image-GS, SIGGRAPH 2025): initialize added
   Gaussians' colors from the local residual (blended through the normalization denominator)
   instead of target colors; already flagged in docs/blockers_and_external_techniques.md.
4. **Budgeted, score-ranked waves** (Taming-3DGS, arXiv 2406.15643): replace top-residual-pixel
   with the composite score sketched in docs/blockers item 2 (residual-under-support ×
   activity × footprint), growing exactly K per wave so final N is deterministic — preserving
   ABL-001's fixed-budget protocol.
5. **MCMC-style relocation** (3DGS-MCMC, NeurIPS 2024, arXiv 2404.09591): teleport low-activity
   Gaussians onto high-error support with a function-preserving rescale adapted to the
   normalized renderer (ADR-0003); keeps N exactly constant. Doubles as the key ABL-004
   control: if relocation erases the init gap, the honest conclusion changes.
6. Stretch: AbsGS |∂L/∂μ| accumulation as the split criterion (arXiv 2404.10484; needs a
   backward-pass reduction); Adan optimizer branch in `_make_optimizer` (GaussianImage's
   default, arXiv 2208.06677 — extend `_carry_adam_state` to its three moment buffers);
   sweep `aa_dilation=0.3` vs 0.0 as default (sub-pixel Gaussians at the 0.35px floor are
   near-deltas with poor positional gradients; it is already a logged config axis).

## Goal
Densification that neither clusters nor spikes the loss, plus a relocation mode that makes
fixed-N capacity allocation self-correcting.

## Acceptance criteria
- [x] `_add_from_residual` oversamples k then applies greedy min-spacing suppression (spacing
      tied to base_scale); test: adds in one wave are pairwise separated.
- [x] Function-preserving `_split_from_residual` variant (weight-corrected, ±σ_major child
      placement) behind `split_mode='fp_duplicate'`; test: PSNR at the split iteration drops
      < 0.05 dB (vs the current visible dip).
- [x] `color_init='residual'` option for normalized-renderer adds; benchmark slice recorded.
- [x] `split_mode='ranked_wave'`: composite score, exactly-K growth, score components logged
      for the stage-influence harness.
- [x] `relocate_every`/`relocate_count` mode implemented with function-preserving rescale under
      the normalized renderer, reusing `_carry_adam_state`; N constant across the fit; test.
- [x] Each new mode registered as a stage-search axis value (ABL-002) with a one-line
      description in benchmarks/README.md.
- [ ] Stretch items each behind flags with one benchmark slice per item; results in notes.

## Notes
- 2026-07-03 partial implementation: residual-add spacing controls
  `split_min_spacing`/`split_oversample`, normalized residual color init via
  `split_color_init='residual'`, CLI flags, and stage-search refine arms
  `residual_add_nms`, `residual_tensor_add_nms`, `residual_add_residual_color`,
  `residual_tensor_add_residual_color`, and combined NMS+residual-color variants.
- Benchmark slice:
  `ara/evidence/fit004-residual-add-controls-2026-07-03/` on four 160px COCO crops,
  budget 256, seed 0, 60 iters, one +32 wave at iter 30. Mean single-stage results:
  no-refine 21.6899 PSNR / 20.112 AUC; residual_add 21.2078 / 19.4778;
  residual_add_nms 21.3046 / 19.5909; residual_tensor_add 21.3282 / 19.5325;
  residual_tensor_add_nms 21.3822 / 19.6472. Residual color init alone was worse in this
  short slice; keep it as an experimental axis, not a default.
- 2026-07-03 core completion: `fp_duplicate` half-opacity parent/child splitting, `ranked_wave`
  composite parent scoring, and constant-N relocation are implemented with tests. Ranked-wave
  rows expose `ranked_wave_score_mean`, `ranked_wave_residual_support_mean`,
  `ranked_wave_activity_mean`, and `ranked_wave_footprint_mean` in stage-search outputs.
- Benchmark slice:
  `ara/evidence/fit004-fp-ranked-relocate-2026-07-03/` on four 160px COCO crops, budget 256,
  seed 0, 60 iters, one +32 split/relocation wave at iter 30. Mean single-stage results:
  no-refine 21.6899 PSNR / 20.1120 AUC; fp_duplicate 21.4186 / 19.6873;
  residual_tensor_add_nms 21.3822 / 19.6472; relocate 21.3245 / 19.7765;
  ranked_wave 21.3140 / 19.5506; residual_add_nms 21.3045 / 19.5909. These modes are useful
  controls, but this short slice still does not justify enabling in-loop restructuring by
  default.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `benchmarks/stage_search.py`,
`tests/test_fit_dynamics.py`. Relocation semantics under the normalized renderer needs a short
ADR (function-preservation math differs from the 3D additive case).

## Depends on
FIT-002 (correctness first), BENCH-002 (fair budgets before ranking variants).
