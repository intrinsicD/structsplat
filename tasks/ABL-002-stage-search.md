# ABL-002: Full stage-combination search

**Status: partial.** Harness implemented in `benchmarks/stage_search.py` (factorial + influence
modes, ADR-0010); full screening/final runs pending.

## Goal
Find the best complete StructSplat configuration, not just the best initialization strategy, and
measure the influence of each stage in isolation (quality, convergence rate, speed).

## Stages Covered
- Tensor operator: central, Sobel, Scharr; color space: luma, rgb (Di Zenzo).
- Density: structure, gradient, variance, hybrid, uniform.
- Sampling: WSE, Poisson-disk dart throwing, Halton (density-warped), CVT/Lloyd,
  farthest-point, density-random, jittered-grid.
- Initialization: flanking/on-edge/isotropic strategies, axis ratio, coherence,
  orientation mode (tensor/random/zero), scale mode (spacing/uniform/knn), scale-cap mode
  (none/hard/feature), and color mode.
- Renderer: normalized, additive, exact CUDA normalized, exact CUDA additive, and gsplat
  comparator modes.
- Fitting: L1/L2/Charbonnier, Adam/AdamW, none/step/cosine LR schedule.
- Refinement: none, pruning, duplicate/support-duplicate split, residual-add, and
  residual-tensor-add densification.
- Pyramid: single-stage or residual pyramid with prefix metrics.

## Acceptance criteria
- [x] Emit tidy JSON/CSV and a ranked markdown summary for complete stage configs.
- [x] Record stage choices, metrics, fit time, final Gaussian count, and optional prefix metrics.
- [x] Record convergence (iters-to-target, PSNR-AUC) and speed (init/fit/seconds-to-target)
      per row so quality/convergence/speed winners are separable.
- [x] Influence mode: one-factor-at-a-time paired deltas vs the baseline (`influence.md`).
- [x] Canonicalize + dedupe configs whose differing stage is provably inert.
- [x] Provide CLI entry point: `structsplat stage-search` (`--mode factorial|influence`).
- [x] Provide screening script: `scripts/run_stage_search_screening.sh`.
- [ ] Run cheap screening on 20-50 COCO images and keep top candidates per stage.
- [ ] Run final confirmation on 100-200 images, 3 seeds, and multiple budgets.
- [ ] Compare the final winner against GaussianImage-RS and AIR under the fair protocol.

## Depends on
CORE-001/002, INIT-001/002/003/004, FIT-001, HIER-001, BENCH-001.
