# ABL-002: Full stage-combination search

**Status: partial.** Harness implemented in `benchmarks/stage_search.py`; full screening/final runs pending.

## Goal
Find the best complete StructSplat configuration, not just the best initialization strategy.

## Stages Covered
- Tensor operator: central, Sobel, Scharr.
- Density: structure, gradient, variance, hybrid, uniform.
- Sampling: WSE, density-random, jittered-grid.
- Initialization: flanking/on-edge/isotropic strategies, axis ratio, coherence, scale mode, color mode.
- Renderer: normalized, additive reference mode.
- Fitting: L1/L2/Charbonnier, Adam/AdamW, none/step/cosine LR schedule.
- Refinement: none, pruning, duplicate split, residual-add densification.
- Pyramid: single-stage or residual pyramid with prefix metrics.

## Acceptance criteria
- [x] Emit tidy JSON/CSV and a ranked markdown summary for complete stage configs.
- [x] Record stage choices, metrics, fit time, final Gaussian count, and optional prefix metrics.
- [x] Provide CLI entry point: `structsplat stage-search`.
- [x] Provide screening script: `scripts/run_stage_search_screening.sh`.
- [ ] Run cheap screening on 20-50 COCO images and keep top candidates per stage.
- [ ] Run final confirmation on 100-200 images, 3 seeds, and multiple budgets.
- [ ] Compare the final winner against GaussianImage-RS and AIR under the fair protocol.

## Depends on
CORE-001/002, INIT-001/002/003/004, FIT-001, HIER-001, BENCH-001.
