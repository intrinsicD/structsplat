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
- [x] Provide screening script: `deprecated_scripts/run_stage_search_screening.sh`.
- [x] Provide a bounded renderer-screen report driver that retains fields, final and intermediate
      reconstruction/error images, temporal metrics, aggregate curves, raw metric tables, and a
      portable `index.html`: `scripts/experiments/abl002_renderer_report.py`.
- [ ] Run cheap screening on 20-50 COCO images and keep top candidates per stage.
- [ ] Run final confirmation on 100-200 images, 3 seeds, and multiple budgets.
- [ ] Compare the final winner against GaussianImage-RS and AIR under the fair protocol.

## Renderer-screen visual diagnostic

The historical lower-level stage-search page is scalar-only and is not a completed experiment
handoff. Use the bounded report driver when screening the normalized/additive renderer axis:

```bash
env PYTHONPATH=src:. python scripts/experiments/abl002_renderer_report.py \
  tests/test_images \
  results/abl002_additive_visual_coco4_m512_i750_b2k5k_s012_20260731_diagnostic \
  --budgets 2000 5000 --seeds 0 1 2 --iters 750 --max-side 512 \
  --log-every 25 --renderers cuda cuda_additive --lpips --device cuda
python scripts/check_report_bundle.py --allow-dirty \
  results/abl002_additive_visual_coco4_m512_i750_b2k5k_s012_20260731_diagnostic
```

This four-image run is a development diagnostic only. It does not satisfy the 20–50-image cheap
screen or the clean-source/prospective-review requirements for a result-bearing default decision.

## Depends on
CORE-001/002, INIT-001/002/003/004, FIT-001, HIER-001, BENCH-001.
