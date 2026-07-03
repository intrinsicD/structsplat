# FIT-002: Fitter correctness fixes (split colors, opacity pruning, history pairing)

**Status: done.** Confirmed defects from the 2026-07-03 repo review.

## Context
1. **`_split_from_residual` injects full target colors under additive renderers.** The
   duplicate-split path assigns `_nearest_image_colors(target, means)` unconditionally, while
   `_add_from_residual` correctly switches to residual colors for additive modes — so
   `duplicate`/`support_duplicate` splits double-count brightness wherever children land in an
   additive fit. (`src/structsplat/fit.py:246`, correct branch at `fit.py:301-306`)
2. **Activity-based pruning ignores opacity.** `gaussian_activity` is opacity-free by design,
   so a Gaussian the optimizer has driven fully transparent is never pruned in
   `opacity_mode='constant'` runs — stranded capacity at fixed N.
   (`src/structsplat/render.py:185-206`, `src/structsplat/fit.py:193-210`)
3. **History rows mix pre/post-restructure state.** On iterations where prune/split fires,
   the logged row pairs a pre-step, pre-restructure PSNR with a post-restructure
   `n_gaussians`. (`src/structsplat/fit.py:380-385`)
4. **`residual_tensor_add` hard-codes anisotropy** (`1 + 3*coherence`) and the 0.35 scale
   floor instead of reusing `InitConfig.max_axis_ratio`/`coherence_power` semantics — the
   densifier and the init disagree about what anisotropy means. (`src/structsplat/fit.py:284-298`)

## Goal
Densification behaves consistently across renderer modes and parameterizations; logged history
rows describe one consistent state.

## Acceptance criteria
- [x] `_split_from_residual` uses residual colors `(target - render_img)` at child positions
      when `cfg.renderer` is an additive mode; test: additive fit with duplicate splits does
      not regress PSNR at the split iteration.
- [x] Pruning multiplies activity by `opacity_values()` when opacities are present (or adds a
      min-opacity criterion); threshold-unit change documented in `FitConfig`; test: a
      zero-opacity Gaussian is pruned.
- [x] History rows log `n_gaussians` captured at the same point as the PSNR (pre-restructure),
      or both post-; one convention, documented.
- [x] `residual_tensor_add` anisotropy threaded from config (a FitConfig field defaulting to
      InitConfig semantics); the 0.35 floor hoisted to one shared module constant.
- [x] `pytest -q` green; one benchmark slice with splits enabled confirms no regression.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/render.py`, `src/structsplat/config.py`,
`tests/test_fit_dynamics.py`. No ADR (bug fixes within existing decisions).

## Depends on
FIT-001, CORE-004.
