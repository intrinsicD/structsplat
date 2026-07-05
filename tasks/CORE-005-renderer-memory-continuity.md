# CORE-005: Reference renderer memory bound + C0-continuous support cutoff

**Status: partial.** From the 2026-07-03 repo review.

## Context
1. **Chunking does not bound real peak memory.** Each flat-tile slice's autograd graph retains
   ~10 budget-sized saved tensors (`gid/px/py/dx/dy/q/w/...`) until backward, so peak memory
   scales with the *total* tile-element count, not the chunk budget — the `render_chunk` knob
   only bounds the forward working set. (`src/structsplat/render.py:76-90`)
2. **Hard support cutoff creates O(1) intensity steps and gradient dead zones.** The Gaussian
   tail is truncated at the sigma_cutoff AABB; under the normalized renderer this paints a
   visible discontinuity at box edges wherever coverage is sparse, and pixels with zero
   coverage render black with exactly zero gradient toward them. (`src/structsplat/render.py:84-93`)
3. The `max(chunk, 64) * 4096` element-budget formula is duplicated in three places
   (`render.py:77,199`, `fit.py:162`) and the unit of `chunk` is undocumented.

## Goal
Peak memory truly O(budget) when requested; C0-continuous compact support in both reference and
CUDA renderers; one budget helper with documented semantics.

## Acceptance criteria
- [ ] Opt-in gradient checkpointing per slice (`torch.utils.checkpoint`) making backward peak
      memory O(budget); measured before/after peak on a large-N fit recorded in the task notes.
- [x] Support fade `w = max(exp(-q/2) - exp(-sigma_cutoff^2/2), 0)` implemented in
      `_accumulate`, `gaussian_activity`, `_support_residual_scores`, and the CUDA kernels,
      behind a config flag; parity test reference-vs-CUDA with the fade on.
- [x] One benchmark slice (fixed seed/budget) comparing fade on/off for PSNR and
      iters-to-target, logged; if the fade wins, flip the default via an ADR note.
- [x] `_element_budget(chunk)` helper hoisted into `render.py`, imported by `fit.py`;
      `FitConfig.render_chunk` docstring states the unit (elements = chunk * 4096).

## Notes

- 2026-07-04 partial implementation: `_element_budget(chunk)` now lives in
  `src/structsplat/render.py`, and `_accumulate`, `gaussian_activity`, and fit's
  `_support_residual_scores` all use it. The helper preserves the previous
  `max(chunk, 64) * 4096` behavior; `FitConfig.render_chunk` now documents the effective element
  budget, and focused coverage pins the 64-unit floor plus 4096-element unit.
- 2026-07-05 partial implementation: Added opt-in `FitConfig.support_fade` / CLI
  `--support-fade`. Reference normalized/additive renderers, `gaussian_activity`, fit's
  `_support_residual_scores`, exact CUDA scatter kernels, and `cuda_tiled` kernels all use the
  same compact-support weight. Codec blobs now store `support_fade` so `decode_and_render` remains
  self-describing. Validation: reference fade tests passed; CUDA fade parity passed under
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` with 22 renderer tests; a tiny
  `FitConfig(renderer="cuda", support_fade=True)` smoke completed on CUDA.
- 2026-07-05 benchmark decision: Added `--support-fade` as a real fair-density benchmark axis
  in `benchmarks/fair_density_control_compare.py` and ran
  `results/fair_density_control_supportfade_difficult4/` on the four current finalist rows
  (`onedge`/`qt-WSE` x residual/tensor), Kodak difficult-four, budgets {2000,5000,10000}, seed 0,
  max-side 768, 1500 iters, exact CUDA. The run completed 48/48 rows and wrote a local
  `index.html` overview. Paired against the matching fade-off rows, support fade improved PSNR
  only at 2k (+0.4209 dB mean, 9/16 wins) and improved AUC in 38/48 cells (+0.1073 mean), but
  lost final PSNR overall (-0.1389 dB mean, 9/48 wins) and added +1.67 s mean fit time. Keep the
  flag opt-in; do not flip the default or add an ADR default-change note from this slice.

## Interfaces touched
`src/structsplat/render.py`, `src/structsplat/cuda_render.py`,
`src/structsplat/cuda/render_ext.cpp`, `src/structsplat/cuda/render_ext.cu`,
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/codec.py`,
`src/structsplat/pyramid.py`, `src/structsplat/cli.py`. The fade changes renderer math only when
opted in; if it becomes the default, add an ADR amendment to ADR-0003.

## Depends on
CORE-003, CORE-004.
