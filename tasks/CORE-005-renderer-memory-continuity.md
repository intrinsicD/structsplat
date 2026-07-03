# CORE-005: Reference renderer memory bound + C0-continuous support cutoff

**Status: todo.** From the 2026-07-03 repo review.

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
- [ ] Support fade `w = max(exp(-q/2) - exp(-sigma_cutoff^2/2), 0)` implemented in
      `_accumulate`, `gaussian_activity`, `_support_residual_scores`, and the CUDA kernels,
      behind a config flag; parity test reference-vs-CUDA with the fade on.
- [ ] One benchmark slice (fixed seed/budget) comparing fade on/off for PSNR and
      iters-to-target, logged; if the fade wins, flip the default via an ADR note.
- [ ] `_element_budget(chunk)` helper hoisted into `render.py`, imported by `fit.py`;
      `FitConfig.render_chunk` docstring states the unit (elements = chunk * 4096).

## Interfaces touched
`src/structsplat/render.py`, `src/structsplat/cuda/render_ext.cu`, `src/structsplat/fit.py`,
`src/structsplat/config.py`. The fade changes renderer math → needs an ADR amendment to
ADR-0003 if it becomes the default.

## Depends on
CORE-003, CORE-004.
