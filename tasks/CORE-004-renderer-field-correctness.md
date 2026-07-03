# CORE-004: Renderer + GaussianField correctness fixes

**Status: done.** Confirmed defects from the 2026-07-03 repo review (each traced to the failing
line; the first four were adversarially verified by execution). CUDA-path fixes (N=0, support
bounds, once_differentiable) are covered by parity tests that require a GPU to execute; the
CPU-testable fixes (value-semantic `from_numpy`, dilation validation) are exercised directly.

## Context
Four correctness bugs in the render/field layer, plus test-coverage gaps that let CUDA
regressions ship silently.

1. **CUDA forward returns uninitialized memory for N=0.** `render_ext.cu` allocates
   `out_flat = torch::empty(...)` and only writes it inside `if (n > 0 && pixels > 0)`; with an
   empty field the caller receives recycled allocator memory. The reference renderer returns
   zeros. (`src/structsplat/cuda/render_ext.cu:231-261`)
2. **`support_bounds` int cast is UB for NaN/huge means.** `static_cast<int>(nearbyintf(mean))`
   saturates to INT_MAX on CUDA; `ix + rx` then signed-overflows and wraps negative, defeating
   both clamps (`x1 - x0 + 1` wraps a second time to a positive tile size), so a fit that
   diverges can read/write out of bounds instead of degrading gracefully like the reference.
   (`src/structsplat/cuda/render_ext.cu:20-29`)
3. **`GaussianField.from_numpy` aliases caller arrays.** `torch.as_tensor` is zero-copy for
   float32 CPU ndarrays, so optimizer steps silently mutate the init module's numpy arrays
   in place (verified with `np.shares_memory`). Any code reusing an init array after fitting
   reads corrupted data. (`src/structsplat/gaussians.py:27-34`)
4. **Negative `aa_dilation` silently produces NaN images.** `conics()` computes
   `1/(s^2 + dilation)` unvalidated; `dilation < -min(s^2)` gives negative inverse variances,
   `exp` overflow, and `inf/inf = NaN` through the normalized division. Nothing validates the
   config anywhere. (`src/structsplat/gaussians.py:129-130`, `src/structsplat/config.py:67`)
5. `_ExactRenderCuda.backward` is not `@once_differentiable` and ignores
   `ctx.needs_input_grad` — double-backward fails with a generic error and all five gradients
   are always computed. (`src/structsplat/cuda_render.py:67-88`)

## Goal
The CUDA path degrades exactly as gracefully as the reference; field construction has value
semantics; invalid configs fail loudly at construction, not as NaNs mid-fit.

## Acceptance criteria
- [x] N=0 CUDA render returns zeros (run `finalize_kernel` unconditionally when `pixels > 0`,
      or early-return a zeros tensor); parity test `cuda(N=0) == reference(N=0)`.
- [x] `support_bounds` guards non-finite/out-of-range means (clamp the float into
      `[-(rx+1), width+rx]` before the cast or compute in int64; early-out on `!isfinite`);
      test: a field containing a NaN/1e12 mean renders without illegal access and matches the
      reference's zero-contribution behavior.
- [x] `from_numpy` always copies (`torch.as_tensor(...).clone()` or `torch.tensor`); regression
      test asserts `not np.shares_memory(...)` for float32 inputs including opacities/scale_max.
- [x] `dilation >= 0` validated (raise `ValueError`) in `conics()`/`radii()` or
      `FitConfig.__post_init__`; test asserts the raise.
- [x] `@torch.autograd.function.once_differentiable` on the CUDA backward.
- [x] CUDA parity tests parametrized over {normalized, additive} × {opacities on/off} ×
      {dilation 0, 0.3}, including `opacities.grad` parity (currently only the
      normalized/no-opacity forward+backward pair is covered, `tests/test_render.py:63-114`).

## Interfaces touched
`src/structsplat/cuda/render_ext.cu`, `src/structsplat/cuda_render.py`,
`src/structsplat/gaussians.py`, `src/structsplat/config.py`, `tests/test_render.py`,
`tests/test_gaussians.py`. No API change; no ADR (pure correctness).

## Depends on
CORE-001, CORE-002.
