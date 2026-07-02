---
name: review
description: Use when reviewing a StructSplat diff or PR, or self-reviewing before commit. A correctness-first checklist for differentiable graphics code — gradient safety, numerical stability, determinism, and performance regressions. Trigger on "review this", "check my changes", or before opening a PR.
---

# Review checklist

## Correctness (highest priority)
- **Math matches an oracle.** Structure-tensor / conic / render formulas have NumPy checks — if you
  changed one, update or add the mirror. Prefer a closed-form or NumPy reference over "looks right".
- **Gradients flow** where intended: renderer must stay differentiable w.r.t. `means`, `conics`,
  `colors`. No `.item()`/`.detach()`/`.long()` on the loss path. `radii` is a tiling quantity and
  may be detached.
- **Shapes and coords**: `(H, W, 3)`, positions `(x, y)`. Guard image-boundary indexing.
- **Determinism**: same seed -> same init. New randomness must thread `InitConfig.seed`.

## Stability
- `log_scales` clamped so Gaussians can't collapse (`fit.py`) or explode past image size.
- Divisions guarded (`+ eps`). No NaNs from `sqrt`/`log` of non-positive values.

## Performance (reference is allowed to be slow, but watch regressions)
- Renderer stays chunked and sorted-by-radius (bounded memory). Flag O(N*H*W) dense paths.
- Sampling stays grid-accelerated; flag accidental O(M^2) neighbor scans.

## Hygiene
- Diff is scoped to the task. NumPy/torch split intact. Public signatures documented.
- Any new user-facing behavior has a test and, if quality-relevant, a benchmark number.
