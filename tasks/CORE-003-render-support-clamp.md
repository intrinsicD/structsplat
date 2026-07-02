# CORE-003: Edge-aware render support window (fix off-image support truncation + tile waste)

**Status: todo.** Latent bug in the reference renderer (predates the Claude/Codex merge; present on
both source branches). Found by the ABL-001 approach review, verified by execution.

## Context
`render._accumulate` (`render.py:57-58`) and `gaussian_activity` (`render.py:114-115`) clamp the
per-axis support half-widths with `rx = radii[:, 0].clamp(max=W)`, `ry = radii[:, 1].clamp(max=H)`.
That constant clamp is only safe for Gaussians whose center is inside the image. Means are
unconstrained during fitting (`fit.py` clamps `log_scales` but never `means`), so:

1. **Correctness.** A large Gaussian whose center drifts *off*-image has its in-image support cut
   mid-image: pixels beyond the clamped window get exactly zero weight, which under the normalized
   renderer paints a hard black band where the true contribution is still large. Verified: a
   Gaussian at `x=-40, sigma=40` on a 64px-wide image (3σ radius 120 → clamped to 64) renders
   columns 25–63 as exactly 0, though the true weight at the cut (x=25, ~1.6σ) is 0.267.
2. **Performance.** For an in-image whole-image Gaussian the `(2W+1)×(2H+1)` tile is ~4× larger than
   any in-image support can be; a handful of large Gaussians can dominate render cost and peak
   memory.

## Goal
Bound each Gaussian's support window by its true reachable extent so no in-image pixel of the real
support is ever dropped, and no off-image pixels are ever visited.

## Acceptance criteria
- [ ] In `_accumulate` and `gaussian_activity`, replace the constant clamp with an edge-aware bound:
      `rx = min(radii[:, 0], max(ix, (W-1)-ix))`, `ry = min(radii[:, 1], max(iy, (H-1)-iy))`,
      clamped to `min=1` (where `ix, iy` are the rounded integer centers). Applies for in- *and*
      off-image centers.
- [ ] Regression test: a Gaussian centered off-image with large σ renders a smooth in-image falloff
      (no exact-zero band); assert the cut column's value matches the analytic weight within tol.
- [ ] Equivalence test: for in-image centers the rendered image is bit-identical to the current
      output (the removed pixels were already `valid`-masked zeros), so no existing test regresses.
- [ ] Numerically confirm the tile-element count drops for whole-image Gaussians (no perf
      regression for the common small-Gaussian case).

## Interfaces touched
`src/structsplat/render.py` (`_accumulate`, `gaussian_activity`). No API change, no ADR
(pure support-window correctness/efficiency; the renderer math is unchanged).

## Notes
- Optional, separate: lower the fit scale ceiling `hi = log(max(H, W))` (`fit.py:216`) toward
  `log(max(H, W) / 3)` so 3σ still spans the image while cutting the worst-case tile ~9×. Track as a
  FIT follow-up if pursued — it changes fitting dynamics, this task does not.
- Keep `radii` detached (tiling quantity only; never on the loss gradient) per the `review` skill.

## Depends on
CORE-001.
