# CORE-001: Differentiable reference rasterizer

**Status: done (reference).** See `render.py`, ADR-0003.

## Goal
A differentiable, sorting-free rasterizer `I = sum c_i G_i / (sum G_i + eps)`.

## Acceptance criteria
- [x] Differentiable w.r.t. `means`, `conics`, `colors`.
- [x] Chunked + sorted-by-radius so memory is bounded and large Gaussians aren't clipped.
- [x] Boundary-safe indexing; NumPy-mirror check of the compositing formula.

## Notes
Reference is Python-loop-per-chunk; the performant tile kernel is PORT-001.
