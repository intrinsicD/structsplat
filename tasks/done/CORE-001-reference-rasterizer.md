# CORE-001: Differentiable reference rasterizer

**Status: done (reference).** See `render.py`, ADR-0003.

## Goal
A differentiable, sorting-free rasterizer `I = sum c_i G_i / (sum G_i + eps)`.

## Acceptance criteria
- [x] Differentiable w.r.t. `means`, `conics`, `colors`.
- [x] Memory bounded and large Gaussians aren't clipped: each Gaussian is evaluated on its own
      per-axis AABB (`radii = (rx, ry)`), batched into flat ragged tiles under an element budget so
      peak memory is bounded even for a single huge Gaussian.
- [x] Boundary-safe indexing; NumPy-mirror check of the compositing formula.

## Notes
Reference is a Python-driven batched loop; the performant tile kernel is PORT-001. The support
window is the AABB of the `sigma_cutoff` ellipse per Gaussian (tighter than a square sized by the
major axis for anisotropic/flanking Gaussians). Optional `aa_dilation` adds an EWA low-pass.
