# PORT-001: CUDA tile rasterizer -> IntrinsicEngine RHI pass

**Status: partial.** StructSplat now owns an exact PyTorch CUDA extension for the reference
normalized/additive equations (`renderer=cuda`, `renderer=cuda_additive`; ADR-0011). The remaining
work is a tiled/culled production kernel and the IntrinsicEngine RHI port.

## Goal
A tiled CUDA rasterizer matching `render.py` numerically, then an IntrinsicEngine RHI pass
(Vulkan/DX12) for real-time decode. Reference stays the correctness oracle.

## Current milestone
- [x] Exact CUDA extension matches the clipped-support normalized and additive reference equations.
- [x] Python autograd wrapper supports forward/backward and uses `@once_differentiable`.
- [x] Parity tests cover normalized/additive, opacity on/off, and dilation variants when CUDA is
      available.
- [x] `renderer=gsplat` remains a separate GaussianImage++ alpha/sum comparator, not a reference
      equivalent.

## Acceptance criteria
- [x] CUDA forward matches reference within tolerance on fixed fields/images.
- [x] CUDA backward matches reference/autograd gradients within tolerance.
- [ ] Tiled/culled gather kernel avoids full Gaussian x pixel work and sizes blocks by tile area.
- [ ] Backward reduces partials with shared-memory/block reductions instead of direct global
      atomics where that becomes the throughput bottleneck.
- [ ] Exact ellipse-tile intersection or a tighter conservative bound is evaluated; current AABBs
      are worst-case for long anisotropic edge Gaussians.
- [ ] Optional top-k normalized mode (Image-GS-style) is implemented as a non-reference renderer
      stage if it proves useful.
- [ ] Deterministic-accumulation option and tolerance policy are documented for reproducibility.
- [ ] Throughput target (e.g. >1000 FPS decode at target N) documented.
- [ ] RHI pass consumes the same packed Gaussian buffer layout; parity test vs reference.

## Depends on
CORE-001.
