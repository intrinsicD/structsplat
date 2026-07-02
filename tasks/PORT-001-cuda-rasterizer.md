# PORT-001: CUDA tile rasterizer → IntrinsicEngine RHI pass

**Status: todo (future).** The reason for the PyTorch/NumPy reference (ADR-0001).

## Goal
A tiled CUDA rasterizer matching `render.py` numerically, then an IntrinsicEngine RHI pass
(Vulkan/DX12) for real-time decode. Reference stays the correctness oracle.

## Acceptance criteria
- [ ] CUDA forward matches reference within tolerance on a fixed field/image.
- [ ] CUDA backward matches autograd grads (finite-difference check).
- [ ] Throughput target (e.g. >1000 FPS decode at target N) documented.
- [ ] RHI pass consumes the same packed Gaussian buffer layout; parity test vs reference.

## Depends on
CORE-001.
