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
- [x] Opt-in `cuda_tiled` / `cuda_tiled_additive` path builds a detached tile-to-Gaussian index
      and renders one CUDA block per image tile, matching the reference forward/backward equations.

## Acceptance criteria
- [x] CUDA forward matches reference within tolerance on fixed fields/images.
- [x] CUDA backward matches reference/autograd gradients within tolerance.
- [x] Tiled/culled gather kernel avoids full Gaussian x pixel work and sizes blocks by tile area.
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

## Notes

- 2026-07-05: Added opt-in `renderer="cuda_tiled"` and `renderer="cuda_tiled_additive"`.
  The Python wrapper constructs a detached tile-to-Gaussian index from the same clipped support
  rectangles as the reference renderer, then calls tiled CUDA forward/backward kernels. Forward
  assigns one block per image tile and one thread per tile pixel, looping only over Gaussians
  whose support overlaps that tile. Backward currently uses direct global atomics for parameter
  gradients, so the shared-memory/block-reduction optimization remains open. Validation:
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. pytest -q
  tests/test_render.py -k 'cuda_exact or cuda_tiled or cuda_empty or cuda_nonfinite'` passed
  14 CUDA renderer tests, and a tiny `FitConfig(renderer="cuda_tiled")` smoke completed on CUDA.
- 2026-07-21: A pure-CUDA/Thrust port feasibility study
  (`docs/research/2026-07-21-cuda-thrust-port-feasibility.md`) reviewed this task's remaining
  milestones against the committed PORT-004/FIT-003/tiled evidence. Conclusion: keep the torch
  harness for training and pursue PORT-002/003-style in-extension work (CUB binning, shared-memory
  staging, tighter ellipse-tile bounds, shape-dispatched backward reductions, graph capture); the
  torch-free "pure CUDA" architecture is right specifically for this task's forward-only
  decode/RHI milestone, where the order-independent normalized sum needs no depth sort and a fixed
  tile-list traversal gives deterministic accumulation cheaply. Analysis only — no status change.
- 2026-07-05 fair-protocol benchmark: Ran
  `results/fair_density_control_cuda_tiled_difficult4/` on the same four fair-density finalist
  rows used for the support-fade slice, Kodak difficult-four, budgets {2000,5000,10000}, seed 0,
  max-side 768, 1500 iters, with `renderer=cuda_tiled`. The run completed 48/48 rows and wrote a
  local `index.html` overview. Paired against exact `renderer=cuda`, `cuda_tiled` averaged
  -0.1328 dB final PSNR, +0.0009 AUC, and +17.63 s fit time, or 1.69x slower. Slowdown was worst
  on `kodim19` (+24.05 s mean, 1.86x) and high-budget rows still averaged 1.68x slower. Keep exact
  CUDA for fair/ABL confirmation training sweeps; prioritize backward reductions and tighter
  ellipse-tile bounds before treating tiled as an acceleration path.
