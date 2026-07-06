# PORT-003: Avoid atomics in tiled backward

**Status: todo.** Throughput bottleneck follow-up for `cuda_tiled`.

## Context
The current tiled backward directly atomically adds parameter gradients for every pixel-Gaussian
pair. That is simple and correct, but likely becomes the bottleneck as tile culling improves and N
or image size grows.

## Goal
Replace direct global atomic accumulation with a reduction strategy that improves tiled backward
throughput without sacrificing numerical parity.

## Approach
1. Prototype per-tile gradient buffers followed by a per-Gaussian reduction.
2. Compare against an alternate one-block-per-Gaussian backward kernel for high-overlap cases.
3. Keep a deterministic accumulation mode in mind for reproducibility-sensitive benchmarks.
4. Select the strategy based on measured time, memory, and occupancy, not just kernel elegance.

## Acceptance criteria
- [ ] New backward path matches reference/CUDA autograd gradients within the established tolerance.
- [ ] Benchmark isolates backward time vs current direct-atomic kernel across low/high N and
      isotropic/anisotropic fields.
- [ ] Memory overhead is bounded and logged; fallback to atomic path remains available.
- [ ] Fit smoke test passes with restructuring events that change N.
- [ ] PORT-001 notes updated with the measured decision.

## Interfaces touched
`src/structsplat/cuda/render_ext.cu`, `src/structsplat/cuda/render_ext.cpp`,
`src/structsplat/cuda_render.py`, `tests/test_render.py`, CUDA benchmark scripts.

## Depends on
PORT-001. Pairs with PORT-002.
