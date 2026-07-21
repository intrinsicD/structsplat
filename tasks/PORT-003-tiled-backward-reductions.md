# PORT-003: Avoid atomics in tiled backward

**Status: implemented (unmeasured).** The tiled backward now reduces gradients warp-level before
touching global memory; acceptance stays open until gradients and timings are validated on a
CUDA device (see PORT-002's 2026-07-21 note for the shared validation protocol).

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

## Notes

- 2026-07-21: Implemented as warp-level shuffle reductions inside the staged tiled backward
  (PORT-002's 2026-07-21 kernel rework): every warp iterates the same staged Gaussian in
  lockstep, reduces the nine gradient components across its 32 lanes, and lane 0 issues at most
  one atomicAdd per component — a 32x cut in global atomic transactions versus the per-pixel
  atomics it replaces, with zero extra memory (registers and shuffles only; approach item 1's
  per-tile gradient buffers were not needed). Launches pad `blockDim` to a warp multiple so the
  full-mask shuffles are defined for every legal tile size; inactive lanes contribute exact
  zeros, and warps with no coverage of a staged Gaussian skip via one ballot. Deviation from
  approach item 3 as written: the per-pixel-atomic tiled kernel was replaced, not kept as a
  runtime fallback — it was measured strictly slower end-to-end (2026-07-05 evidence) and the
  untiled `cuda` renderer remains the shipped default and production fallback; the old kernel
  stays available in git history for paired comparison. Deterministic accumulation
  (reproducibility item 3) remains open: per-warp reduction order is fixed, but cross-warp
  atomicAdd ordering is not.
- Measurement (open): `python -m benchmarks.tiled_render_profile` isolates backward medians
  across the N/overlap/anisotropy grid against exact `cuda`; PORT-001's notes take the measured
  decision once run on hardware.
