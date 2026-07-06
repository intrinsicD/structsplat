# PORT-002: GPU-native tile index + fused loss/backward

**Status: todo.** Performance follow-up for the CUDA tiled renderer.

## Context
`cuda_tiled` currently builds the tile-to-Gaussian index in Python/Torch and sorts it each call.
The benchmark notes show this path is slower than exact CUDA despite lower theoretical work. The
next acceleration step is to move binning, prefix sums, and repeated buffer management onto the GPU,
then fuse render and loss work where training uses a stable objective.

## Goal
Make the tiled path a real acceleration path by removing Python-side indexing overhead and reducing
memory traffic during training.

## Approach
1. Implement GPU binning of Gaussian support rectangles into tile lists.
2. Use prefix-sum/compaction kernels and preallocated work buffers sized from worst-case or cached
   capacity.
3. Add optional fused render + L1/SSIM partial accumulation for training loops.
4. Investigate CUDA graph capture when N, image size, and tile capacity are stable.

## Acceptance criteria
- [ ] Tile index construction runs fully on GPU after input tensors are on device.
- [ ] Reuses preallocated buffers across iterations without hidden CPU synchronization.
- [ ] Forward parity vs existing `cuda_tiled` and reference renderers.
- [ ] Fused training loss path matches unfused loss within tolerance on fixed fixtures.
- [ ] Benchmark shows tile-index time, render time, backward time, and total fit time before/after.
- [ ] If CUDA graphs are added, fallback path remains available for dynamic-N fits.

## Interfaces touched
`src/structsplat/cuda_render.py`, `src/structsplat/cuda/render_ext.cpp`,
`src/structsplat/cuda/render_ext.cu`, `src/structsplat/fit.py`, CUDA tests and benchmarks.

## Depends on
PORT-001, FIT-003.
