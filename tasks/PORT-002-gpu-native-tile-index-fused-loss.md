# PORT-002: GPU-native tile index + fused loss/backward

**Status: implemented (unmeasured) for the index/kernel work; fused loss and CUDA graphs remain
open.** The 2026-07-21 implementation was authored without a CUDA device, so every acceptance
box stays unchecked until the parity tests and the preregistered profile run on real hardware.

## Context
`cuda_tiled` currently builds the tile-to-Gaussian index in Python/Torch and sorts it each call.
The benchmark notes show this path is slower than exact CUDA despite lower theoretical work. The
next acceleration step is to move binning, prefix sums, and repeated buffer management onto the GPU,
then fuse render and loss work where training uses a stable objective.

## Goal
Make the tiled path a real acceleration path by removing Python-side indexing overhead and reducing
memory traffic during training.

## Approach
1. Implement GPU binning of Gaussian support rectangles into tile lists, with a tighter
   ellipse-tile intersection test than the current AABB overlap — the loose bound is worst for
   exactly the elongated Gaussians this method produces, and the 2026-07-05 cuda_tiled test
   (`ara/evidence/fair-density-control-cuda-tiled-difficult4-2026-07-05/`) named tighter bounds
   a prerequisite (with PORT-003's backward reductions) for tiled ever beating exact CUDA.
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

## Notes

- 2026-07-21 implementation (approach items 1–2 plus PORT-003's reduction; unmeasured, authored
  without a CUDA device):
  - `ext.build_tile_index` moves binning into the extension: a per-Gaussian tile-count kernel
    over the same clipped support rectangles as the reference, `cub::DeviceScan` for pair
    offsets, a pair-expansion kernel, `cub::DeviceRadixSort::SortPairs` on packed 32-bit
    `(tile_id, gid)` keys restricted to the live key bits, and a binary-search kernel for
    per-tile ranges. One intentional scalar device→host sync sizes the sort buffers; all other
    allocation goes through the torch caching allocator (the buffer-reuse mechanism). Radix
    sorting is stable, so within-tile order is ascending gid and the GPU-built index is
    deterministic run-to-run — unlike the torch `argsort` builder it replaces, which is kept as
    `tile_index_backend="torch"` for parity testing.
  - The tiled forward/backward kernels now cooperatively stage Gaussians (params + bounds +
    int32 gid) into shared memory in batches of 256, evaluating `support_bounds` once per
    Gaussian per block instead of once per pixel-Gaussian pair.
  - Tighter-than-AABB culling (approach item 1's ellipse test): `tile_min_q` computes the exact
    minimum of the conic quadratic over the tile∩support rectangle (convex closed form on the
    four edges). Pairs with `min q > sigma_cutoff^2` take a sentinel key and sort out of every
    tile range. Exact only under `support_fade`, where the visible weight is exactly zero beyond
    the cutoff; the wrapper auto-disables the cull otherwise. Culled contributions are exact
    zeros, so forward and backward are unchanged (tested).
  - Validation on a CUDA machine: `pytest -q tests/test_render.py -k 'cuda'` (existing tiled
    parity tests now exercise the new path; new tests cover index-backend equivalence and cull
    on/off equality), then `python -m benchmarks.tiled_render_profile` whose preregistered gate
    (frozen before any timing) requires the representative 512²/N=8192/overlap-16/ratio-6 tiled
    step to be ≤ 1.00x exact `cuda`, every N=8192 cell to hold that direction, GPU index build
    ≤ 15% of the tiled step, and CVs ≤ 5%. Passing authorizes only the fair-protocol end-to-end
    fit benchmark; the shipped GPU default stays `cuda`.
  - Remaining scope for this task: fused render+L1/SSIM partial accumulation (approach item 3)
    and CUDA graph capture with the dynamic-N fallback (approach item 4).
