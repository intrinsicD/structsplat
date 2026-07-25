# PORT-003: Avoid atomics in tiled backward

**Status: implemented, gradient-validated, and measured under PORT-002's passed profile
(2026-07-25, RTX 3050, ADR-0024 parity amendment).** The tiled backward reduces gradients
warp-level before touching global memory. RTX 4090 parity passed on 2026-07-22. The measured
direction is uniformly favorable, but the pass authorizes only the fair-protocol end-to-end fit
benchmark; the shipped GPU default stays untiled `cuda`.

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
- [x] New backward path matches reference/CUDA autograd gradients within the established tolerance.
- [x] Benchmark isolates backward time vs current direct-atomic kernel across low/high N and
      isotropic/anisotropic fields. (`benchmarks/tiled_render_profile.py`, passed 2026-07-25.)
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
- 2026-07-24 profile run (RTX 3050, PyTorch 2.9.0+cu128): the backward isolation now exists, but
  **the governing frozen gate did not pass**, so no decision is taken and PORT-001's notes are
  left unchanged. The failure is entirely in PORT-002's parity precondition and is attributable to
  a single near-singular-denominator pixel in the *baseline* renderer, not to the warp-reduced
  backward; the full diagnosis, artifact paths, and the explicit do-not-retune constraint live in
  PORT-002's matching note. Recorded backward medians (ms, `cuda` -> `cuda_tiled_gpu_index_cull`)
  are uniformly favorable to the reduction — e.g. 512²/N=8192/overlap-16/ratio-6
  `5.7897 -> 2.1361`, 512²/N=8192/overlap-16/ratio-1 `4.9357 -> 1.6486`, 256²/N=8192/overlap-4/
  ratio-1 `4.3489 -> 0.9574`, and 512²/N=2048/overlap-16/ratio-6 `3.4526 -> 1.6087`. The
  cull-vs-nocull spread also isolates the exact ellipse test's contribution at high anisotropy
  (`2.1361` vs `2.7177` backward at the representative cell). Treat all of these as recorded
  measurements pending a valid precondition, not as an authorized speedup claim.
- 2026-07-25: the precondition was resolved by ADR-0024 (governing parity scoped to
  candidate-vs-baseline; the failure was a float32 conditioning property of the *baseline* at a
  single near-cutoff pixel, with an algebraic `expm1` fix tested and rejected as ineffective on
  total error). The profile then **passed**. Backward medians at the passing run
  (`results/port002_tiled_render_profile_rtx3050_adr0024/`, ms, exact `cuda` -> warp-reduced
  tiled): 512²/N=8192/ov16/ar6 `5.960 -> 2.110`. The warp reduction is favorable in every measured
  cell. PORT-001's notes may now record the measured decision, with the standing scope limit: this
  is one consumer-GPU microprofile, it authorizes the fair-protocol end-to-end fit benchmark, and
  it does not authorize a default flip or a cross-GPU claim. Deterministic accumulation
  (approach item 3) remains open — cross-warp `atomicAdd` ordering is still unfixed.
- Correctness (2026-07-22): the complete CUDA renderer selection
  (`pytest -q tests/test_render.py -k cuda`) passed 29/29 tests on an NVIDIA RTX 4090 with PyTorch
  2.12.0+cu132, including tiled gradient parity and exact ellipse-cull parity. No performance
  conclusion follows from this test run. The exact dirty-worktree command, base commit, source
  hashes, numeric-tolerance caveat, and evidence scope are recorded in PORT-002's matching note.
