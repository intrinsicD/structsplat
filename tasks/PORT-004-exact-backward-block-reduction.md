# PORT-004: Exact-backward block reduction

**Status: implemented/screened; keep benchmark-only and opt-in; default unchanged.** The initial
BENCH-010 actionability profile authorized implementation, but the block-reduction experiment does
not pass the governing frozen gate after independent results audit.

## Decision question

Does replacing the untiled exact backward's per-thread final atomics with one block reduction and
one write per Gaussian/component reduce exact-backward and representative fit-step device time
without changing StructSplat's normalized renderer gradients?

The existing kernel already assigns one CUDA block to each Gaussian. Each of 256 threads accumulates
its strided pixel contributions locally, then all threads atomically add the same Gaussian's two
mean, three conic, three color, and optional opacity gradients. The candidate changes only that
within-block reduction schedule.

## Source-bound profiling result

Frozen representative cell: RTX 3050, `256 x 256`, `N=2048`, requested support overlap `16`, exact
untiled normalized CUDA renderer, opacity/support fade enabled, default `0.7 L1 + 0.3 SSIM` loss and
Adam learning rates.

- exact backward median: `1.120256 ms`;
- representative device-side micro-fit step median: `3.377120 ms`;
- exact-backward share: `33.1719%`, above the frozen `>25%` gate;
- raw repeat ranges: `1.109--1.200 ms` backward and `3.292--3.583 ms` full step;
- all eight untiled/tiled grid parity cells passed; maximum image/gradient differences were
  `2.9802e-7` / `2.4331e-8`;
- Nsight Compute hardware counters were unavailable with `ERR_NVGPUCTRPERM`.

The source structure and timing share authorize this experiment. The `4,718,592` representative
atomic-call figure is a source-issued call model, not a measured hardware counter or predicted
speedup.

The historical actionability run and its exact eight-file executed-source snapshot are preserved at
`results/bench010_exact_backward_profile/`. The live profiler now owns the follow-on comparison, so
do not point it at that historical output directory. Its snapshot-manifest SHA-256 is
`4b6cf8805132fbb4e1110c046cf6e785dca1b59240b7731807ae0a2244d92884`.

Reproduce the current block-reduction comparison into a fresh directory with:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.exact_backward_profile \
    --outdir results/bench010_exact_backward_block_reduce_rerun
```

## Frozen implementation gate

Keep the current kernel as the default and add an explicitly experimental selector. On the same
source-bound representative cell, the block-reduction path survives only if it:

1. matches forward and every parameter gradient within the established reference/CUDA tolerance;
2. reduces median exact-backward device time by at least `10%`;
3. reduces median representative fit-step device time by at least `3%`;
4. increases peak PyTorch allocator memory by no more than `5%`;
5. retains the direction on the frozen count/overlap grid with stable repeat distributions.

Failure keeps the prototype benchmark-only or removes it; no threshold or cell may be retuned after
timing. Passing authorizes an end-to-end fit benchmark on disjoint images. It does not by itself
authorize a default flip, cross-GPU claim, quality/convergence claim, or compression claim.

## Block-reduction experiment result

The opt-in `cuda_block_reduce` path replaces the nine baseline final atomic callsites with warp/
shared-memory block reductions and one direct write per Gaussian/component. The original `cuda`
path remains unchanged and default. Both reference calibrations and all eight paired grid cells
passed forward/gradient parity; maximum image/gradient absolute differences were `2.980232e-7` /
`1.245644e-8`.

At the representative RTX-3050 cell, the primary run measured:

- exact backward: `1.177600 -> 0.500736 ms` (`-57.478%`);
- representative device-side step: `3.425120 -> 2.558976 ms` (`-25.288%`);
- peak incremental PyTorch allocator memory: `15,386,624` bytes for both arms;
- maximum registered representative timing CV: `2.927%`.

An identical independent run repeated the large representative reductions (`-51.284%` backward,
`-24.760%` full step) but failed the unchanged stability limit: candidate exact-backward CV was
`5.1154%`, above `5%`. More importantly, the governing prose gate required direction retention on
the whole frozen count/overlap grid, while the executable predicate checked performance only at the
representative cell. The measured grid violates that condition: all four `N=512` exact-backward
candidate/baseline ratios were `1.0290`, `1.0100`, `0.9938`, and `1.0231`, and one full-step ratio
was `1.0052`.

Therefore the literal primary artifact pass is incomplete relative to this task's frozen gate.
Do not retune or rewrite it. Keep the selector benchmark-only; no end-to-end fit, default, quality,
convergence, compression, expressiveness, cross-GPU, or universal-speed authorization follows.
The source-bound claim table, all eight ratios, raw recomputation, and exact rerun commands are in
`results/bench010_exact_backward_block_reduce/audit.md`.

## Depends on

CORE-001/004, ADR-0011, PORT-001, PORT-003, BENCH-002.
