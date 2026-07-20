# BENCH-010 block-reduction results audit

Date: 2026-07-15
Device: NVIDIA GeForce RTX 3050
Disposition: **keep the selector opt-in and benchmark-only; do not promote a speed claim or flip a default.**

## Claim disposition

| Claim | Disposition | Evidence |
|---|---|---|
| The candidate implements one block per Gaussian, a warp/shared-memory reduction, and no final gradient atomics. | Confirmed, source-bound. | Executed-source inspection reports nine baseline atomic callsites and zero candidate callsites; the exact source snapshot is preserved. |
| Candidate forward and parameter gradients match the exact normalized equation at the frozen tolerance. | Confirmed on this grid/device/source. | Both reference calibrations and all eight paired grid cells pass; maximum image/gradient absolute differences are `2.980232e-7` / `1.245644e-8`. |
| The candidate accelerates the representative `256x256`, `N=2048`, overlap-16 micro-fit. | Narrowed to a descriptive source/device/synthetic result. | Primary: backward `1.177600 -> 0.500736 ms` (`-57.478%`), step `3.425120 -> 2.558976 ms` (`-25.288%`). Identical confirmation: backward `1.196032 -> 0.582656 ms` (`-51.284%`), step `3.407872 -> 2.564096 ms` (`-24.760%`). |
| The frozen implementation gate passed robustly. | Refuted. | The primary executable gate passes, but the identical confirmation fails its unchanged `CV <= 5%` criterion (`5.1154%` candidate-backward CV). The task's all-grid direction requirement is also absent from the executable predicate and is violated by measured small-N cells. |
| The optimization is universally faster on the frozen grid. | Refuted. | At `N=512`, exact-backward candidate/baseline ratios are `1.0290`, `1.0100`, `0.9938`, and `1.0231`; one full-step ratio is `1.0052`. |
| The candidate improves quality, convergence, compression, expressiveness, end-to-end fit time, or cross-GPU performance. | Unauthorized / untested. | The experiment is a synthetic CUDA-event microprofile only. |

## Independent recomputation

- Parsed `16` aggregate rows and all `1,200` raw CUDA-event samples.
- Recomputed every phase median and population CV from `samples.csv`; all values match `rows.json` exactly.
- Recomputed every candidate/baseline ratio, representative reduction, memory ratio, and decision input; all stored arithmetic matches.
- Both arms use the same normalized forward equation, loss, Adam scope, deterministic field, support fade, opacity, warmup, and repeat count.
- Peak incremental PyTorch allocator memory is `15,386,624` bytes for both representative arms in both runs (`0%` regression). This is not whole-device VRAM.
- Nsight Compute hardware counters were not collected because the host returned `ERR_NVGPUCTRPERM`; source-issued atomic counts remain explanatory proxies, not hardware counters.

Primary grid ratios (`candidate / baseline`):

| H | N | overlap | exact backward | full step |
|---:|---:|---:|---:|---:|
| 128 | 512 | 4 | 1.029015 | 1.005164 |
| 128 | 512 | 16 | 1.010025 | 0.998908 |
| 128 | 2048 | 4 | 0.526870 | 0.922734 |
| 128 | 2048 | 16 | 0.508721 | 0.925012 |
| 256 | 512 | 4 | 0.993814 | 0.924566 |
| 256 | 512 | 16 | 1.023124 | 0.937926 |
| 256 | 2048 | 4 | 0.463527 | 0.754525 |
| 256 | 2048 | 16 | 0.425217 | 0.747120 |

## Integrity and protocol finding

`tasks/PORT-004-exact-backward-block-reduction.md` requires the candidate to retain the direction
on the frozen count/overlap grid. `PREREGISTERED_GATE` checks parity over the grid but performance
and repeat stability only at the representative cell. Because the all-grid performance condition
was not executable and the observed small-N ratios violate it, the primary artifact's literal
`opt_in_microprofile_gate_pass` is insufficient for promotion. This finding must not be repaired by
post-hoc threshold or predicate changes; a new protocol would be a new experiment.

## Source binding

- Baseline executed-source snapshot manifest SHA-256:
  `4b6cf8805132fbb4e1110c046cf6e785dca1b59240b7731807ae0a2244d92884`.
- Candidate executed-source snapshot manifest SHA-256:
  `e4e2f18dd7f6a782d3b621e906ab76618020edad9abba5d87893ed36af491f24`.
- All eight candidate snapshot files match the primary artifact's recorded worktree SHA-256s.
- The confirmation artifact records the same eight critical-source SHA-256s as the primary.

## Validation and rerun commands

Validation completed with `59 passed`, Ruff, Python compilation, and `git diff --check`:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  pytest -q tests/test_exact_backward_profile.py tests/test_render.py \
    tests/test_covariance_filter.py tests/test_cli.py
```

Primary and independent identical confirmation:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python3 -m benchmarks.exact_backward_profile \
    --outdir results/bench010_exact_backward_block_reduce_rerun

LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python3 -m benchmarks.exact_backward_profile \
    --outdir results/bench010_exact_backward_block_reduce_confirmation_rerun
```

The current `cuda` path remains unchanged and default. `cuda_block_reduce` remains an explicit,
experimental, untiled selector.
