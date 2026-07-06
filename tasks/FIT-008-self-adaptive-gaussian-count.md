# FIT-008: Self-adaptive Gaussian count

**Status: todo.** Target-quality / target-rate fitting mode.

## Context
Most current CLI flows assume a fixed `num_gaussians`. Recent self-adaptive 2DGS image
representations emphasize choosing primitive count from image complexity and target quality or rate,
which is also useful for fair compression and speed comparisons.

## Goal
Add a fitting mode that grows, relocates, or prunes Gaussians until a target quality, rate, or
compute budget is reached instead of requiring a fixed final N.

## Approach
1. Add a controller that monitors PSNR/MS-SSIM/AUC improvement and optional estimated bpp.
2. Grow in budgeted waves while marginal quality per new Gaussian remains above a threshold.
3. Prune or relocate low-activity Gaussians when the controller is over budget or stalled.
4. Report the selected N as an output metric, not a hidden side effect.

## Acceptance criteria
- [ ] CLI accepts target modes such as `--target-psnr`, `--target-ms-ssim`, `--target-bpp`, and
      `--max-gaussians`.
- [ ] Fit history records selected N, growth/prune events, marginal gain, and stopping reason.
- [ ] Fixed-N behavior remains unchanged when target mode is off.
- [ ] Tests cover target reached, max-N reached, and no-growth/stalled stopping paths.
- [ ] Benchmark compares adaptive-N vs fixed-N sweeps on quality, fit time, and final N.
- [ ] Stage-search and rate-distortion outputs include enough metadata to keep comparisons fair.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`benchmarks/rate_distortion.py`, `benchmarks/stage_search.py`, tests for fit history/control.

## Depends on
FIT-004, BENCH-002. Pairs with COMP-004 and FF-001.
