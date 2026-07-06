# FIT-008: Self-adaptive Gaussian count

**Status: done (2026-07-06).** Target-quality / target-rate fitting mode.

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
- [x] CLI accepts target modes such as `--target-psnr`, `--target-ms-ssim`, `--target-bpp`, and
      `--max-gaussians`.
- [x] Fit history records selected N, growth/prune events, marginal gain, and stopping reason.
- [x] Fixed-N behavior remains unchanged when target mode is off.
- [x] Tests cover target reached, max-N reached, target-BPP, and no-growth/stalled stopping paths.
- [x] Benchmark compares adaptive-N vs fixed-N sweeps on quality, fit time, and final N.
- [x] Stage-search and rate-distortion outputs include enough metadata to keep comparisons fair.

## Implementation notes

- `FitConfig.adaptive_count=True` enables the controller; fixed-N remains the default.
- Stops on `target_psnr`, `target_ms_ssim`, raw-attribute `target_bpp`, `max_gaussians`,
  `stalled`, `no_growth`, or `iteration_limit`.
- Growth reuses existing residual densification modes via `adaptive_split_mode`.
- History records `adaptive_events`, `adaptive_stop_reason`, and `adaptive_selected_n`.
- Stage-search rows record `adaptive_count`, event counts, stop reason, selected N, and
  `estimated_bpp`.
- Rate-distortion rows record `n_gaussians`, `adaptive_stop_reason`, and `estimated_raw_bpp`.

## Evidence

- Focused tests passed: `6 passed in 1.88s`.
- Full suite after implementation passed: see commit notes.
- Smoke evidence: `ara/evidence/fit008-adaptive-count-smoke-2026-07-06/run.md`.
- The tiny smoke selected the 32-Gaussian cap correctly, but fixed-32 beat adaptive 16->32 on
  mean PSNR/AUC and fit time, so this remains opt-in controller infrastructure.

## Interfaces touched
`src/structsplat/fit.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`benchmarks/rate_distortion.py`, `benchmarks/stage_search.py`, tests for fit history/control.

## Depends on
FIT-004, BENCH-002. Pairs with COMP-004 and FF-001.
