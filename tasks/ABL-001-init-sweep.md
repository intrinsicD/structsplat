# ABL-001: Init-strategy × budget sweep (core experiment + fitness)

**Status: partial.** Harness implemented in `benchmarks/ablation.py`; full multi-seed run pending.

## Goal
Answer turn-3 empirically and expose the result as a co-scientist fitness signal.

## Acceptance criteria
- [x] Sweep `{random, grid, iso_blue_noise, aniso_onedge, aniso_flanking} × {budgets}` on fixed
      images, configurable seeds/cell, fixed fitter/iters.
- [x] Report PSNR/MS-SSIM/LPIPS at fixed budget AND iters-to-target; mean ± std.
- [x] Emit tidy JSON/CSV + a markdown summary table + PSNR-vs-budget and PSNR-vs-iters plots.
- [x] Expose a scalar aggregate (e.g. mean PSNR at target budget, or AUC of PSNR-vs-iters) as
      `fitness(strategy)` for search.
- [ ] Run the full ≥3 seed/cell experiment and record the empirical winner.

## Hypothesis (to confirm or refute)
`aniso_flanking ≥ aniso_onedge > iso_blue_noise > grid > random` at low budgets; gap → 0 as budget
grows. Record the truth either way.

## Depends on
INIT-003, INIT-004, BENCH-001.
