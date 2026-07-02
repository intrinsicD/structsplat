# FIT-001: Adam fitter

**Status: done (reference).** See `fit.py`.

## Acceptance criteria
- [x] Per-group LRs (means/scales/rot/color); loss `(1-w)L1 + w(1-SSIM)`.
- [x] `log_scales` clamped (no collapse/explosion).
- [x] Records PSNR history + iters-to-target; returns render + metrics dict.

## Follow-ups
- [x] LR schedules. Step decay is now a pure function of the global iteration (`_lr_factor`), so it
      survives the optimizer rebuild after a prune/split instead of silently resetting.
- [x] Pruning of inactive/dead Gaussians.
- [x] Residual-driven split/densify during fit.
- [x] Adam moment continuity across prune/split (`_carry_adam_state`): survivors keep their
      `exp_avg`/`exp_avg_sq`, new Gaussians start clean. Previously every densification event wiped
      all optimizer state and spiked the loss.
- [x] Position LR retuned: the default `lr_means=2e-3` left Gaussians nearly frozen in pixel units
      (Adam step ≈ lr px/iter). See ADR-0008 and `benchmarks/` sweep.
- [ ] Optional opacity parameterization.
