# FIT-001: Adam fitter

**Status: done (reference).** See `fit.py`.

## Acceptance criteria
- [x] Per-group LRs (means/scales/rot/color); loss `(1-w)L1 + w(1-SSIM)`.
- [x] `log_scales` clamped (no collapse/explosion).
- [x] Records PSNR history + iters-to-target; returns render + metrics dict.

## Follow-ups
Optional opacity/pruning of dead Gaussians; LR schedules; densify/split during fit (later).
