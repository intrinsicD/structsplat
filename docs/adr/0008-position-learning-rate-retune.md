# ADR-0008: Retune the fitter learning rates (positions were effectively frozen)

## Context
`FitConfig` shipped with `lr_means = 2e-3`. Positions are stored in **pixel** coordinates, and under
Adam the per-step update is approximately `lr` in parameter units once the moment estimates settle
(the raw gradient magnitude is normalized away). So `lr_means = 2e-3` moves a Gaussian on the order
of 2e-3 px/iter — a few pixels over the entire 2000-iteration budget. The structure-tensor init
places Gaussians well, which masked the problem, but any Gaussian that needed to migrate to correct
an error essentially could not, and convergence to a target PSNR was far slower than necessary.

## Decision
Raise the default learning rates to values validated by a sweep (kodim19 crop, 256², N=2000, 300
iters, `aniso_flanking`):

| config | lr_means | psnr@300 | iters to PSNR 26 | iters to PSNR 28 |
|---|---|---|---|---|
| old default | 2e-3 | 27.22 | 141 | — (never) |
| means×10 | 2e-2 | 28.45 | 91 | 254 |
| all raised | means 2e-2, others 3–5e-2 | 29.00 | 27 | 137 |
| positions fast | means 1e-1 | 29.22 | 23 | 69 |
| + step decay | means 1e-1, decay | **29.51** | 23 | 73 |

New defaults: `lr_means 5e-2, lr_scales 3e-2, lr_rot 1e-2, lr_color 3e-2`. This sits in the
validated fast regime while staying below the most aggressive setting for robustness across image
sizes and the prune/split schedule. Net effect: ~6× faster to a target PSNR and ~+2 dB at fixed
iterations, on this cell.

Learning-rate **decay stays opt-in**, not on by default: `StepLR`'s period must scale with the
total iteration budget (the sweep used decay-every-100 at 300 iters; the same period at the default
2000 iters would halve the LR 20 times and stall). When used, set `lr_decay_every ≈ iters/4`. The
decay factor is now computed from the global iteration (ADR: see `_lr_factor` in `fit.py`) so it
survives the optimizer rebuild after a prune/split instead of resetting.

## Consequences
+ Large convergence-speed and low-budget-quality win at zero architectural cost — directly serves
  the ABL-001 "iters-to-target" metric and the co-scientist fitness signal.
+ Because the ablation holds the fitter fixed across all cells, the retune shifts every strategy
  equally; relative strategy comparisons are unaffected, absolute numbers improve.
- LRs in pixel units are image-scale dependent in principle; Adam's normalization makes them robust
  in practice, but a very different resolution regime may warrant a per-run sweep. The values are
  plain config, exposed on the CLI and the benchmark, so they remain a tuning surface.
- Higher LRs make the `log_scales` clamp and the split/prune stabilization matter more; both are in
  place (`fit.py`).

## Links
Amends FIT-001 defaults. Interacts with the Adam-moment-continuity change (`_carry_adam_state`) and
the global-iteration LR schedule, both in `fit.py`.
