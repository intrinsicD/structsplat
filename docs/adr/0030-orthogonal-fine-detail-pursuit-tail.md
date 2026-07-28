# ADR-0030: Optional orthogonal fine-detail pursuit tail

- Status: accepted (experimental interface); one exposed-image screen, default remains off
- Date: 2026-07-28
- Tasks: FIT-039, FIT-040, FIT-041
- Related: ADR-0022, ADR-0025, ADR-0029, CORE-010/011/012, FIT-031/033/038

## Context

ADR-0029's error-only tail ranks foreground RGB MAE and then globally recovery-fits an expanded
field. Its committed FIT-031 screen improved global and boundary metrics, but it did not predeclare
or measure a deep fine-detail band. On the requested full masked Janelle frame, the equal-base
FIT-041 control activates all 2,777 requested rows yet every appended row ends with its center
within 2.24 px of the mask boundary. It improves foreground PSNR by `0.320788 dB`, but changes
deep sigma-1.5 high-pass MSE by only `0.000042%` relative and Laplacian MSE by `0%`.

The desired intervention is different: spend a small number of ordinary Gaussians only where the
current deep-interior high-pass residual remains, remeasure after each cohort, and never disturb
the 11,000 already accepted rows.

## Decision

1. `scripts/convert.py --fine-detail-pursuit` enables a distinct terminal stage after
   `safe_polish`. It is disabled by default and mutually exclusive with ADR-0029's
   `--fine-detail`.
2. The stage is allowed only for masked jobs, dynamic storage, and normalized renderers. It uses
   the existing certified mask constraint and full protected commit gate.
3. Each wave selects 128 maxima of the current RGB mean-square sigma-1.5 high-pass residual,
   strictly deeper than `margin + 6 px`. A 5x5 NMS applies within the wave. Across waves, only
   exact prior sites are forbidden: FIT-039's killing ablation showed that excluding adjacent
   sites suppresses distinct fabric residual lobes.
4. Every new row is an ordinary constant-color Gaussian with 0.35 px isotropic scale and opacity
   0.8. All inherited rows remain bit-for-bit frozen.
5. After each wave, deterministic regularized conjugate gradients jointly solve the colors of
   every accumulated pursuit row under the exact normalized-compositor denominator. Geometry,
   opacity, and inherited colors do not move.
6. A wave commits only if the unchanged protected metric vector is safe. The stage stops at the
   first committed state reaching both `25%` deep sigma-1.5 high-pass reduction and `20%`
   Laplacian reduction, or on a protected rejection, insufficient sites, or 2,048 added rows.
7. Config, history, result rows, and reports persist the two detail baselines/targets, every wave's
   site/order and canonical site-set hashes, solve diagnostics, protected decision, row counts,
   termination reason, and time. Canonical site-set hashes treat a CUDA `topk` tie permutation as
   the same field basis while preserving the ordered hash for forensic diagnosis.

## Consequences

- The option targets a predeclared deep-detail band. It is not a replacement for ADR-0029 when
  global foreground or boundary PSNR is the objective.
- Dynamic storage may grow by at most 2,048 rows, but the first target-satisfying wave stops
  earlier. No global post-tail optimizer is run.
- The Janelle thresholds and geometry are development defaults for this explicit opt-in, not a
  general optimum. Independent images, replicated seeds, and rate/work-matched controls are still
  required before any recipe/default or efficiency promotion.
- The primitive and NPZ format remain unchanged. The result is a sparse allocation/solve policy,
  not a codec-rate, novel-primitive, convergence, or state-of-the-art claim.

## Development screen

On the exposed full `1200x1038` masked Janelle `frame_00008/C0001` target, seed 0, RTX 3050, the
stage reaches its first safe target at six waves and 768 rows. Deep sigma-1.5 high-pass MSE falls
`25.9262%`, Laplacian MSE falls `27.3157%`, and raw LPIPS falls `10.46%` relative. Foreground PSNR
changes only `+0.034326 dB`; boundary metrics and exact-zero outside-mask values remain safe.

The same-base ADR-0029 control uses 2,777 rows (`3.616x` as many), improves global foreground PSNR
more (`+0.320788 dB`), but contributes zero rows to the predeclared deep region and does not
materially change either fine-detail stop metric. This establishes a source-bound objective
trade-off on one exposed trajectory, not a general superiority or equal-rate result. C59/C60 and
`ara/evidence/fit040-orthogonal-detail-pursuit-janelle-2026-07-28/` bind the implementation and
measurement.
