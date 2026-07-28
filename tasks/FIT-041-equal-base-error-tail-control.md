# FIT-041: Equal-base error-tail control

## Status

Completed same-base control on the exposed full-frame Janelle target. The audit passes and shows
that FIT-031's tail wins global foreground PSNR while FIT-040 wins the predeclared fine-detail
objective with 3.616x fewer rows; no general or equal-rate claim is authorized.

## Context

FIT-031's committed evidence used a `1200x437` cropped target, while FIT-039/040 use the requested
full `1200x1038` masked frame. The raw `4,608` versus `768` addition counts therefore cannot
support an efficiency comparison. A same-base control is required before judging whether the
new pursuit is more row-efficient than the user's latest-commit error-only tail.

## Goal

Measure both terminal tails from the same 11,000-row field, decoded target pixels, mask, renderer,
and protected metric baseline.

## Frozen protocol

- Reuse `runs/fit032_current_base_20260728/runs/current/C0001/seed_0`.
- Disable every ordinary schedule phase because the persisted field is already post-polish.
- Make the schedule-entry color solve an exact float32 no-op with one iteration and finite
  `1e30` ridge, matching FIT-040's replay adapter.
- Run the shipped FIT-031 defaults unchanged: fraction `0.5`, 512-row maximum batches, eight-row
  minimum, `1.25` maximum scale, full protected gate, and fixed-point convergence phase.
- Recompute the same deep sigma-1.5 high-pass and Laplacian metrics used by FIT-039/040.
- Compare rows, detail reductions, protected results, and measured tail time to the already
  source-bound FIT-040 result. Do not reinterpret wall time as a general performance result.

## Acceptance criteria

- [x] Base field, target pixels, mask, row count, and no-op constraint hashes/checks match FIT-040.
- [x] Persist the full error-tail estimator, requested/activated rows, allocation/convergence
      reasons, protected metrics, fine-detail metrics, field hash, history, and environment.
- [x] State the direction of the same-base row/detail comparison without generalizing beyond this
      exposed image, seed, GPU, and trajectory.
- [x] Carry the result into FIT-040's evidence note and claim boundary before closing either task.

## Depends on

FIT-031/039/040, CORE-012, ADR-0029.

## Result

All field/target/mask/constraint bindings match FIT-040. FIT-031 activates all 2,777 requested
rows and gains `0.320788 dB` foreground PSNR, but all 2,777 centers remain within 2.24 px of the
mask boundary: deep sigma-1.5 high-pass changes only `0.000042%`, Laplacian `0%`, and raw LPIPS
improves `3.21%` relative. FIT-040 uses 768 deep rows, gives `25.926%/27.316%` detail reductions
and `10.46%` LPIPS improvement, but less global PSNR. The adversarial JSON audit passes.
