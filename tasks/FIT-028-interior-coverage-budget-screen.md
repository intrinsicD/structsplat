# FIT-028: Measure the interior coverage budget and decide its default

## Status

Mechanism implemented (ADR-0026), default `0.0` — behaviour unchanged. **The measurement below
has not run.** No claim is authorized until it does.

## Context

ADR-0026 records the finding that motivates this task: in the BENCH-017 exploratory pass,
`interior_holes_regressed` accounted for **82 of 110 rejected blocks (75%)**, and `safe_polish`
committed **0 of 3,276** attempted steps across 7 of 7 images while every pixel-error metric
improved in every rejected block.

That is a diagnosis of *waste*, not evidence that spending coverage is *better*. A block whose
pixel error improves while interior holes grow may be a genuine improvement the strict gate
refuses, or it may be the optimizer discovering that deleting coverage is a cheap way to reduce
mean error. Those two are indistinguishable in the rejection log and are distinguished only by
measuring terminal quality at a fixed budget.

## Goal

Decide whether `hole_regression_budget` should be nonzero in the ADR-0025 recipe, and if so, at
what value and for which arm.

## Approach (preregister before running)

1. **Arms**: `hole_regression_budget` in `{0.0, 1e-4, 5e-4, 2e-3}`, everything else at the shipped
   recipe. Both arms of ADR-0025 — full-frame on Kodak-24, masked on the dome fixture — because
   the interior/boundary distinction only exists in the masked case.
2. **Equal budget**: same `capacity`, same `step_scale`, same seed set (>= 3 seeds). Report
   terminal *and* Pareto-selected state.
3. **Report per cell**: PSNR / MS-SSIM / LPIPS, terminal `interior_hole_fraction`, attempted and
   accepted steps, wall-clock, and the rejection-reason histogram.
4. **Frozen gate**, declared before the first fit: a nonzero budget is promoted only if it wins
   mean PSNR with a 95% CI excluding zero, does not lose MS-SSIM or LPIPS beyond a declared slack,
   **and** terminal `interior_hole_fraction` stays below a declared absolute ceiling. Recovering
   wasted steps is not by itself a reason to promote — the terminal image is the deliverable.

## Acceptance criteria

- [ ] Preregistered protocol, arms, and gate committed before any target is fitted.
- [ ] All cells complete with per-row config/provenance (BENCH-002 rules).
- [ ] Frozen gate evaluated once; the decision recorded in `ara/logic/claims.md` either way.
- [ ] If promoted: `pipeline.py` `RECIPE` and `PipelineConfig` updated in one commit, citing the
      claim, per the ADR-0025 update rule. If not: ADR-0026 amended to say the knob stays off.

## Depends on

ADR-0026, ADR-0025, BENCH-002, BENCH-017, FIT-023

## Notes

The observation that motivated this is `ara/staging/observations.yaml` O87, which is staging, not
a claim. Do not cite the 75% figure as evidence *for* the budget — it is evidence that the gate
discards work, which is the reason to run the experiment, not its result.

A negative result is a good outcome and would sharpen FIT-030: if buying pixel error with
coverage loses at every budget, then coverage is not the thing to relax and the real fix is
proposal-time repair.
