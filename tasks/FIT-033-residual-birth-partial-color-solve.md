# FIT-033: Residual-birth partial color solve

## Status

Completed exposed-image mechanism positive. The exact frozen-base partial solve passes its
development gate at 128 rows, but independent-image confirmation and any pipeline/default
promotion remain unauthorized.

## Context

FIT-031's error tail assigns each new row the target color at its selected pixel. On the valid
max-side-1200 current-pipeline C0001 state, 32/64/128 such rows are not uniformly safe or
efficient, and FIT-032's gauge-lifted dipoles reduce deep-interior high-pass residual by only
`0.13--0.16%`. The normalized renderer is linear in colors once geometry and opacity are fixed,
but adding rows changes its denominator. A solve restricted to the new rows can account for that
change while leaving the already accepted base colors untouched.

A bounded development screen found that the color solve alone improves foreground error on
FIT-031 geometry but does not materially change high-pass error. The frozen candidate therefore
also changes allocation: rank the sigma-1.5 high-pass rendering residual, retain 5x5-NMS peaks
strictly deeper than `margin + 6 px`, and add 0.35-pixel isotropic rows at opacity 0.8. These
values are fixed before the result-bearing equal-budget run.

## Goal

Test whether exact, frozen-base least-squares colors make each ordinary residual birth carry
substantially more foreground and fine-detail correction per added Gaussian.

## Acceptance criteria

- [x] Implement only under `benchmarks/`, `scripts/experiments/`, focused tests, and evidence
      documents.
- [x] Use the exact normalized-renderer denominator and an implicit `A_new.T A_new` operator;
      do not approximate the compositor as additive.
- [x] Freeze every inherited row and solve only the added cohort with deterministic regularized
      conjugate gradients.
- [x] Verify the implicit operator and solve against a materialized small-fixture least-squares
      solution.
- [x] At equal net-row budgets `{32, 64, 128}`, compare the frozen high-pass allocation with
      target-pixel versus partial-solved colors, FIT-031 error-ranked allocation with both color
      rules, and the strongest FIT-032 arm on the same persisted current-pipeline C0001 state.
- [x] Log conditioning/residual diagnostics, color range, protected metrics, high-pass residual
      MSE, containment, runtime, and exact row count.
- [x] Advance only to independent-image confirmation if the solved high-pass cohort obtains at
      least `2x` the immediate foreground-MSE reduction per row of its identical-geometry
      target-color control at two budgets, reduces deep-interior sigma-1.5 high-pass residual by
      at least `5%` at one budget no larger than 128, and passes every protected metric.
      Otherwise close it as negative. This exposed-image gate cannot authorize a pipeline or
      default change.

## Depends on

FIT-005, FIT-017, FIT-025, FIT-031, FIT-032, CORE-012, BENCH-002.

## Notes

This is an allocation/initialization mechanism for ordinary serialized Gaussians, not a new
primitive, codec, or general convergence claim. The exposed C0001 state is development evidence
only.

## Result

The solved high-pass cohort beats its identical-geometry target-color control by more than `2x`
at all three budgets, remains protected-safe, and reaches `6.473%` deep sigma-1.5 high-pass
reduction at 128 rows. This validated the exact partial solve and seeded the bounded FIT-034--039
search; it is not independent-image confirmation.
