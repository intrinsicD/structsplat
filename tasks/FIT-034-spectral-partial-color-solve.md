# FIT-034: Spectral partial color solve

## Status

Completed negative on the exposed Janelle state. Spectral weighting does not materially beat
the ordinary RGB solve and misses the frozen `15%/10%` gate; no pipeline/default change is
authorized.

## Context

FIT-033's 128-row high-pass/NMS geometry plus an ordinary frozen-base RGB color solve passes its
development gate and reduces deep sigma-1.5 high-pass residual MSE by `6.47%`. Orthogonal
sigma-0.75, sigma-3, Laplacian, and raw deep-residual metrics move in the same direction, but
the effect is not yet the order-of-magnitude detail improvement motivating this line.

Once geometry and opacity are fixed, the normalized renderer is linear in color. The same
implicit normal-equation machinery can therefore solve the new colors after a linear high-pass
observation operator. A small raw-RGB term keeps the solution anchored to total reconstruction
quality; inherited rows remain exactly frozen.

## Goal

Test whether a high-pass-observation partial color solve extracts substantially more fine-detail
correction from the same 32/64/128 ordinary-Gaussian cohorts than FIT-033's RGB solve.

## Acceptance criteria

- [x] Implement only under `benchmarks/`, `scripts/experiments/`, focused tests, and evidence
      documents.
- [x] Use the exact normalized-renderer denominator and the exact transpose of a symmetric,
      zero-padded Gaussian high-pass observation; do not approximate composition as additive.
- [x] Freeze inherited rows and solve only the new cohort with deterministic regularized
      conjugate gradients.
- [x] Verify the implicit weighted system against a materialized small-fixture solve.
- [x] Use a bounded exposed-image screen only to select the raw-RGB safety weight, then freeze it
      before an equal-budget `{32, 64, 128}` comparison against FIT-033's RGB solve and target
      colors on identical geometry.
- [x] Log solver residuals, color/render range, fixed-point drift, protected metrics, three
      high-pass bandwidths, Laplacian, Sobel, raw deep residual, runtime, and exact row count.
- [x] Advance only to independent-image confirmation if at no more than 128 rows the spectral
      solve reduces deep sigma-1.5 high-pass MSE by at least `15%`, Laplacian MSE by at least
      `10%`, passes every protected metric, and keeps new raw colors within `[-2, 2]`.
      Otherwise close it as negative. This exposed-image gate cannot authorize a pipeline or
      default change.

## Depends on

FIT-005, FIT-017, FIT-025, FIT-031, FIT-032, FIT-033, CORE-012, BENCH-002.

## Notes

This remains an initialization/allocation mechanism for ordinary serialized Gaussians. It is
not a new primitive, codec, convergence result, or broad quality claim.

## Result

The best protected 128-row spectral arm reaches `6.548%` sigma-1.5 high-pass and `7.515%`
Laplacian reduction, essentially tied with FIT-033 and far below the gate. Objective weighting is
not the active bottleneck.
