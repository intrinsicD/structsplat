# FIT-037: Minimum-row fine-detail target

## Status

Completed negative for the frozen static allocator. No tested cohort through 2,048 rows reaches
the predeclared `25%/20%` detail target; only iterative pursuit was authorized next.

## Context

FIT-033 establishes a protected-safe 128-row high-pass/NMS plus partial-color-solve candidate
with `6.47%` deep sigma-1.5 high-pass reduction. FIT-034--036 show that spectral color
weighting, affine appearance, and residual-tangent anisotropy do not lift the safe 128-row result
past `8.90%`. The active bottleneck is therefore the amount of residual support addressed, not
the coefficient solver or local covariance.

FIT-031 added 4,608 rows without an equal-count detail metric. The remaining decision is the
smallest ordinary-Gaussian cohort that achieves a materially large fine-detail reduction.

## Goal

Measure a nested row-count/detail Pareto curve and select the minimum ordinary-row count that
crosses a frozen large-effect target.

## Acceptance criteria

- [x] Use FIT-033's frozen 0.35-pixel, opacity-0.8, 5x5-NMS high-pass sites and exact partial RGB
      solve; do not retune geometry or colors.
- [x] Evaluate nested added-row budgets `{128, 256, 384, 512, 768, 1024, 1536, 2048}` from the
      same 11,000-row persisted current-pipeline state.
- [x] Log protected metrics, three high-pass bandwidths, Laplacian, Sobel, raw deep residual,
      solver/color/render diagnostics, runtime, and exact row count.
- [x] Define success before the curve as at least `25%` deep sigma-1.5 high-pass reduction and
      at least `20%` Laplacian reduction with every protected metric safe.
- [x] Select the first nested budget satisfying both targets. If none does, close this static
      allocator as negative and authorize only a separately preregistered iterative pursuit.
- [x] Report rows relative to FIT-031's 4,608 accepted additions and the 11,000-row base.
      This exposed-image result cannot authorize a pipeline/default change.

## Depends on

FIT-031/033/034/035/036, CORE-012, BENCH-002.

## Notes

This is an exposed-image rate/detail development curve in Gaussian rows, not bytes. The current
raw format costs nine stored scalars per constant-opacity row; actual codec claims remain out of
scope.

## Result

The static nested curve saturates at 2,048 added rows with `15.01%` sigma-1.5 high-pass and
`12.04%` Laplacian reduction. No minimum passing count exists, so the stale one-shot ranking is
closed and FIT-038's separately preregistered iterative residual pursuit was authorized.
