# FIT-038: Orthogonal fine-detail pursuit

## Status

Completed negative for 5x5 cross-wave exclusion. Iterative remeasurement improves the static
curve but misses the frozen target through 2,048 rows; only the FIT-039 exclusion killing ablation
was authorized.

## Context

FIT-037's one-shot nested high-pass ranking saturates: 2,048 added rows reduce deep sigma-1.5
high-pass MSE by only `15.01%` and Laplacian MSE by `12.04%`, below the frozen `25%/20%`
large-effect target. The ranking is stale after the first corrections because it never observes
the residual induced by its own accepted cohort.

The normalized renderer is linear in colors at fixed geometry. A matching-pursuit tail can add a
small batch, exactly re-solve every detail-row color, recompute the rendered residual, and spend
the next batch only on still-unrepresented neighborhoods.

## Goal

Find the minimum 128-row increment at which iterative residual pursuit reaches FIT-037's frozen
large-effect detail target.

## Acceptance criteria

- [x] Reuse FIT-033's 0.35-pixel, opacity-0.8, 5x5-NMS births and exact partial RGB solve.
- [x] Start from the same persisted 11,000-row field; add exactly 128 rows per stage and jointly
      re-solve all accumulated detail-row colors while inherited rows stay frozen.
- [x] Recompute sites from the current exact render after every solve. Forbid a 5x5 neighborhood
      around every prior site so duplicate basis columns cannot masquerade as capacity.
- [x] Stop at the first protected-safe stage reaching at least `25%` deep sigma-1.5 high-pass
      reduction and `20%` Laplacian reduction, or after 2,048 rows.
- [x] Log every stage's protected vector, three high-pass bandwidths, Laplacian, Sobel, raw deep
      residual, solver/color/render diagnostics, runtime, row identities, and exact count.
- [x] Compare the reached count with FIT-037's static curve and FIT-031's 4,608 additions. If no
      stage reaches the target, close ordinary-Gaussian pursuit as negative.
- [x] This exposed-image result cannot authorize a pipeline/default change.

## Depends on

FIT-031/033/037, CORE-012, BENCH-002.

## Notes

This is an allocation schedule for existing ordinary serialized Gaussians, not a new primitive,
codec, convergence theorem, or general-quality claim.

## Result

At 2,048 rows, iterative pursuit with radius-2 prior-site exclusion reaches `20.22%` sigma-1.5
high-pass and `16.21%` Laplacian reduction. This is better than FIT-037 but still below both frozen
targets; the over-strong cross-wave exclusion became FIT-039's sole remaining variable.
