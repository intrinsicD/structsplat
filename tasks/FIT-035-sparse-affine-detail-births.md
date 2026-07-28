# FIT-035: Sparse affine detail births

## Status

Completed negative on the exposed Janelle state. The richer affine cohort improves the local
screen but misses the frozen gate and costs extra coefficients; no format, codec, pipeline, or
default change is authorized.

## Context

FIT-033's 128 constant-color high-pass births reduce deep detail error by `6.47%`. FIT-034 shows
that directly changing the color objective reaches only `6.55%`, so constant-color basis rank,
not objective mismatch, is the active bottleneck. CORE-006 already provides scale-normalized
local affine RGB gradients per Gaussian and NPZ round-trip support, although the v1 codec rejects
them and the maintained exact color solver intentionally fails closed.

BENCH-009's inherited-row affine packets are a negative/unavailable lineage and remain closed.
This task tests a different finite action: newly allocated, high-pass-selected ordinary
footprints with jointly solved constant and affine coefficients. It does not retune BENCH-009's
selector or promote its claims.

## Goal

Test whether local affine RGB on only the residual detail cohort produces a much larger
fine-detail reduction at the same Gaussian count.

## Acceptance criteria

- [x] Implement only under `benchmarks/`, `scripts/experiments/`, focused tests, and evidence
      documents.
- [x] Match the repository's normalized affine renderer exactly, including opacity, support
      fade, scale-normalized rotated local coordinates, and the shared denominator.
- [x] Verify implicit apply/transpose adjointness, native render parity, and the solve against a
      materialized small fixture.
- [x] Freeze inherited rows; solve only each new row's constant RGB and two local RGB gradients
      with deterministic regularized conjugate gradients.
- [x] Use a bounded exposed-image screen for footprint scale and gradient ridge only, then freeze
      them before equal-row `{32, 64, 128}` comparison.
- [x] Compare equal Gaussian count against FIT-033 constant births and disclose raw scalar cost:
      15 scalars per affine row versus 9 per current constant-opacity row. Include a
      scalar-matched constant-row diagnostic.
- [x] Log protected metrics, three high-pass bandwidths, Laplacian, Sobel, raw deep residual,
      coefficient/render range, fixed-point drift, runtime, exact row count, and renderer A/A.
- [x] Advance only to independent-image confirmation if at no more than 128 affine rows the
      candidate reduces deep sigma-1.5 high-pass MSE by at least `15%`, Laplacian MSE by at least
      `10%`, passes every protected metric, and has finite coefficients with maximum absolute
      local gradient no greater than 2. Otherwise close it as negative. This exposed-image gate
      cannot authorize a pipeline or default change.

## Depends on

CORE-006/012, FIT-005/017/025/031/032/033/034, BENCH-002/009/011.

## Notes

The optimization target is Gaussian count, not byte rate. Any positive result must remain
explicitly unavailable for codec/rate claims until affine coefficients have an actual encoded
representation and a complete-byte comparison.

## Result

The selected protected 128-row affine arm (scale `0.5`, gradient ridge `0.01`) reaches `8.895%`
sigma-1.5 high-pass and `9.723%` Laplacian reduction with bounded gradients. It misses both frozen
thresholds and uses 15 scalars per row versus nine for the ordinary constant-opacity row.

A 2026-07-28 evidence-completeness rerun found that the renderer A/A harness conflated a valid
identity result with a scientific gate pass: the strict gate returns `no_material_gain` for exact
identity. The harness now permits only that reason for A/A while still failing closed on any
protected regression. The source-bound rerun passes A/A and leaves this task's negative
disposition unchanged. See
`ara/evidence/fit031-new-method-stages-janelle-2026-07-28/run.md`.
