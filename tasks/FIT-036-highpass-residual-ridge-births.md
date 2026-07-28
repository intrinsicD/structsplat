# FIT-036: High-pass residual-ridge births

## Status

Completed negative on the exposed Janelle state. Residual-tangent anisotropy remains near the
isotropic FIT-033 result and misses the frozen gate; no pipeline/default change is authorized.

## Context

FIT-033's 128 isotropic 0.35-pixel high-pass births reduce deep sigma-1.5 residual MSE by
`6.47%`. FIT-034 shows the color objective is not limiting, and FIT-035's richer affine colors
reach only `8.90%` under its frozen safety/gradient constraints. The remaining low-rate option
inside the current ordinary-Gaussian grammar is to make each footprint cover a coherent
same-sign residual ridge rather than one nearly isolated pixel.

Unlike FIT-025's target-structure detail tail, this task estimates tangent and coherence from the
current high-pass rendering residual at already frozen high-pass/NMS sites. It retains ordinary
constant-color serialized Gaussians and the exact partial RGB solve.

## Goal

Test whether residual-tangent anisotropy substantially increases fine-detail reduction per
ordinary added Gaussian without violating mask containment.

## Acceptance criteria

- [x] Implement only under `benchmarks/`, `scripts/experiments/`, focused tests, and evidence
      documents.
- [x] Reuse FIT-033's deterministic high-pass/NMS sites and exact partial RGB solve.
- [x] Set the short scale to 0.35 pixels and derive only the long scale and angle from local
      high-pass residual coherence/tangent; preserve exact mask containment.
- [x] Use a bounded exposed-image screen for maximum long scale and coherence exponent, then
      freeze them before equal-row `{32, 64, 128}` comparison.
- [x] Log protected metrics, three high-pass bandwidths, Laplacian, Sobel, raw deep residual,
      constraint deltas, color/render range, runtime, and exact row count.
- [x] Advance only to independent-image confirmation if at no more than 128 ordinary rows the
      candidate reduces deep sigma-1.5 high-pass MSE by at least `15%`, Laplacian MSE by at least
      `10%`, and passes every protected metric. Otherwise close it as negative. This exposed
      image cannot authorize a pipeline or default change.

## Depends on

FIT-017/025/031/032/033/034/035, CORE-010/011/012, BENCH-002.

## Notes

This is a geometry allocation rule for the existing serialized primitive, not a new texture
carrier, codec, general convergence result, or novelty claim.

## Result

The best protected 128-row ridge arm (maximum long scale `1.0`, coherence power `2`) reaches only
`6.565%` sigma-1.5 high-pass and `7.506%` Laplacian reduction. Geometry anisotropy is not the
active bottleneck at this cohort size.
