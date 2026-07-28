# FIT-039: Detail-pursuit exclusion radius

## Status

Completed exposed-image development positive. Exact-site-only cross-wave exclusion reaches the
frozen target first at 768 rows; this authorized only FIT-040's separate default-off integration.

## Context

FIT-038's 5x5 prior-site exclusion improves the 2,048-row result over FIT-037's static ranking
from `15.01%/12.04%` to `20.22%/16.21%` sigma-1.5/Laplacian reduction, but misses the frozen
`25%/20%` target. A 5x5 exclusion is stricter than basis uniqueness: adjacent 0.35-pixel
Gaussians are distinct and may be required for neighboring lobes of fabric detail.

## Goal

Determine whether exact-site-only or 3x3 prior-site exclusion reaches the existing target without
more than 2,048 rows.

## Acceptance criteria

- [x] Reuse FIT-038 unchanged except for prior-site exclusion radius `{0, 1}`; radius `2` is the
      already executed control.
- [x] Keep 128-row stages, within-stage 5x5 NMS, joint exact color re-solve, inherited-row freeze,
      geometry, opacity, metrics, protected gate, and `25%/20%` target identical.
- [x] Stop each radius at its first target pass or 2,048 rows and record all row identities and
      diagnostics.
- [x] Select the smallest passing row count, breaking ties in favor of larger exclusion. If no
      arm passes, close this selector family as negative and require a new primitive/format for
      larger gains.
- [x] This exposed-image result cannot authorize a pipeline/default change.

## Depends on

FIT-038, FIT-037, FIT-033, CORE-012, BENCH-002.

## Result

Radius 0 reaches the first protected-safe target at six 128-row waves and 768 unique sites:
`25.926%` sigma-1.5 high-pass and `27.316%` Laplacian reduction. Radius 1 needs more rows and the
radius-2 control fails by 2,048. Cold replay, fixed-point, containment, perceptual, and spatial
audits pass; no default or generality promotion follows.
