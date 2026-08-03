# FIT-046 — Additive appearance variable projection

## Context

For fixed Gaussian geometry, a direct additive RGB-coefficient renderer is linear in appearance.
The current all-Adam fit ignores that structure and spends many expensive raster/backward passes
recovering coefficients that can instead be solved as a bounded least-squares subproblem. A dense
pixel-by-Gaussian design matrix would erase the memory advantage, so the useful question is
whether a matrix-free, renderer-backed solve improves time-to-quality at production scale.

## Goal

A default-off matrix-free variable-projection method for the semantic candidate selected by
BENCH-020, with measured convergence benefit over matched all-Adam controls and no dense design
matrix materialization.

## Non-goals

- Selecting field semantics, alpha policy, topology changes, or a compressed format.
- Solving geometry in closed form or claiming that the full non-convex problem becomes convex.
- Replacing the maintained optimizer before BENCH-021 and BENCH-022 confirm an end-to-end win.

## Method and comparisons

- Expose renderer-backed forward and adjoint products for the linear appearance block; include
  masks, alpha/DC policy, selected coefficient domain, regularization, and color-space conventions
  in the operator contract.
- Implement a numerically guarded iterative least-squares solver without a dense `P x N` matrix.
  If structural mass is independently supervised, keep it in a separately constrained block.
- Compare solve-once initialization, alternating geometry/appearance solves, every-`K` solves,
  and final polishing against all-Adam at equal renderer calls, sampled pixels, and wall time.
- Record conditioning, solver residual, iterations, operator calls, and fallbacks. Fail closed to
  the last finite field when an inner solve is singular or exceeds its frozen budget.

## Acceptance criteria

- [ ] Tiny dense fixtures agree with an explicit least-squares oracle in value and coefficients;
      adjoint tests pass in float64 and production dtype tolerances are documented.
- [ ] Production code never allocates a dense pixel-by-Gaussian matrix; peak-memory tests and
      telemetry make this checkable at representative dimensions.
- [ ] The solver preserves finite parameters, masks, alpha semantics, deterministic replay, and
      import boundaries; constrained quantities cannot cross their declared domains and signed
      fields are never passed through a nonnegative-only solver.
- [ ] A pre-registered screen reports PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective,
      time-to-target, PSNR-time AUC, renderer/operator calls, wall time, and peak memory against
      matched all-Adam arms on development and sealed-confirmation data.
- [ ] BENCH-021 receives one frozen schedule or an explicit negative result; no post-hoc schedule
      selection is hidden in the production profile.
- [ ] Focused tests, report/audit artifacts, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Additive operator/optimizer modules under `src/structsplat/`, fit configuration and telemetry,
focused numerical/performance tests, bounded experiment driver and report bundle,
`docs/additive_field_v2.md`, this task, and the Index.

## Depends on

BENCH-020, CORE-013, FIT-005/010, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

The first implementation should prefer a small, inspectable matrix-free method over a general
solver framework. Promotion is evidence-gated; a correct method that does not improve the
quality-time frontier remains a research result, not production complexity.
