# INIT-010 — Field V2 initializer transfer screen

## Context

StructSplat's structure-tensor/WSE initialization evidence was produced under normalized rendering
and specific optimizer schedules. Direct additive coefficients, optional structural mass, alpha
gating, and variable projection change early gradients and may change which geometry prior is
useful. Reusing the old winner without a transfer screen would confound the BENCH-021 convergence
recipe.

## Goal

Select one deterministic geometry initializer for the BENCH-020 field semantics at matched row,
raw-byte, initialization-time, and early-fit work budgets, or retain the simplest baseline if no
structured initializer transfers.

## Candidate arms

- regular grid and seeded uniform/random controls;
- gradient/error-density sampling available without outcome leakage;
- current quadtree-WSE and progressive WSE ordering;
- structure-tensor anisotropic on-edge placement;
- any BENCH-020-compatible initialization supplied by the incumbent native-additive control.

Appearance is initialized or conditionally solved through one identical frozen rule after geometry
creation. Final-count versus progressive-count staging belongs to FIT-048, not this task.

## Non-goals

- Inventing a new sampler, changing field/loss semantics, or using learned prediction (FF-002).
- Crediting an initializer for extra rows, fit steps, color solves, or hidden preprocessing time.
- Promoting the normalized initializer evidence as an additive claim.

## Acceptance criteria

- [ ] A preregistered protocol freezes arms, source/prepared inputs, masks, exact N/raw-byte lanes,
      seeds, appearance initialization/solve, fit work, metrics, timing boundary, and killing rule.
- [ ] Deterministic fixtures verify coordinate, covariance/orientation, alpha/mask, row-count,
      progressive-prefix, and seed contracts under the Field V2 adapter.
- [ ] Every arm receives identical post-init optimizer/loss/topology settings; initialization and
      any structure-tensor/WSE preprocessing time and memory are included.
- [ ] Report initial and early/full PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, coverage/
      boundary diagnostics, displacement/survival, iterations/time-to-target, renderer work,
      canonical bytes, init/total wall time, and peak memory.
- [ ] One exact initializer/config advances to BENCH-021 only under a frozen Pareto/noninferiority
      rule; otherwise a simple grid/native baseline is retained and the negative transfer recorded.
- [ ] Portable report, independent audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Field V2 initialization adapters/configuration, existing sampling/init modules, bounded experiment
driver, focused deterministic tests and report tooling, `docs/additive_field_v2.md`, this task, and
the Index.

## Depends on

BENCH-020, CORE-013, INIT-003/005/006/009, BENCH-002/004

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

This is a transfer test, not a new initialization method. It prevents the production recipe from
paying for structured preprocessing whose benefit disappears under additive semantics.

