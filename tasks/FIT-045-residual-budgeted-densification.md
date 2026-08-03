# FIT-045 — Direct-control regional allocation and merge screen

## Context

The earlier task proposed distributing a fixed residual-birth budget across tiles, but it depended
on FIT-030 and therefore put a cheap allocation mechanism behind the rate-controller design that
needs its result. It also predates LocoADC's region-wise error/gradient-coherence allocation and
similarity-driven merging. The question remains useful only as a direct-control replication and
matched mechanism screen, not as a novelty claim.

## Goal

Determine whether any regional birth/merge policy improves the selected BENCH-020 field at equal
row, proposal, optimizer-work, and wall-time budgets, and freeze at most one allocator for the
BENCH-021 recipe and FIT-030's byte-priced controller.

## Candidate arms

- `fixed_full_n`: initialize the final count and never change topology.
- `global`: current global residual ranking at the same per-event proposal/birth budget.
- `uniform_regions`: equal regional allocation followed by unchanged local site selection.
- `residual_proportional`: deterministic allocation proportional to regional residual energy,
  with a small predeclared exponent set.
- `locoadc_direct_control`: reproduce LocoADC-style regional error and gradient-coherence grouping,
  allocation, and similarity-driven merge as faithfully as the 2D field contract permits.
- `regional_without_merge` and `regional_with_merge`: isolate allocation from merge value.

Native LocoADC evidence and the local transplant are reported separately. If an exact mechanism is
unavailable from primary sources/code, label the arm approximate and prohibit a reproduction claim.

## Non-goals

- Choosing the total byte budget, topology stopping point, or precision (FIT-030 owns that).
- Inventing another residual score; FIT-017/018 supply existing within-region controls.
- Treating fixed tiles as object segmentation or changing the field/alpha semantics.
- Changing defaults or claiming regional densification/merging as StructSplat novelty.

## Acceptance criteria

- [ ] A preregistered protocol freezes region construction, update cadence, arms, proposal/birth/
      merge budgets, row and work matching, seeds, datasets/splits, metrics, and tie/failure rules.
- [ ] The direct-control provenance table maps every local LocoADC mechanism to primary paper/code
      behavior and records exact, adapted, unavailable, or omitted status before outcomes.
- [ ] Deterministic integer allocation/redistribution and merge rules have tests for zero-error,
      exhausted candidates, ties, disconnected regions, boundary masks, and exact total budgets.
- [ ] Screen reports PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, time-to-target/AUC,
      regional residual dispersion, births/merges/rows, proposals, renderer work, wall time, and
      memory on matched development arms.
- [ ] One frozen allocator/merge policy advances to BENCH-021 only if it clears predeclared quality
      and overhead guards against `fixed_full_n`, `global`, and `uniform_regions`; otherwise the
      negative result lowers FIT-030's topology scope.
- [ ] Portable report, independent results audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Residual allocation/merge hooks in the selected fitter, configuration and telemetry, direct-control
adapter/experiment driver, focused tests, maintained report, `docs/additive_field_v2.md`, this task,
and the Index.

## Depends on

FIT-017/018, BENCH-020, CORE-013, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Prior-art mapping and protocol review must occur before implementation outcomes.

## Notes

This re-scope deliberately removes the dependency on FIT-030. FIT-045 supplies controlled evidence
about where capacity should move; FIT-030 later decides whether a move is worth its complete bytes.
