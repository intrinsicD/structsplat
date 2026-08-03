# FIT-048 — Additive scale/topology stage-order screen

## Context

The live conversion's bootstrap/coverage/detail/closure/redistribution/polish order is unvalidated
for additive semantics, and repeated transactional blocks are expensive. Existing HIER-003/004
evidence shows that coarse-to-fine schedules can trade early convergence against terminal quality
under normalized rendering, while SGI and EDGS motivate multiscale and full-count controls in
other Gaussian settings. None transfers automatically to Field V2.

## Goal

Select the simplest scale/topology order that reaches the selected Field V2 quality/downstream
target with the best measured work-time frontier, before BENCH-021 composes optimizer mechanisms.

## Candidate arms

- full-resolution, final-N from initialization, no densification-stage delay;
- full-resolution progressive N under one fixed proposal schedule;
- coarse-to-fine image targets with final N present from the start;
- coarse-to-fine plus progressive N, using the frozen HIER-004 schedule as the local control;
- the current transactional staged order as a diagnostic control, not the assumed baseline winner.

All arms use the same BENCH-020 semantics, INIT-010 initializer family where compatible, FIT-049
loss, row/byte cap, selected checkpoint rule, and total pixel-render/optimizer-work budgets.

## Non-goals

- Tuning the loss, initializer, allocation score, or coefficient solver inside this task.
- Claiming SGI or EDGS reproduction; they are prior-art/system controls for stage organization.
- Retaining phases whose only function is historical naming rather than measured value.

## Acceptance criteria

- [ ] A reviewed protocol freezes target scales, interpolation/filtering, stage boundaries, N
      trajectories, events, exact pixel-render/optimizer work, wall cap, seeds, data/splits,
      checkpoint/stopping rule, and elimination margins.
- [ ] A/A fixtures prove the single-scale final-N baseline matches its direct fit, scale transforms
      preserve crop/alpha coordinates, and cumulative rows/work cannot exceed the frozen budget.
- [ ] Report full-resolution PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, boundary/coverage,
      time/iterations/rendered-pixels-to-target, AUC, stage transitions, births/merges, rejected
      work, wall time, and peak memory.
- [ ] Compare arms at equal cumulative rendered pixels/parameter updates and separately at equal
      wall time; low-resolution pixels are not counted as full-resolution iterations.
- [ ] BENCH-021 receives one exact stage order or a negative result selecting full-resolution
      final-N. No stage is retained without an observable responsibility and telemetry boundary.
- [ ] Portable report, independent audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Field V2 fit/pyramid/target schedule adapters, stage telemetry/configuration, bounded experiment
driver, parity/work-accounting tests, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

BENCH-020, CORE-013, INIT-010, FIT-049, HIER-003/004, BENCH-002/004

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

This task directly answers whether the current stage order is useful. A simple full-N single-scale
winner is a positive simplification result, not a failed research outcome.

