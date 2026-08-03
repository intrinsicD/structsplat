# BENCH-021 — Additive convergence portfolio

## Context

BENCH-020 selects semantics, while INIT-010 and FIT-044/045/046/047/048/049 test complementary
ways to initialize the field, define the objective, order stages, and spend optimizer work.
Evaluating every cross-product at full scale would be slow and invite post-hoc storytelling;
evaluating each feature only in isolation would miss interactions. A frozen successive-halving
portfolio is the decision gate between method prototypes and one production recipe.

## Goal

Select one reproducible additive fitting recipe on a quality-time-memory-downstream frontier, or
stop the additive production branch if no candidate materially improves the matched controls.

## Non-goals

- Changing the semantic contract, codec, downstream protocol, or confirmation set after outcomes.
- Exhaustive hyperparameter optimization or a full factorial of every implementation detail.
- Treating intermediate successive-halving ranks as formal confirmation.

## Protocol requirements

- Freeze the candidate grammar and all elimination rules before target access: selected
  initialization, objective, scale/topology order, parameter-group schedule, regional allocation,
  variable projection, tile sampling, and a small set of declared interactions.
- Include incumbent native additive, normalized plain, and current normalized pipeline controls.
- Use a staged screen with metadata-selected development data and one untouched confirmation set.
  Allocate identical seeds and initial work to every arm; promote by a predeclared Pareto rule,
  not a mutable scalar score.
- Run matched fixed-row, equal-canonical-byte, equal-renderer-work, and equal-wall-time views where
  meaningful. Full accounting includes preparation, initialization, rejected work, evaluation,
  checkpointing, and method-specific setup.

## Acceptance criteria

- [ ] A reviewed protocol manifest freezes arms, interaction shortlist, datasets/splits, seeds,
      budgets, metrics, target thresholds, elimination/tie rules, missing policy, and source/env
      hashes before the first result-bearing run.
- [ ] The report includes convergence curves, iterations/time-to-target, PSNR-time AUC,
      PSNR/MS-SSIM/LPIPS, BENCH-019 downstream response, canonical bytes/rows, renderer and sampled
      work, wall time, peak memory, and failure rate for every arm.
- [ ] Successive-halving decisions can be replayed from raw rows and do not pool views as
      independent samples; development and sealed-confirmation results are visibly separated.
- [ ] One exact recipe/config digest is frozen for CORE-014, including stopping policy and safe
      fallback, or the architecture records that no additive recipe cleared the gate.
- [ ] Interaction conclusions are restricted to tested combinations; losing components are
      retired or left explicitly optional rather than silently retained.
- [ ] Portable report, independent results audit, ARA disposition, architecture/task updates, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Maintained benchmark/ablation configuration and drivers, telemetry/report tooling,
`ara/evidence/`, `docs/additive_field_v2.md`, relevant fit task files, this task, and the Index.

## Depends on

BENCH-020, INIT-010, FIT-044/045/046/047/048/049, BENCH-002/004

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Protocol and results reviewers must be distinct from the experimental Driver.

## Notes

This task is intentionally a portfolio rather than a serial chain of defaults. It is the sole
authority for composing the selected optimizer parts into a recipe before production integration.
