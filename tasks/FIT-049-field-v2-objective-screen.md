# FIT-049 — Field V2 objective and loss screen

## Context

The live arms cannot identify loss quality because they change renderer, mask policy, containment,
topology, and commit gates together. Current StructSplat runs also do not use spherical harmonics;
their density-like terms govern coverage/proposals rather than SH color. Once BENCH-020 fixes the
field semantics, the training objective can be tested without mistaking a representation change
for a loss improvement.

## Goal

Select one explicit image/structure/downstream training objective and checkpoint guard for the
Field V2 convergence portfolio, or retain masked/matted RGB MSE when no additional term earns its
quality, downstream, convergence, and compute cost.

## Sequential candidate screen

1. At frozen geometry/topology and appearance initialization, compare RGB MSE, L1 plus the existing
   SSIM weight, and a predeclared Charbonnier setting.
2. Confirm the appearance-loss finalists under short geometry optimization at equal renderer work.
3. Only if BENCH-019 validates an independently defined structural target or differentiable
   downstream surrogate, add that term to the appearance winner in a separately isolated arm.
4. Keep LPIPS/MS-SSIM as evaluation/checkpoint guards unless BENCH-019 and a frozen cost pilot
   authorize a direct training term; do not add all metrics to one weighted blend.

## Non-goals

- Changing field semantics, alpha/target matting, stage order, topology, or coefficient domain.
- Calling initialization density, proposal ranking, or raw-denominator floors a color/SH loss.
- Outcome-visible weight tuning or a composite score whose trade-offs cannot be inspected.

## Acceptance criteria

- [ ] Protocol freezes exact target/matting/color space, candidate equations/weights, geometry and
      topology state, seeds, work/time budgets, metrics, checkpoints, missing policy, and sequential
      elimination rules before target access.
- [ ] Closed-form and autograd fixtures verify every term, mask/alpha denominator, zero-area case,
      gradient finiteness, scale with resolution/batch, and declared interaction with coefficient
      solves; no loss silently clamps render values before authoritative evaluation.
- [ ] Frozen-geometry screen isolates objective value; short full-fit confirmation reports
      PSNR/MS-SSIM/LPIPS, BENCH-019 downstream response, boundary/coverage diagnostics,
      time/render-work-to-target, AUC, final quality, loss-term time, and peak memory.
- [ ] Structural/downstream terms are present only with the BENCH-019 target definition and are
      compared as one-factor additions to the appearance winner with explicit image-quality guards.
- [ ] One exact objective/config advances to FIT-048/BENCH-021, or RGB MSE remains the candidate;
      negative losses and expensive metric terms are not retained as precautionary complexity.
- [ ] Portable report, independent audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Field V2 loss/config and metric adapters, fit telemetry, bounded experiment driver, numerical and
mask/alpha tests, maintained report, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

BENCH-019/020, CORE-013, FIT-012/016, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

The default hypothesis is intentionally simple: RGB MSE for a PSNR-oriented additive field, with
perceptual/downstream metrics as selection guards. The task exists to falsify or retain that
hypothesis under matched semantics.

