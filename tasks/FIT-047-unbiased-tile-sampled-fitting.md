# FIT-047 — Unbiased tile-sampled fitting

## Context

Full-frame loss and backward work are repeatedly paid even when only a minority of tiles contains
high residual or currently useful support. Biased hard-example cropping can converge quickly to
the wrong image, especially at masks and low-texture regions. A probability-recorded tile sampler
can focus compute while retaining an unbiased full-image objective through inverse-propensity
weighting and periodic full evaluations.

## Goal

A deterministic, default-off tile-sampled optimizer whose gradient estimator is empirically
unbiased, whose work accounting is exact, and whose quality-time frontier is tested against
full-frame fitting.

## Non-goals

- Changing the selected field equation, inventing a new perceptual objective, or using sampled
  validation metrics as stopping evidence.
- Hiding tile-index construction, probability updates, or periodic full evaluations from timing.
- Promoting a sampler that wins only by evaluating fewer pixels at an unmatched quality target.

## Method and comparisons

- Build/cache the Gaussian-to-tile incidence needed by the selected additive renderer and record
  all setup/update cost.
- Sample from a frozen mixture of uniform coverage and residual/gradient-informed probabilities;
  record inclusion probabilities and apply the exact inverse-propensity estimator.
- Define mask and crop sampling at the same authoritative coordinate boundary as full-frame loss.
- Compare full-frame, uniform-tile, biased hard-tile diagnostic, and unbiased adaptive sampling at
  equal sampled pixels, renderer work, and wall time. Periodic full-frame evaluation is mandatory.

## Acceptance criteria

- [ ] Selecting every tile reproduces full-frame loss and gradients within frozen tolerances on
      CPU oracle and accelerated paths.
- [ ] Every sampled training term has an explicit per-pixel/tile decomposition and inclusion
      probability; non-decomposable terms are evaluated in full or excluded from the sampled arm,
      never approximated under the “unbiased” label.
- [ ] Monte Carlo fixtures demonstrate the estimated loss/gradient mean agrees with the exact
      full-frame result; variance, probability floors, and weight clipping (if any) are reported.
- [ ] Seeded sample streams, probability maps, masks, and tile-index updates replay exactly;
      zero-probability authoritative pixels are impossible by construction.
- [ ] Telemetry reports sampled/full pixels, tile-index setup and refresh time, renderer calls,
      wall time, peak memory, convergence curves, final full-frame metrics, and the BENCH-019
      downstream objective.
- [ ] A frozen development/confirmation screen identifies one sampler for BENCH-021 or records a
      negative result; exact work and quality noninferiority rules are set before outcome access.
- [ ] Focused tests, portable report/audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

Tile-index/sampler/fit modules under `src/structsplat/`, additive renderer hooks, telemetry,
focused statistical/parity tests, bounded experiment driver and maintained report,
`docs/additive_field_v2.md`, this task, and the Index.

## Depends on

BENCH-020, CORE-013, BENCH-002, ADR-0024

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

The sampler is a compute policy, not a new scientific loss. Any clipped or self-normalized
estimator must be labelled biased and cannot satisfy the primary arm without a separate argument.
