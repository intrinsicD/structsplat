# ADR-0026: Coverage is a budgeted constraint, not an unconditional veto

## Status

Accepted (mechanism only). The shipped default is unchanged: `hole_regression_budget = 0.0`
reproduces the historical strict gate exactly. Promoting a nonzero default requires the
FIT-028 measurement and a claim row.

## Context

`safe_commit_decision` is a Pareto gate: a candidate block commits only if no protected metric
regresses. Interior hole fraction is one of those metrics, and until now it was compared with
strict monotonicity — `CommitTolerances.hole_absolute` defaults to `0.0`, and that field is
documented as numerical slack, not a trade-off budget.

The BENCH-017 exploratory pass over Kodak-24 (full-frame arm, capacity 5,000 and 11,000, seed 0,
7 images fitted) recorded the consequence. Across every rejected block:

| rejection reason | count |
|---|---:|
| `interior_holes_regressed` | 82 |
| `cvar99_mse_regressed` | 10 |
| `foreground_mse_regressed` | 7 |
| `boundary_mse_regressed` | 7 |
| `no_safe_topology_trial` | 4 |

**75% of all discarded work is attributable to one term.** The `safe_polish` phase is the clean
case: 0 of 3,276 attempted steps committed across 7 of 7 images, and in all 7 rejected blocks
*every* pixel-error metric improved. A representative block:

```
foreground_mse   0.00207991 -> 0.00204614   improved
cvar99_mse       0.027548   -> 0.026957     improved
p99_mse          0.0213841  -> 0.0209885    improved
interior_holes   0.000882   -> 0.001465     regressed  -> 468 steps discarded
```

At 768x512 that hole regression is roughly 229 pixels out of 393,216. The gate discarded a block
of strictly-better optimization to avoid it, and — because the gate operates at `block_steps`
granularity — the unit of discarded work is up to 250 steps, not one.

The design fault is that coverage is expressed as a *veto on a global metric vector* when it is
really a *constraint on the field*. A veto cannot express "repair this locally"; it can only
reject everything that happened alongside the regression.

## Decision

Add an explicit, named interior coverage trade-off budget, separate from numerical slack:

- `safe_commit_decision(..., hole_regression_budget: float = 0.0)`;
- `SafeScheduleConfig.hole_regression_budget`, `PipelineConfig.hole_regression_budget`, and
  `structsplat convert --hole-regression-budget`, all defaulting to `0.0`.

The budget is the interior hole fraction a block may add and still commit. It is deliberately
**not** placed on `CommitTolerances`, whose docstring commits that class to numerical slack only;
conflating a quality trade-off with float tolerance is what made the strict behaviour invisible.

**Boundary holes are never budgeted.** `boundary_hole_fraction` keeps its exact gate regardless of
the interior budget. The masked arm's boundary closure is the one part of the schedule with
confirmed behaviour, and a hole at the mask boundary is a different failure from a hole in open
interior.

## Consequences

- Default behaviour is bit-identical; this ADR ships a knob, not a policy change. The recipe in
  `pipeline.py` keeps `0.0` until FIT-028 measures the trade-off curve.
- The gate can now express "spend a bounded amount of coverage for pixel-error progress", which
  is what the strict form could not say at any setting.
- A nonzero budget makes coverage genuinely negotiable, so the interior hole fraction must be
  reported in any result that used one. It is already in the metric vector and in
  `<stem>_pipeline.json`.
- This is the *cheap* half of the fix. The principled form — enforce coverage as a proposal-time
  feasibility predicate with local re-seed repair, so a 229-pixel hole costs a local repair rather
  than a global rollback — is FIT-030 and remains open. A budget bounds the damage the veto does;
  it does not make coverage a constraint in the right place.

## Alternatives rejected

- **Raise `CommitTolerances.hole_absolute`.** Violates that class's stated contract and hides a
  quality trade-off inside a field readers are told is float slack.
- **Drop the hole term from the gate.** Unbounded: nothing else in the metric vector protects
  coverage, and "no holes" is a real requirement of the representation.
- **Shrink `block_steps` alone.** Reduces the *unit* of discarded work without reducing the
  *rate* of rejection. Worth measuring on its own (BENCH-018) and composes with this change, but
  it does not address the cause.

## References

- FIT-028 (the measurement that would authorize a nonzero default), FIT-029 (`safe_polish`),
  FIT-030 (proposal-time coverage), BENCH-018 (`block_steps` granularity).
- ADR-0025 (the entrypoint whose recipe would carry any promoted default).
- FIT-023 / C50 (Pareto-safe checkpointing, the part of the gate that is working).
