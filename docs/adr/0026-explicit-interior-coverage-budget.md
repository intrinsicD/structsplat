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

## Outcome — 2026-08-08 (masked arm): the knob stays off

FIT-028's masked arm has run. **`hole_regression_budget` remains `0.0` in the ADR-0025 recipe.**
Per FIT-028's acceptance criteria this section is the required amendment for a non-promotion.

The mechanism is sound. Across `{0.0, 1e-4, 5e-4, 2e-3}` on masked Janelle `frame_00008/C0001` at
max-side 1200, capacity 11,000, seeds 0/1/2, step acceptance rises monotonically
`8.71 -> 9.26 -> 9.53 -> 10.48%` and rejected blocks citing `interior_holes_regressed` collapse
`63 -> 18 -> 6 -> 0`. The budget does exactly what this ADR specified.

It buys nothing. No arm wins the frozen PSNR gate (paired deltas `+0.0865 / -0.0683 / +0.2116 dB`,
all 95% intervals containing zero, response non-monotonic), and **the number of rejected blocks does
not move**: `73 -> 68 -> 72 -> 72`. At `2e-3` the interior-hole veto is gone entirely and discarded
work is unchanged, because rejections migrate to `cvar99_mse_regressed`, whose sole-cause count rises
`6 -> 39`. Substitution is complete.

This refutes the Context section's causal reading. Only 4 of 73 baseline rejected blocks (5.5%) were
vetoed by the hole term *alone*, which bounded the recoverable set before any arm ran — a Pareto gate
with several protected terms cannot be repaired by relaxing one of them, because the binding
constraint simply moves. The interior-hole veto is a **symptom**; on the masked arm the cause is the
CVaR99 tail guard together with the boundary pixel-error terms.

`budget2e3` is instructive: largest PSNR point estimate, only nominal LPIPS gain, highest acceptance —
and the only arm retaining terminal interior holes (`0.00131%` versus `0.00000%`). It breaches
FIT-028's pre-declared guardrail, so its gain is at least partly bought by deleting coverage, which is
the failure mode that task existed to detect.

The Consequences section already called this "the *cheap* half of the fix" and named FIT-030 —
proposal-time coverage feasibility with local re-seed repair — as the principled form. That reasoning
is strengthened, not weakened: the cheap half is measurably inert on this arm, so the remaining value
is in FIT-030 and in whatever addresses the tail guard.

Scope: one exposed development image, one capture group, three seeds, one GPU, provisional self-review
only. The **full-frame Kodak-24 arm is still unscreened**, so this amendment governs the masked arm and
the default; it is not a general result. See `C64`, `C65`, and
`ara/evidence/fit028-hole-budget-janelle-2026-08-08/run.md`.

## Correction — 2026-08-08 (surface and accounting)

Two defects in the text above were found while executing FIT-028's masked arm. Neither changes the
mechanism or the `0.0` default; both change what this ADR may be cited for.

**1. The promised CLI flag was never implemented.** The Decision section lists
`structsplat convert --hole-regression-budget` alongside the two Python surfaces. The Python
surfaces exist as stated (`safe_commit_decision(..., hole_regression_budget=...)`,
`SafeScheduleConfig.hole_regression_budget`, `PipelineConfig.hole_regression_budget`). The CLI flag
does not exist anywhere in `src/`, `scripts/`, or `benchmarks/`, and `scripts/convert.py` — the sole
conversion CLI under ADR-0025 — does not expose it. The budget is reachable only from Python, from
`PipelineConfig.schedule_overrides`, or from the `hole_budget` stage registered in
`workflows.STAGE_VARIANTS` for FIT-028. Adding a knob to the deliberately minimal ADR-0025
entrypoint is a separate decision and is not made here.

**2. The headline rejection table is a reason-occurrence count, not a block count.** The five counts
sum to exactly 110, which is the signature of counting reason *occurrences*: one rejected block
records a list of reasons and can cite several at once. The surrounding prose reads "82 of 110
rejected **blocks**" and "75% of all discarded work is attributable to one term", which are
block-level and attribution claims that an occurrence total cannot support. No BENCH-017 evidence
bundle exists — there is no `ara/evidence/bench017-*` and no persisted rejection log — so the
original footing cannot be recovered.

The distinction is material, not pedantic. On FIT-028's masked-arm baseline cell
(`frame_00008/C0001`, seed 0, max-side 1200) the two footings differ by roughly 2x: 44 reason
occurrences across 23 rejected blocks. `interior_holes_regressed` is cited by 19 of those 23 blocks
(82.6%), but it is the **sole** reason in only 4 (17.4%); the other 15 are co-vetoed, mostly by
`cvar99_mse_regressed`. Relaxing one term cannot revive a block that several terms rejected, so the
sole-reason count — not the citation count — bounds what a budget can recover. Cite this ADR for the
mechanism and the design argument, not for the 75% attribution.

Both the occurrence and block footings, plus the sole-reason subset, are now reported by
`workflows._gate_telemetry` and
`scripts/experiments/fit028_bench018_gate_screen_report.py`, labelled so they cannot be conflated
again.

## References

- FIT-028 (the measurement that would authorize a nonzero default), FIT-029 (`safe_polish`),
  FIT-030 (proposal-time coverage), BENCH-018 (`block_steps` granularity).
- ADR-0025 (the entrypoint whose recipe would carry any promoted default).
- FIT-023 / C50 (Pareto-safe checkpointing, the part of the gate that is working).
