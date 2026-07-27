# FIT-030: Rate-aware continuous allocation (design)

## Status

Design-only. This is the architectural project the other gate tasks are the cheap precursors to.
**No implementation should start before FIT-028 and BENCH-018 report** — they measure whether the
current schedule's waste is a tuning problem or a structural one, and that answer changes this
design.

## Context

Three observations from the BENCH-017 exploratory pass point at the same structural issue.

**1. Coverage is a veto, not a constraint.** 75% of discarded blocks died on
`interior_holes_regressed` (ADR-0026). A veto on a global metric vector cannot say "repair this
locally"; it can only reject everything that happened alongside the regression. Coverage is
spatially localized and cheap to repair — a hole is some pixels with no support — so paying for it
with a global rollback of up to 250 steps is a bad exchange rate by construction.

**2. Error is used as a verdict, not a map.** Every attempted step computes a per-pixel residual in
its backward pass, and the schedule collapses it to five scalars used to accept or reject. The
signal that would say *where* to place the next Gaussian is already being computed and discarded.
A per-region residual EMA is nearly free on top of the gradient already being paid for.

**3. Rate is decided implicitly by a quality mechanism.** At `capacity=11,000` the schedule placed
**8,584 rows and reported converged**. The stopping point of growth — a rate decision — is an
emergent side effect of the commit gate. For a representation whose point is compression, the
rate should be chosen, not discovered.

Related: the phase structure imposes fixed row targets at 5/11, 8/11, 10/11 of capacity, inherited
from the Janelle schedule (5,000 / 8,000 / 10,000 / 11,000 rows). Those fractions are a schedule
where an allocation policy would do better.

## Direction (not yet a decision)

Separate the three concerns the phase structure currently fuses:

- **Coverage as a proposal-time feasibility predicate.** Never accept a birth/death/prune that
  opens an interior hole; when one opens anyway, repair it locally by re-seeding that region
  rather than rolling back the block. Cost is bounded by the hole, not by the block.
- **Detail as continuous residual-driven allocation.** Maintain a per-region residual EMA from the
  existing backward pass; birth where marginal distortion reduction per **bit** is highest, kill
  where it is lowest. No phase boundaries and no fixed row targets.
- **Acceptance as a rare global safety net.** Keep FIT-023 / C50 Pareto-safe checkpointing, which
  is the part of the gate with confirmed value (+0.502/+0.519 dB). Demote or drop the per-block
  trial-and-rollback.

And make the objective rate-aware: minimize `D + lambda * R` where **R is bits, not rows**. Rows
are not equal cost — a Gaussian's bit cost depends on quantization precision for position, scale,
rotation, and color — so a rate-blind allocator will spend bytes buying negligible dB in regions
where they are worthless. Consequences: sweep `lambda` rather than `capacity` to produce an RD
curve, and let precision vary per Gaussian. `codec.py` and `benchmarks/rate_distortion.py` already
exist; it is the *fitter* that is rate-blind, not the codec.

## Open questions (answer before designing further)

- Does relaxing the veto (FIT-028) recover most of the loss? If yes, this project is a smaller
  refactor than it looks and should be scoped down.
- What is the actual cost of a per-region residual EMA at the working resolutions, measured rather
  than assumed?
- Does the Pareto-safe checkpoint alone preserve the C50 gain without the per-block gate? That is
  the load-bearing question for dropping trial-and-rollback, and it is answerable cheaply.
- Which `lambda` parameterization is stable across images? A fixed `lambda` gives variable rate per
  image; a rate target gives variable quality. The dome use case likely wants the latter.

## Relationship to the boundary

The masked arm's boundary handling is the one part of the pipeline with confirmed good behaviour,
and it generalizes: it seeds tangent-aligned Gaussians on a *known discontinuity*. Unmasked images
have discontinuities too — the structure tensor's edge/corner labels find them, and
`aniso_onedge` / flanking already target them (ADR-0004, ABL-006). The mask boundary is the special
case where the discontinuity is known exactly and is infinitely sharp. A unified treatment is
preferable to two boundary stories, but that unification is downstream of this task, not part of
it.

## Depends on

FIT-028, BENCH-018, FIT-027, ADR-0026, ADR-0025, COMP-001, COMP-003

## Notes

Everything above is motivated by a single-seed, single-arm exploratory pass on 7 images
(`ara/staging/observations.yaml` O87-O89). None of it is claimed. The purpose of FIT-028 and
BENCH-018 is to find out whether this design is solving a real problem before anyone builds it.
