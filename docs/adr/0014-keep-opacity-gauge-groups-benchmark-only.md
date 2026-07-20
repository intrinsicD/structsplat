# ADR-0014: Keep opacity-gauge groups benchmark-only

## Context

FIT-019 tested an exact symmetry of StructSplat's normalized renderer. Replacing one Gaussian by
co-located copies whose opacity fractions sum to the original preserves the rendered image, but a
row-wise top-k allocator can give the copies multiple selection tickets. Aggregate-first scoring
over externally supplied exact groups restores canonical actions.

The frozen procedural guard confirmed this commutation property on all 16 target/seed checkpoints.
It did not confirm recovery utility. Against raw alpha-1 selection in the gauge view, quotient
selection gained `+0.2111 dB` after 20 fresh-optimizer steps but won only 5/8 target families and
fell to `-0.6007 dB` after 100. It also missed the post-20 floor against canonical support. A
source-frozen replay reproduced every non-timing value exactly.

Ordinary split siblings are not exact gauge classes: moment-preserving splitting changes means,
scales, and support. Production lineage would therefore be approximate state with maintenance,
codec, and operator semantics not justified by the exact-equivalence result.

## Decision

Do not add equivalence-group or quotient-allocation state to production from FIT-019; keep exact
opacity-gauge grouping in the benchmark and investigate recovery trajectories before reconsidering
an approximate grouping mechanism.

## Consequences

+ `GaussianField`, the fitter, CLI, renderer, defaults, and codec remain unchanged.
+ The exact refinement and aggregate-first scorer remain reusable correctness oracles in
  `benchmarks/gauge_equivalence_audit.py`.
+ Gauge invariance may be required as a correctness property for a future allocator, but it is not
  itself evidence of better quality, convergence, performance, or compression.
+ The next quality/convergence experiment should control distinct-site coverage and measure dense
  perturb--recover trajectories with fresh preregistered data.
- No natural-image confirmation, lineage metadata, or approximate sibling grouping is authorized
  by FIT-019.

## Links

Depends on FIT-018 and FIT-019. It does not supersede an earlier ADR because quotient state was
never a shipped architecture decision.
