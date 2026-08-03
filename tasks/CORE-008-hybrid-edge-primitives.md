# CORE-008: Hybrid Gaussian + frequency-bearing primitive control

**Status: todo, conditional and off the Field V2 critical path.** WIPES is the direct
frequency-bearing primitive baseline. BENCH-007's texture regression is allocation-specific—the
same Gaussian basis under the gradient control performs better—so it does not establish a
primitive-family failure. Reopening requires BENCH-022 to expose a repeatable basis-limited
residual after the base field/codec are frozen.

## Context
Some high-frequency edges and textures are inefficient for blob-only Gaussian bases. WIPES already
demonstrates localized wavelet visual primitives, so a Gaussian-windowed edge/DoG/Gabor arm is
scientifically useful only as a controlled basis-family experiment with actual bytes—not as a
generic claim that frequency-bearing primitives are new.

## Goal
Identify whether StructSplat's residual failure is caused by the Gaussian basis lacking phase/sign
degrees of freedom, after controlling for parameter count, actual stream bytes, optimization
compute, and WIPES/native wavelet evidence.

## Approach
1. Require BENCH-022 or a separately preregistered residual audit to show a repeatable
   basis-limited texture or thin-line failure at complete actual rate; BENCH-007 did not isolate
   one.
2. Reproduce or run WIPES first where official code permits. Record native evidence separately.
3. Build a synthetic linear-dictionary oracle: pure Gaussian, Gaussian derivative, Gabor, and
   localized wavelet atoms under exact coefficient/metadata costs. This is the cheapest killing
   test and does not require production renderer changes.
4. Only if a phase-bearing family clears thresholds frozen before target access on the synthetic
   oracle and disjoint image pilot should a separate versioned hybrid-field/renderer task be
   proposed. CORE-008 does not mutate Field V2 itself.

## Acceptance criteria
- [ ] BENCH-022 (or an independent preregistered audit) provides a repeatable failure regime and
      frozen image IDs that were not selected by inspecting rich-atom outcomes.
- [ ] WIPES prior-art/native-control status is recorded before implementation.
- [ ] Synthetic hard edges, junctions, thin lines, chirps, and oriented textures are evaluated with
      exact atom metadata and coefficients counted.
- [ ] Pure Gaussian, extra-Gaussian-count, derivative/Gabor, and localized-wavelet controls receive
      equal RDO search and compute.
- [ ] Killing/promotion margins, actual-bpp targets, image split, compute, and decode/query guards
      are reviewed and frozen before execution; complete COMP-013-compatible bytes are counted.
- [ ] A failed spike closes the basis-family task without modifying the production field format.
- [ ] Portable report, independent results audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched
Start as a standalone benchmark/dictionary spike. Production field, renderer, fit, codec, and ADR
changes are explicitly out of scope until the gate passes.

## Depends on
BENCH-022/025, COMP-013/014, CORE-013, BENCH-002/007. Optional: CORE-007.

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Prior-art/native-control review occurs before any result-bearing implementation.

## Notes

This is a WIPES-controlled basis-family killing test, not a promised production feature or a
novelty lane. A positive result creates a new implementation/codec task with its own ADR and
compatibility plan.
