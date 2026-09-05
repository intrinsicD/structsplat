# ADR-0034: Keep code-driven refinements experimental after bounded utility tests

- Status: accepted (experimental interfaces and evidence disposition); defaults unchanged
- Date: 2026-09-05
- Tasks: FIT-050, PORT-007, FIT-051
- Related: ADR-0003, ADR-0011, ADR-0025

## Context

Code inspection identified rejected full color solves and repeated coverage/tail work as simple
potential improvements. Three distinctly reviewed, clean-source development studies completed
214 cells. Artifact integrity passes do not imply that a method passes its utility gate.

FIT-050 is compatibility-limited: 21/24 ray transactions abort before a direction is evaluated.
PORT-007 observes roughly 10× aggregate component ratios for coverage reuse, but exact null-gain
decisions differ even in legacy repeats and complete pipeline A/A trajectories are unstable.
FIT-051 obtains small actual-render backtracking gains, but the +0.005293 dB median is far below
its frozen +0.1 dB threshold. The native VJP provides no measured quality advantage.

## Decision

1. Retain the color-ray modules and same-call quality options as tested opt-in experimental tools.
   Keep the normalized equation, training backward, maintained schedule and defaults unchanged.
2. Close the three bounded studies as completed negative-utility/no-promotion assays, not as failed
   implementations. Do not reinterpret component timings as execution-equivalent pipeline speed.
3. Preserve original source-bound reports and all cells. A changed mechanism needs a new protocol,
   exact prospective approval and new output directory, as done for FIT-051 after FIT-050.
4. Canonical CPU reporting in FIT-051 is a pre-run backend contract, not permission to compare its
   perceptual values directly with FIT-050's older reporting backend or widen frozen tolerances.
5. Retain only invariant mask geometry in the post-run PORT validator cache, scoped to one check.
   Every raw image, full quality vector and gate decision is still independently recomputed.
   This checker-only optimization is not measured method performance.

## Evidence and consequences

This disposition explicitly depends on staged observations O184–O186, crystallized as C73–C75
through this artifact commitment; no user affirmation of the interpretations is inferred.
See [findings](../research/2026-09-05-code-driven-findings.md) and
[evidence/audit](../../ara/evidence/code-driven-method-research-2026-09-05/run.md).

The experiments establish small tolerance-safe backtracking progress and an opportunity to remove
component work. They do not establish useful-scale perceptual improvement, generic gradient or
line-search failure, causal reuse-induced pipeline divergence, isolated end-to-end acceleration,
held-out generalization or novelty. Baseline null-decision stability and rejected work are an
open diagnostic question, not a selected remedy. No more complex solver is justified by default.
