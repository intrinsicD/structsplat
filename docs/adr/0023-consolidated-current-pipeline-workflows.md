# ADR-0023: Consolidated current-pipeline workflows

- Status: accepted
- Date: 2026-07-24

## Context

`scripts/` mixed routine conversion, broad benchmark launchers, native environment provisioning,
paper utilities, and source-bound one-image experiments. The newest Janelle evidence also had a
clear bounded winner, but its runner hardcoded the Janelle source adapter and boundary path, so an
unmasked invocation could not exercise the same optimizer and topology schedule.

FIT-025's generic +512 arm is the bounded Janelle development winner: fixed 12,024-row physical
storage, an 11,512-row ordinary active ceiling, no detail tail, global refinement, Pareto-safe
checkpoints every 50 steps, and event color solve off. This is single-image development evidence,
not authorization for a repository-wide quality or speed claim.

## Decision

Expose exactly four supported operational scripts: `convert.py`, `benchmark.py`, `ablation.py`,
and `stage_search.py`. Freeze their default as the source-bound
`safe_schedule_2026_07_24` profile in `structsplat.pipeline`; keep orchestration and portable report
generation in `structsplat.workflows`.

The masked and unmasked paths share the 5,000-row start, 12,024-row physical storage, 11,512-row
ordinary active ceiling, phase ordering and budgets, optimizer settings, proposal auction,
checkpoint policy, and polish. The masked path uses 4,500 general WSE rows plus 500 boundary rows
and enables mask containment, boundary losses, metrics, and proposals. The unmasked path uses
5,000 general WSE rows, disables containment and every boundary-specific term, and uses
count-matched coverage/detail proposals in the same closure slot.

Move previous task-specific launchers to `deprecated_scripts/`. Preserve evidence and trace files
with their originally executed paths; the deprecation README explains the new archive prefix.

## Consequences

- Routine users have four stable entry points and one resolved profile rather than duplicated
  shell defaults.
- Conversion, benchmark, ablation, and stage search share raw rows, provenance, failures, curves,
  timings, target/intermediate/final/error images, and a portable `index.html`.
- Official GaussianImage and Image-GS adapters remain isolated native executions and are optional
  subprocesses, never relabeled as common-harness controls.
- The operational profile can be changed only by updating this decision, the frozen profile,
  masked/unmasked equivalence tests, and usage documentation together.
- Calling this profile "current" describes the workflow selection. It does not promote the
  single-image Janelle result to a general superiority, throughput, compression, or SOTA claim.
