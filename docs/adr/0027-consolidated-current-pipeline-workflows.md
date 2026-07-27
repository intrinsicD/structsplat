# ADR-0027: Consolidated current-pipeline workflows

- Status: accepted
- Date: 2026-07-24

## Context

`scripts/` mixed routine conversion, broad benchmark launchers, native environment provisioning,
paper utilities, and source-bound one-image experiments. The Janelle evidence established a clear
measured recipe, but its runners hardcoded the Janelle source adapter and boundary path, so an
unmasked invocation could not exercise the same optimizer and topology schedule. CORE-012/ADR-0025
subsequently established `structsplat.pipeline.RECIPE`, `PipelineConfig`, and `run_pipeline` as the
single maintained definition of that recipe.

## Decision

Expose exactly four supported operational scripts: `convert.py`, `benchmark.py`, `ablation.py`,
and `stage_search.py`. They delegate to `run_pipeline`; orchestration, registered variants, raw
rows, and portable report generation live in `structsplat.workflows`. They do not freeze or infer
a second definition of "current best."

The default masked and unmasked paths share the 5,000-row start, dynamic 11,000-row budget, phase
ordering and step ceilings, optimizer settings, proposal auction, checkpoint policy, and polish.
The masked path uses 4,500 general WSE rows plus 500 boundary rows and enables mask containment,
boundary losses, metrics, and proposals. The unmasked path uses 5,000 general WSE rows, disables
containment and every boundary-specific term, and uses count-matched coverage/detail proposals in
the same closure slot and budget.

Move previous task-specific launchers to `deprecated_scripts/`. Preserve evidence and trace files
with their originally executed commands; the deprecation README explains the archive prefix.
New one-off experiment drivers belong in `scripts/experiments/`.

## Consequences

- Routine users have four stable entrypoints and one recipe definition rather than duplicated
  shell defaults.
- Conversion, benchmark, ablation, and stage search share raw rows, provenance, failures, curves,
  timings, target/intermediate/final/error images, and a portable `index.html`.
- Official GaussianImage and Image-GS adapters remain isolated native executions and are optional
  subprocesses, never relabeled as common-harness controls.
- A recipe change happens in `RECIPE`/`PipelineConfig` under ADR-0025; the workflow layer consumes
  it automatically. Changes to masked/full-frame parity also update this decision, equivalence
  tests, and usage documentation.
- Calling the recipe "current" describes the operational selection. It does not promote the
  single-image Janelle result to a general superiority, throughput, compression, or SOTA claim.
