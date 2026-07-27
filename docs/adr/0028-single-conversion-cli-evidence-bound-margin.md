# ADR-0028: One conversion CLI and an evidence-bound mask margin

- Status: accepted
- Date: 2026-07-27

## Context

ADR-0025 established one programmatic definition of the current-best recipe, but the repository
then exposed two command-line paths to it: the single-image `structsplat convert` subcommand and
the file/folder `scripts/convert.py` workflow. Their defaults drifted. In particular,
`PipelineConfig.mask_margin` and `structsplat convert` used `1.5`, while the workflow and every
resolved FIT-023/024/025 Janelle arm underlying C50/C51/C52 used `0.75`.

The two values encode a real containment trade-off: a larger margin is more conservative, while a
smaller margin leaves less unreachable boundary area. There is no matched margin comparison, so
the repository cannot call either value generally superior. It can, however, identify which value
reproduces the named evidence-bearing recipe (C56).

## Decision

1. **`scripts/convert.py` is the sole supported current-best conversion CLI.** It accepts one
   image or a recursively scanned folder. A single image may use `--mask`; a parallel image tree
   may use `--mask-dir`. Omitting both selects the full-frame arm.
2. **Remove `structsplat convert`.** The `structsplat` console keeps `fit`, `render`, batch,
   ablation, stage-search, and generation research tools, but it no longer carries a second
   best-default conversion surface.
3. **Keep `structsplat.pipeline.run_pipeline` as the canonical programmatic definition.** It is
   the implementation API consumed by `structsplat.workflows`, not a second command-line
   entrypoint. Non-default recipe experiments use `PipelineConfig` directly.
4. **Set the recipe margin to `0.75` px.** C56 establishes this as the executed Janelle
   safe-schedule value, not as a quality win over `1.5`. `PipelineConfig` owns the default and
   every workflow parser derives from it. Explicit overrides remain available above the `0.72`
   containment floor and are recorded in result provenance.
5. **Bump the recipe version.** A run produced after this decision identifies
   `safe-commit-schedule@2026-07-27.1`; older outputs retain their original resolved
   configuration.

## Consequences

- There is one documented command to convert either masked or full-frame inputs, with one artifact
  and report contract.
- The duplicate CLI parser, rendering, serialization, and report implementation is removed.
- Routine conversion stays on the fixed current profile. Broader capacity, schedule, renderer, and
  commit-gate experiments remain available through the programmatic API or dedicated evaluation
  workflows rather than expanding the default CLI.
- `0.75` is reproducible with the evidence that selected the current profile. This decision makes
  no claim about a margin sensitivity curve or superiority on other images.

## Supersession and links

This ADR supersedes only ADR-0025's statement that `structsplat convert` is the command-line
entrypoint and ADR-0026's exposure of `--hole-regression-budget` on that removed command. It amends
ADR-0027 by making `convert.py` the only conversion script among its four operational workflows;
the benchmark, ablation, and stage-search scripts remain evaluation entrypoints. The canonical
recipe, arm-selection, and shared-schedule decisions in ADR-0025 remain accepted.

Evidence: C56 and
`ara/evidence/core012-janelle-mask-margin-provenance-2026-07-27/run.md`. Realizes the interface
correction to CORE-012.
