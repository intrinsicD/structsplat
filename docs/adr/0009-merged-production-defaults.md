# ADR-0009: Production defaults for the merged (Claude + Codex) StructSplat

> Update: ADR-0013 supersedes this ADR's initialization strategy default. The shipped init
> strategy is now `quadtree_wse`; the other production defaults recorded here still stand unless
> separately superseded.

## Context
MERGE-001 combined two experimental branches into one codebase:
- **Claude** optimized the core: a ragged tight-support renderer, a ~30x faster anisotropic WSE
  sampler, retuned fitter learning rates (ADR-0008), Adam-moment continuity across prune/split, an
  optional EWA `aa_dilation`, and a first quantization codec (ADR-0007).
- **Codex** broadened the search space: a `stage-search` harness (ABL-002) and the config knobs it
  sweeps — gradient operator (central/sobel/scharr), density mode (structure/gradient/variance/
  hybrid/uniform), sampling mode (wse/density_random/jittered_grid), color mode (bilinear/
  local_mean/two_sided), scale mode, per-Gaussian opacity, additive renderer (ADR-0006), pixel loss
  (l1/l2/charbonnier + warmup), optimizer (adam/adamw), LR schedule (none/step/cosine), and split
  mode (duplicate/residual_add).

Every new axis is now a config field, so the question is which *values* ship as defaults versus
which stay opt-in candidates for the screening to promote later.

## Decision
The production defaults are Claude's fast, validated core; every Codex breadth axis defaults to its
conservative option and stays a candidate:

| axis | default | rationale |
|---|---|---|
| renderer | `normalized` | ADR-0003; additive (ADR-0006) is opt-in and unproven for fitting |
| split_mode | `duplicate` | residual_add is a candidate, not shown to win yet |
| opacity_mode | `none` | 8-param budget; opacity is a candidate for generation/pruning |
| pyramid | off (single-stage) | LOD study (HIER-001) not yet a proven default |
| lr_means/scales/rot/color | `5e-2 / 3e-2 / 1e-2 / 3e-2` | ADR-0008; positions were frozen at the old 2e-3 |
| lr_schedule | `none` | opt-in; step period must scale with `iters` |
| gradient_operator | `central` · density `structure` · sampling `wse` · color `bilinear` | the validated init path |

The renderer's normalized and additive modes now share one accumulator (`render._accumulate`): the
ragged tight-support flat-tile core, with per-Gaussian opacity threaded through both. So opacity and
the additive mode ride on Claude's fast path rather than the old square-tile loop.

## Consequences
+ `structsplat fit` with no extra flags gives the fast, retuned, normalized behavior; the whole
  Codex axis set is reachable by flag and by `stage-search` for principled promotion.
+ `stage-search` can reproduce the Claude defaults as a named cell, so the screening measures every
  variant against the shipped baseline on equal footing.
- The defaults are chosen from small-scale evidence plus the prior single-branch screenings, not the
  full COCO confirmation (MERGE-001 acceptance criteria, still pending a GPU + dataset run). Any
  promotion of an opt-in axis to default must cite that larger run.

## Links
Supersedes neither ADR-0003 nor ADR-0006 — it records which of their modes is the default. Depends
on ADR-0007 (codec), ADR-0008 (LRs). Realizes the reconcile/demote steps of MERGE-001.
