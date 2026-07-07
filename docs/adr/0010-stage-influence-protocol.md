# ADR-0010: Stage-influence protocol (OFAT deltas around a fixed baseline)

## Context

ABL-002's factorial stage search ranks *complete* configurations, but a ranking cannot answer
"what does stage X contribute?": in a (partial, shuffled) factorial, per-stage marginal means are
confounded by whatever else co-varied, and the full product wastes cells on configs that only
differ in a stage that provably cannot affect the output (e.g. the tensor operator under
`strategy=random`). We also had no convergence-rate or speed signal per row — only final PSNR —
so "max quality / max convergence / max speed" candidates could not be separated.

## Decision

`benchmarks/stage_search.py` gains an **influence mode**: one-factor-at-a-time around a fixed
baseline (the first value of every stage axis; defaults = ADR-0009 production defaults, with
ADR-0013 superseding the init strategy default). Every
variant differs from the baseline in exactly one stage, and the summary (`influence.md`) reports
**paired deltas** — per (image, budget, seed) cell — for quality (PSNR, MS-SSIM, LPIPS),
convergence (iters-to-target, PSNR-AUC over the training trajectory), and speed (init/fit
seconds, seconds-to-target). Factorial mode remains for finding the best combination; both modes
now **canonicalize and deduplicate** configs that provably produce the identical initial field
(random/grid ignore tensor/density/sampling and have zero angles under both `tensor` and `zero`
orientation; `jittered_grid` placement ignores density; `two_sided` color equals `bilinear`
outside `aniso_flanking`). Orientation is deliberately *not* pinned for isotropic inits: equal
initial axes still break symmetry through fitting — the rotation decides which axis each scale
gradient feeds.

## Consequences

* Per-stage influence is measured directly (paired, same-cell deltas), not inferred from
  confounded marginals; the marginal-means table stays in `summary.md` but is labeled
  observational.
* Interaction effects are deliberately out of scope for influence mode: a stage whose value only
  helps in combination with another (e.g. additive renderer + opacity) shows its solo effect.
  Finding interactions is factorial mode's job; the two modes share one runner and row schema so
  results compose.
* OFAT cost is linear in the number of options (1 + Σ(k_i − 1) runs per cell), which makes
  multi-seed, multi-budget influence runs affordable and the deltas honest (mean ± std over
  cells).
* Deduplication changes factorial cell counts versus the pre-ADR harness when inert combinations
  were requested; `--no-dedupe` restores the old behavior for reproducing historical sweeps.

## Protocol note: factored refinement axes (FIT-009)

The original `refine` stage mixed site selection, growth primitive, and sampled-add NMS into flat
strings such as `residual_tensor_add_nms_residual_color`. FIT-009 splits that into explicit
stage fields: `refine_site`, `refine_primitive`, and `refine_nms`, with `refine_color`,
`refine_prune`, and `refine_relocate` as orthogonal flags. Legacy `--refine-modes` aliases remain
accepted and are normalized into those fields before labels, dedupe, budget accounting, and
summaries are produced. Canonicalization pins primitive/NMS/color when `refine_site=none`, and pins
NMS/color for non-`sampled_add` primitives because those controls are inert there.

## Protocol note: LR schedule in pyramid vs single-stage cells (HIER-002)

The `lr_schedule` axis must mean the same thing in every cell. A `cosine` schedule spans the
**whole run**: `fit` accepts a `(sched_offset, sched_total)` span, and `fit_pyramid` passes the
level's nominal start (`lvl * iters_per_level`) and the pyramid-wide iteration count
(`levels * iters_per_level`), so a pyramid cosine is one decay across all levels — not a per-level
warm restart — and is directly comparable to a single-stage cosine over the same total budget.
Early stops do not reshape the schedule (it is a function of the nominal planned iteration).
Level budgets are placed by largest-remainder allocation so they sum exactly to `num_gaussians`,
and pyramid runs report `iterations_run`/`stopped_early` aggregated across levels.
