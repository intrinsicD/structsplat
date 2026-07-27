# CORE-012: One maintained entrypoint for the current best pipeline

## Status

Implemented on 2026-07-26 (ADR-0025). `structsplat.pipeline.run_pipeline` and `structsplat
convert` are the maintained best-pipeline entrypoint, with a masked arm and a full-frame arm over
one shared schedule. No library default changed. The full-frame arm has no benchmark screen yet —
BENCH-017 owns it — so this task claims a maintained composition and a mechanism extension, not a
quality result on unmasked images.

## Context

The 2026-07-25 repository review found no artifact defining the pipeline end to end. Three partial
definitions disagreed: the conservative library defaults (ADR-0009/0013), the pinned benchmark
recipe `structsplat_best_default` (C12), and the safe-commit schedule (`safe_schedule.py`, C50-C52)
that carries the best measured quality but was reachable only through `scripts/fit_janelle_*.py`,
absent from `docs/architecture.md`, and governed by no ADR. Several confirmed wins also ship
default-off (C25, C50), so "what should I run today" required cross-referencing the claim ledger,
task statuses, and README prose.

Dome capture work is alpha-matted, so the masked path is the primary consumer, but the same
question applies to ordinary images and the answer should not be a second method.

## Goal

One entrypoint that (a) names the current best pipeline in one machine-readable place, (b) serves
masked and unmasked inputs from the same schedule, differing only in the mask/boundary stages, and
(c) is updated by a single reviewable diff when a new approach wins.

## Approach

1. **`src/structsplat/pipeline.py`.** `RECIPE` (name, version, per-choice evidence, evidence
   scope) plus `PipelineConfig` whose defaults are the recipe: `quadtree_wse` +
   `wse_progressive_order` init (ADR-0013, C25), Pareto-safe checkpoints every 50 steps (C50),
   event color solve off (C50), no specialized detail tail (C52), dynamic storage (C51), global
   refinement. Phase targets derive from `capacity` by the Janelle ratios (5/11, 8/11, 10/11), so
   `capacity` alone rescales the schedule and the defaults reproduce 5,000/500/8,000/10,000.
2. **Arm selection on `mask`.** `run_pipeline(image, mask=None, cfg)`; nothing else selects an arm.
3. **Full-frame arm by degeneration, not duplication.** `run_safe_schedule(mask=None)` builds an
   all-true frame mask. `mask.signed_distance` already clips an empty complement to the image
   diagonal, so caps are inert, the projection interior is the whole frame, the boundary band is
   empty, and `_visible_boundary` is empty. Boundary-specific work is disabled and count-matched
   general coverage/detail proposals use the same closure slot and budget (ADR-0027).
4. **`structsplat convert`.** A small flag surface over `PipelineConfig`; `structsplat fit` keeps
   its knob-level surface unchanged.
5. **`_safe_quantile`.** `torch.quantile` rejects inputs above `2**24` elements. A mask ROI rarely
   reaches that; a full frame does. Below the limit the value is unchanged, so previously recorded
   metrics stay comparable.

## Acceptance criteria

- [x] `run_pipeline` runs both arms end to end and returns `recipe`/`arm`/`init`/`device`/
      `pipeline_config` provenance with the schedule payload.
- [x] `mask=None` runs the same closure slot with general proposals; the masked arm uses boundary
      proposals.
- [x] Both arms share the schedule apart from explicit boundary switches and the closure label.
- [x] The masked arm preserves the ADR-0017/0019 guarantee (`outside_max_abs == 0`).
- [x] Recipe defaults do not leak into `FitConfig`/`InitConfig`.
- [x] `PipelineConfig` imports without torch, so `structsplat convert --help` works without it.
- [x] Fast tests cover the recipe surface and config->schedule translation; `slow` tests cover both
      arms end to end.
- [x] ADR-0025 records the decision; README and `docs/architecture.md` point at the entrypoint.

## Interfaces touched

`src/structsplat/pipeline.py` (new) · `src/structsplat/safe_schedule.py` (`mask=None` arm,
`_safe_quantile`) · `src/structsplat/cli.py` (`convert`) · `tests/test_pipeline.py` (new).

## Depends on

FIT-023, FIT-024, FIT-025, CORE-010, CORE-011, INIT-009, ADR-0013, ADR-0017, ADR-0019, ADR-0025.

## Notes

- The recipe's quality evidence is one masked image, one seed, one GPU (C50/C51/C52). This task
  makes that recipe reachable and maintained; it does not broaden its evidence.
- `structsplat_best_default` (C12) is intentionally left alone: it is the fair-density harness's
  comparison baseline at its own proxy regime, not a pipeline definition.
- Deferred: `fit.py`'s remaining full-frame `torch.quantile` calls (lines ~1769/1779/2445) have the
  same `2**24` ceiling. They are equally reachable from the masked arm today, so they are a
  separate robustness fix rather than part of this change.
- Follow-up: BENCH-017 (full-frame arm screen vs the plain-fit path), and a `--resume` story for
  long dome runs, which currently lives only in `scripts/fit_janelle_mask_contained.py`.
