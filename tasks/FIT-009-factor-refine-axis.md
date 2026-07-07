# FIT-009: Factor the refine axis into orthogonal sub-axes

**Status: todo.** The 19 refine modes conflate three independent decisions; the best combination
is currently inexpressible.

## Context
`refine` mode strings like `residual_tensor_add_nms_residual_color` bundle three orthogonal
choices: **where** capacity is added (residual / residual x tensor / absgrad / ranked /
freq-violation), **how** it is instantiated (duplicate / fp_duplicate / moment_preserving /
residual-sampled add), and **dedup** (NMS on/off). Evidence says the winning ingredients live in
different bundles: `residual_tensor_add` had the best site selection
(`fit004-residual-add-controls-2026-07-03`, `fit006-frequency-violation-smoke-2026-07-06`), while
`moment_preserving` is the best primitive — split dip -0.21 dB vs -1.16/-2.01, recovery ~3 iters
vs 6+ (`fit007-moment-preserving-smoke-2026-07-06`). `residual_tensor x moment_preserving` cannot
be written today.

## Goal
Replace the flat mode list with three composable config fields, keep every legacy string working
as an alias, and measure the previously inexpressible combinations.

## Design
- `FitConfig.refine_site in {none, residual, residual_tensor, absgrad, ranked, freq_violation}`
- `FitConfig.refine_primitive in {duplicate, fp, moment_preserving, sampled_add}`
- `FitConfig.refine_nms in {off, on}`; existing `prune`/`relocate`/`split_color_init` stay
  orthogonal flags as they already are.
- A single alias table maps every legacy `split_mode`/refine string to a
  (site, primitive, nms, prune, color) tuple — reproducibility of old configs is exact.
- Stage-search: the `refine` axis becomes three axes; `_canonicalize` marks primitive/nms inert
  when `site=none` (dedupe rule), and `_refine_adds_capacity` is derived from (site, primitive)
  so the BENCH-002 equal-budget start-below-budget correction is preserved.

## Acceptance criteria
- [ ] Alias table covers all modes currently in `stage_search.py`'s refine axis; a test asserts
      each legacy string produces bit-identical fit behavior on a fixed CPU fixture and seed.
- [ ] Equal-budget capacity accounting proven by test: no factored combination ends above budget.
- [ ] Smoke matrix over site x primitive (CPU, tiny) runs via stage-search with correct metadata.
- [ ] A fair-regime slice (difficult-4 protocol) compares `residual_tensor x moment_preserving`
      against `residual_tensor_add` and `moment_preserving` alone; evidence committed.
- [ ] Docs: `benchmark` skill's stage list and ADR-0010 axis notes updated in the same commit.

## Interfaces touched
`src/structsplat/config.py`, `src/structsplat/fit.py`, `benchmarks/stage_search.py`,
`tests/test_fit_dynamics.py`, `tests/test_stage_search.py`, `.claude/skills/benchmark/`.

## Depends on
FIT-004, FIT-006, FIT-007. Feeds ABL-005 (its `refine=moment_preserving` arm becomes
site/primitive-explicit once this lands).
