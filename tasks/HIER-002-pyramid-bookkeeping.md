# HIER-002: Pyramid bookkeeping fixes (iteration accounting, budgets, schedules)

**Status: todo.** Confirmed defects from the 2026-07-03 repo review (item 1 adversarially
verified by execution).

## Context
1. **`fit_pyramid` returns the last level's `iterations_run`/`stopped_early` only.** The
   returned dict is the final level's fit output with history patched in, but
   `iterations_run` and `stopped_early` are never aggregated — pyramid benchmark rows
   under-report iterations by ~levels×. (`src/structsplat/pyramid.py:118,142-148`;
   downstream symptom at `benchmarks/optimization_followup.py:234`)
2. **Phantom iterations on early stop.** `iter_offset += level_cfg.iters` advances by the full
   level budget even when the level early-stopped, inserting phantom iterations into the
   combined iters-to-target and PSNR-AUC. (`src/structsplat/pyramid.py:131`)
3. **Level budgets do not sum to N.** `max(1, round(total*frac))` per level means pyramid runs
   place a slightly different budget than the nominal N they are compared at.
   (`src/structsplat/pyramid.py:96`)
4. **Cosine schedule silently becomes per-level warm restarts.** `_lr_factor` uses
   `cfg.iters = iters_per_level`, so the `lr_schedule='cosine'` axis measures a different
   schedule in pyramid vs single-stage cells. (`src/structsplat/fit.py:53-54`,
   `src/structsplat/pyramid.py:94`)

## Goal
Pyramid runs report exactly what they did, place exactly the nominal budget, and schedule
axes mean the same thing in every cell.

## Acceptance criteria
- [ ] `iterations_run` summed across levels; `stopped_early = any(level stopped)`; test.
- [ ] `iter_offset += out["iterations_run"]`; test: an early-stopping level produces no gap in
      the combined history iteration axis.
- [ ] Largest-remainder allocation of level budgets (floor shares, distribute the remainder by
      largest fractional part); test: `sum(level budgets) == num_gaussians` for
      representative (N, fractions) pairs including tiny N.
- [ ] Cosine phase either spans the whole pyramid run (pass a global (offset, total) into
      `fit`) or the warm-restart semantics are documented in `PyramidConfig` and ADR-0010's
      protocol notes — one or the other, decided and recorded.
- [ ] `pytest -q` green; `tests/test_pyramid.py` extended for all three accounting fixes.

## Interfaces touched
`src/structsplat/pyramid.py`, `src/structsplat/fit.py`, `src/structsplat/config.py`,
`tests/test_pyramid.py`.

## Depends on
HIER-001, FIT-001.
