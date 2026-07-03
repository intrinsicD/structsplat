# BENCH-002: Benchmark harness experimental-validity fixes

**Status: partial.** From the 2026-07-03 repo review. **This task gates every future sweep** —
conclusions drawn before these fixes are confounded. The budget-fairness item is the single
highest-severity finding in the repo.

The science-gating fixes are done and tested: equal budgets (refine arms capped at the cell
budget, `n_gaussians` per row), per-cell error isolation + resumable JSONL, `config.json`
(resolved args + device + versions) from every harness, best-config `fitness()`, the corrected
`_canonicalize`, the reconciled `scale_cap=none` baseline, symmetric render timing, one clamped
(display-referred) metric convention per cross-repo row, the GPU-nondeterminism caveat, and the
removal of all `/home/...` defaults (Instant-GI via `STRUCTSPLAT_INSTANT_GI`). **Remaining:** a
multi-seed axis (`--seeds`, mean ± std) for the four cross-repo *comparison* scripts
(`coco_fit_compare`, `cross_repo_matrix_compare`, `quadtree_init_compare`,
`optimization_followup`) — those are single-seed and require COCO data + CUDA to exercise, so they
are left as follow-up; the primary sweep harnesses (`ablation`, `stage_search`) already sweep seeds.

## Context
1. **Budget unfairness in refine arms (high).** `stage_search.py` and
   `optimization_followup.py` leave `max_gaussians=None` while forcing `split_count`, so
   refine=residual_add/duplicate arms run with up to +25% more Gaussians than the baselines
   they are ranked against. Any "refine wins" conclusion is confounded with capacity.
   (`benchmarks/stage_search.py:400`, `benchmarks/optimization_followup.py:134-153`;
   contrast `coco_fit_compare`/`cross_repo`, which cap correctly)
2. **Baseline divergence.** Stage-search pins `scale_cap_modes: ("feature12",)` as baseline
   while `config.py` ships `scale_cap_mode='none'` — the benchmark baseline and shipped
   defaults have silently diverged from ADR-0009/0010. (`benchmarks/stage_search.py:47,57`)
3. **No run-config persistence.** `ablation.py`, `stage_search.py`, `rate_distortion.py`
   never write the resolved config/args/device/package-version into their output dirs —
   invariant 5 ("reproducible from logged config + seed") is only half-met.
   (`benchmarks/ablation.py:141`)
4. **No per-cell error isolation; results written only at sweep end.** One failing arm (e.g.
   renderer=cuda on a CPU box) aborts hours of completed cells. (`benchmarks/stage_search.py:496`)
5. **`fitness()` pooling bias.** `fitness(rows, strategy, budget)` and the psnr-vs-budget plot
   pool all hyperparameter variants of a strategy into one mean, biasing the headline
   comparison whenever flank/ratio/threshold sweeps are active. (`benchmarks/ablation.py:256`)
6. **`_canonicalize` drops distinct configs.** The jittered_grid density pin wrongly applies
   to quadtree strategies, silently deduping genuinely different cells from factorial sweeps.
   (`benchmarks/stage_search.py:152`)
7. **Crashes that void whole runs.** `_write_summary`/`_make_grid` raise `StatisticsError`
   when any method has zero ok rows — guaranteed on machines without `/home/alex/...`.
   (`benchmarks/coco_fit_compare.py:275`, `benchmarks/cross_repo_matrix_compare.py:475`)
8. **Single-seed comparisons.** All four follow-up comparison scripts have no seed axis —
   rankings against random-init baselines carry no variance estimate.
   (`benchmarks/cross_repo_matrix_compare.py:693`)
9. **Asymmetric timing.** The render-speed benchmark times the reference renderer *with*
   autograd graph construction but the tile prototype under `no_grad`, inflating the reported
   speedup. (`benchmarks/optimization_followup.py:383`)
10. **Misc:** `summarize_influence` "Baseline means" reports one arbitrary baseline cell as
    the mean (`benchmarks/stage_search.py:647`); hardcoded `/home/alex/...` dataset and
    Instant-GI paths (`benchmarks/optimization_followup.py:492`,
    `benchmarks/cross_repo_matrix_compare.py:686`); row-internal metric inconsistency —
    mse/mae/edge metrics on the clamped render, psnr/ssim from fit() on the unclamped one
    (`benchmarks/cross_repo_matrix_compare.py:344`); GPU renders are nondeterministic
    (atomicAdd / CUDA index_add) while the cross-repo benchmark defaults to renderer=cuda —
    the reproducibility invariant needs a documented caveat + logged renderer/device/versions.

## Goal
A sweep result is trustworthy by construction: equal budgets, resumable, reproducible from its
own artifacts, statistically honest.

## Acceptance criteria
- [x] Refine arms capped at the cell budget by default (`max_gaussians=budget` unless
      overridden, or refine configs start at budget − planned additions); an `n_gaussians`
      column added to result rows; test asserting no arm exceeds its budget.
- [x] Stage-search baseline scale_cap reconciled with shipped defaults: either ADR-0009 is
      amended to promote feature12 (citing a held-out run) or the baseline reverts to 'none';
      the `stage_search.py:57` comment matches reality.
- [x] Every harness writes `config.json` (resolved args + dataclass dicts + device + torch/
      package versions) into its outdir; rows carry strategy/iters where missing
      (`rate_distortion.py`).
- [x] Per-cell try/except with status/error rows; incremental JSONL appends (or periodic
      rewrite) so partial sweeps are recoverable; summary writers guard empty methods
      (`_mean_or_none` pattern) — a sweep with one broken arm completes and says so.
- [x] `fitness()` aggregates per full config key (or max-over-configs, documented); plot uses
      best-config-per-strategy.
- [x] `_canonicalize` pins restricted to strategies that actually route through
      `_blue_noise_positions`; unit test enumerating quadtree × sampling combinations.
- [ ] Comparison scripts accept `--seeds` and report mean ± std across images × seeds. (follow-up)
- [x] Timing comparisons run both closures under the same grad mode with symmetric setup.
- [x] Dataset/Instant-GI paths are CLI arguments with clear skip behavior; no `/home/alex`
      defaults anywhere (`grep -r /home/alex benchmarks/` is empty).
- [x] One metric convention per row (clamped vs unclamped) across all columns, documented.
- [x] GPU nondeterminism caveat documented (README + benchmark skill), renderer/device/version
      logged in all results.

## Interfaces touched
`benchmarks/ablation.py`, `benchmarks/stage_search.py`, `benchmarks/optimization_followup.py`,
`benchmarks/coco_fit_compare.py`, `benchmarks/cross_repo_matrix_compare.py`,
`benchmarks/rate_distortion.py`, `tests/test_ablation.py`, `tests/test_stage_search.py`.

## Depends on
— (gates ABL-004; do first).
