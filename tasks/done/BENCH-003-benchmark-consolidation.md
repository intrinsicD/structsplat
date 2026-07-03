# BENCH-003: Benchmark script consolidation + documentation

**Status: done.** Completed 2026-07-03. From the 2026-07-03 repo review.

## Context
Seven benchmark scripts have accreted with heavily duplicated, already-drifting helpers:
4 copies of `_load_image`, 3 of `_psnr_auc` (with differing rounding), 2 of
`_bilinear`/tile-labeling helpers, and two conflicting definitions of the "StructSplat" method
row. `coco_fit_compare.py` is effectively superseded by `cross_repo_matrix_compare.py`.
`benchmarks/README.md` documents only `ablation.py`; the other six scripts — including the
ABL-002 stage-search harness that has its own CLI subcommand — are undocumented, and only 2 of
7 have any tests.

## Goal
One shared helper module, no drifting duplicates, every script documented with purpose/task-ID/
caveats, smoke-tested.

## Acceptance criteria
- [x] `benchmarks/common.py` extracted: image loading, `_psnr_auc` (one rounding), bilinear,
      row/summary writing, method-analogue builders, Instant-GI loader; all scripts import it.
- [x] `coco_fit_compare.py` deleted or marked superseded in its docstring + README, with the
      unique bits (held-out image protocol) folded into `cross_repo_matrix_compare.py`.
- [x] `benchmarks/README.md`: one paragraph per remaining script (purpose, task ID, caveats,
      example invocation), per the docs-sync skill.
- [x] Smoke tests for `rate_distortion.run_rd` and each comparison summary writer using toy
      images and stubbed method rows (must cover the zero-ok-rows path fixed in BENCH-002).
- [x] `pytest -q` green; a short stage-search screening run produces byte-identical summary
      output before/after the refactor (helper extraction must not change numbers).

## Completion notes

Evidence lives in `ara/evidence/bench003-common-refactor-2026-07-03/`.

The short stage-search screening summary was byte-identical before/after the helper extraction:
`48b65d95e01ec254b7aad7bb5b18d945019381449cdf368e5fffbf84608aff6f`.

## Interfaces touched
`benchmarks/*` (all), `tests/test_ablation.py`, `tests/test_stage_search.py`, new
`tests/test_benchmark_common.py`.

## Depends on
BENCH-002 (land validity fixes first so the refactor doesn't rebase over them).
