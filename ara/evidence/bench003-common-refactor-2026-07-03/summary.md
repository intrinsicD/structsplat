# BENCH-003 benchmark common refactor evidence

Date: 2026-07-03

Scope:
- Extracted shared benchmark helpers into `benchmarks/common.py`.
- Updated benchmark CLIs/writers to import the common helper surface.
- Marked `coco_fit_compare.py` as superseded by `cross_repo_matrix_compare.py`.
- Expanded `benchmarks/README.md` with purpose, task ID, caveats, and example invocation per script.
- Added `tests/test_benchmark_common.py` covering shared helpers, rate-distortion smoke output, zero-ok summary cells, and summary-writer stubs.

Stage-search stability check:
- Before ref: `56bc1f2`
- After ref: BENCH-003 working tree before commit
- Screening: one 16x16 toy image, budget 16, seed 0, 2 iterations, CPU normalized renderer, one baseline config.
- Result: `summary.md` byte-identical before/after.
- SHA-256 before and after: `48b65d95e01ec254b7aad7bb5b18d945019381449cdf368e5fffbf84608aff6f`

Files:
- `stage_search_summary.md`: copied byte-identical summary.
- `stage_search_summary.sha256`: before/after hash lines from the ignored `results/bench003_stage_byte` run.
