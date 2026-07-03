# INIT-006: Init-time performance (quadtree, spacing, run-lengths, pair discovery)

**Status: partial.** From the 2026-07-03 repo review. All items are asymptotic/memory fixes with
no behavior change — outputs must stay bit-identical (or identical up to documented tie-breaks).

## Context
1. **`_quadtree_leaves` is O(n²)** with per-split Python recomputation of every leaf's
   priority; measured ~13.4s at n=4,000, extrapolating to minutes at the default 20,000
   budget. (`src/structsplat/init.py:107-139`)
2. **`_nn_spacing` allocates ~650 MB** of float64 temporaries at N=20k
   (`chunk × N × 2` broadcast). (`src/structsplat/init.py:436-448`)
3. **`_feature_run_lengths` walks pixels in Python** — O(N · max_steps) interpreter
   iterations for `scale_cap_mode='feature'`. (`src/structsplat/init.py:236-272`)
4. **`_neighbor_pairs` builds dense per-cell tables sized ncx·ncy** — O(bounding-box cells),
   not O(points); ~405 MB measured for 50k points on a 4096-px domain.
   (`src/structsplat/sampling.py:77-87`)
5. **`dart_throwing` scans with the global max radius × global metric inflation**, making the
   ablation baseline ~2× slower than the WSE it is meant to price.
   (`src/structsplat/sampling.py:218-236`)
6. **`_pair_d2` allocates a fresh (M,2,2) averaged-metric temporary per `farthest_point`
   iteration** — O(n·M) allocation churn. (`src/structsplat/sampling.py:190-197`)
7. **The 99th-percentile energy is recomputed independently** in `st.compute`,
   `density_from_energy`, and `_feature_run_lengths` per init call.
   (`src/structsplat/structure_tensor.py:139`)

## Goal
Feature-aware init stays interactive at the default 20k budget on 1–4 MP images.

## Acceptance criteria
- [ ] `_quadtree_leaves` uses a max-heap keyed by (mass, area), priorities computed once per
      cell at push time → O(n log n); leaf sets identical to the current implementation on the
      test images (or differences limited to documented tie-breaking).
- [x] `_nn_spacing` uses the GEMM form (|a|²+|b|²−2ab) or the sampling grid hash; peak
      allocation < 200 MB at N=20k; results equal within fp tolerance.
- [ ] `_feature_run_lengths` vectorized ((N, S) coordinate grid + argmax over the first
      failing step); equal results; ≥10× faster at N=20k.
- [x] `_neighbor_pairs` indexes only occupied cells (np.unique + searchsorted mapping); memory
      O(points); identical pair sets.
- [ ] `dart_throwing` uses a per-cell/local max accepted radius for its reach bound; identical
      accepted sets given the same rng.
- [ ] `_pair_d2` hoists flat metric components (m00/m01/m11) once per call site.
- [ ] The percentile ref is computed once in `st.compute`, stored on `StructureTensor`, and
      consumed by density/init (falling back to local computation when absent).
- [ ] A timing table (before/after per function at N∈{5k, 20k}) recorded in this file's notes.

## Notes
- 2026-07-03 partial implementation: `_nn_spacing` now uses a chunked GEMM distance matrix and
  caps the effective chunk with `_NN_SPACING_MAX_MATRIX_ELEMS = 16_000_000` (128 MB at
  float64). Focused test coverage compares the result to the old broadcast formula across
  multiple chunks.
- 2026-07-03 partial implementation: `_neighbor_pairs` now stores only occupied cell starts and
  counts via `np.unique(cid_sorted, return_index=True)`, then maps queried neighbor cells through
  `np.searchsorted`. Pair-set parity is tested against a brute-force reference for Euclidean and
  anisotropic metrics, including a sparse coordinate range that would be pathological for dense
  per-cell tables.
- Timing/memory check on random 4096x3072-domain points, default requested chunk 2048:

  | Function | N | Old seconds | New seconds | Max abs diff | Old matrix | New matrix |
  |---|---:|---:|---:|---:|---:|---:|
  | `_nn_spacing` | 5,000 | 0.3390 | 0.0736 | 7.85e-10 | 163.8 MB | 81.9 MB |
  | `_nn_spacing` | 20,000 | skipped | 1.0657 | n/a | 655.4 MB formula | 128.0 MB |

## Interfaces touched
`src/structsplat/init.py`, `src/structsplat/sampling.py`, `src/structsplat/structure_tensor.py`,
`src/structsplat/density.py`. NumPy-only invariant preserved. No ADR (no math change).

## Depends on
INIT-003, INIT-005 (land robustness first so perf work doesn't rebase over it).
