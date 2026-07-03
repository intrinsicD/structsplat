# INIT-005: Init-math robustness, flanking unification, and WSE test coverage

**Status: done.** From the 2026-07-03 repo review. The dart-throwing metric leak was
adversarially verified by execution.

## Context
1. **99th-percentile references collapse on sparse-content images.** Every threshold and
   normalization (`density_from_energy`, structure-tensor labels, `_feature_run_lengths`)
   references `np.percentile(energy, 99)`. On a mostly-blank image with sensor noise the
   reference is noise-scaled: labels saturate to 100% non-flat and the density binarizes,
   making `density_power` a no-op. (`src/structsplat/density.py:16-24`,
   `src/structsplat/structure_tensor.py:139-145`)
2. **`density_mode='hybrid'` double-normalizes** — it mixes two already-normalized pmfs and
   re-passes the mixture through `density_from_energy`, so `density_power`/`density_base` mean
   different things for this mode. (`src/structsplat/density.py:43-46`)
3. **Border jitter atoms.** `sample_candidates` clips the half-pixel jitter onto the border
   line, creating probability atoms / coincident candidates at image edges.
   (`src/structsplat/density.py:125-128`)
4. **Silent exact-N contract break.** All samplers return all M indices when `n >= M` without
   warning (misconfigured `candidate_oversample` passes silently). (`src/structsplat/sampling.py:150`)
5. **`dart_throwing`'s shortfall fill ignores the metric** — the fill loop is Euclidean
   radius-normalized only, mixing distance semantics inside one ablation arm.
   (`src/structsplat/sampling.py:260-275`; `_pair_d2` already implements the metric distance)
6. **Flanking code is duplicated and diverged.** `_flank_edge_points` (used by
   quadtree_wse/hybrid) and the inline `aniso_flanking` block are copy-identical except the
   helper lacks the two_sided color correction from commit a455e98. Also: the flank offset
   floors at `edge_w * flank_offset_frac` (1px at defaults), not the blur width the comment
   claims, and the two_sided probe `eps >= ~2*grad_sigma` can overshoot thin structures and
   sample the far side. (`src/structsplat/init.py:218-233,562-589`)
7. **Test gap on the thesis path.** The WSE spacing assertion `d.min() > 0` is vacuous, and
   the anisotropic (metric) elimination path — the project's central contribution — has no
   direct test. (Verified nuance: total WSE breakage *is* caught indirectly by
   `test_wse_density_adaptivity`; the metric path is not.) (`tests/test_sampling.py:15`)

## Goal
Init math that degrades gracefully on degenerate images, one flanking implementation, and
tests that would actually catch a regression in the anisotropic sampler.

## Acceptance criteria
- [x] Percentile references computed over structured pixels only (energy above an absolute
      noise floor) or floored (`ref = max(percentile, k*median_energy)`); test on a synthetic
      near-blank noisy image asserting labels are not saturated and density is not binary.
- [x] `hybrid` mixes raw normalized features once through a single base/power/normalize step;
      test asserting `density_power` changes the hybrid pmf.
- [x] Border jitter folded asymmetrically (no probability atoms at x=0/W-1); test.
- [x] Samplers warn or raise on `n > M`; `InitConfig` validates `candidate_oversample >= 1`.
- [x] `dart_throwing` fill uses `_pair_d2` with the metric (or flags fill activation in the
      return for ablation filtering); test.
- [x] `_flank_edge_points` returns `(pts, color_pts)` with a `two_sided` option; the inline
      `aniso_flanking` branch calls it; the duplicate block is deleted; existing flanking
      tests pass unchanged.
- [x] Flank offset floor applied after the fraction (`max(s_across*frac, edge_w)`) or the
      comment/ADR corrected to state the frac-scaled floor is intentional; two_sided `eps`
      capped by local across-edge feature width.
- [x] New tests: non-vacuous isotropic WSE spacing bound (min distance above random-subset
      expectation, like the dart-throwing test) and an anisotropic WSE test asserting
      along/across-edge NN-displacement statistics reflect the metric.

## Interfaces touched
`src/structsplat/density.py`, `src/structsplat/structure_tensor.py`,
`src/structsplat/sampling.py`, `src/structsplat/init.py`, `src/structsplat/config.py`,
`tests/test_sampling.py`, `tests/test_init_stages.py`. NumPy-only invariant preserved.

## Depends on
INIT-003.
