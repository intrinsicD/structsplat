# ABL-004: Killer controls + the full ABL-001 run + committed evidence

**Status: partial.** From the 2026-07-03 repo review. This is the repo's actual experiment — the
initialization thesis has never been tested by its own protocol. All recorded evidence sits at
one operating point (4 COCO images, 512 Gaussians, 80 iters, max-side 160).

## Context
Three gaps between what the repo claims and what it has measured:

1. **ABL-001 has never run.** No multi-seed strategy × budget sweep at realistic budgets and
   resolutions exists. (tasks/ABL-001-init-sweep.md is the protocol; this task is its
   execution plus the missing control arms.)
2. **Missing controls that could kill (or crown) the thesis.**
   - *Floyd–Steinberg dithering* of the structure-tensor density map (Instant-GI's
     placement primitive, ICCV 2025): O(HW), milliseconds, blue-noise-like spectra. If WSE
     doesn't beat it, the blue-noise-spectrum claim dies; if it does, that IS the paper.
     NumPy-only, belongs in `sampling.py` + the sampler axis.
   - *Relocation/error-driven growth* (FIT-004's MCMC mode): 3DGS literature shows relocation
     reduces sensitivity to initialization — it must run as a control arm, not just an
     improvement.
   - *Image-GS-style gradient-weighted random* is already present (`density_random`); ensure
     it is in every headline table.
3. **Evaluation-tuning contamination.** The cross-repo matrix compares the *searched*
   StructSplat config against repo-inspired analogues on the same four images used to select
   that config, and O14/N17 frame this as PSNR dominance. Needs held-out images, a
   shipped-defaults row, and honest reframing.

## Goal
A committed, reproducible answer to the README's open question: does structure-tensor
anisotropic blue-noise flanking improve low-budget quality or convergence speed under matched
conditions and honest controls?

## Acceptance criteria
- [x] `floyd_steinberg` sampler in `src/structsplat/sampling.py` (NumPy-only, torch-free),
      registered in the sampler axis + ablation; unit test (exact mass ≈ N, spacing sanity).
- [ ] ABL-001 executed per its protocol: ≥3 seeds × budgets {2k, 5k, 10k, 20k} × all
      strategies + {floyd_steinberg, density_random, relocation-enabled random} controls, on
      Kodak-24 + a pinned COCO/DIV2K subset at realistic resolution; PSNR/MS-SSIM(/LPIPS),
      iters-to-target, init/fit seconds.
- [x] Cross-repo matrix re-run on images disjoint from the four used for config selection,
      with a `structsplat_shipped_defaults` row alongside `structsplat_current`; O14/N17
      reworded to "best-searched config vs repo-inspired policy analogues".
- [ ] Summary artifacts (summary.md, metrics.csv, config.json) committed under
      `ara/evidence/` with evidence-index entries (results/ stays gitignored; curated evidence
      does not) — closing the "every quantitative claim cites absent files" gap.
- [ ] README's hypothesis section updated with the measured answer (either direction), and
      `ara/logic/claims.md` populated with the promoted/parked claims.
- [ ] ABL-001's status updated accordingly (done or failed-with-findings).

## Progress notes

- 2026-07-04: Added the missing Floyd-Steinberg control arm and made `benchmarks.ablation`
  resumable/shardable. Prepared Kodak-24 plus the pinned COCO fixtures under ignored
  `results/datasets/abl004/`. First bounded full-protocol shard completed one cell
  (`kodim01`, random, 2000 Gaussians, seed 0, 1500 iters, max-side 768) in 780.38 s fit time
  on the local RTX 3050; evidence: `ara/evidence/abl004-first-shard-2026-07-04/`. This is
  runtime calibration only, not the completed ABL-001 sweep.
- 2026-07-04: Added `structsplat_shipped_defaults` to `benchmarks.cross_repo_matrix_compare`.
  A tiny CPU smoke ran the row through the matrix loop. The held-out cross-repo rerun itself is
  still pending.
- 2026-07-04: Ran a small held-out Kodak cross-repo matrix (`kodim01`-`kodim04`, max-side 160,
  80 iters, seed 0) with `structsplat_current` and `structsplat_shipped_defaults`; evidence:
  `ara/evidence/abl004-kodak4-cross-repo-2026-07-04/`. README/ARA claim rewording now frames
  the result as best-searched StructSplat policy vs local repo-inspired analogue rows.
- 2026-07-04: Added `--renderer` to `benchmarks.ablation` and changed the full ABL-004 wrapper
  to default to `RENDERER=cuda` (overrideable). Exact CUDA requires
  `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` in this workspace because conda's
  `libstdc++` lacks `CXXABI_1.3.15`. Five full-iteration calibration cells on `kodim01`,
  max-side 768, budget 2000, seed 0, 1500 iters, exact CUDA, finished with mean fit time
  21.606 s and mean init+fit time 21.801 s; evidence:
  `ara/evidence/abl004-cuda-calibration-2026-07-04/`. The full 3,696-cell matrix now estimates
  to ~0.93 GPU-days at flat 2k cost, or ~4.31 GPU-days under linear budget scaling across
  {2k, 5k, 10k, 20k}. This is feasible as a scheduled/job-queue run but still not interactive.

## Decision-grade staged protocol

The full ABL-001/ABL-004 matrix remains the gold-standard protocol. If it is not run as a
scheduled multi-day job, use this predeclared staged protocol so stopping is evidence-based:

1. **Screen.** Run all 11 arms on a balanced 8-image subset, budgets {2k, 5k}, seed 0,
   exact CUDA, max-side 768, 1500 iters. Eliminate arms that lose clearly on paired PSNR and
   convergence metrics to both the thesis arm and at least one killer control.
2. **Confirm.** Run the top 3-4 arms plus required controls on all 28 images, seeds {0,1,2},
   budgets {2k, 5k, 10k}. Report paired image/seed/budget deltas with confidence intervals.
3. **High-budget check.** Run only finalists at 20k on all 28 images and seeds {0,1,2}, because
   the ABL-001 hypothesis already expects gaps to shrink at high budget.
4. **Promotion rule.** README/ARA claims can be updated only from paired results. If
   `aniso_flanking` does not beat `floyd_steinberg`, `density_random`, and `random_relocate`
   under the confirm/high-budget stages, record the thesis as weakened or failed-with-findings
   instead of continuing the exhaustive grid.

## Interfaces touched
`src/structsplat/sampling.py`, `src/structsplat/init.py` (sampler registration),
`benchmarks/ablation.py`, `benchmarks/cross_repo_matrix_compare.py`, `scripts/run_abl004_full_ablation.sh`,
`ara/evidence/`,
`ara/logic/claims.md`, `README.md`, `tasks/ABL-001-init-sweep.md`, `tests/test_sampling.py`.

## Depends on
BENCH-002 (validity fixes are prerequisites — running the sweep before them wastes the
compute), ABL-003 (know what baseline you're standing on), FIT-004 (relocation control arm;
can run without the stretch items).
