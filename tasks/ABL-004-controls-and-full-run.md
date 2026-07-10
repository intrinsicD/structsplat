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
- [x] Summary artifacts (summary.md, metrics.csv, config.json) committed under
      `ara/evidence/` with evidence-index entries (results/ stays gitignored; curated evidence
      does not) — closing the "every quantitative claim cites absent files" gap.
- [x] README's hypothesis section updated with the measured answer (either direction), and
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
- 2026-07-04: Reran the two-arm high-budget smoke as a real exact-CUDA 1500-iteration check
  (`kodim01`, budget 20000, seed 0). Evidence:
  `ara/evidence/abl004-high-budget-cuda-2026-07-04/`. Floyd-Steinberg slightly beat
  anisotropic flanking on this cell (30.7803 vs 30.7510 PSNR; 0.98379 vs 0.98311 MS-SSIM) and
  was faster overall (31.91 s vs 43.92 s init+fit). This is single-image/single-seed evidence
  only, but it reinforces that Floyd-Steinberg must remain a required killer control through
  the staged confirmation.
- 2026-07-04: Completed the staged 8-image exact-CUDA screen within the user's 6-hour cap
  (`kodim01,04,07,10,13,16,19,22`, budgets {2000,5000}, seed 0, 11 arms, 1500 iters,
  176/176 cells). Evidence: `ara/evidence/abl004-stage-screen-8img-cuda-2026-07-04/`.
  Runtime was 74.82 min wall-clock. The original `aniso_flanking` thesis arm was close but
  not best: `aniso_onedge` won mean PSNR at 2000 (26.9861 vs 26.7436 for flanking; 7/8 paired
  wins), while `quadtree_wse`/`quadtree_hybrid` led at 5000 (30.2148/30.2097 vs 30.0470 for
  flanking). Floyd-Steinberg did not generalize from the one-image 20k warning, ranking 11/11
  at 2000 and 8/11 at 5000 by mean PSNR, with clear failures on `kodim07`. Confirmation should
  test `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`, `iso_blue_noise`,
  and Floyd-Steinberg.
- 2026-07-04: Generated visual audit sheets for five Kodak examples (`kodim01`, `kodim07`,
  `kodim10`, `kodim13`, `kodim22`) across the six small visual levels used in the earlier
  matrix (`160/240/320 px` x `80/200 iters`) and the current finalist/control set. Evidence:
  `ara/evidence/abl004-visual-examples-2026-07-04/`; live outputs:
  `results/abl004_visual_examples_5img/`. This is inspection support for the staged-screen
  interpretation, not a replacement for confirmation.
- 2026-07-05: Added `benchmarks.abl004_confirmation`, a decision-grade confirmation wrapper
  around the existing resumable ablation harness. It writes the 1,512-cell expected manifest
  for the current confirmation protocol, runs bounded shards, and analyzes rows into
  missing-cell, leaderboard, paired-delta/bootstrap-CI, per-image/seed baseline-loss, pairwise,
  and rank-stability artifacts. A tiny CPU run verified the `run -> analyze` path; the default
  `plan` resolves to 28 images x 3 seeds x 3 budgets x 6 variants = 1,512 cells.
- 2026-07-05: After inspecting the difficult-4 visual comparison, adopted a standing artifact
  convention: any substantial ABL-004 benchmark/visual-audit result should include a local
  `index.html` overview that embeds the headline diagrams and comparison images and states whether
  images are original saved renders or matched reruns.
- 2026-07-05: Tested feature-adaptive scale caps on the four fair-density finalist rows in
  `results/fair_density_control_difficult4/`, scaled from the original 12 px / 160 px evidence
  to a 57.6 px cap at max-side 768. The expanded exact-CUDA run completed 204/204 cells with
  48 feature-cap rows. Feature caps were faster but not finalist-worthy here: paired against the
  matching uncapped rows, they averaged -2.0531 dB PSNR, -0.8960 AUC, and -4.05 s fit time, with
  wins only on `kodim07` at 10k (4/48 paired PSNR wins). Do not promote feature-cap variants to
  the confirmation shortlist from this evidence.
- 2026-07-05: Started the default ABL-004 confirmation set after the fair-density follow-ups did
  not change the shortlist or renderer. `results/abl004_confirmation/confirmation_plan.csv`
  contains 1,512 expected cells (28 images x 3 seeds x 3 budgets x 6 variants). The first bounded
  exact-CUDA shard completed 18/18 requested cells: `kodim01`, budget 2000, all six variants,
  seeds {0,1,2}. `results/abl004_confirmation/index.html` now links the confirmation analysis,
  scalar plots, missing-cell report, leaderboards, and paired-delta tables. This shard is partial
  smoke/evidence only: 1,494 cells remain before confirmation is decision-grade. On this slice,
  `aniso_onedge` led mean PSNR (23.3672), followed by Floyd-Steinberg (23.2256) and
  `quadtree_hybrid` (23.1588).
- 2026-07-06: Continued the default exact-CUDA confirmation run with a second bounded shard
  (`kodim01`, budget 5000, all six variants, seeds {0,1,2}). The confirmation artifact now has
  36/1,512 completed cells with zero errors, leaving 1,476 cells. At 5000 Gaussians on this
  single-image slice, `aniso_onedge` and `iso_blue_noise` are statistically indistinguishable by
  mean PSNR (25.5627 vs 25.5584), followed by `quadtree_wse` (25.4492), `aniso_flanking`
  (25.3828), `quadtree_hybrid` (25.2802), and Floyd-Steinberg (25.2144). Treat this as continued
  shard evidence only; the confirm stage still needs all remaining images/budgets before task
  retirement or README claim promotion.
- 2026-07-06: Completed the next bounded confirmation shard (`kodim01`, budget 10000, all six
  variants, seeds {0,1,2}), finishing the full `kodim01` block for confirmation budgets {2000,
  5000, 10000}. The artifact now has 54/1,512 cells with zero errors, leaving 1,458 cells. At
  10000 Gaussians on this single-image slice, `quadtree_wse` leads mean PSNR (27.8561), followed
  by `iso_blue_noise` (27.7928), Floyd-Steinberg (27.7214), `aniso_flanking` (27.7189),
  `aniso_onedge` (27.6979), and `quadtree_hybrid` (27.6669). This remains shard evidence only:
  one image is not enough to retire ABL-004 or promote README claims.
- 2026-07-06: Completed the next bounded confirmation shard (`kodim02`, budget 2000, all six
  variants, seeds {0,1,2}). The artifact now has 72/1,512 cells with zero errors, leaving
  1,440 cells. At 2000 Gaussians on this single-image slice, `aniso_onedge` leads mean PSNR
  (31.2651), followed by `quadtree_wse` (31.1883), Floyd-Steinberg (30.8809),
  `iso_blue_noise` (30.8091), `quadtree_hybrid` (30.7088), and `aniso_flanking` (30.5071).
  Floyd-Steinberg remains slower to PSNR 30.0 on this slice (124.3 mean iters) than the
  tensor-aware WSE rows (62.3-66.0). This is still partial confirmation evidence only.
- 2026-07-06: Completed the next bounded confirmation shard (`kodim02`, budget 5000, all six
  variants, seeds {0,1,2}). The artifact now has 90/1,512 cells with zero errors, leaving
  1,422 cells. At 5000 Gaussians on this single-image slice, `aniso_onedge` leads mean PSNR
  (33.5770), followed by `quadtree_wse` (33.4950), `quadtree_hybrid` (33.3821),
  `iso_blue_noise` (33.3478), `aniso_flanking` (33.3136), and Floyd-Steinberg (33.2920).
  The 5000-Gaussian aggregate now spans two images and six paired units; `aniso_onedge` remains
  first at 29.5698 mean PSNR, with `quadtree_wse` second at 29.4721.
- 2026-07-06: Completed the next bounded confirmation shard (`kodim02`, budget 10000, all six
  variants, seeds {0,1,2}), finishing the full `kodim02` block for confirmation budgets {2000,
  5000, 10000}. The artifact now has 108/1,512 cells with zero errors, leaving 1,404 cells. At
  10000 Gaussians on this single-image slice, `aniso_onedge` leads mean PSNR (35.7905),
  followed by `quadtree_wse` (35.7700), `iso_blue_noise` (35.7505), `quadtree_hybrid`
  (35.7436), Floyd-Steinberg (35.5835), and `aniso_flanking` (35.5701). Across the current
  two-image 10000-Gaussian aggregate, `quadtree_wse` narrowly leads mean PSNR (31.8131).
- 2026-07-07: Completed ABL-006, the predeclared successive-halving replacement for the remaining
  flat confirmation. Evidence: `ara/evidence/abl006-complete-2026-07-07/`. The staged run completed
  728/728 cells with 0 missing, using Kodak-24 + COCO4, exact CUDA, max-side 768, 1500 iterations,
  and 3-seed finalist confirmation. `aniso_flanking`, `quadtree_hybrid`, `iso_blue_noise`, and
  Floyd-Steinberg were eliminated at stage 1. `quadtree_wse` is the significant budget-5000 PSNR
  winner (+0.0930 dB vs `aniso_onedge`, 95% CI [+0.0168, +0.1700]) and has a small non-significant
  budget-10000 PSNR lead; `aniso_onedge` remains the low-budget/MS-SSIM alternative. README and
  `ara/logic/claims.md` now cite this measured answer. ABL-004 remains partial only because its
  broader original scope still includes the flat full ABL-001/20k/control matrix.
- 2026-07-10: Added an explicit default-promotion check to the fair-density harness summary and
  recomputed `results/fair_gaussian_variants_20260709_best_candidates/` with its local
  `index.html` overview. No candidate passed the predeclared gate of paired mean improvements on
  PSNR, MS-SSIM, AUC, fit seconds, and total seconds. `loss_weight=tensor` improved AUC only
  (+0.0028) while losing PSNR (-0.0201 dB) and MS-SSIM (-0.00084); final color solve improved
  quality (+0.1555 dB PSNR, +0.00084 MS-SSIM) but lost AUC (-0.0030) and speed (+0.1079 s fit).
  Keep `structsplat_best_default` unchanged.
- 2026-07-10: Recomputed the same-hyperparameter fair Gaussian-variant benchmark with the local
  Instant-GI hook enabled (`STRUCTSPLAT_INSTANT_GI=/home/alex/Documents/Instant-GI/quard_image.py`),
  producing `results/fair_gaussian_variants_20260710_full_external_same_hparams/index.html` and
  committed curated evidence under
  `ara/evidence/fair-gaussian-variants-full-external-same-hparams-2026-07-10/`. The rerun completed
  232/232 cells (the prior linked artifact had 224/232 because Instant-GI was unset) at the same
  four COCO images, budget 640, max-side 160, 500 iters, seeds 0/1, exact CUDA. No default candidate
  passed the promotion gate: tensor loss gained PSNR/AUC/speed but lost MS-SSIM, and final color
  solve gained PSNR/MS-SSIM but lost AUC and speed. Keep `structsplat_best_default` unchanged.
- 2026-07-10: Upgraded the proxy artifact analysis with `default_dominance.csv` and image-clustered
  paired 95% confidence intervals across PSNR, MS-SSIM, AUC, fit time, and total time. The result
  now states the review conclusion mechanically: every equal-budget headline reference analogue
  is a tradeoff against the pinned default, and adaptive 1.5x is explicitly not comparable.
  BENCH-005 also added the first true native external path and completed the full COCO4 x seeds
  0/1 review-proxy slice at max-side 160, cap 640, and 500 steps. Paired against a fresh LPIPS-
  enabled StructSplat-default rerun, native GaussianImage++ is faster (+0.4284 s fit-time gain,
  95% CI [+0.2704, +0.6549]) but loses PSNR (-5.0678 dB), proxy MS-SSIM (-0.05142), LPIPS
  (native gain -0.1886), and AUC (-7.1638), with all four quality/convergence CIs below zero.
  Keep this short-horizon matched-axis result separate from native-default/full-resolution claims.
- 2026-07-10: Diagnosed the 5k terminal regression and added FIT-015's post-transition
  `best_psnr_final_count` policy plus a within-trajectory audit. On COCO4 x seeds 0/1, N=640,
  5,000 steps, 7/8 runs restored an earlier state at the identical final count, gaining +0.7702 dB
  PSNR, +0.00892 MS-SSIM, and +0.0076 LPIPS on average over their own terminal states. The 500-step
  guard retained the terminal state in 7/8 runs and was effectively neutral. Keep the pinned
  general default unchanged; expose checkpoint selection as the long-horizon quality candidate.
- 2026-07-10: Rebuilt/reran Image-GS in its official Python 3.11/Torch 2.4/CUDA 12.4 environment
  and added an official base-GaussianImage Python 3.10/Torch 2.0/cu118 runner. At 500 steps,
  StructSplat wins Image-GS final quality familywise and base GaussianImage is far from convergence.
  At 5k, both native methods are tradeoffs against the checkpoint candidate: Image-GS and
  GaussianImage retain proxy-MS-SSIM/speed advantages respectively, while StructSplat has higher
  PSNR and better LPIPS. None is a global dominance result.

## Decision-grade staged protocol

The full ABL-001/ABL-004 matrix remains the gold-standard protocol. If it is not run as a
scheduled multi-day job, use this predeclared staged protocol so stopping is evidence-based:

1. **Screen.** Run all 11 arms on a balanced 8-image subset, budgets {2k, 5k}, seed 0,
   exact CUDA, max-side 768, 1500 iters. Eliminate arms that lose clearly on paired PSNR and
   convergence metrics to both the thesis arm and at least one killer control.
2. **Confirm.** Run the top screen arms plus required controls on all 28 images, seeds {0,1,2},
   budgets {2k, 5k, 10k}. Current confirmation set after the 8-image screen:
   `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`, `iso_blue_noise`, and
   `floyd_steinberg`. Report paired image/seed/budget deltas with confidence intervals and write a
   local `index.html` overview embedding the key metric plots and visual grids.
3. **High-budget check.** Run only finalists at 20k on all 28 images and seeds {0,1,2}, because
   the ABL-001 hypothesis already expects gaps to shrink at high budget.
4. **Promotion rule.** README/ARA claims can be updated only from paired results. If
   `aniso_flanking` does not beat `floyd_steinberg`, `density_random`, and `random_relocate`
   under the confirm/high-budget stages, record the thesis as weakened or failed-with-findings
   instead of continuing the exhaustive grid.

## Interfaces touched
`src/structsplat/sampling.py`, `src/structsplat/init.py` (sampler registration),
`benchmarks/ablation.py`, `benchmarks/abl004_confirmation.py`, `benchmarks/cross_repo_matrix_compare.py`, `benchmarks/abl004_visual_examples.py`, `scripts/run_abl004_full_ablation.sh`,
`ara/evidence/`,
`ara/logic/claims.md`, `README.md`, `tasks/ABL-001-init-sweep.md`, `tests/test_sampling.py`.

## Depends on
BENCH-002 (validity fixes are prerequisites — running the sweep before them wastes the
compute), ABL-003 (know what baseline you're standing on), FIT-004 (relocation control arm;
can run without the stretch items).
