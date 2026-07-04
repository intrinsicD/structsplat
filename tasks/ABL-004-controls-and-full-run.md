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
- [ ] Cross-repo matrix re-run on images disjoint from the four used for config selection,
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

## Interfaces touched
`src/structsplat/sampling.py`, `src/structsplat/init.py` (sampler registration),
`benchmarks/ablation.py`, `benchmarks/cross_repo_matrix_compare.py`, `ara/evidence/`,
`ara/logic/claims.md`, `README.md`, `tasks/ABL-001-init-sweep.md`, `tests/test_sampling.py`.

## Depends on
BENCH-002 (validity fixes are prerequisites — running the sweep before them wastes the
compute), ABL-003 (know what baseline you're standing on), FIT-004 (relocation control arm;
can run without the stretch items).
