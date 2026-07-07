# HIER-004: Pyramid convergence repair and promotion decision

**Status: done.** Explicit per-level iteration schedules are implemented. The 150/1350
two-level schedule repairs HIER-003's AUC loss while preserving the final-quality gain on the
difficult-four slice. Keep `pyramid=single` as shipped/default until larger confirmation; promote
150/1350 as the pyramid quality candidate.

## Context
HIER-003 ran the fair-regime difficult-four diagnosis and found:

- `pyramid_split_1500` won final PSNR in 16/16 pairs against single-stage at equal nominal
  iterations: +1.0000 dB mean PSNR, +0.01399 MS-SSIM, edge MAE better in 11/16 pairs.
- The same arm lost PSNR AUC in 16/16 pairs: -1.3540 mean AUC.
- Extra coarse iterations (`pyramid_fullfield_iters`) did not explain the result; 0.1/0.9 +
  cosine was worse for AUC and edge MAE; the matched residual-add refine twin lost both final PSNR
  and AUC.

So the pyramid idea is not failed, but the schedule is not default-ready.

## Goal
Find a schedule that keeps most of the final-PSNR gain while reducing the AUC loss, or close the
default candidacy honestly.

## Protocol
Use the same difficult-four exact-CUDA slice as HIER-003 unless a cheaper proxy is first used for
screening and then confirmed at the full fair regime.

Candidate arms:

1. `pyramid_split_1500` control from HIER-003.
2. delayed-level insertion variants that give the final full field more of the 1500-iteration
   horizon (for example 0.2/0.8, 0.25/0.75, 0.5/0.5 iteration shares).
3. coarse-level loss/metric warmup variants: lower coarse LR, lower coarse SSIM weight, or shorter
   coarse horizon so early AUC is not dominated by an intentionally under-capacity field.
4. optional prefix-aware AUC reporting: if pyramid is meant as a streaming LOD method, report
   prefix AUC separately from final-full-field AUC rather than hiding the tradeoff.

## Acceptance criteria
- [x] At least one full fair-regime candidate is compared against `single_1500` and
      `pyramid_split_1500`.
- [x] Promotion requires final PSNR within 0.1 dB of `pyramid_split_1500` and AUC loss no worse
      than -0.25 dB versus `single_1500`; otherwise default remains `pyramid=single`.
- [x] Claims/docs updated with the candidate/default decision.
- [x] Evidence committed under `ara/evidence/hier004-*/`.

## Outcome

Evidence: `ara/evidence/hier004-pyramid-convergence-repair-2026-07-07/`.

Implemented:

- `PyramidConfig.level_iters: list[int] | None` for explicit coarse-to-fine iteration schedules.
- `structsplat fit --level-iters ...`.
- `stage_search --pyramid-level-iters ...`.
- Stage-search rows record `level_iters` and `level_budgets`.

Fair-regime difficult-four result:

| arm | dPSNR vs 750/750 pyramid | dAUC vs single | decision |
|---|---:|---:|---|
| `pyramid_leveliters_150_1350` | +0.0601 | +0.0011 | pass |
| `pyramid_leveliters_200_1300` | -0.0429 | -0.1319 | pass |
| `pyramid_leveliters_300_1200` | -0.0562 | -0.3567 | fail AUC |
| `pyramid_leveliters_375_1125` | -0.0531 | -0.5126 | fail AUC |
| `pyramid_leveliters_500_1000` | -0.1543 | -0.8244 | fail PSNR/AUC |

`150/1350` is best: it beats single-stage by +1.0601 dB final PSNR and is AUC-neutral
(+0.0011), while slightly beating the HIER-003 750/750 pyramid control. This repairs the local
convergence failure. Decision-grade default promotion still needs a larger multi-seed
confirmation; until then, `pyramid=single` remains the broad default and `150/1350` is the pyramid
quality candidate.

## Interfaces touched
`src/structsplat/pyramid.py`, `src/structsplat/config.py`, `src/structsplat/cli.py`,
`benchmarks/stage_search.py`, `tests/test_pyramid.py`, `tests/test_stage_search.py`, benchmark
docs, and task/claim files. Renderer/compositing remains untouched; CORE-009 owns the
background-layer route.

## Depends on
HIER-001, HIER-003, BENCH-004.
