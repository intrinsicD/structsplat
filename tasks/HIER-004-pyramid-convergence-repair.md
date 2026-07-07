# HIER-004: Pyramid convergence repair and promotion decision

**Status: todo.** HIER-003 revived pyramid as a final-quality arm but exposed a severe AUC loss.
Decide whether that can be repaired, or document pyramid as an offline-quality / prefix-LOD mode
only.

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
- [ ] At least one full fair-regime candidate is compared against `single_1500` and
      `pyramid_split_1500`.
- [ ] Promotion requires final PSNR within 0.1 dB of `pyramid_split_1500` and AUC loss no worse
      than -0.25 dB versus `single_1500`; otherwise default remains `pyramid=single`.
- [ ] If no candidate meets the rule, update claims/docs to say pyramid is an offline-quality or
      prefix-LOD feature, not a default fitting schedule.
- [ ] Evidence committed under `ara/evidence/hier004-*/`.

## Interfaces touched
Likely `src/structsplat/pyramid.py`, `benchmarks/stage_search.py`, benchmark docs, and task/claim
files. Avoid renderer/compositing changes here; CORE-009 owns the background-layer route.

## Depends on
HIER-001, HIER-003, BENCH-004.
