# HIER-001: Progressive pyramid

**Status: done.** Reference densification and prefix metrics work (`pyramid.py`). HIER-003
found the current two-level pyramid is no longer a final-PSNR loser, HIER-004 repaired the AUC
loss with a 150/1350 level-iteration schedule, and the optional additive-renderer comparison was
measured on 2026-07-07. Treat 150/1350 under the normalized renderer as the pyramid quality
candidate; `pyramid=single` stays the broad shipped default until larger multi-seed confirmation.

## Goal
Coarse→fine construction where finer Gaussians are placed where the *residual* structure tensor has
energy; append order forms an LOD prefix.

## Acceptance criteria
- [x] Level 0 from image density; finer levels from residual density; re-fit whole field.
- [x] Verify prefixes as LOD candidates by rendering prefix metrics after fitting.
- [x] Multi-scale tensor `rho` per level.
- [x] Budget schedule diagnosis: HIER-003 showed final-quality upside but poor convergence/AUC.
- [x] AUC/convergence repair and promotion decision (HIER-004): 150/1350 is the candidate,
      default remains `single` pending larger confirmation.
- [x] Optional: additive-renderer mode (ADR-0006) to enable true residual summation (compare).

## Outcome

Evidence:

- `ara/evidence/hier003-pyramid-diagnosis-2026-07-07/`
- `ara/evidence/hier004-pyramid-convergence-repair-2026-07-07/`
- `ara/evidence/hier001-additive-pyramid-2026-07-07/`

The pyramid implementation places level 0 from image density, finer levels from residual density,
tracks prefix metrics when field restructuring does not invalidate prefixes, and supports explicit
per-level iteration schedules. The local quality candidate is the normalized-renderer two-level
schedule with `level_iters=[150, 1350]`: HIER-004 measured +1.0601 dB final PSNR over single-stage
with AUC effectively tied on the difficult-four 2k/5k slice.

The optional additive comparison did not improve the decision. On a matched 512-Gaussian Kodak4
slice, `cuda_additive` lost to `cuda` and additive+pyramid lost to additive+single in every PSNR
pair (-0.3743 dB mean). Additive rendering remains supported and tested, but it is not part of the
HIER-001 promoted recipe.

## Depends on
INIT-002, FIT-001.
