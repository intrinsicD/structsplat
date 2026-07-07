# HIER-001: Progressive pyramid

**Status: partial.** Reference densification and prefix metrics work (`pyramid.py`). HIER-003
found the current two-level pyramid is no longer a final-PSNR loser (+1.0000 dB mean over the
difficult-four 2k/5k slice), but it loses PSNR AUC badly. Treat it as an offline/final-quality
candidate until HIER-004 repairs or bounds the convergence cost.

## Goal
Coarse→fine construction where finer Gaussians are placed where the *residual* structure tensor has
energy; append order forms an LOD prefix.

## Acceptance criteria
- [x] Level 0 from image density; finer levels from residual density; re-fit whole field.
- [x] Verify prefixes as LOD candidates by rendering prefix metrics after fitting.
- [x] Multi-scale tensor `rho` per level.
- [x] Budget schedule diagnosis: HIER-003 showed final-quality upside but poor convergence/AUC.
- [ ] AUC/convergence repair and promotion decision (HIER-004).
- [ ] Optional: additive-renderer mode (ADR-0006) to enable true residual summation (compare).

## Depends on
INIT-002, FIT-001.
