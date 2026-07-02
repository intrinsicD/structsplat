# HIER-001: Progressive pyramid

**Status: partial.** Reference densification and prefix metrics work (`pyramid.py`); full LOD study pending.

## Goal
Coarse→fine construction where finer Gaussians are placed where the *residual* structure tensor has
energy; append order forms an LOD prefix.

## Acceptance criteria
- [x] Level 0 from image density; finer levels from residual density; re-fit whole field.
- [x] Verify prefixes as LOD candidates by rendering prefix metrics after fitting.
- [x] Multi-scale tensor `rho` per level.
- [ ] Budget schedule study.
- [ ] Optional: additive-renderer mode (ADR-0006) to enable true residual summation (compare).

## Depends on
INIT-002, FIT-001.
