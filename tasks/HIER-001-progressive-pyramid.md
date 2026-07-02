# HIER-001: Progressive pyramid

**Status: partial.** Reference densification works (`pyramid.py`); ordering guarantees open.

## Goal
Coarse→fine construction where finer Gaussians are placed where the *residual* structure tensor has
energy; append order forms an LOD prefix.

## Acceptance criteria
- [x] Level 0 from image density; finer levels from residual density; re-fit whole field.
- [ ] Verify prefixes are usable as LOD (quality degrades gracefully when truncated).
- [ ] Multi-scale tensor `rho` per level; budget schedule study.
- [ ] Optional: additive-renderer mode (ADR-0006) to enable true residual summation (compare).

## Depends on
INIT-002, FIT-001.
