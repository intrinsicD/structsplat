# CORE-002: RS Gaussian parameterization + conics

**Status: done (reference).** See `gaussians.py`, ADR-0002.

## Acceptance criteria
- [x] 8 params/Gaussian: mean(2), log_scales(2), rotation(1), color(3).
- [x] `conics()` returns inverse-covariance `(a,b,c)`; validated exact vs `inv(R S^2 R^T)`.
- [x] `radii()` from max std; save/load round-trips.
