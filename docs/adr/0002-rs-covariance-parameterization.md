# ADR-0002: Rotation-Scaling (RS) covariance parameterization

## Context
A 2D Gaussian's covariance can be parameterized by a Cholesky factor (3 free params) or by
rotation + per-axis scale (RS). The init sets orientation and anisotropy *directly* from the
structure tensor's eigenvectors/eigenvalues.

## Decision
Use RS: per Gaussian store `mean(2), log_scales(2), rotation(1), color(3)` = 8 params. Covariance
`Sigma = R S^2 R^T`; render uses the inverse-covariance "conic" `(a,b,c)`. `log_scales` keeps
scales positive and clamps cleanly.

## Consequences
+ Structure-tensor init maps straight onto parameters: `theta = along-edge angle`, `sx = s_along`,
  `sy = s_across`. Orientation and extent are decoupled and independently clampable.
+ Matches GaussianImage's 8-parameter budget; opacity is folded into unbounded color.
- Slightly more trig in the conic computation than Cholesky (negligible; validated in tests).
