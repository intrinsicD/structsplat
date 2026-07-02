# ADR-0003: Normalized weighted-sum renderer (not additive/alpha, not sorted)

## Context
2D-Gaussian image rasterizers vary: sorted alpha-compositing (3DGS heritage), unnormalized additive
accumulation (AIR — enables clean stage-wise residuals), or normalized weighted summation
(GaussianImage — sorting-free, fast). Our renderer must be differentiable and simple to port.

## Decision
Use **normalized weighted summation**: `I = sum_i c_i G_i / (sum_i G_i + eps)`. No depth sort.

## Consequences
+ Order-independent, sorting-free, cheap, differentiable — the fast-decode property GaussianImage
  reports; a good CUDA port target.
+ Partition-of-unity behavior suits smooth-region tiling; a lone Gaussian renders its flat color.
- Residual composition is **not** additive, so the hierarchy is *densification* (add Gaussians where
  residual error is high, re-fit all) rather than residual summation.
- If we later want AIR-style predicted residuals, add an additive-mode renderer under a new ADR;
  it interacts with the pyramid and with how colors initialize.
