# Problem

Single-image 2D Gaussian representations can fit images compactly, but they usually rely on random
or weakly structured initial placement plus iterative optimization to discover geometry. That makes
low-budget quality, convergence speed, and density-control behavior difficult to reason about.

StructSplat asks whether a deterministic, feature-aware initial field can make the representation
more predictable:

- use one structure tensor for density, orientation, and flat/edge/corner classification;
- place Gaussians with density-adaptive blue-noise sampling rather than clumped random samples;
- treat edge discontinuities explicitly by comparing on-edge and flanked placement;
- keep the renderer/loss/optimizer policy explicit enough that ablations can isolate each stage.

The open scientific question is not "can a fitter eventually overfit one image?" It is: under
matched budgets, metrics, renderers, and stopping rules, which placement, scale, renderer, and
densification stages actually improve quality, convergence, or storage?
