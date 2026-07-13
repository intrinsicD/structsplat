# ADR-0001: PyTorch reference implementation first, CUDA/Vulkan later

**Status update (2026-07-13):** the reference-first decision remains the correctness policy, but
the exact CUDA renderer was subsequently implemented under ADR-0011. PORT-001 now denotes the
remaining production/tiled/Vulkan/RHI path, not the absence of a CUDA research renderer.

## Context
The original research question was about *initialization* (structure-tensor anisotropic blue noise)
and its effect on 2D-Gaussian image reconstruction across budgets. That demanded fast iteration on init
variants, an autograd fitter, optional perceptual metrics, and possibly a learned feed-forward
predictor — all of which are cheapest in PyTorch and match the reference ecosystem (GaussianImage,
Image-GS, AIR). The end goal is still a CUDA/Vulkan rasterizer inside IntrinsicEngine.

## Decision
Build the research + ablation repo in PyTorch. Keep the differentiable rasterizer and the sampler
as clearly bounded modules. Defer the CUDA tile rasterizer and the IntrinsicEngine RHI pass to
`PORT-001`, with the PyTorch/NumPy versions as the correctness oracle.

## Consequences
+ Fast to run the ablation; autograd for free; easy metrics and future feed-forward net.
+ Clean seam for porting (`render.py`, `sampling.py`).
- The reference renderer is slow at large N (Python-driven chunk loop). Acceptable for research on
  small budgets/images; not the deployment path.
