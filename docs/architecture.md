# Architecture

## Pipeline
```
image (H,W,3) in [0,1]
      │
      ▼
structure_tensor.compute ──► StructureTensor{ lam1,lam2, across_edge_angle, coherence, energy, label }
      │                                   │                 │                         │
      │ energy                            │ eigenvectors    │ eigenvalue pattern      │
      ▼                                   ▼                 ▼                         │
density.py  ── pmf ──►  sampling.eliminate (WSE)  ◄── anisotropy_metric ◄─────────────┘
                          │  exact-N blue noise, density- & anisotropy-adaptive
                          ▼
init.build_field ──► GaussianField{ means, log_scales, rotations, colors }   (RS params)
                          │
                          ▼
fit.fit  ──►  render.render (normalized weighted sum, differentiable)  ──►  Adam (L1 + SSIM)
                          │
                          ▼
pyramid.fit_pyramid: level 0 from image density; finer levels add Gaussians where the *residual*
structure tensor has energy (densification); append order = coarse→fine = LOD prefix.
```

## Module responsibilities
- **NumPy, init-time, no autograd:** `structure_tensor`, `density`, `sampling`, `config`.
- **torch, autograd:** `gaussians`, `render`, `metrics`, `init` (bridge), `fit`, `pyramid`.
- **entry:** `cli` (`structsplat fit` / `structsplat ablation`).

## Extension seams
- Init strategies: `init.STRATEGIES` (the ablation variables).
- Renderer variants (e.g. additive for AIR-style residuals): behind ADR-0006, keep reference oracle.
- Performance: `PORT-001` CUDA tile rasterizer → IntrinsicEngine RHI pass; reference stays the oracle.
- Feed-forward init predictor (`FF-001`) and compression codec (`COMP-001`) attach after the fitter.
