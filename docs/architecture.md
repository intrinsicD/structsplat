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
- **NumPy, init-time, no autograd:** `structure_tensor` (selectable central/sobel/scharr operator),
  `density` (structure/gradient/variance/hybrid/uniform modes), `sampling`, `config`.
- **torch, autograd:** `gaussians` (RS + optional opacity), `render` (normalized default +
  additive, ADR-0006, sharing one accumulator), `metrics`, `init` (bridge), `fit` (selectable
  loss/optimizer/LR-schedule/split-mode), `pyramid`, `codec` (post-fit quantization, ADR-0007).
- **entry:** `cli` (`structsplat fit` / `ablation` / `stage-search`).

## Stage-search (ABL-002)
`benchmarks/stage_search.py` sweeps *complete* configurations across every swappable stage — tensor
operator, density mode, sampling mode, init strategy, color mode, scale mode, opacity, renderer,
loss, optimizer, LR schedule, refinement, pyramid — and emits ranked JSON/CSV/markdown. The shipped
defaults (ADR-0009) are one named cell in that space; everything else is a candidate the screening
can promote. `benchmarks/ablation.py` (ABL-001) stays the focused init-strategy × budget sweep.

## Performance notes (reference is the oracle; these keep it usable at N~20k on CPU)
- `sampling.eliminate` builds the WSE conflict graph vectorized over grid-cell offsets (only the
  greedy heap removal stays in Python); the anisotropic search reach is bounded per receiver by the
  metric's minimum eigenvalue, so no long-range along-edge conflict is missed. ~30x faster than the
  original per-pair Python loops at N=20k.
- `render` evaluates each Gaussian on the axis-aligned bounding box of its `sigma_cutoff` ellipse
  (per-axis radii `(rx, ry)`), laid out as one ragged flat tensor — no padding to a shared square
  tile. Elongated anisotropic Gaussians get a tight rectangle instead of a square sized by the major
  axis (~3x forward speedup on a flanking init). Still fully differentiable; radii stay detached.
- `render`/`conics` take an optional EWA-style `aa_dilation` (Sigma + d·I) low-pass for sub-pixel
  Gaussians — off by default; exact under RS since it only shifts the per-axis variances.

## Extension seams
- Init strategies: `init.STRATEGIES` (the ablation variables).
- Renderer variants (e.g. additive for AIR-style residuals): behind ADR-0006, keep reference oracle.
- Performance: `PORT-001` CUDA tile rasterizer → IntrinsicEngine RHI pass; reference stays the oracle.
- Feed-forward init predictor (`FF-001`) and compression codec (`COMP-001`) attach after the fitter.
