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
- **NumPy, init-time, no autograd:** `structure_tensor` (selectable central/sobel/scharr operator;
  luma or Di Zenzo rgb color space), `density` (structure/gradient/variance/hybrid/uniform modes +
  the inverse-CDF warp for low-discrepancy samplers), `sampling` (WSE blue noise, Poisson-disk
  dart throwing, farthest-point, CVT/Lloyd, Halton, and opt-in terminal-set-preserving progressive
  WSE order), `config`.
- **torch, autograd:** `gaussians` (RS + optional opacity + optional per-Gaussian scale caps,
  ADR-0012), `render` (normalized default + additive, ADR-0006, exact CUDA variants, ADR-0011,
  and gsplat comparator, sharing one accumulator where semantics match), `metrics`, `init`
  (bridge), `fit` (selectable loss/optimizer/LR-schedule/split-mode), `pyramid`, `codec`
  (post-fit quantization, ADR-0007).
- **entry:** `cli` (`structsplat fit` / `ablation` / `stage-search`).

## Stage-search (ABL-002, protocol in ADR-0010)
`benchmarks/stage_search.py` sweeps configurations across every swappable stage — tensor operator,
tensor color space, density mode, sampling mode, orientation mode, init strategy, color mode,
scale mode, opacity, renderer, loss, optimizer, LR schedule, factored refinement
(`refine_site`, `refine_primitive`, `refine_nms`, sampled-add score, plus
color/prune/relocate flags), pyramid — in
two modes:
**factorial** (full product, ranked, for the best complete config) and **influence**
(one-factor-at-a-time paired deltas vs the baseline = first value of each axis; emits
`influence.md` with ΔPSNR/ΔMS-SSIM/ΔAUC/Δiters-to-target/Δseconds per stage option). Configs
whose differing stage is provably inert are canonicalized and deduplicated. Every row records
quality (PSNR/MS-SSIM/LPIPS), convergence (iters-to-target, PSNR-AUC), and speed (init/fit
seconds) so max-quality, max-convergence-rate, and max-speed candidates can be read from the same
run. The shipped defaults (ADR-0009 plus ADR-0013's init-default update) are one named cell in
that space; everything else is a candidate the screening can promote. `benchmarks/ablation.py`
(ABL-001) stays the focused
init-strategy × budget sweep.

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
- `renderer=cuda` and `renderer=cuda_additive` call StructSplat's owned exact CUDA extension for
  the same clipped-support equations. `renderer=gsplat` is kept as a separate alpha/sum comparator
  because it is not numerically equivalent to the normalized reference.
- `scale_cap_mode=feature` gives each Gaussian a local support ceiling from the structure tensor's
  feature run length. `scale_cap_mode=feature_rel` instead derives the cap from local density
  radius / quadtree leaf side with separate along/across multipliers. The fitter clamps optimized
  scales to the field-owned cap, preventing long edge spikes without changing the renderer
  equation. Both cap modes are searchable and default off after INIT-008's fair-density negative.

## Extension seams
- Init strategies: `init.STRATEGIES` (the ablation variables).
- Renderer variants (e.g. additive for AIR-style residuals): behind ADR-0006, keep reference oracle.
- Performance: `PORT-001` CUDA tile rasterizer → IntrinsicEngine RHI pass; reference stays the oracle.
- Feed-forward init predictor (`FF-001`) and compression codec (`COMP-001`) attach after the fitter.
