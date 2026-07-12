---
name: core
description: Load at the start of any StructSplat work session. Repo map, invariants, naming, and the layered architecture (structure-tensor init -> anisotropic blue-noise sampling -> RS Gaussians -> normalized renderer -> fit -> pyramid). Use when orienting in the codebase, deciding where code belongs, or before editing any module.
---

# StructSplat — core conventions

Feature-aware, hierarchical, anisotropic **blue-noise 2D Gaussian image representation**. Research
reference in PyTorch; the rasterizer + sampler are the pieces later ported to CUDA/Vulkan and into
IntrinsicEngine as an RHI pass.

## Layer map (data flows top to bottom)
- `structure_tensor.py` (NumPy) — J = G_rho * (grad I grad I^T); eigen-analysis gives **density**,
  **orientation**, and **flat/edge/corner** labels. One operator, three jobs (ADR-0004).
- `density.py` (NumPy) — energy -> density pmf; residual density for pyramid levels.
- `sampling.py` (NumPy) — Weighted Sample Elimination: exact-N blue noise, density-adaptive via
  per-point radius, anisotropic via a per-point metric tensor (ADR-0005), with opt-in
  terminal-set-preserving progressive survivor order (INIT-009).
- `gaussians.py` (torch) — `GaussianField`, RS parameterization, conics, radii (ADR-0002).
- `render.py` (torch) — differentiable normalized-weighted-sum rasterizer, no sort (ADR-0003).
- `init.py` (torch bridge) — the five strategies in `STRATEGIES` (the ablation variables).
- `fit.py` (torch) — Adam fitter, L1+SSIM, records PSNR history + iters-to-target.
- `pyramid.py` (torch) — progressive densification driven by residual structure tensor.

## Invariants (do not break without an ADR)
1. Init-time math stays **NumPy and importable without torch**. Autograd stays in torch modules.
2. Images are `(H, W, 3)` float32 in `[0, 1]`; positions are `(x, y)` in pixel coords.
3. RS parameterization: `theta` = angle of the `sx` axis; edge Gaussians elongate **along** the
   tangent (`sx = s_along`, `sy = s_across`).
4. The renderer is **normalized** (divides by summed weight). Residual work is densification, not
   additive summation — if you need additive, write ADR-0006 first.
5. Every experiment is reproducible: seed flows InitConfig.seed -> RNG; log the config with results.

## Where things go
- New init strategy -> `init.py` + add to `STRATEGIES` + register in `benchmarks/ablation.py`. See `method`.
- New metric -> `metrics.py`, wired through `fit.py` output dict and `BENCH-001`.
- Perf/CUDA -> `PORT-001`; keep the NumPy/torch reference as the correctness oracle.

## Naming
`structsplat` is a **placeholder project name** — if it changes, update `pyproject.toml`, imports,
`README`, and this file in one commit (see `docs-sync`).
