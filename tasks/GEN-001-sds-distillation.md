# GEN-001: Generative 2D Gaussians via score-distillation of a pretrained diffusion model

**Status: todo (design).** Route 1 — no Gaussian dataset required. Reuses the renderer (CORE-001)
and fitter (FIT-001).

## Context
Generate images *as* 2D-Gaussian sets by distilling a **frozen** pretrained text-to-image (latent)
diffusion model through the differentiable renderer — the VectorFusion / SVGDreamer pattern (score
distillation through a differentiable vector rasterizer, with **no** vector dataset), adapted from
Bézier paths to our 2D Gaussians. Target the niche where 2D Gaussians beat pixels first:
flat / iconographic / poster / texture / line-art prompts, with resolution-free, editable output.

## Goal
Optimize a `GaussianField` θ so that `render(θ)` is a high-probability sample under a frozen
pretrained diffusion model, conditioned on a prompt, using score distillation.

## Approach (sketch)
1. **SDS gradient.** For x = render(θ): draw t, ε; form `x_t = α_t x + σ_t ε`; query the frozen
   U-Net `ε_φ(x_t, t, y)`; apply `∇_θ L = E_{t,ε}[ w(t) (ε_φ − ε) · ∂x/∂θ ]`. Do **not** backprop
   through `ε_φ` — its output *is* the gradient direction; backprop `∂x/∂θ` through the renderer into
   the Gaussian params.
2. **Latent variant (default for SD-class models).** Encode `z = E(x)` with the frozen VAE encoder,
   run SDS in latent space; gradient flows `ε_φ(z_t) − ε` → through `E` → through the renderer → θ.
3. **VSD / VPSD upgrade.** Replace the SDS mean-seeking term with a variational score (a LoRA on the
   frozen model modeling the particle distribution) to cut SDS over-saturation and mode collapse
   (SVGDreamer's VPSD). Heavier, much better quality/diversity; tracked as GEN-003 once the SDS
   baseline is debuggable.
4. **Init from a raster sample (fast path).** Sample one raster image from the pretrained model, fit
   it with StructSplat (`init` + `fit`), then SDS/VSD-refine in Gaussian space. Reuses the whole
   analysis stack and converges far faster than from scratch (VectorFusion initializes from an image
   sample for the same reason).
5. **Compositing.** Use the additive / alpha renderer (ADR-0006) so overlapping Gaussians can
   occlude; opacity also gives a pruning signal for variable cardinality.
6. **Coarse-to-fine (optional).** Drive the Gaussian count with the progressive ordering (HIER-001):
   few Gaussians at high noise levels, densify as t decreases.

## Acceptance criteria
- [ ] SDS gradient path end-to-end (pixel and latent variants); gradient finite and non-NaN.
- [ ] Runs from a frozen pretrained pipeline behind an optional `[gen]` extra (diffusers,
      transformers, accelerate). **Core repo still installs and `pytest -q` passes without `[gen]`.**
- [ ] Classifier-free guidance exposed (high CFG for vanilla SDS; normal CFG for VSD).
- [ ] `raster-sample → fit → refine` pipeline wired, reusing `init` / `fit`.
- [ ] Uses the additive renderer mode; opacity + scale regularization prevents degenerate Gaussians.
- [ ] VSD / LoRA variant behind a flag, or explicitly handed off to GEN-003 after the baseline.
- [ ] Eval: CLIP score (prompt alignment) + multi-resolution renders of the same θ (demonstrating
      resolution independence) + Gaussian count; optional FID on a class-conditional set.
- [ ] `structsplat generate "<prompt>" --n 5000 --steps N` subcommand; saves θ (.npz) + PNG(s).

## Interfaces touched
`src/structsplat/generate.py` (new) · `render.py` (additive mode, ADR-0006) · `gaussians.py`
(opacity param) · `config.py` (`GenConfig`) · `cli.py` (`generate` subcommand) · `pyproject.toml`
(`[gen]` extra).

## Depends on
CORE-001 (renderer) · ADR-0006 (additive / alpha compositing) · ADR-0002 / CORE-002 (opacity param).
Optional: FIT-001 + `init` (raster-init path) · HIER-001 (coarse-to-fine).

## Notes
- SDS pitfalls: over-saturated, over-smoothed, low-diversity samples — move to VSD once the SDS
  baseline works; vanilla SDS typically needs very high CFG.
- No ground truth ⇒ no PSNR. Evaluate with CLIP / FID + qualitative, and lean on the primitive
  advantages (arbitrary resolution, editability, compactness) as the story.
- This is the **distillation** route (no dataset). The **native** route — fit a corpus to Gaussians,
  bake them into a splatter-grid, and run latent diffusion on the grid (DiffSplat-style) — is a
  separate future task (**GEN-002**) and yields a fast feed-forward sampler once trained.
