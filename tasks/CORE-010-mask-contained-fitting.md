# CORE-010: Mask-contained fitting for alpha-masked images

## Status

Implemented (opt-in, default off) on 2026-07-21. Everything is opt-in and default-off; no default
change, no compression claim, no novelty claim. Distinct from CORE-007/008, which stay design-only:
this task targets *external* alpha masks (sprite/matte inputs) and containment of the fitted field,
not internal-boundary mixing, and it needs no BENCH-007 authorization because it promotes no
structural-prior or rate claim.

Landed: `src/structsplat/mask.py` (NumPy EDT/SDF/erosion/nearest-inside/color-dilation), the
hard-containment loop (`fit._MaskConstraint`: mean projection + dynamic signed-distance scale caps,
ADR-0017), the soft out-of-mask coverage penalty, `loss_weighting="mask"` with SSIM matting,
`init.build_masked_field`, CLI flags (`--mask`, `--mask-contain`, `--mask-margin`,
`--mask-coverage-weight`, `--loss-weighting mask`), and `tests/test_mask.py`. Follow-up (own tasks):
the committed five-arm containment-cost benchmark with numbers on a real masked dataset (still
deferred; CORE-011's modes become additional arms), the directional cap extension (landed
2026-07-22 as CORE-011's certified anisotropic caps, ADR-0019, together with the boundary-band
under-coverage penalty and boundary tangent densification), and pyramid support.

## Context

Fitting a matted image (object over a constant matte) leaks Gaussians across the mask boundary,
and the normalized renderer (ADR-0003) makes the leak look worse than in additive systems:

- Outside the mask an overhanging Gaussian is typically the **only** contributor to a pixel, so
  the compositor renders its full color at any tail weight (`w*c/(w+eps)` in `render.py`).
  Overhang appears as full-strength stamps, not faded tails.
- At lone-contributor pixels the geometry gradient of the rendered color vanishes
  (`d/dw [w/(w+eps)] = eps/(w+eps)^2 ~ 0`), so color supervision outside the mask — matte
  targets or randomized fill — cannot push geometry back inside; it mostly drags boundary
  colors toward the outside target. That is the observed bleed + dark-fringe equilibrium, and
  the reason a random-color-outside fill is rejected as the mechanism here.
- With the default `support_fade=False`, rendered support is the whole AABB tile of the
  sigma_cutoff ellipse (CORE-003). Only `support_fade=True` (CORE-005) makes support exactly
  the cutoff ellipse, which is what makes an exact zero-outside statement possible.
- The loss cannot currently ignore outside pixels: `_prepare_loss_weight_map` (FIT-012) maps any
  provided map into `[1, 1+beta]`, and the SSIM term is unweighted full-image.

Prior-art boundary: region/segmentation-constrained rasterization exists (Contour-Aware 2DGS,
already the CORE-007 baseline), random-background supervision is standard in masked NeRF/3DGS
training, sprite-border color dilation is standard practice, and distance-transform constraints
are classical. This task claims a capability for this codebase, not a method contribution.

## Goal

Given `(image, binary mask)`, fit a field whose **effective** sigma_cutoff support is contained
in the mask, so a cold decode — the codec intentionally does not store `scale_max` — renders
exactly zero outside the mask **without access to the mask**, while losing as little in-mask
quality as possible versus unconstrained fitting.

Containment definition: with `support_fade=True`, the sigma_cutoff ellipse of the *effective*
covariance (base scales + `filter_variance` + `aa_dilation`; see
`GaussianField.effective_scales`) lies inside the mask. The guarantee uses the sufficient
isotropic cap `sigma_cutoff * max(sx_eff, sy_eff) <= SDF(mu) - margin` with `margin >= 1 px`
(integer tile rounding + bilinear SDF interpolation error).

## Approach

1. **Mask/SDF module (NumPy, invariant 1).** New `src/structsplat/mask.py`: exact separable EDT
   (Felzenszwalb two-pass; scipy is not a dependency), signed distance, erosion,
   nearest-inside-pixel table, push--pull color dilation. Importable without torch.
2. **Mask-consistent loss.** `loss_weighting="mask"`: use the provided `(H,W)` map raw (zeros
   allowed) for the pixel term; matte both render and target before SSIM so boundary windows
   see consistent zeros on both sides.
3. **Mask-aware init.** Density multiplied by the eroded mask (the `density=` override of
   `build_field` already exists); structure tensor stays on the *matted* image — the matte edge
   is a useful boundary density/orientation attractor (tangent-aligned, thin-across seeds per
   invariant 3); colors sampled from the *color-dilated* image to avoid matte contamination;
   mask caps composed into the existing `_scale_caps` output with `np.minimum` (ADR-0012).
4. **Fit-time hard containment.** In the existing post-step `no_grad` clamp block in `fit.py`:
   project means into the eroded mask via the nearest-inside table, then refresh the mask
   component of `scale_max` from `SDF(mu)`. Split/relocate/adaptive-add children derive their
   mask cap from SDF at the child position instead of `_nearest_scale_caps` inheritance.
   Requires an ADR: this extends ADR-0012 enforcement from static caps to position projection
   plus dynamic caps.
5. **Soft out-of-mask coverage penalty** (the principled replacement for random fill):
   `L_out = lambda * mean_{p not in M} den(p)` on the raw unnormalized weight sum, which keeps
   full-strength geometry/opacity gradients precisely because it is the gauge that
   normalization cancels. Used at small `lambda` to smooth optimization before hard clamps
   bind, and usable alone as a soft mode.
6. **Optional quality extension**, only if the isotropic cap's in-mask cost is too high:
   directional cap `sigma_cutoff * sqrt(n^T Sigma n) <= d_normal` along the local boundary
   normal (keeps edge Gaussians long-along-tangent), guarded by the isotropic cap near corners.

## Non-goals

- Masked-domain rendering (weights times mask at rasterization): guarantees zero bleed but
  requires the mask at decode and a renderer-semantics ADR; separate task if that use-case
  materializes.
- Reopening CORE-007/008 internal-boundary questions or any BENCH-007-gated claim.
- Default changes: with no mask supplied, behavior must be unchanged.

## Acceptance criteria

- [x] `mask.py` EDT/SDF/nearest-inside match brute force on random small masks including
      concave shapes and holes; module imports without torch (test).
- [x] Masked init places every seed in the eroded mask and every initial effective cutoff
      ellipse inside the mask (property test).
- [x] Hard-contained fit keeps means and effective cutoff ellipses contained, including
      prune/split/relocate events; with `support_fade=True` the rendered image is exactly zero
      outside the mask at the end of the fit (test).
- [x] `loss_weighting="mask"` zero-weights outside pixels and the SSIM path mattes both sides
      (test); the coverage penalty reduces out-of-mask weight sum on a synthetic overhang
      fixture with nonzero geometry gradients (test).
- [ ] Containment cost measured on a small masked benchmark with arms: current behavior (matte
      target), random-color control, coverage-penalty-only, hard-contained, hard+penalty.
      Report in-mask PSNR/MS-SSIM, boundary-band (<=2 px inside) PSNR, out-of-mask energy, and
      composite-over-random-background PSNR, reproducible from logged configs + seeds. Honest
      trade-off reporting; no promotion rule — expect an in-mask cost (INIT-008 measured
      -0.3733 dB mean for feature caps on the fair-density screen). **Deferred:** the arms are all
      runnable from CLI flags today (the CLI already prints out-of-mask mean |render|), but a
      committed benchmark script + dataset numbers is follow-up.
- [x] No-mask paths unchanged (regression test); NumPy/torch split intact; ADR-0017 written for
      the projection/dynamic-cap decision; README/architecture/INDEX updated in the same commit.

## Interfaces touched

`src/structsplat/mask.py` (new), `src/structsplat/config.py` (opt-in knobs),
`src/structsplat/init.py`, `src/structsplat/fit.py`, `benchmarks/` (small masked-arm script),
`tests/`. Not `src/structsplat/render.py`.

## Depends on

CORE-003, CORE-005, INIT-002/003, FIT-012, ADR-0003, ADR-0012 (+ INIT-008 cap-cost evidence).
