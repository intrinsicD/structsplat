# ADR-0017: Mask-contained fitting via mean projection + dynamic signed-distance scale caps

## Context

Fitting an alpha-masked image (an object matted over a constant background) leaks Gaussians across
the mask boundary. The normalized compositor (ADR-0003) makes the leak conspicuous: outside the
mask an overhanging Gaussian is usually the only contributor to a pixel, so `w*c/(w+eps)` renders
its full color at any tail weight, and the geometry gradient at a lone-contributor pixel is
`eps/(w+eps)^2 ~ 0`, so color supervision (matte targets, or randomized background fill) cannot
push the geometry back inside — it only drags boundary colors. Randomized-background supervision is
an alpha-compositing device with no port to a normalized weighted-sum renderer.

ADR-0012 already stores an optional per-Gaussian `scale_max` and has the fitter clamp optimized
scales to it after each step. That machinery caps *extent* but not *position*, and its caps are
static (assigned at init from local structure). Containment additionally needs positions kept
inside the mask and caps that track a Gaussian as it moves relative to the boundary.

## Decision

Add an opt-in mask-containment mode (`FitConfig.mask_contain`, default off) that, wherever the
fitter already clamps to `scale_max` — after every optimizer step, after each prune/split/relocate/
growth event, at fit entry, and before the final render — additionally:

1. **projects** each mean into the eroded mask interior using a precomputed nearest-inside feature
   transform, and
2. **overwrites** `scale_max` with a fresh isotropic cap derived from the signed distance,
   `sigma_cutoff * sqrt(sx_eff^2 + extra) <= SDF(mu) - margin` (extra = filter variance + AA
   dilation), so the sigma_cutoff ellipse of the *effective* covariance stays inside the mask.

Because the cap is recomputed from the current position each call, `mask_contain` **owns**
`scale_max` and does not compose with `scale_cap_mode` (the two cap systems are mutually
exclusive; `build_masked_field(contain=True)` and combining a pre-capped field with `mask_contain`
are rejected). With `support_fade=True` (ADR-0005/CORE-005) the renderer weight is exactly zero
outside the ellipse, so a contained field renders exactly zero outside the mask and a cold decode
reproduces that **without the mask** — `scale_max` is deliberately not encoded (COMP codec), yet
containment is baked into the parameter values.

Two supporting, independently-usable pieces ship alongside it: a differentiable soft
`mask_coverage_weight` penalty on the raw out-of-mask weight sum (full-strength geometry/opacity
gradients, the principled replacement for random fill; usable without hard containment), and a
`loss_weighting="mask"` mode that drops out-of-mask pixels from the pixel loss and mattes both
sides before SSIM. The exact separable distance transform, signed distance, erosion, nearest-inside
feature transform, and boundary color dilation live in `mask.py` as pure NumPy (core invariant 1;
scipy is not a dependency). The renderer is unchanged — this is a field/fitter constraint, not a
compositing change, so ADR-0003 stands.

## Consequences

+ The fitted field is provably contained: means inside, and the sigma_cutoff ellipse of the
  effective covariance inside the mask (sufficient isotropic condition, `margin >= 1` px covers
  integer tile rounding and nearest-pixel SDF error). With `support_fade=True`, exact zero outside.
+ Reuses the ADR-0012 enforcement point and the existing `density=`/`tensor=` init overrides;
  masked density restricts seeds, boundary color dilation removes matte contamination while the
  matte edge is intentionally retained in the structure tensor as a boundary attractor.
+ Clean codec story: containment survives a mask-free cold decode.
- Hard caps cost in-mask fidelity near the boundary (INIT-008 measured -0.3733 dB mean for feature
  caps on the fair-density screen); the isotropic cap is conservative at corners. A directional
  cap along the boundary normal is a documented future extension (CORE-010).
- Mutually exclusive with `scale_cap_mode`; not yet wired through the pyramid.
- The guarantee assumes `support_fade=True` for exact-zero and `filter_variance=0` when the cap's
  min-scale clamp would otherwise bind; both are the default masked-fitting configuration.

## Links

Amends ADR-0012's `scale_max` enforcement (static -> projection + dynamic caps). Implements
CORE-010. Related: ADR-0003 (normalized renderer), ADR-0005/CORE-005 (compact support fade),
FIT-012 (loss weighting), COMP codec (`scale_max` intentionally unencoded).
