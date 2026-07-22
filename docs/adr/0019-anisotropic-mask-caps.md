# ADR-0019: Anisotropic mask caps via station-ball SDF certificates

## Context

ADR-0017's containment cap is isotropic: `sigma_cutoff * max_axis <= SDF(mu) - margin`. That is
the only sound cap derivable from a *single* SDF sample (the Lipschitz ball argument), but it is
exactly what starves the boundary of mask-contained fits (CORE-011): near the boundary the cap
approaches the minimum scale in *every* direction, so edge Gaussians cannot elongate along the
boundary tangent (invariant 3) and the band can only be tiled by O(perimeter) confetti. The
observed result is a hole band hugging the mask edge after convergence.

Any anisotropic relaxation needs more boundary information than one SDF sample. Candidate
sources: a boundary-normal field (slab condition `sigma_cutoff * sqrt(n^T Sigma n) <= d`), local
curvature estimates, or additional SDF samples. Normal/curvature models fail exactly where the
cap matters most (corners, thin structures, mask holes) and would need separate guards.

## Decision

Opt-in `FitConfig.mask_cap_mode="anisotropic"` (default `"isotropic"`, ADR-0017 behavior
bit-unchanged). The short axis keeps the isotropic cap. The long axis gets the largest cap from a
geometric ladder (up to ~18.6x the isotropic reach) that passes a **station-ball certificate**
built from additional SDF samples:

Cover the sigma_cutoff ellipse (along half-extent `L`, across half-extent `w`) with `2J` balls
centred on its long axis at stations `t_j = +-(j+0.5)*delta`, `delta = L/J <= ~1 px`. The slab of
the ellipse owned by station `j` fits in a ball of radius `r_j = sqrt(w(t_in)^2 + (delta/2)^2)`
(`w(t)` the cross-section halfwidth at the slab edge nearer the centre). Since the SDF is
1-Lipschitz and bilinear interpolation of grid samples errs by at most ~0.71 px (already absorbed
by `margin > ~0.71`, same accounting as ADR-0017), `SDF(c_j) >= margin + r_j` at every station
implies the whole ellipse — interior included, so enclosed mask holes are covered too — lies
strictly inside the mask. No normal or curvature model is trusted: corners, curvature, thin
structures, and holes bind the cap through the probes themselves. The ladder's floor is the
always-valid isotropic cap; the ball inflation over the exact slab is `O(delta^2 / d)` (station
spacing shrinks near the frontier), so near-touching, thin, tangent-aligned Gaussians — the
boundary coverage workhorses — certify with sub-0.1 px loss.

Certification runs on a cadence (`mask_cap_refresh_every`, default 10) plus fit entry, every
restructure event, and the terminal apply; between refreshes only mean projection and clamping to
the stored caps run. Certified caps depend on the row's current rotation and across scale, so a
mid-fit state between refreshes is approximate in the same sense ADR-0017's per-step cap lag
already is; every post-refresh state — in particular the returned/terminal field — is certified
at its actual parameters, so the ADR-0017 guarantee is preserved: with `support_fade=True` the
fitted field renders exactly zero outside the mask on a cold decode without the mask.

Two companion CORE-011 features ship alongside (opt-in, independent): the boundary-band
under-coverage hinge on the raw weight sum (the in-mask twin of the CORE-010 coverage penalty,
same normalization-gauge argument), and boundary tangent densification seeded by
`mask.boundary_normals` (pure NumPy, smoothed-SDF gradient; invariant 1 upheld).

## Consequences

+ Boundary Gaussians elongate along the tangent instead of tiling: certified caps reach ~19x the
  isotropic reach on straight edges, automatically shrinking at corners/curvature.
+ Guarantee unchanged: containment claims attach to post-refresh states exactly as ADR-0017's
  attach to post-apply states; the exact-zero-outside test remains the acceptance oracle.
+ The dead band (pixels with SDF below ~margin) is untouched by construction — that is a
  `mask_margin` trade-off, not a cap-mode issue (CORE-011 documents it).
- More SDF gathers: active rows x ladder x stations per refresh (vectorized; cadence bounds the
  cost). Near-isotropic rows gain nothing (the ball inflation is tight only for slim
  cross-sections) — acceptable, they are not the boundary mechanism.
- Certified caps are state-dependent (rotation/across-scale), so `scale_max` is even more
  explicitly fit-state, not codec state (`scale_max` remains unencoded, ADR-0017/COMP).

## Links

Amends ADR-0017 (cap derivation; enforcement points unchanged). Implements CORE-011 with
CORE-010 as the base capability. Related: ADR-0003 (normalized renderer), ADR-0005/CORE-005
(compact support fade), ADR-0012 (`scale_max` machinery).
