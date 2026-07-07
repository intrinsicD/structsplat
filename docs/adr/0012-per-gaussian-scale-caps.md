# ADR-0012: Per-Gaussian scale caps

## Context

Unbounded scale optimization can produce long, spiky Gaussians that blur across unrelated
structures, hide missing detail, and waste renderer support area. The issue is most visible for
anisotropic edge Gaussians: a broad tangent-axis scale can preserve PSNR while creating artifacts
and increasing fit/render time.

Feature-adaptive scale-cap experiments showed that capping support by local structure can improve
mean PSNR and reduce runtime in the small COCO screens. A hard global cap is useful as a control,
but local features are a better prior: long support is reasonable in smooth regions and risky near
short or interrupted edges.

## Decision

`GaussianField` may carry an optional per-Gaussian `scale_max`. Initialization can attach caps
through `InitConfig.scale_cap_mode`:

- `none`: no cap, historical behavior;
- `hard`: a global absolute sigma cap;
- `feature`: a local feature-aware cap from structure-tensor run lengths and configured minimums.
- `feature_rel`: a feature-relative cap from local density radius / quadtree leaf side with
  separate along/across multipliers and loose sparse-region rows.

The fitter clamps optimized scales to the owned field cap after optimizer steps. The cap is a field
constraint, not a renderer change: rendering still uses the same Gaussian equations and
`sigma_cutoff` support policy.

Stage-search exposes scale caps as a searchable axis so the project can promote or demote caps by
evidence rather than making them unconditional.

2026-07-07 update: INIT-008 tested `feature_rel` on the fair-density difficult-four protocol that
previously rejected resolution-scaled `feature` caps. `feature_rel` repaired most of the old loss
relative to absolute `feature` caps, but still averaged -0.3733 dB PSNR versus matching uncapped
rows and failed the no-loss criterion at 2000/5000 budgets. Scale caps remain opt-in/searchable,
not a default candidate for the current fair-density protocol.

## Consequences

+ Reduces long-support artifacts and pathological support-window cost.
+ Makes scale-control experiments reproducible because the cap is stored with the field instead of
  being an external post-processing convention.
+ Provides a clean hook for densification: capped broad Gaussians can be paired with residual
  children rather than stretching to cover missing detail.
- Too-small caps can reduce MS-SSIM or suppress useful broad low-frequency support, so caps remain
  a stage-search axis and not a universal guarantee.
- Current fair-density evidence closes the feature-cap default candidacy; future cap work needs a
  new task and a new promotion rule, not reuse of the old `feature12` claim.
- Code that subsets/appends/saves/loads fields must preserve `scale_max`; tests cover this.

## Links

Amends CORE-002's GaussianField responsibilities and ADR-0009's production-default discussion.
Related active tasks: FIT-004, CORE-005, INIT-006, ABL-004.
