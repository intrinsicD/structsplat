# CORE-011: Boundary coverage for mask-contained fitting

## Status

Implemented (opt-in, default off) on 2026-07-22. Extends CORE-010; everything is opt-in and
default-off. No default change, no compression claim, no novelty claim (anisotropic constrained
fitting, penalty methods, and contour-seeded densification are standard toolbox items; this task
claims the capability for this codebase). The committed multi-arm benchmark on a real masked
dataset remains deferred (shared with CORE-010's deferred containment-cost benchmark; the new
modes become additional arms).

## Context

The CORE-010 hard-containment fit converges with an under-covered boundary: a black hole band
hugging the mask edge (observed on the C0001 mask-contained densified run, 2026-07-22; see
`ara/evidence/core010-c0001-densified-fit-2026-07-22/run.md`). The starvation is structural, not
a tuning artifact. With `sigma_cutoff=3`, `mask_margin=1.5`, min scale 0.35, the ADR-0017
isotropic cap gives a Gaussian at signed distance `s` the reach `s - margin`, so:

1. **Dead band.** The SDF is 1-Lipschitz, so any pixel covered by a contained Gaussian satisfies
   `SDF(p) >= margin` (up to the nearest-pixel SDF slack the margin already accounts for). The
   outermost ~`margin` px of the interior are *provably uncoverable*: they render 0 forever.
2. **Starved band.** The isotropic cap `(SDF - margin) / sigma_cutoff` reaches a 1 px scale only
   at SDF = 4.5 px, and forbids tangent elongation, so tiling the boundary band costs
   O(perimeter / px) near-minimum Gaussians.
3. **Densification poisoning.** Uncovered boundary pixels render black against real object
   colors at loss weight 1, so they are permanently the top residuals; residual-driven
   densification keeps spending spawn budget on sites whose children are projected ~2.55 px
   inward and capped to confetti, starving legitimate interior sites.

## Goal

Close the *reachable* boundary band (holes and error) of mask-contained fits while preserving the
ADR-0017 guarantee — exact zero outside the mask on a cold, mask-free decode — and leave the dead
band to an explicit `mask_margin` trade-off.

## Approach (all opt-in `FitConfig` knobs, inert without a mask)

1. **Anisotropic mask caps** (`mask_cap_mode="anisotropic"`, ADR-0019). Keep the isotropic cap on
   the short axis; certify a longer long-axis cap with a station-ball SDF cover: the cutoff
   ellipse is covered by balls along its long axis, each certified by one bilinear SDF probe with
   the 1-Lipschitz slack (`SDF(c_j) >= margin + sqrt(w_j^2 + (delta/2)^2)`). Corners, curvature,
   and mask holes bind through the probes themselves — no trusted normal model. Certified caps
   ladder up to ~18.6x the isotropic reach; recertification runs at `mask_cap_refresh_every` plus
   entry/every restructure event/terminal, so the returned field is always certified at its final
   parameters. Boundary Gaussians elongate along the tangent (invariant 3) instead of tiling.
2. **Under-coverage penalty** (`mask_undercoverage_weight`, band/tau knobs). The in-mask twin of
   the CORE-010 coverage penalty: a hinge `mean(max(0, tau - den)) / tau` on the raw unnormalized
   weight sum over reachable band pixels (`margin < SDF <= margin + band`). Acts on the gauge the
   normalized compositor cancels, so it pulls geometry/opacity with full-strength gradients and
   cannot be satisfied by recoloring. Support-limited by design (same tiles as the renderer).
3. **Boundary tangent densification** (`mask_boundary_add_every/count/band/spacing`). Spawns
   tangent-aligned children at boundary-band residual peaks: Euclidean NMS spacing on the thin
   band approximates arc-length spacing; children are projected to the eroded interior, oriented
   by the smoothed SDF gradient (`mask.boundary_normals`), across-scale bound to the reachable
   depth, along-scale seeded from the spacing and then certified/trimmed by the next cap refresh.
   Composable with any `refine_site`/adaptive mode; recorded as `boundary_add_events`.
4. **Diagnostics.** The CLI prints boundary-band (<= 2 px) PSNR next to the out-of-mask energy,
   so boundary improvements are visible (whole-image PSNR hides them).

Dead-band handling stays an explicit trade-off: reducing `--mask-margin` toward the ~0.71
bilinear-SDF bound shrinks the dead band (the config comment and ADR-0017 accounting still hold);
the last ~1 px can also be read as intentional alpha falloff when compositing.

## Non-goals

- Masked-domain rendering (mask at decode) — unchanged CORE-010 non-goal, still the ceiling arm.
- Reopening CORE-007/008 internal-boundary questions or any BENCH-007-gated claim.
- Default changes: without a mask, and with `mask_cap_mode="isotropic"` (default), behavior is
  unchanged.
- 1D arc-length ribbon primitives (constraint-manifold parameterization) — CORE-008 territory.

## Acceptance criteria

- [x] `mask.boundary_normals` / `gaussian_smooth` are pure NumPy, module imports without torch
      (existing import test), normals unit inward on straight edges and zero where unreliable
      (test).
- [x] Anisotropic caps certify tangent elongation on straight edges, bind at corners, never fall
      below the isotropic cap, and keep the exact-zero-outside render on random elliptical masks
      with random rotated fields and through a full fit with refresh cadence (tests).
- [x] Under-coverage hinge: positive with correct gradient direction on an uncovered band, exactly
      zero when covered, and reduces uncovered band pixels over a fit (tests).
- [x] Boundary tangent add: spawns contained, tangent-oriented (invariant 3: sx along), eroded-
      interior children; events recorded; exact zero outside preserved; requires a mask (tests).
- [x] Config validation for every new knob; no-mask and isotropic-default paths unchanged
      (existing regression tests stay green).
- [ ] Committed multi-arm benchmark on a real masked dataset with boundary-band PSNR (extends the
      deferred CORE-010 five-arm benchmark with anisotropic / +penalty / +spawn arms; report
      in-mask, boundary-band, out-of-mask energy, composite-over-background). Deferred: runnable
      from CLI flags today.

## Interfaces touched

`src/structsplat/mask.py` (normals + smoothing), `src/structsplat/fit.py` (`_MaskConstraint`
cap modes + penalties, `_boundary_tangent_add`, loop wiring), `src/structsplat/config.py`,
`src/structsplat/init.py` (`build_masked_field(cap_mode=...)`), `src/structsplat/cli.py`,
`tests/test_mask.py`. Not `src/structsplat/render.py`.

## Depends on

CORE-010, ADR-0017, ADR-0019, CORE-003/005, FIT-012.
