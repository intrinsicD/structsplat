# FIT-022: Coverage-matching regularizer (feature-targeted fit-time Gaussian blue noise)

## Context

The 2026-07-23 ideation audit (`docs/research/2026-07-23-coverage-matching-gbn-ideation.md`)
classified the push–pull / Gaussian-blue-noise idea as N2-T: the attraction–repulsion mechanism
is prior art (Electrostatic Halftoning 2010; Gaussian Blue Noise 2022; MMD gradient flows), but
one recipient-specific formulation is apparently unexplored — a differentiable, two-sided
**coverage-matching energy on the normalized compositor's own denominator**, optimized jointly
with reconstruction:

```
E = mean_inside ( S(p) − c(p) )²,   S(p) = Σ_i detach(o_i) · G_{Σ_i}(p − μ_i)
```

`S` is the raw weight sum the renderer normalizes by (the same accumulation as the CORE-010/011
mask penalties), and `c` is a feature-weighted target profile normalized to the current detached
total kernel mass, so the term only redistributes coverage. Expanding the square yields
attraction to under-covered features plus pairwise anisotropic Gaussian repulsion in surplus
regions — push and pull as one term, evaluated in field form at O(N·support).

The central claim to test (not asserted): under the *normalized* compositor, redundant capacity
is a loss-null gauge direction, so this term transports capacity nearly orthogonally to the data
term; under the *additive* compositor the identical term is predicted to be harmful. The audit's
preregistered fixed-N killing experiment (four arms, WSE-init baseline, additive-differential
arm, abandonment criterion) is the required screen before any promotion talk; this task only
lands the mechanism as a searchable knob per ADR-0010.

## Goal

Implement the regularizer in `fit.py`, default off:

- `coverage_match_weight` — loss weight; 0 disables (default).
- `coverage_match_target` — `tensor` (1 + beta·normalized structure-tensor energy),
  `tensor_boundary` (adds a mask-boundary band boost; requires a mask),
  `error_blend` (tensor profile blended with the smoothed current |residual|, detached).
- `coverage_match_beta`, `coverage_match_boundary_boost`, `coverage_match_boundary_band`,
  `coverage_match_error_alpha` — profile shape knobs.
- `coverage_match_decay_frac` — 0 keeps the weight constant; otherwise a cosine ramp to zero at
  this fraction of the (global, pyramid-aware) schedule, so the term organizes layout early and
  releases the fit to the data term late.
- Opacities are **detached** inside `S` (fixed, not a knob): the term must transport geometry,
  not dim rows — otherwise it fights the pooled park/merge logic and the opacity gauge.
- The uniform floor in the `tensor` profile is deliberate: every pixel needs S > 0 under the
  normalized compositor, so flat regions keep a baseline coverage target.
- Composes with FIT-021 pooled triage (parked rows contribute exactly zero to `S`; the mass
  normalization tracks live mass automatically) and with the mask machinery (profile restricted
  to the mask interior; the out-of-mask side stays owned by the CORE-010 coverage penalty).

## Acceptance criteria

1. Default behavior unchanged: `coverage_match_weight=0` produces byte-identical fits.
2. Gradient discipline: the term's gradient reaches means/log-scales/rotations and is exactly
   zero on opacities (detached). Tested directly on the helper.
3. Transport sanity: pure-`E` gradient descent on means measurably reduces the coverage
   mismatch on a synthetic case. Tested.
4. Mass neutrality: `c` is renormalized to the detached current total of `S` every evaluation,
   so the term is scale-free by construction (asserted via the helper's target sum).
5. Maskless, masked, and masked+pooled (`triage_every`) fits all run with the term enabled;
   the mask-contained arm preserves exact zero outside the mask. Tested end-to-end (CPU-small).
6. `tensor_boundary` without a mask fails with a clear error; knob ranges validated in
   `FitConfig.__post_init__`.
7. History logs the term (`coverage_match_loss`) at the logging cadence, and the decayed weight
   reaches zero after `coverage_match_decay_frac` of the schedule. Tested.
8. `pytest -q` green for the new suite and the fit-adjacent suites.

## Interfaces touched

`fit.py` (module-level raw weight map shared with `_MaskConstraint.raw_weight_map`, profile
preparation, schedule factor, loss term, history), `config.py` (knobs + validation), `cli.py`
(flags through the shared fit-option surface), `tests/test_coverage_match.py` (new),
`tasks/INDEX.md`, `docs/architecture.md`.

## Depends on

Ideation audit 2026-07-23 (provenance + preregistered screen), CORE-010/011 (weight-sum
machinery and mask domain), FIT-012 (structure-tensor energy map pattern), FIT-021/ADR-0020
(pooled compatibility), ADR-0003 (normalized compositor — the gauge argument), ADR-0010
(searchable-axis protocol).

## Notes / deferred

- The killing experiment itself (four arms incl. the additive-differential, difficult-four
  proxy, preregistered grid, abandonment rule) is benchmark work, not part of this task's
  implementation; run it before any default/promotion discussion, honoring the FIT-012/013/016
  regularizer track record.
- Sharing one accumulation with the CORE-010/011 penalties when both are enabled is deferred:
  those penalties need live opacity gradients while this term detaches them, so v1 keeps
  separate accumulations rather than compromising either semantics.
- Any external claim additionally requires the audit's flagged 2025–26 compact-GS prior-art
  sweep.
