# ADR-0024: Scope renderer parity to candidate-vs-baseline and document normalized fade-cutoff conditioning

## Context

PORT-002's frozen timing profile (`benchmarks/tiled_render_profile.py`) gates the GPU-native tiled
renderer on a parity precondition. The 2026-07-24 RTX 3050 run failed that precondition while
passing every performance sub-check by a wide margin (representative step ratio `0.6983` against a
`<= 1.00` limit, all seven N=8192 grid cells keeping direction, GPU index share `1.110%` against a
`<= 15%` limit).

The failure was diagnosed rather than assumed. All four recorded failures are one cell,
512²/N=8192/overlap-4/ratio-1, counted once per arm, and every arm reported the identical
`max_abs = 8.877516e-4` — including the unchanged, shipped-default untiled `cuda` baseline and the
legacy `cuda_tiled_torch_index` builder that the rework replaced. Direct measurement at that cell:

- candidate vs baseline: `max 1.788139e-7` for all three tiled arms;
- each arm vs the torch reference: exactly `1` of `786,432` values exceeds tolerance
  (`0.000127%`); median gap `0`, q99 `1.19e-7`, q99.99 `5.36e-7`;
- the offending value is pixel `(row=235, col=348)`, where the normalized denominator is
  `1.822550e-7` contributed by a **single** Gaussian sitting at its support-fade cutoff.

The mechanism is float32 conditioning of the fade weight near the cutoff, not a defect in the
tiled path. Under `support_fade`, `w = exp(-q/2) - exp(-c^2/2)` with `c = sigma_cutoff`, so
`w -> 0` as `q -> c^2` and `w` is approximately proportional to `(c^2 - q)`. Near `q = 9`, float32
spacing is `ulp(q) ~ 9.54e-7`, giving `w` a relative error of roughly `ulp/(c^2 - q)`, which
diverges at the cutoff. At the failing pixel `(c^2 - q) ~ 3.3e-5` and the measured relative error
in `w` is `1.11e-2`. ADR-0003's normalize-by-denominator then converts that into output error with
amplification `eps/(D + eps)^2`; with `D = 1.82e-7` and `render._EPS = 1e-8` this predicts
`~7e-4`, matching the observed `8.88e-4`.

An algebraic reformulation was tested and rejected. The identity
`exp(-q/2) - exp(-c^2/2) == exp(-c^2/2) * expm1((c^2 - q)/2)` removes the cancellation in the
subtraction and does improve the formula's own contribution at the failing point
(`3.76e-4 -> 5.51e-8` relative, measured at fixed `q_f32`). It does **not** improve the total
error (`1.11e-2` naive vs `1.15e-2` expm1), because the dominant term is the float32
representation of `q` itself, which both forms inherit. There is no float32-local fix; the
conditioning is a property of the primitive, not of an implementation choice.

The precondition as written therefore gates PORT-002/PORT-003 on a pre-existing numerical property
of the baseline renderer they do not modify.

## Decision

Split the renderer parity precondition into a governing criterion and a diagnostic one.

1. **Governing: candidate versus baseline.** Every tiled arm must match the untiled exact `cuda`
   renderer at the *same* `PARITY_ATOL`/`PARITY_RTOL` already frozen (`5e-4`). No new tolerance
   constant is introduced. This is the comparison PORT-002 and PORT-003 actually claim: the tiled
   path computes what the exact renderer computes, faster.

2. **Diagnostic: agreement with the torch reference.** Still computed and recorded per arm. A cell
   where the untiled `cuda` baseline *also* mismatches the reference is labelled
   `baseline_attributable` and does not fail the gate, because the baseline is unchanged by this
   work and its reference agreement is owned by PORT-001 / ADR-0011. A cell where the baseline
   agrees with the reference but any candidate arm does not remains a hard gate failure.

Rule 2's discriminator is deliberately non-tunable: it excuses a mismatch only when the unmodified
baseline exhibits it too, and any error the tiled path introduces on its own makes candidate differ
from baseline and fails rule 1. No threshold, cell, or tolerance was retuned.

Document the near-cutoff regime as a known conditioning property of the normalized compositor under
support fade rather than treating it as a bug to be suppressed.

## Consequences

+ PORT-002/PORT-003 are gated on the claim they make, not on a baseline property they do not touch.
+ `PARITY_ATOL`, `PARITY_RTOL`, the representative cell, every timing threshold, and the grid are
  unchanged. The performance side of the gate is untouched and was never in question.
+ Reference disagreement is now *reported* rather than silently collapsed into one boolean, so a
  genuine future renderer regression against the reference stays visible in the artifact.
+ The renderer keeps its current fade formulation. `expm1` was measured and offers no total-error
  benefit, so adopting it would add a kernel change with no accuracy justification.
- This re-preregistration was authored **after** the 2026-07-24 timings were seen. That ordering is
  recorded in the profile artifact and in PORT-002's notes. The re-run it authorizes is therefore
  informed-order evidence, and its scope is unchanged from the original gate: it authorizes only
  the fair-protocol end-to-end fit benchmark. It does not authorize a default flip, a cross-GPU
  claim, or any quality/compression claim.
- Users fitting at very low effective coverage (pixels reached by a single Gaussian at its cutoff)
  should expect float32 output error up to `~1e-3` at those pixels under the normalized compositor.
  `render._EPS = 1e-8` is far below such denominators and does not stabilize them.

## Links

Constrains the precondition of PORT-002 and PORT-003. Depends on ADR-0003 (normalized compositor)
and ADR-0011 (owned exact CUDA renderer). Does not supersede either: the renderer's mathematics,
defaults, and shipped GPU path are unchanged.
