# CORE-008: Hybrid Gaussian + frequency-bearing primitive control

**Status: design-only, not authorized by BENCH-007.** WIPES is the direct frequency-bearing
primitive baseline. BENCH-007's texture regression is allocation-specific—the same Gaussian basis
under the gradient control performs better—so it does not establish a primitive-family failure.

## Context
Some high-frequency edges and textures are inefficient for blob-only Gaussian bases. WIPES already
demonstrates localized wavelet visual primitives, so a Gaussian-windowed edge/DoG/Gabor arm is
scientifically useful only as a controlled basis-family experiment with actual bytes—not as a
generic claim that frequency-bearing primitives are new.

## Goal
Identify whether StructSplat's residual failure is caused by the Gaussian basis lacking phase/sign
degrees of freedom, after controlling for parameter count, actual stream bytes, optimization
compute, and WIPES/native wavelet evidence.

## Approach
1. Require a future independent mechanism study to show a repeatable basis-limited texture or
   thin-line failure at <=1 bpp; BENCH-007 did not isolate one.
2. Reproduce or run WIPES first where official code permits. Record native evidence separately.
3. Build a synthetic linear-dictionary oracle: pure Gaussian, Gaussian derivative, Gabor, and
   localized wavelet atoms under exact coefficient/metadata costs. This is the cheapest killing
   test and does not require production renderer changes.
4. Only if a phase-bearing family has a >=20% sparse-coding RD advantage on the frozen synthetic
   set and the eight-image pilot should a versioned hybrid field/renderer be proposed.

## Acceptance criteria
- [ ] BENCH-007 provides the failure regime and frozen image IDs.
- [ ] WIPES prior-art/native-control status is recorded before implementation.
- [ ] Synthetic hard edges, junctions, thin lines, chirps, and oriented textures are evaluated with
      exact atom metadata and coefficients counted.
- [ ] Pure Gaussian, extra-Gaussian-count, derivative/Gabor, and localized-wavelet controls receive
      equal RDO search and compute.
- [ ] Promotion requires >=20% favorable RD at the synthetic sparse frontier and >=0.25 dB at both
      0.5 and 1.0 actual bpp on the eight-image pilot without >20% decode-time regression.
- [ ] A failed spike closes the basis-family task without modifying the production field format.

## Interfaces touched
Start as a standalone benchmark/dictionary spike. Production field, renderer, fit, codec, and ADR
changes are explicitly out of scope until the gate passes.

## Depends on
CORE-001, INIT-001, FIT-001, BENCH-007. Optional: CORE-007.
