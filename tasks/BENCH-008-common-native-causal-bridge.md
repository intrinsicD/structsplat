# BENCH-008 — Common/native causal bridge

**Status:** blocked on BENCH-007 Stage 1 and native field-schema feasibility.

## Decision this task owns

Separate placement, representation/renderer, optimizer, and codec effects before interpreting
cross-repository rankings. Native-authentic results remain the external comparison; this bridge is
an intervention study, not a replacement for them.

## Hypothesis

The apparent gap between StructSplat and current 2D Gaussian methods contains a substantial
initialization-by-renderer/objective interaction. The null is that allocation rankings are stable
when the renderer/optimizer axis changes, in which case further cross-replay work should stop.

## Staged protocol

### Stage A — schema and forward parity

- Define an explicit lowest-common-denominator field: means, covariance/RS, constant RGB, optional
  opacity, image extent, and compositing convention.
- Export the same frozen fields from tensor-WSE, SLIC/Sobel, Image-GS gradient, and uniform-WSE
  allocation.
- Replay only where semantics can be represented exactly. Record unsupported attributes rather
  than silently dropping them.
- Compare StructSplat normalized rendering, Image-GS top-K normalized rendering, and
  GaussianImage/GaussianImage++ alpha/sum rendering on synthetic fields and eight Stage-1 images.
- Require central raw-float metrics, contributor/coverage maps, cold artifact reload, and
  implementation/extension provenance.

### Stage B — crossed optimization

For feasible cells, cross `initial field × native optimizer/renderer` at fixed original pixels,
initial/final N, requested steps, seed, and synchronized timing. Each optimizer retains its native
loss only in a separately labeled row; add a shared-L2 or shared-L1 control where feasible.

Use a factorial interaction model and image-cluster bootstrap. The primary observable is the
change in an initializer's paired PSNR rank across renderer/optimizer columns; secondary
observables are edge-band error, LPIPS, convergence AUC, fit time, and count trajectory.

### Stage C — native-authentic confirmation

Run published/default protocols separately for available official implementations:
Structure-Guided Allocation, Image-GS, GaussianImage++, GaussianImage, SAD, WIPES, AIR, and
Instant-GI. A missing public implementation stays `not run`; do not synthesize a native row from a
local analogue. Rate definitions must be split into actual stream bpp, analytical/parameter bpp,
checkpoint bytes, and null.

## Killing and promotion rules

- Stop after Stage A if fewer than two renderer families can replay the shared field without
  semantic loss.
- Stop the research lane after Stage B if initializer rankings are stable and every preregistered
  interaction is below 0.25 dB; retain the benchmark result.
- Promote to the full crossed matrix only if an interaction exceeds 0.25 dB on the eight-image
  pilot with an image-bootstrap interval excluding zero and a predicted mechanism map changes in
  the same direction.

## Deliverables

- Versioned field-schema document and exact/unsupported conversion table.
- Synthetic forward-parity tests and frozen-field artifacts.
- Crossed raw rows, interaction estimates, central metrics, provenance, and failure table.
- Separate native-authentic RD/time table with no synthetic leaderboard across incompatible rate
  definitions.

## Depends on

BENCH-005, BENCH-007 Stage 1, CORE-001/003/005, COMP-002.
