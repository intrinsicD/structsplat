# HIER-033 — Pixel-gradient operator oracle

## Context
The August 12 anatomy derives additive gradients but leaves finite operator value untested.

## Goal
Compare parameter-family signals against finite local edits and fixed optimizer recovery.

## Non-goals
- No default change, downstream 3D transfer, sealed images, or quality claim from synthetic cases.

## Acceptance criteria
- [ ] Analytic sums and curvature match renderer/autograd.
- [ ] Freeze fixtures, candidate bank, controls, recovery, metrics, gates, and source digest.
- [ ] Distinct prospective review and clean immutable run with portable raw artifacts.
- [ ] Independent audit, synchronized task/docs/ARA, full verification.

## Interfaces touched
Pixel-gradient reference, focused tests, and task-scoped driver.

## Depends on
HIER-031/032, ADR-0006

## Agent workflow
- Driver: codex-root
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log
Protocol in preparation; no formal outcome exists.

## Notes
Design: docs/research/2026-08-12-hier-pixel-gradient-anatomy.md.

## Experiment design in preparation

The planned bounded assay uses six procedural defects (translation, width, rotation, RGB,
two lobes, and residual outside current finite support), seeds0/1/2, 64x64 images and three
initial Gaussians. This is mechanism evidence, not a synthetic-to-natural quality claim.

All finite trial states keep N=3. Continuous move/scale/rotation/RGB edits act on the designated
parent row at two frozen magnitudes. Symmetric splits halve the parent's RGB, preserve its
scales, and explicitly delete one of two donor rows. Residual-peak births replace one of those
same donors. A no-op control is retained. The resulting fixed bank has15 candidates per case;
every donor's reconstruction damage is included in the measured objective.

Pixel signed/absolute gradients, local Gram blocks and split matrices propose local scores,
but finite trial renders are authoritative. Compare prediction regret with the unrestricted,
continuous-only and split-only finite oracles, explicitly labelled privileged controls.
Evaluate every candidate immediately and after the same20-step Adam recovery, reset optimizer
state for every trial, and preserve all recovered states. The central question is whether
gradient cancellation identifies an edit family, not whether an unrestricted candidate search
can improve training loss. CPU reference execution avoids the occupied GPU and has no speed claim.

These choices are a design draft until bound by an executable protocol, a clean source commit
and a distinct prospective approval. No formal run is authorized by this draft alone.
