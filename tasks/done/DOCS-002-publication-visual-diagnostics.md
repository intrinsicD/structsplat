# DOCS-002 — Publication visual diagnostics

**Status: done.** Completed 2026-07-14.

## Context

StructSplat had numerical benchmark figures but no reusable visualization of the actual
structure-tensor maps, density, tensor-aligned WSE field, Gaussian RS geometry, or normalized
renderer responsibilities. Readers therefore could not inspect the mechanism that BENCH-007 is
meant to test. Publication figures must be reproducible and must not imply that initialization-only
images are optimized-quality or held-out results.

## Goal

Add a deterministic CPU-capable figure generator that exposes the existing production method and
normalized-renderer diagnostics on a real or user-supplied image, with individual lossless panels,
a labeled montage, raw diagnostic arrays, and source/config/hash provenance.

## Acceptance criteria

- [x] Uses the production structure-tensor, density, initializer, and normalized renderer paths.
- [x] Provides a deterministic vector overview separating source-only encoder analysis from the
      transmitted decoder-complete stream and cold-decode path.
- [x] Shows input, tensor energy/coherence/classes/tangents, sampling density/sites, initialized RS
      ellipses, reconstruction, coverage denominator, effective contributors, responsibility
      entropy, and dominant ownership.
- [x] Saves deterministic PNG/NPZ/JSON artifacts with resolved configuration, source-image SHA-256,
      direct implementation-source hashes, output hashes, coordinate/angle conventions, and an
      initialization-only disclaimer.
- [x] Provides a CPU CLI with image, crop/max-side, seed/count/strategy, tensor, glyph, ellipse, and
      output controls.
- [x] Tests orientation convention, renderer-denominator parity, diagnostic identities/ranges,
      determinism/provenance, and CLI smoke.
- [x] Generates one pinned example under `ara/evidence/` and documents regeneration plus claim
      limits.

## Interfaces touched

- `src/structsplat/visualize.py`
- `deprecated_scripts/render_paper_figures.py`
- `tests/test_visualize.py`
- `docs/publication_figures.md`
- `ara/evidence/docs002-publication-visual-diagnostics-2026-07-14/`

## Depends on

INIT-001, INIT-003, CORE-001, CORE-002, BENCH-007.

## Completion evidence

- Focused visualization/tensor/renderer suite: 92 passed.
- Complete repository suite: 449 passed.
- Ruff lint and formatting checks passed.
- The pinned manifest reports 384 initialized Gaussians, full pixel coverage, zero responsibility
  identity error, and zero effective-count/entropy bound violations.

This task closes explanatory and diagnostic infrastructure only. Actual-rate RD, causal control,
edge-band, convergence, and success/failure result figures remain BENCH-007 deliverables.
