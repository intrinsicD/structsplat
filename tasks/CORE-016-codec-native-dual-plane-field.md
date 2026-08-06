# CORE-016 — Codec-native dual-plane Gaussian observation field

## Context

HIER-005--009 show that reducing an explicit pixel-Gaussian field by local contraction can retain
average quality while still producing unacceptable lattice, hole, or redistributed local-error
artifacts.  The same explicit rows currently carry appearance, renderer weight, structural meaning,
and most of the encoded geometry cost.  Realtime-gs already exposes a pluggable point-query backend,
so a Stage-1 representation does not need to materialize every appearance sample as an independently
stored and lifted Gaussian.

## Goal

Implement and kill-test a default-off, self-contained reference packet with two coupled but
semantically distinct planes:

1. an implicit pixel-lattice normalized-Gaussian appearance field whose decoded coefficients are
   carried by a conventional image codec; and
2. a sparse nonnegative anisotropic structural Gaussian measure used for proposals and 2D-to-3D
   lifting.

The packet must support continuous point queries, complete-byte accounting, exact alpha gating,
deterministic cold decode, and a paired realtime-gs `ObservationQueryBackend` adapter without
changing the maintained StructSplat pipeline or realtime-gs checkout.

## Non-goals

- Do not claim a new Laplacian pyramid, RBF method, image codec, structured Gaussian grammar, or
  global compression result; those mechanisms have direct prior art.
- Do not promote Field V2 semantics, replace COMP-013, change `GaussianField`, alter the conversion
  CLI/defaults, or write into the currently dirty realtime-gs worktree.
- Do not treat an exposed Janelle diagnostic as held-out or confirmation evidence.
- Do not describe a sparse structural field without its paired appearance backend as a faithful
  teacher.

## Acceptance criteria

- [x] A NumPy/Pillow reference packet has a strict versioned grammar, canonical metadata and
      checksums, bounded decoding, exact complete-byte accounting, and deterministic repeated
      encode/decode for supported codecs.
- [x] The appearance plane implements a finite, numerically continuous normalized Gaussian-lattice
      query at arbitrary crop/canvas coordinates; pixel-center replay, alpha gating, constant-field
      reproduction, boundary behavior, and malformed inputs have focused tests.
- [x] A deterministic exact-count structural allocator reuses the repository structure tensor,
      keeps nonnegative mass separate from appearance, records the seed/config, and exports the
      structure plane through `ObservationField2D`.
- [x] A lazy optional adapter produces a paired realtime-gs structural
      `GaussianObservationField` plus `ObservationQueryBackend`; tests or a local compatibility
      check verify query/weight/coordinate parity without importing realtime-gs or torch at base
      module import time.
- [x] A bounded diagnostic driver reports source and canonical-PNG bytes, packet/component bytes,
      original-file ratio, PSNR/SSIM/MS-SSIM when available, gradient/high-pass error, worst-pixel
      and multiscale-patch error, encode/cold-decode/query/render time, structural count, and a
      quality--bytes--time curve with visual originals/reconstructions/errors.
- [x] The first Janelle C0001 diagnostic compares fixed codec/quality and structural-count ladders
      against the extant `.rtgsv`/HIER-009 evidence without relabelling incompatible metrics.  Kill
      the architecture if no packet is artifact-safe, smaller than the exact source, materially
      faster to encode than the extant iterative fit, and query-compatible with realtime-gs.
- [x] The research portfolio, prior-art threats, architecture boundary, task state, and generated
      session brief are synchronized; focused tests and `./scripts/verify.sh` pass.

## Interfaces touched

`src/structsplat/codec_native_field.py`, an optional realtime-gs adapter module, focused tests,
`scripts/experiments/`, `docs/adr/`, `docs/research/`, `docs/architecture.md`, ARA staging/evidence
only if a diagnostic is retained, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

CORE-013, BENCH-019, BENCH-020, COMP-013, BENCH-025, HIER-005/009, BENCH-002, ADR-0006

## Agent workflow

- Driver: codex-root
- Reviewer: pending-distinct
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.  Any
formal result beyond the exposed diagnostic requires a distinct prospective protocol review before
execution.

### Handoff

#### Objective

Review the default-off v2 packet, continuous Gaussian-lattice decoder, independent structural
measure, realtime-gs adapter, and the narrow exposed C0001 killing-test interpretation.

#### Changes

Added the strict `.sgdp` producer/decoder and exact byte ledger, cardinal Gaussian prefilter,
structure-tensor/Halton exact-count allocator, lazy paired realtime-gs adapter, task-local diagnostic,
focused tests, ADR-0032, research portfolio, results audit, and ARA evidence note. No maintained
pipeline or format dispatch changed.

#### Evidence

`PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src pytest -q
tests/test_codec_native_field.py` passes 13 tests. `./scripts/verify.sh` passes with 1,673 passed, 6
skipped, and all five structural checkers green. The selected ignored bundle's 25-file custom
manifest, packet component sum, and cold byte-identical resave were independently replayed.

#### Assumptions

The C0001 source/mask are exposed development data. Bilinear decoded-raster interpolation is only an
off-grid control. The historical `.rtgsv`, iterative fit, and HIER-009 rows are contextual rather
than rate/work/preprocessing-matched baselines.

#### Uncertainties

Cardinal ringing remains measurable, the custom diagnostic schema is not accepted by
`check_report_bundle.py`, and neither 512-row structural sufficiency nor real multiview quality has
been established. The selected sigma/step setting is post-hoc.

#### Review focus

Audit finite-kernel coordinate/boundary parity, signed-coefficient conditioning, packet decode
bounds and canonicality, exact byte accounting, crop/full-frame rate wording, and whether the paired
backend is propagated through a real multiview lift without falling back to structural colors.

#### Protected actions not taken

No realtime-gs file, maintained StructSplat default, renderer equation, Field V2 semantic selection,
claim-ledger row, held-out split, or existing evidence bundle was changed or consumed.

#### Recommended next action

Obtain a distinct code/results review, then preregister the analytic/supersampled ringing assay and
a matched full-frame real-multiview downstream experiment before any promotion.

## Notes

The key falsifiable systems claim is narrower than novelty: separating the query-quality plane from
the lift-structure plane should remove explicit per-row geometry from the appearance byte budget and
avoid nonlinear per-image fitting, while still giving realtime-gs continuous colors and a bounded
set of physical lifting proposals.  The conventional codec is a charged component, not a free
baseline or hidden source image.  The reversal path is deletion of this default-off module and task
lineage; all maintained formats and defaults remain unchanged.

The exposed development pilot survives the frozen systems killing rule but does not select a
default. The post-hoc selected v2 packet is 3,896,344 complete bytes, gives below-display error at
decoded pixel centers, and has query parity plus a synthetic two-view CompactCarve smoke. Its
3.662x source ratio compares a full frame with a crop packet; crop-local canonical PNG ratio is
1.139x. Off-grid bilinear-control sampling retains 3.784% local-envelope and 0.0244% global-range
escape. The custom diagnostic manifest passes independent size/hash replay but is not accepted by
the maintained `check_report_bundle.py` schema. See ADR-0032, the
`docs/research/2026-08-06-codec-native-dual-plane-portfolio.md` portfolio, the paired results audit,
and `ara/evidence/core016-codec-native-dual-plane-janelle-2026-08-06/run.md`. Distinct scientific
review, held-out full-frame rate evidence, and a real multiview downstream assay remain open.
