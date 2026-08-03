# COMP-013 — Observation Field V2 codec

## Context

Rows and NumPy payload size are not compression metrics. Field V2 requires a complete, versioned
bitstream whose geometry, appearance, optional structural channel, alpha, indexes, tables, and
headers are all counted. Compression choices must also preserve cold decode, point queries, and
random access well enough for realtime-gs ingestion; an offline entropy number alone is
insufficient.

## Goal

A deterministic, complete Field V2 codec with canonical raw accounting, actual byte-rate control,
quality-aware quantization, spatial entropy coding, cold decode/query benchmarks, and strict
round-trip semantics.

## Non-goals

- Choosing field semantics before BENCH-020 or changing the legacy SSPL1 format in place.
- Reporting estimated entropy, compressed tensor bodies, or row counts as complete stream bytes.
- Making temporal or neural decoding part of the first production format.

## Proposed progression

1. Define the canonical uncompressed Field V2 byte representation and complete stream grammar.
2. Establish per-attribute scalar quantization/QAT and exact distortion sensitivities before more
   complex codebooks; include position, scale, rotation, appearance, structural mass, and alpha.
3. Add deterministic spatial ordering (for example Morton/tile order), context models, and a
   portable range/ANS backend only when each step beats simpler zlib/PNG/VQ controls.
4. Support target-byte operation through measured complete bytes and bounded search, not an
   estimated rate proxy alone.
5. Benchmark cold decode, resident memory, full render, point query, and random tile access.

## Acceptance criteria

- [ ] A versioned spec defines magic/version, canvas/crop transform, renderer/support semantics,
      tensor shapes/dtypes, quantizers, ordering, entropy tables, alpha/index payloads, checksums,
      coefficient/DC domain, and fail-closed limits; every emitted byte is attributable.
- [ ] Canonical raw bytes and complete compressed bytes are reported separately. Headers, alpha,
      background/DC, side tables, codebooks, padding, indexes, and container overhead are included
      in totals.
- [ ] Encode/decode are deterministic across repeated supported-platform runs; truncated,
      corrupted, unknown-version, oversized, nonfinite, and metadata-inconsistent streams fail
      closed without partially trusted output.
- [ ] Lossless/reference and quantized tiny fixtures validate equations and field semantics;
      quantized output remains finite and within frozen per-attribute/error tolerances.
- [ ] A frozen RD experiment compares current zlib/PNG/VQ baselines and each codec increment at
      equal complete bytes, including practical active-crop target rates selected before execution.
- [ ] Reports include PSNR/MS-SSIM/LPIPS, BENCH-019 downstream response, complete bpp/bytes,
      encode/cold-decode/query time, throughput, resident/peak memory, and random-access overhead.
- [ ] Public API/CLI integration is default-off and legacy streams remain readable and unchanged.
- [ ] Spec, tests, portable report/audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

New codec modules/spec under `src/structsplat/` and `docs/`, Field V2 IO/API, rate-distortion and
query benchmarks, CLI hooks, tests/fuzz fixtures, `docs/additive_field_v2.md`, this task, and Index.

## Depends on

CORE-013, BENCH-020, COMP-002/004/008/009, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Codec/security review and result review must be recorded before promotion.

## Notes

The initial production candidate deliberately favors independently testable scalar quantization
and spatial coding. More expressive VQ or learned entropy models must earn their decoder and
random-access complexity on complete-byte RD curves.
This direct-row stream is the mandatory baseline for BENCH-025. Seed-generated local structure is
a different decoder grammar owned conditionally by COMP-014, not an implicit COMP-013 feature.
