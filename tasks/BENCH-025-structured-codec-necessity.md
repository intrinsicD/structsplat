# BENCH-025 — Structured-codec necessity gate

## Context

COMP-008/009 tested an SGI-inspired conditional entropy mechanism on existing direct attributes;
they did not test SGI's seed-local generation, multi-scale fitting, or resulting decoder grammar.
COMP-013 establishes the strongest honest direct Field V2 codec. Before accepting its compression
ceiling—or building a much more complex generator—the project needs to measure whether decoded
local structure buys complete bytes at the Janelle/public resolutions and realtime query contract.

## Goal

Decide whether the production candidate should retain the direct COMP-013 stream or authorize
COMP-014 to build a seed-structured Field V2 codec, using native SGI evidence and a bounded local
generator oracle with complete-byte, cold-decode, and query accounting.

## Candidate evidence

- strongest direct COMP-013 stream and its scalar/context/VQ increments;
- native-authentic SGI at its intended high-resolution protocol when official code/checkpoints run;
- a separately labelled local seed/block oracle that predicts groups of BENCH-020-selected field
  attributes from compact seed state and counts generator, seed, residual, index, and alpha bytes;
- conventional image-codec references at equal pixels/bytes for reconstruction context, while
  explicitly reporting that they do not expose Gaussian point queries without full decode.

The oracle may use an explicit small linear/MLP generator, but it cannot receive free shared
weights or future/confirmation data. Oracle and deployable measurements are never conflated.

## Non-goals

- Claiming an SGI reproduction from a local transplant or comparing only estimated entropy.
- Implementing a production seed stream, changing Field V2 semantics, or promoting a default.
- Selecting blocks/hard images after inspecting structured-arm outcomes.

## Acceptance criteria

- [ ] A prior-art/provenance table maps SGI's seeds, generators, multi-scale fit, adaptive
      quantization, context model, and stream semantics to native, local-exact, adapted, omitted,
      or unavailable evidence.
- [ ] A reviewed protocol freezes high-resolution development/confirmation roles, image/crop
      policy, rates, seeds, generator capacity, training/fit work, complete-byte syntax, cold
      expansion/query definitions, metrics, and killing thresholds before target access.
- [ ] The local oracle counts every per-file/shared parameter under explicit deployment scenarios;
      no training prior, generator weight, alpha, index, header, or expanded resident field is free.
- [ ] Report float and cold-decoded PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, complete
      bytes/bpp, encode/fit/training time, cold decode/expansion, query latency, resident/peak
      memory, random access, and failures against direct and conventional controls.
- [ ] The decision is one of: direct COMP-013 is sufficient; authorize exactly one COMP-014
      grammar; or unavailable. A positive decision requires a predeclared complete-byte advantage
      at acceptable quality/downstream/decode/query cost on disjoint confirmation.
- [ ] Native and local rows remain separate, correlated crops are not treated as independent, and
      a distinct auditor recomputes the decision from raw rows.
- [ ] Portable report, ARA disposition, architecture/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

BENCH-005/native adapters, bounded seed-generator oracle under `scripts/experiments/`, COMP-013
rate/query adapters, report/audit tooling, `ara/evidence/`, `docs/additive_field_v2.md`, this task,
and the Index.

## Depends on

BENCH-020, COMP-008/009/013, CORE-013, BENCH-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`. Protocol and results reviewers must be distinct from the Driver.

## Notes

This is a necessity/killing test, not a promise to reproduce SGI. It exists because an entropy
model over independent rows and a decoder that generates correlated rows are different hypotheses.
