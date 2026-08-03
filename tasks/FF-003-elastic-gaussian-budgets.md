# FF-003 — Complete-byte elastic Field V2 predictor

## Context

An amortized encoder that requires a separate checkpoint for every Gaussian count or byte rate is
costly to train, store, and maintain. Row count is also the wrong production budget: Field V2
geometry, appearance, optional structure, alpha, indexes, entropy context, and headers all
contribute to complete bytes. After FF-002 selects a viable predictor, elasticity should target a
small complete-byte ladder through one checkpoint and the same COMP-013 decoder.

## Goal

One predictor checkpoint that emits deterministic candidates for a frozen ladder of complete-byte
targets and remains competitive with separately trained per-rate FF-002 checkpoints at matched
training compute.

## Method boundary

- Choose one output contract before the screen: nested prefix candidates, deterministic
  confidence/top-N candidates, or a spatial candidate map with a byte-conditioned selector.
- Condition on target complete bytes (and optionally a discrete rate embedding), then use a
  bounded COMP-013-aware selection/quantization step to meet the requested rate.
- Freeze practical active-crop target rates after inspecting only metadata and product/storage
  constraints. Always report full-canvas bpp and exact bytes too.

## Non-goals

- Starting before FF-002 selects a predictor or modifying COMP-013's grammar.
- Calling predicted rows, analytical bits, or entropy estimates the achieved rate.
- Adaptive per-image quality targets, temporal prediction, or production-default promotion.
- Formal amortization/generalization claims; BENCH-023 owns confirmation.

## Acceptance criteria

- [ ] One checkpoint serves the complete-byte ladder; tests prove prefix/top-N/candidate-map
      contract, deterministic ties, exact schema, and valid Field V2 output for every rate.
- [ ] Budget sampling, embeddings, codec-aware selection, training compute, and seeds are fully
      logged and reproducible. Every evaluation output is encoded and cold-decoded before scoring.
- [ ] Development RD table reports target/actual complete bytes and active/full bpp,
      PSNR/MS-SSIM/LPIPS, BENCH-019 downstream objective, `0/50/200/500` refinement latency,
      encode/decode/query time, memory, and failures for elastic and per-rate dedicated models.
- [ ] Rate miss and elastic-versus-dedicated quality deficits are explicit at every target; no
      aggregate hides a failed rate or difficult image.
- [ ] A frozen rule decides whether the elastic checkpoint advances to BENCH-023, dedicated
      checkpoints remain, or the learned branch is stopped.
- [ ] Focused tests, portable report/audit, ARA disposition, docs/task synchronization, and
      `./scripts/verify.sh` pass.

## Interfaces touched

`src/structsplat/predictor.py`, COMP-013 rate/encode API, training/evaluation drivers, tests/report
artifacts, `docs/additive_field_v2.md`, this task, and the Index.

## Depends on

FF-002, BENCH-025, COMP-013/014

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff`, `### Review`, and pre-run `### Protocol review` blocks using
`tasks/README.md`.

## Notes

The byte ladder replaces the historical `{128,256,512,1024}` row ladder. Row count remains useful
telemetry but cannot define an equal-rate comparison.
