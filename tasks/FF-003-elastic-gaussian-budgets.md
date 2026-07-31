# FF-003 — Elastic Gaussian-count predictor

## Context

The FF-001 predictor is fixed to one `num_gaussians`: its final layer is a dense
`num_gaussians * 10` head, so every budget needs its own trained checkpoint and the parameter
count scales linearly with N. StructSplat's stated goal is compression, which means operating
points along a rate–distortion curve, not one field size. Once FF-002 has replaced the global
row-tensor representation with a spatial/occupancy form, budget elasticity becomes a small
extension instead of a rewrite: the same predictor can be trained across several N and asked for
any of them at inference.

## Goal

One predictor supporting `N in {128, 256, 512, 1024}` from a single checkpoint, trained with
budget sampling and a budget embedding, using one of three frozen output forms (choose one
before the screen, record the reason):

- nested outputs — the first 128 slots are valid for every larger budget (strict prefix
  property);
- occupancy/confidence logits followed by top-N selection;
- a spatial candidate map from which N candidates are selected.

Evaluated as the complete rate–distortion curve
`(N, encoded bytes, PSNR, refinement time)` on held-out images, against per-budget dedicated
FF-002 checkpoints at matched training compute.

## Non-goals

- Starting before FF-002 reaches a terminal disposition; this task inherits its representation
  and its held-out protocol.
- New codec work; encoded bytes use the existing NPZ/SSPL1 persistence as-is.
- Budgets outside the frozen set or adaptive per-image budget selection (a separate question).
- Any default change.

## Acceptance criteria

- [ ] One checkpoint serves all four budgets; a test proves the chosen output form's contract
      (prefix validity, top-N determinism at fixed seed, or candidate-map exact-N selection).
- [ ] Budget embedding and budget-sampling schedule are logged config; training is reproducible
      from config + seed.
- [ ] Held-out rate–distortion table `(N, encoded bytes, PSNR, refinement time)` for the elastic
      checkpoint versus per-budget dedicated checkpoints at matched training compute, plus
      elastic-vs-dedicated deltas per budget.
- [ ] Degradation bound recorded: the elastic checkpoint's PSNR deficit versus dedicated at each
      N is reported explicitly, including negative outcomes.
- [ ] Outcome recorded as an ARA observation or claim row; Index status updated in the same
      commit.
- [ ] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/predictor.py`, training/export drivers under `scripts/experiments/`,
`src/structsplat/codec.py` (read-only byte accounting), `tests/`, `tasks/INDEX.md`, `ara/`.

## Depends on

FF-002

## Agent workflow

- Driver: pending
- Reviewer: pending
- Turn: driver
- Reviewed revision: pending

### Handoff log

Append exact `### Handoff` and `### Review` blocks using the schema in `tasks/README.md`.
Before a formal result-bearing run, append the prospective `### Protocol review` block from that
document and bind the exact frozen protocol digest.

## Notes

This connects the learned predictor to the compression goal: a single amortized encoder that
emits any point on the frozen budget ladder. If the elastic checkpoint's deficit versus
dedicated checkpoints exceeds what the storage savings justify, record that as the negative
result and keep dedicated checkpoints. BENCH-007's discipline applies: no compression-adjacent
novelty claim from this task alone.
