# COMP-006: Marginal cold-stream rate--distortion attribution

## Status

Completed negative on 2026-07-15. The frozen 36-cell development screen and an independent
same-source replay each produced 33,840 validated cold streams and match exactly on protocol,
source, cells, streams, selections, aggregates, and decision. The development gate failed, so the
decision is **stop** and the odd-variant confirmation split remains unfit and unscored. This task
is benchmark-only; it changed no SSPL1 syntax, production allocator, fitter default, renderer,
configuration, or CLI behavior.

## Question and claim boundary

The operational question is whether one extra standard Gaussian is a better use of a small
self-contained SSPL1 byte allowance than either replacing an existing Gaussian at equal count or
spending the allowance on global attribute precision.

At the target-specific integer cap `matched_no_edit_bytes + 16`, the preregistered claim is:

> the best of 16 frozen residual-birth actions has at least `+0.15 dB` paired cold-decoded PSNR
> advantage over the strongest feasible union of no-edit recovery, 16 matched count-neutral
> birth-for-death actions, and the exhaustive SSPL1 precision-only envelope.

The null is that the control envelope equals or beats the structural-birth oracle. This is a
matched operational audit, not a deployable selector and not a novelty claim for exact-byte RDO.
Classical operational codec RDO, EBCOT, tree-structured RDO, sparse approximation, and MDL already
establish rate-priced mode and structure selection. The search-bounded delta is the full-container
counterfactual audit across heterogeneous edits in this explicit 2D Gaussian codec.

## Frozen protocol

### Targets and independence

- Deterministic `64x64` float32 targets from six new families: curved chromatic step,
  three-region junction, thin Bezier strokes, sparse hard-edged disks, annular sectors, and a
  smooth quadratic patch.
- Six frozen variants per family. Even variants `{0,2,4}` are development; odd variants
  `{1,3,5}` are an untouched confirmation split.
- Target pixels, metadata, split, and manifest are hash-bound before scoring. Every target hash
  must be disjoint from FIT-020.
- Seeds `{0,1}` are repeated initialization/fitting measurements and are averaged before target-
  level inference. Targets, not seeds, are the independent units.
- Confirmation pixels may be generated for hash freezing and tests, but they may not be fit,
  encoded, or scored unless development passes every gate below.

### Shared parent

For each target/seed:

1. Build one `N=64`, constant-RGB, no-opacity field with shipped `quadtree_wse` initialization.
2. Run 120 ordinary normalized-renderer fit steps, then 40 base-QAT steps.
3. Encode a complete zlib-9 SSPL1 stream at `(means, scales, rotation, colors)=(12,6,6,6)`.
4. Persist, hash, cold-read, and decode that exact stream. Every action branch starts from an
   independent detached clone of this cold-decoded field with fresh Adam state.

The cold parent deliberately discards hidden float state. `fit()` and QAT both reconstruct Adam,
so the evidence is fresh-optimizer recovery and makes no optimizer-continuation claim.

### Matched actions and compute

- Generate 16 unique, spacing-controlled residual-local-maximum birth rows once from the shared
  parent with the existing sampled-add constructor. Freeze their order and action hashes.
- `birth_i`: append candidate row `i` (`N=65`).
- `replace_i`: delete the one deterministically lowest-activity parent row and append the exact
  same candidate row `i` (`N=64`). The donor is fixed across all 16 candidates in a cell.
- `no_edit`: clone the parent (`N=64`).
- Score every action at recovery checkpoints 0 and 20. Checkpoint 20 receives the same 20-step
  base-bit QAT recovery with fresh Adam; checkpoint 0 receives no optimization.
- Candidate generation, ordinary fitting, QAT, loss, learning rates, renderer, support, seed,
  and metric code are otherwise fixed.

This paired birth/replacement construction isolates the extra row. A moment split is excluded
because it would introduce an opacity stream when the no-opacity parent has none. Affine color,
WIPES-like carriers, and other richer atoms are excluded because codec v1 rejects them; no proxy
byte count is allowed in this task.

### Exact-rate and precision rules

- Each counterfactual is encoded as an independent, complete, self-describing SSPL1 container.
- `delta_bytes = len(counterfactual_blob) - len(matched_no_edit_blob)` includes magic, header,
  ranges, framing, every compressed stream, and any Morton/zlib context change.
- This is a **counterfactual complete-stream delta**, not an incremental patch cost, additive local
  price, per-row byte cost, or sequential-edit budget. Negative/non-monotone deltas are valid and
  are handled by integer-cap Pareto selection, never division by a near-zero delta.
- Precision-only candidates exhaust the 875 global bit mixes with means in `10..16` and scales,
  rotation, and colors independently in `4..8`. They encode the matched recovered no-edit field
  with its frozen QAT ranges. Stable selection is by lowest cold MSE, then smallest bytes, then
  lexicographic bit mix.
- Primary cap is `matched_no_edit_bytes + 16`. `+0`, `+8`, and `+32` are descriptive sensitivity
  caps only and cannot rescue the primary decision.
- Direct precision encodes do not receive per-mix QAT. This intentionally makes the screen a
  cheap killing test. A positive result authorizes, but does not replace, a stronger per-mix-QAT
  confirmation; a structural loss against this weaker control is already decisive.

### Metrics, integrity, and replay

- Candidate selection uses cold-decoded display-clamped MSE. Report PSNR, SSIM, MS-SSIM, total
  bytes/bpp, nominal raw bits, component-byte deltas, stream hashes, encode/decode-render time,
  and action/recovery time. Timing is descriptive CPU evidence only.
- Persist every stream before scoring; read it back; verify its SHA-256, header schema/dimensions,
  decoded field state, finite central render, and exact `blob_components()` partition.
- Fail closed on missing/duplicate cells, target or action hash drift, branch mutation, wrong
  counts, cap violations, unexpected schema, nonfinite values, or source/config mismatch.
- Use deterministic single-thread CPU execution, append-only resumable JSONL, environment and
  source fingerprints, and a complete source snapshot.
- Replay the completed permitted split. All non-timing rows, stream hashes/bytes, selections,
  aggregates, and the decision must match exactly.

## Development gate

At recovery step 20 and cap `+16 bytes`, development authorizes the untouched confirmation split
only if all conditions pass:

1. at least 90% of target/seed cells have both a feasible birth and a feasible control;
2. mean target-level paired birth advantage over the strongest control is at least `+0.15 dB`;
3. a family-stratified target bootstrap 95% lower bound is greater than zero;
4. median target-level advantage is at least `+0.10 dB`;
5. at least four of six family means are positive; and
6. every integrity and replay check passes.

If development fails, stop and do not score confirmation. Ranking disagreement without material
RD advantage is benchmark infrastructure, not an allocator result. If development passes, run the
same frozen task once on confirmation and require the identical gate before proposing any natural-
image or learned price-model study.

## Explicitly excluded claims

Even a positive confirmation would not establish a sequential/deployable selector, additive
entropy attribution, a richer atom, a natural-image or external-codec advantage, a rendering or
training speedup, codec-independent behavior, production readiness, or image-compression SOTA.

## Outcome

At the preregistered recovery step 20 and `matched_no_edit_bytes + 16` cap, all 36 target/seed
cells had a feasible birth and control. Birth nevertheless trailed the strongest control by
`-1.0714 dB` mean paired PSNR and `-0.9533 dB` median target-level PSNR. The family-stratified
bootstrap 95% interval was `[-1.2873, -0.8417] dB`, and all six family means were negative.

| Gate | Frozen requirement | Result | Pass |
|---|---:|---:|:---:|
| Feasible cells | at least 90% | 36/36 | yes |
| Mean birth advantage | at least +0.15 dB | -1.0714 dB | no |
| Family-bootstrap lower bound | greater than 0 | -1.2873 dB | no |
| Median birth advantage | at least +0.10 dB | -0.9533 dB | no |
| Positive family means | at least 4/6 | 0/6 | no |
| Integrity and exact replay | all checks | exact match | yes |

The exact-byte oracle and nominal-raw-bit proxy chose the identical row in only 14/36 primary
cells, so complete-stream accounting is useful audit infrastructure. That disagreement does not
rescue structural birth: exact and proxy selection agreed on the broad action class in 34/36
cells, and birth was the exact overall winner in only 5/36. The strongest control was precision
reallocation in 23/36 cells and count-neutral replacement in 13/36. Confirmation is prohibited by
the unchanged gate. See
[`docs/research/2026-07-15-marginal-cold-stream-rd.md`](../docs/research/2026-07-15-marginal-cold-stream-rd.md)
and ADR-0016.

## Acceptance criteria

- [x] Frozen target/action/precision manifests and integrity invariants have focused tests.
- [x] Development raw streams/rows, aggregates, source snapshot, and exact replay are complete.
- [x] Positive or negative outcomes are reported without retuning cap, targets, actions, bits, or
      horizon after inspection.
- [x] Confirmation is run only after an unchanged passing development decision; it was not run.
- [x] Task/index, benchmark docs, dated research report, and ARA agree.
- [x] Focused/full tests, Ruff, source-snapshot verification, and diff hygiene pass.

## Interfaces allowed

New benchmark, tests, task/research documentation, ignored result evidence, and ARA records only.
No production codec, fitter, renderer, configuration, CLI, or default changes are authorized.

## Depends on

COMP-001/002/003/004, BENCH-002/007, FIT-004/017, and E4 in the 2026-07-15 research portfolio.
