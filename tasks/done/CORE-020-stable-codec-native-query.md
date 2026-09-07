# CORE-020 — Stable codec-native observation queries

## Context

CORE-016 accepts positive narrow lattice sigmas, but direct Gaussian exponentiation can
underflow before normalization. A constant packet queried halfway between pixels can then
return black in NumPy or lose both appearance and structural support through the float32
realtime-gs adapter. This is a query arithmetic defect in the existing normalized lattice.

## Goal

Preserve the normalized appearance equation and alpha/structural pairing when its unnormalized
Gaussian weights underflow.

## Non-goals

- No renderer, packet grammar, codec, structural allocator, maintained pipeline or default change.
- No new lift method, GPU experiment, dataset access, quantitative quality/performance claim,
  research promotion, or external artifact transfer.

## Acceptance criteria

- [x] CPU NumPy and optional CPU realtime-gs queries reproduce constant appearance at pixel
      midpoints for narrow accepted sigmas, with exact crop and alpha gating.
- [x] Adapter query, appearance-only query and weight-only query agree with the corresponding
      packet planes; appearance underflow cannot erase independent structural density.
- [x] Pixel-center replay and signed cardinal-prefilter reconstruction remain within the existing
      tolerances, and a closed-form midpoint check verifies finite coordinate gradients.
- [x] Empty/outside batches remain finite and correctly shaped.
- [x] Configuration rejects squared sigmas outside the finite normal float32 range used by the
      adapter; both representable variance limits replay constant appearance.
- [x] A distinct reviewer checks the exact diff and focused CPU tests; task state is synchronized.
- [x] `./scripts/verify.sh` passes.

## Interfaces touched

`src/structsplat/codec_native_field.py`, `src/structsplat/realtime_gs_adapter.py`,
`tests/test_codec_native_field.py`, this task, `tasks/INDEX.md`, and `tasks/SESSION-BRIEF.md`.

## Depends on

CORE-016, ADR-0032

## Agent workflow

- Driver: Codex-structsplat-engineering
- Reviewer: Codex-cross-repo-review
- Turn: none
- Reviewed revision: source/test diff SHA-256 c1b63dbd2b367b1deb710dae8f7f51fa6c8e7b208a85280ec76de0bc015e5e7b

### Handoff log

### Handoff

#### Objective
Repair narrow-sigma normalized appearance without dropping the independent structural plane.
#### Changes
Cancel the common lattice Gaussian factor before exponentiation, safely zero outside rows,
and reject variance settings outside float32's finite normal range. Add boundary/gradient tests.
#### Evidence
The original focused regressions failed in six cases before the fix. The updated focused file
passes 36 tests with one CUDA skip under PyTorch 2.9.0+cpu; the full portable gate is running.
#### Assumptions
The adapter uses float32 arithmetic; unusable extreme configurations may fail at cold decode.
#### Uncertainties
CUDA parity and reconstruction-quality effects have not been measured.
#### Review focus
Check NumPy/torch parity, signed prefilter coefficients, coordinate derivatives, crop/alpha
validity, variance endpoints, and unchanged default replay tolerances.
#### Protected actions not taken
No dataset access, sealed outcome consumption, formal experiment, commit, push, or artifact upload.
Claude code review remains pending explicit network-payload approval.
#### Recommended next action
Independently review the bound source/test diff, then reconcile the full gate and terminal record.

### Review

#### Verdict
Accepted
#### Self-reviewed
No
#### Correctness
Codex-cross-repo-review found no blocker. Normalized appearance, signed coefficients,
crop/alpha gating and structural pairing are preserved. Explicit variance bounds reject
configurations incompatible with the float32 adapter (ARA C76).
#### Evidence quality
The reviewer independently reproduced 36 focused passes and one CUDA skip, then checked
18 lattice configurations with a scalar oracle, analytical derivatives, variance endpoints,
radii 1/2/16, signed prefiltering and full cross-repository query pairing.
#### Simplicity
Small localized arithmetic and domain checks; no production dependency changes.
#### Missing cases
CUDA and calibrated reconstruction effects remain outside this CPU contract.
#### Required changes
None. The independent review's full-gate condition is now satisfied.
#### Optional improvements
None required.

### Final verification (2026-09-07)
The complete portable verify.sh gate passed under torch 2.9.0+cpu and Ruff 0.15.20 with both
repository source trees on PYTHONPATH, MKL_THREADING_LAYER=GNU, and two BLAS/OpenMP threads.
Its local receipt is /tmp/structsplat-rtgs-collab/structsplat-verify-final.log. All reviewed
source/test digests remained unchanged. Tracked tests and the exact review boundary are the
durable evidence; local independent probe receipts remain under /tmp/structsplat-rtgs-collab/.
No commit, push, scientific/default promotion, or external artifact transfer was performed.
Claude review remains unperformed pending specific approval for the prepared code-only payload.

### Claude static review and follow-up (2026-09-07)

The user subsequently approved transmission of the prepared code-only review payload. Claude
(`claude-fable-5-1`, tools disabled) completed an independent static review of the combined
cross-repository patch and reported: "No blocking findings within this patch scope."
Claude did not execute tests. The successful patch packet SHA-256 is
66de54dc3b22162caccb5664e484d0f1082d41c063eb603c6522a464cdbc1527;
the verbatim review SHA-256 is
6aecbb0b1a517f94edceaf7758ca6ec381cf729f0d3970733df3500655b81ad7.
Local review and invocation receipts are in /tmp/structsplat-rtgs-collab/claude-review.md
and claude-review-receipt.json. The earlier full-packet attempt returned unusable tool-call
text with tools disabled and is not counted as a review. No datasets, images, masks,
credentials, saved models, or experimental artifacts were included.

Claude suggested matching the variance-endpoint test's square-root operation to the validator.
The test now uses math.sqrt(float(variance)); production code is unchanged. Independent reviewer
Codex-cross-repo-review accepted this two-line test-only delta (self-reviewed: No), verified
both production hashes were unchanged and reversal reproduced the prior test-file hash, and
passed six variance-boundary cases. The updated CORE-020 source/test diff SHA-256 is
c1b63dbd2b367b1deb710dae8f7f51fa6c8e7b208a85280ec76de0bc015e5e7b.
The pre-Claude implementation review was bound to source/test diff SHA-256
5c9fb8a478d1d3bc8c140543dda399fa088a5e9b5f4ffc1578a718d52a959ae9.
A complete portable verification rerun passed for the final fixtures: 2,495 passed, 77 skipped,
515 deselected in 327.12 seconds, followed by all lint/documentation/claim/task/workflow checks.
Its local receipt is /tmp/structsplat-rtgs-collab/structsplat-verify-claude-final.log.
This satisfies the independent follow-up review's full-gate condition. The rerun also covers
the independently reviewed CORE-021 timestamp fixture correction recorded in that task.
No production code changed after Claude's static review; no commit or push was made.

## Notes

CPU tests use synthetic RGB/masks constructed in memory. This numerical repair does not reopen
the rejected CORE-017--019 lift lineage or authorize scientific acceptance of CORE-016.
Previously accepted extreme sigma configurations whose variance underflows or overflows the
adapter's float32 arithmetic are rejected during configuration/cold decode. Existing realistic
packet settings and the serialized schema stay unchanged.
