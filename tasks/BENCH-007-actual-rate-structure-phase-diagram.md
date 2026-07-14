# BENCH-007 — Actual-rate structure phase diagram

**Status:** in-progress — preregistered 2026-07-13; implementation started 2026-07-14.

## Decision this task owns

Determine whether StructSplat's tensor-driven blue-noise allocation improves rate-distortion after
all transmitted bytes are counted, relative to the strongest direct handcrafted structure prior
and simple allocation controls. This task replaces the 168 KiB analytical-payload lane as the
decision benchmark for compression claims.

The broad question “does image structure help 2D Gaussians?” is already occupied by
[Structure-Guided Allocation](https://arxiv.org/abs/2512.24018), Image-GS, and related work. The
narrow, falsifiable question is:

> At a self-contained SSPL1 rate of 0.25–4.0 bpp, does tensor-metric WSE provide a reproducible
> advantage over SLIC/Sobel allocation, gradient sampling, uniform WSE, and random allocation
> when renderer, fitter, codec search, compute, and decoded pixels are shared?

## Hypotheses

- **H1:** tensor-WSE has a positive rate-distortion effect only in a sparse contributor regime,
  concentrated in edge bands rather than the overcomplete regime.
- **H0:** tensor-WSE has no positive PSNR BD-rate advantage over the strongest nonlearned direct
  prior on held-out images after actual stream bytes and equal encoder search are enforced.
- **Mechanism prediction:** any real advantage co-occurs with lower signed cross-edge bleed or
  edge-band MSE at matched rate, without a compensating texture-band failure.

## Scope and controls

Common-renderer arms:

1. tensor density/orientation + on-edge WSE (the narrow StructSplat claim);
2. shipped `quadtree_wse` (engineering/default control, not the novelty claim);
3. a documented SLIC/Sobel structure-class allocator matching the allocation logic of
   Structure-Guided Allocation as closely as the public specification/code permits;
4. Image-GS-style gradient-weighted sampling;
5. uniform Euclidean WSE;
6. seeded random placement.

All arms use the same constant-color RS field, normalized weighted-sum renderer equation,
optimizer steps, checkpoint rule, QAT allowance, bit-mix/count candidate grid, and SSPL1 encoder.
Native-resolution scientific runs freeze the owned parity-checked exact-CUDA implementation of
that equation; the PyTorch implementation remains the oracle. Paper-name labels are
forbidden unless the native implementation is actually executed; common-renderer transplants must
be labeled `local_<mechanism>_control`.

Outside-class context is reported separately and does not enter the tensor-WSE promotion gate:
lossless PNG as a file-size sanity check; JPEG plus at least one available modern conventional
codec (AVIF or JPEG XL) swept to actual output bytes; and an official learned-codec curve only when
its executable/checkpoint and complete stream are available. Freeze encoder versions, color space,
chroma mode, quality grid, metadata policy, and central RGB round-trip scoring. These rows prevent a
within-Gaussian win from being mislabeled overall image-compression SOTA.

## Dataset and staging

- **Stage 0a — plumbing only:** the four pinned COCO fixtures. Results cannot support a research
  conclusion.
- **Stage 0b — rate calibration only:** DIV2K training images
  `0002, 0268, 0534, 0800`. Use them to estimate stream bytes per Gaussian and freeze the
  resolution-normalized count ladder; do not compare method quality on these images.
- **Stage 1 — development killing pilot:** DIV2K training images
  `0001, 0115, 0229, 0343, 0457, 0571, 0685, 0799`, native orientation, seed 0, targets
  `{0.5, 1.0}` bpp. These IDs and target hashes must be frozen before metric inspection. This is
  a resource/claim killing gate, not held-out evidence.
- **Stage 2 — confirmation only after Stage 1 passes:** all DIV2K validation images `0801–0900`
  as the untouched held-out primary set; Kodak-24 as a development-exposed replication set. Freeze
  tensor-WSE versus the strongest direct nonlearned Stage-1 control as the two-arm primary
  confirmation; run the remaining controls on a separately predeclared diagnostic subset rather
  than multiplying the full matrix post hoc. Use all five target rates
  `{0.25, 0.5, 1.0, 2.0, 4.0}` bpp. Seed 0 is primary; repeat Kodak with seed 1 as a stochastic
  sensitivity analysis.

No crop or resize may silently change the denominator. Rate is
`8 * len(self_contained_stream) / (original_width * original_height)`.
If native-resolution memory is limiting, change only parity-checked chunking/tiling/checkpointing;
do not resize the scientific target to make the cell fit.

## Equal-search rate protocol

1. Predeclare one resolution-normalized Gaussian-count ladder and codec bit-mix ladder for every
   arm. Derive candidate N from original pixel count and Stage-0b's baseline bytes-per-Gaussian
   estimate so the ladder brackets every target (including 4 bpp on native DIV2K). Use fixed
   bracketing multipliers around each estimated target count and the existing four codec bit mixes.
   Freeze the formula, integer rounding, minimum/maximum N, and resulting per-image candidates
   before Stage 1; every arm receives exactly the same N candidates. A fixed absolute 20k ceiling
   is invalid because it cannot reach the high-rate points on multi-megapixel images.
2. Fit each count independently. Apply identical fit and QAT iteration budgets. Do not obtain one
   method's low-count rows by truncating another method's optimized field.
3. Encode, cold-decode, and centrally rescore every candidate. Count the complete SSPL1 file,
   including header, ranges, opacity metadata when present, and entropy payload.
4. At each target rate, select the lowest-MSE (equivalently highest-PSNR) feasible candidate under
   the byte cap. This encoder-side RDO is allowed because the original is available, but every arm
   receives the same candidate grid and all search time is charged. If no candidate fits, record a
   missing point; never substitute analytical or interpolated bytes.
5. Construct a monotone nondominated RD envelope. Compute BD-rate only over the common measured
   PSNR interval, with no endpoint extrapolation. Report the raw points as the primary audit trail.
6. Verify cold-decode equality with the in-memory decoded field and record stream SHA-256,
   source/target hashes, commit/diff provenance, environment, fit/QAT/search time, decode time,
   render latency, peak RSS, and peak VRAM.

## Endpoints and statistics

- **Primary:** paired PSNR BD-rate versus the strongest nonlearned direct control on DIV2K
  validation; paired PSNR at 0.5 and 1.0 bpp.
- **Secondary:** MS-SSIM, LPIPS, encode/search time, cold-decode time, render FPS, stream component
  bytes, edge-band MSE, texture-band MSE, signed cross-edge bleed, and effective contributor count.
- Bootstrap source images after averaging correlated seeds; do not treat rate points as
  independent samples. Report 95% image-cluster intervals and Holm-adjusted intervals for the
  preregistered tensor-WSE comparisons. Preserve per-image curves and failures.

## Gates

Advance tensor-WSE to held-out confirmation only if Stage 1 shows either:

- PSNR BD-rate of at most -10% (candidate versus control; lower rate is favorable), provided at
  least four overlapping measured envelope points make BD-rate defined; or
- at least +0.25 dB at both 0.5 and 1.0 bpp,

with the unadjusted image-bootstrap interval above zero, no more than 10% fit-plus-search time
regression, and the edge/coverage mechanism moving in the predicted direction. This gate only
authorizes Stage 2; it is not itself positive held-out evidence.

Reject or reframe the compression claim if the gain:

- appears only above 4 bpp or only under analytical/count proxies;
- disappears after cold encoding;
- loses to the SLIC/Sobel direct control;
- is caused by unequal candidate search, resize denominators, or a renderer/fitter mismatch; or
- does not survive the full held-out confirmation.

## Deliverables

- A rate-targeted, resumable benchmark with dry-run planning, per-cell journals, explicit missing
  and failed rows, stream validation, and tests for byte caps, denominators, RDO selection,
  monotone envelopes, and BD-rate edge cases.
- Frozen Stage 1 and Stage 2 manifests with source hashes.
- Raw CSV/JSON, central metrics, per-image curves, component byte tables, statistical summaries,
  resource telemetry, and a portable HTML index.
- ARA evidence and a bounded claim update. Negative results are a valid completion.

## Implementation ledger

- [x] Exact-N local SLIC/Sobel control with frozen fidelity assumptions and strategy tests.
- [x] Complete SSPL1 header/stream component accounting with malformed-stream tests.
- [x] Frozen source/pixel hashes, resolution-normalized equal candidate ladders, dry-run planning,
  append-only fit/candidate/error journals, hash-checked resume, and resource telemetry.
- [x] Exact integer byte caps, cold decode/parity, central scoring, RDO selection, explicit missing
  rows, nondominated envelopes, and no-extrapolation BD-rate edge cases.
- [x] Predeclared edge/texture/bleed/contributor metrics, image-cluster bootstrap, Holm adjustment,
  strongest-direct-control rule, and executable Stage-1 gate.
- [x] Automatic F5--F9, retained selected streams/reconstructions, raw CSV/JSON, and portable HTML.
- [x] Separately labeled lossless PNG, JPEG-444, and AVIF-444 context sweep.
- [x] Complete Stage-0a plumbing validation on all four pinned COCO fixtures. The clean-commit
  run completed 144/144 fits, 576/576 cold-encoded candidates, and 48/48 exact-cap selections
  with no failed cell; see `ara/evidence/bench007-stage0a-plumbing-2026-07-14/run.md`.
- [x] Validate persisted-stream cold parity at the decoded-field boundary rather than comparing
  duplicate exact-CUDA renders whose atomic accumulation order is not bit-reproducible; retain the
  frozen `1e-6` tolerance and provide candidate-only revalidation without refitting.
- [x] Complete Stage-0b calibration on the four preregistered DIV2K training IDs. All 8/8 cells
  completed and froze the median `8.614970513660953 B/G`; see
  `ara/evidence/bench007-stage0b-calibration-2026-07-14/run.md`.
- [ ] Freeze and complete Stage 1; obey its stop/go decision without post-hoc rescue.
- [ ] Freeze and run Stage 2 only if the Stage-1 gate authorizes it.

## Non-goals

- No SOTA claim from Stage 0 or Stage 1.
- No comparison of SSPL1 actual bpp with another method's parameter-count proxy.
- No tuning on DIV2K validation. Stage 2 uses the Stage-1-frozen method/protocol unchanged except
  for the preregistered expansion in images and target rates.
- No new learned model, renderer, or primitive in this task.

## Depends on

BENCH-001/002/003/004/006, COMP-001/002/004, INIT-003/009, ABL-004.
