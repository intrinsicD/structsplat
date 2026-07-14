# COMP-005 — Decoder-synchronized structural geometry

**Status:** not authorized — BENCH-007 Stage 1 failed, tensor structure did not survive the
strongest actual-rate control, and explicit layout bytes were not established as the binding loss.

## 2026-07-14 gate decision

Do not run the eight-image spike below as a post-hoc rescue. Decoder-synchronized geometry may be
reconsidered only under a materially new claim, refreshed prior-art audit, new null, and disjoint
development screen.

## Candidate claim

A transmitted low-rate base layer can serve as a decoder-synchronized structural side channel:
encoder and decoder deterministically reconstruct the same tensor/WSE enhancement geometry from
the decoded base, so the enhancement stream pays primarily for appearance and small residual
geometry rather than absolute means, scales, and rotations.

This is not a claim that progressive coding, structured seeds, or implicit geometry is new.
P-GSVC, SGI, Structure-Guided Allocation, and learned codecs are direct threats. The potentially
new relationship is specifically **deterministic tensor-metric blue-noise geometry derived from
already transmitted reconstruction state**, with its rate saving and error propagation measured.
A fresh primary-source novelty search is required before any publication claim.

## Cheapest killing test

Use the eight frozen BENCH-007 Stage-1 DIV2K-training images at total targets 0.5 and 1.0 bpp:

1. Encode a self-contained base layer at a frozen fraction of total rate
   `{0.10, 0.20, 0.35}`.
2. Cold-decode the base at both encoder and decoder.
3. Run a versioned deterministic CPU structure-tensor + WSE procedure on the decoded base to derive
   enhancement means/orientations/scales. Require bit-exact geometry hashes across two processes.
4. Fit and transmit enhancement colors, plus an optional bounded residual-position code.
5. Compare at equal total actual bytes with:
   - explicit SSPL1 enhancement geometry;
   - a deterministic uniform-WSE enhancement layout;
   - Morton-delta explicit layout;
   - base-only reconstruction;
   - an oracle geometry derived from the original image, clearly labeled as an unattainable upper
     bound.

All base bytes, enhancement bytes, codebooks/ranges, version metadata, and residual geometry count
toward rate. Decoder runtime and peak memory are part of the result.

## Predictions and null

- **Prediction:** structural geometry saves at least 20% of enhancement layout bytes while losing
  no more than 0.15 dB at equal total rate; residual position bits concentrate near unstable
  fine-scale edges.
- **Null:** base-derived geometry provides no net RD gain after the base cost and deterministic
  version metadata are counted.
- **Failure signature:** tensor topology changes under base distortion, causing geometry mismatch
  or large edge-band error exactly where sparse coding matters.

## Gate

Advance beyond the eight-image spike only if, at both 0.5 and 1.0 bpp:

- geometry hashes match across cold processes;
- layout bytes fall by at least 20%;
- equal-rate PSNR is within 0.15 dB of explicit layout or improves by at least 0.20 dB;
- the image-bootstrap interval does not admit a loss worse than 0.25 dB; and
- total decode-plus-render time regresses by no more than 20%.

Otherwise close the task with the error-propagation map. Do not rescue it by changing the base
codec, tensor parameters, WSE seed, and residual code on the same evaluation images.

## Required artifacts

- Bitstream syntax and deterministic-algorithm versioning ADR.
- Base/enhancement/component byte tables and cold-decode parity tests.
- Geometry hashes, instability maps, RD points, timing/memory telemetry, and per-image failures.
- Prior-art audit updated at the execution date.

## Depends on

BENCH-007 Stage 1, COMP-001/002/003/004, INIT-001/003/009.
