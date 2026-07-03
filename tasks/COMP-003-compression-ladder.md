# COMP-003: Compression-ratio ladder (toward GaussianImage++-class rate-distortion)

**Status: todo.** From the 2026-07-03 repo review + SOTA survey. Ordered rungs — each is
independently valuable, measured on `benchmarks/rate_distortion.py`, and stacks with the ones
before it. Stop climbing when the RD curve is where you need it.

## Context
The current codec (uniform quantization + Morton/delta + per-stream zlib, ADR-0007) is a solid
v1 but leaves measured, cheap rate on the table, and published 2D-Gaussian codecs
(GaussianImage ECCV 2024; GaussianImage++ arXiv 2512.19108) sit several rungs up this exact
ladder.

## Rungs (acceptance criteria)
- [ ] **1. Per-image scale ranges.** Quantize log-scales over the fitted min/max stored in the
      header (mirroring `color_ranges()`) instead of the static clamp range
      `[log 0.35, log max(H,W)]` — review measured >1 of 8 bits wasted at the static range.
      Freeze ranges during QAT exactly as color ranges are. (`src/structsplat/codec.py:117`)
- [ ] **2. Cheap stream wins (both measured on real fields in the review):** delta-code the
      color stream along the Morton curve (~7%); byte-planar-split the 16-bit means-delta
      stream (~8.5% on that stream); header flag per stream so the decoder stays
      self-describing. (`src/structsplat/codec.py:121-126`)
- [ ] **3. Circular rotation quantization.** 2^bits bins over [0, π) — the current linear
      [0, π] top bin aliases bin 0 (wasted code point; up to a half-step avoidable error near
      the wrap). Update `_quant`/`_dequant` call sites + `quantized_view`'s theta branch.
      (`src/structsplat/codec.py:118`)
- [ ] **4. Sorted attribute planes.** Reshape Morton-ordered attribute streams to
      ~√N×√N planes and encode with PNG (optionally AVIF) instead of `_pack`+zlib
      (Self-Organizing Gaussians, ECCV 2024, arXiv 2312.13299 — the 2D case needs no PLAS,
      Morton order is already spatial). Est. 1.5–3× over per-stream zlib, zero training
      change.
- [ ] **5. LSQ-style learnable quantization steps** during QAT (GaussianImage++,
      arXiv 2512.19108): per-attribute-group learnable lo/hi (equivalently step) with
      straight-through gradients; serialize learned ranges in the header. Est. 10–30% bpp at
      equal PSNR, most at 6–8 bit widths.
- [ ] **6. VQ colors + covariance.** Residual-VQ the (r,g,b) stream and plain-VQ the
      (log_sx, log_sy, θ) triples with an STE codebook-assignment fine-tune reusing the QAT
      plumbing; positions stay on the Morton-delta scalar path (never VQ positions —
      established failure mode). (GaussianImage RVQ, arXiv 2403.08551; Compact3D)
- [ ] **7. Morton-context entropy coding.** Replace zlib with a range/arithmetic coder
      (e.g. `constriction`) driven by an autoregressive context along the Morton order
      (predict next symbol from previous k); the cheap version of HAC-style modeling
      (ECCV 2024, arXiv 2403.14530). Stretch: a tiny learned feature-plane context.
- [ ] **8. Joint RD training (stretch).** `loss += lambda_rate * bits_estimate(field)` with a
      factorized prior over the quantized symbols during the QAT phase; sweep lambda in
      `rate_distortion.py` to trace the full curve. (arXiv 2406.01597)
- [ ] Every rung: RD curve (bpp vs PSNR/MS-SSIM) before/after on the pinned image set,
      committed under `ara/evidence/`; round-trip exactness test per format change.

## Interfaces touched
`src/structsplat/codec.py`, `src/structsplat/fit.py` (QAT phase), `benchmarks/rate_distortion.py`,
`tests/test_codec.py`. Rungs 4+ change the blob format → ADR-0007 amendment or ADR-0011+.

## Depends on
COMP-002 (correctness first — RD numbers must be unbiased before optimizing them), BENCH-002
(config logging).
