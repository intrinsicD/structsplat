# COMP-001: Quantization + entropy/VQ codec

**Status: partial.** First codec implemented in `src/structsplat/codec.py` (ADR-0007); RD sweep in
`benchmarks/rate_distortion.py`. VQ colors + learned entropy model still open.

## Goal
Turn the fitted field into a compact bitstream (rate–distortion), following GaussianImage-style
quantization + entropy/VQ coding.

## Acceptance criteria
- [x] Quantize params (STE during fine-tune); entropy code them.
      Uniform per-attribute quantization + zlib; `qat_finetune` runs a straight-through-estimator
      fine-tune through the quantized renderer (test_codec.py asserts it recovers coarse-bit loss).
- [x] Rate–distortion measurement (actual-bitstream bpp vs PSNR/MS-SSIM) in
      `benchmarks/rate_distortion.py`, sweeping bit mixes ± QAT.
- [ ] Vector-quantized / residual-VQ colors and a learned entropy model (GaussianImage reports
      ~56 bits/Gaussian with residual-VQ colors — the next rate win).
- [ ] Progressive bitstream that exploits the pyramid LOD-prefix (per-level streams).
- [ ] Rate–distortion vs JPEG / GaussianImage baselines on the standard Kodak set.

## Design notes (ADR-0007)
- Positions: Morton-reorder (free under the order-independent normalized renderer, ADR-0003) then
  delta-code; 16-bit fixed point over the image extent by default.
- Rotation canonicalized to `[0, pi)` (Gaussian is invariant under `theta + pi`).
- Colors are unbounded (opacity folded in) → quantized over per-channel data range in the header,
  not a fixed [0,1].

## Depends on
FIT-001.
