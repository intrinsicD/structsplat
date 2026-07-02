# COMP-001: Quantization + entropy/VQ codec

**Status: todo (future).**

## Goal
Turn the fitted field into a compact bitstream (rate–distortion), following GaussianImage-style
quantization + entropy/VQ coding.

## Acceptance criteria
- [ ] Quantize params (STE during fine-tune); entropy or VQ code them.
- [ ] Rate–distortion curve (bpp vs PSNR/MS-SSIM) vs baselines on the standard set.

## Depends on
FIT-001.
