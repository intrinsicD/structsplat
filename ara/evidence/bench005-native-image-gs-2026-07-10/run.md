# Native Image-GS proxy evidence (2026-07-10)

Two provenance-checked lanes ran official Image-GS commit
`03088368d42684fb54225c981cfd94b58cc0393a` with its own bundled `gsplat` build and
release-era `fused-ssim` commit `b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3`.
Target pairs require run-recorded decoded-pixel SHA-256 equality. Central PSNR, small-image proxy
MS-SSIM, and LPIPS come from float reconstructions. Native AUC remains diagnostic because history
clamping/cadence differs; timing protocols are not strict-dominance comparable; Image-GS rate is
an analytical attribute-bit estimate, not a packed stream.

The active isolated environment was Python 3.12.9 / Torch 2.9.0+cu128 / CUDA 12.8 rather than the
official Python 3.11.10 / Torch 2.4.1 / CUDA 12.4 stack. Dependency versions and source/binary
hashes are retained in configs/rows.

## Fixed-N short lane

- COCO4, max-side 160, cap/start 640, 500 steps, seeds 0/1; 8/8 cells successful.
- Image-GS minus StructSplat gains: PSNR -3.6011 dB (95% CI [-4.3059, -2.7527]); proxy MS-SSIM
  -0.01879 [-0.02937, -0.00822]; LPIPS gain -0.1842 [-0.2658, -0.1135].
- The familywise final-quality relation supports StructSplat for this bounded fixed-N ablation.

## SIGGRAPH25 algorithm-profile lane

- Same COCO4 pixels, cap 640/start 320, 5,000 steps, seed 0; 4/4 cells successful.
- Image-GS minus StructSplat gains: PSNR -0.3840 dB (95% CI [-2.3698, +1.1997]); proxy MS-SSIM
  +0.01608 [+0.00074, +0.03142]; LPIPS gain -0.0443 [-0.0652, -0.0243].
- This is a heterogeneous tradeoff, not a winner or native-authentic/full-resolution result.

Local HTML artifacts remain under `results/native_image_gs_matched_fixedn_proxy/` and
`results/native_image_gs_siggraph25_proxy_seed0/`.
