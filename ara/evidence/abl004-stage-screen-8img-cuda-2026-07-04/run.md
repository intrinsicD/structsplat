# ABL-004 staged 8-image exact-CUDA screen

Date: 2026-07-04

Purpose: run the predeclared ABL-004 screen within the user's 6-hour cap. This is a
screening run, not final confirmation: 8 Kodak images, 2 budgets, seed 0 only.

Images:

- `kodim01`
- `kodim04`
- `kodim07`
- `kodim10`
- `kodim13`
- `kodim16`
- `kodim19`
- `kodim22`

Command:

```bash
timeout 6h env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
python -m benchmarks.ablation \
  results/datasets/abl004/kodak24/kodim01.png \
  results/datasets/abl004/kodak24/kodim04.png \
  results/datasets/abl004/kodak24/kodim07.png \
  results/datasets/abl004/kodak24/kodim10.png \
  results/datasets/abl004/kodak24/kodim13.png \
  results/datasets/abl004/kodak24/kodim16.png \
  results/datasets/abl004/kodak24/kodim19.png \
  results/datasets/abl004/kodak24/kodim22.png \
  --budgets 2000 5000 \
  --seeds 0 \
  --iters 1500 \
  --target-psnr 35 \
  --max-side 768 \
  --renderer cuda \
  --device cuda \
  --resume \
  --no-plots \
  --outdir results/abl004_stage_screen_8img_cuda
```

Plots and derived tables were generated after the completed run from `ablation.json`.

## Completion

- Cells: 176/176
- Images: 8
- Budgets: 2000, 5000
- Strategies: 11
- Wall-clock from config write to final summary: 4489.46 s (74.82 min)
- Sum of per-cell init+fit timings: 4454.34 s (74.24 min)
- Mean cell init+fit: 25.31 s

## Headline

At 2000 Gaussians, `aniso_onedge` is the clear screen winner:

- `aniso_onedge`: 26.9861 dB mean PSNR, 0.90834 MS-SSIM
- `aniso_flanking`: 26.7436 dB mean PSNR, 0.90537 MS-SSIM
- Paired `aniso_onedge - aniso_flanking`: +0.2425 dB, 7/8 image wins

At 5000 Gaussians, quadtree variants edge out the anisotropic variants:

- `quadtree_wse`: 30.2148 dB mean PSNR, 0.95159 MS-SSIM
- `quadtree_hybrid`: 30.2097 dB mean PSNR, 0.95205 MS-SSIM
- `aniso_onedge`: 30.1034 dB mean PSNR, 0.95102 MS-SSIM
- `aniso_flanking`: 30.0470 dB mean PSNR, 0.94992 MS-SSIM
- Paired `quadtree_wse - aniso_flanking`: +0.1678 dB, 7/8 image wins

Floyd-Steinberg did not generalize from the single-image 20k warning:

- 2000: 24.3337 dB mean PSNR, rank 11/11
- 5000: 28.9019 dB mean PSNR, rank 8/11
- Paired vs `aniso_flanking`: -2.4099 dB at 2000, -1.1451 dB at 5000
- Clear failure case: `kodim07`

## Interpretation Boundary

This screen weakens the specific `aniso_flanking` thesis arm: the flank offset is not
currently justified by the screen. It does not weaken the broader structure-tensor
initialization direction: `aniso_onedge` is strongest at 2000, and quadtree/tensor hybrids
are strongest at 5000. Confirmation should keep `aniso_onedge`, `aniso_flanking`,
`quadtree_wse`, `quadtree_hybrid`, `iso_blue_noise`, and Floyd-Steinberg as the required
killer control.
