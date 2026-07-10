# Fair Gaussian Variants Full Same-Hyperparameter Benchmark — 2026-07-10

## Command

```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/alex/Documents/structsplat
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
PYTHONPATH=src:. \
STRUCTSPLAT_INSTANT_GI=/home/alex/Documents/Instant-GI/quard_image.py \
python -m benchmarks.fair_density_control_compare \
  --outdir results/fair_gaussian_variants_20260710_full_external_same_hparams \
  --images \
    tests/test_images/COCO_train2014_000000000009.jpg \
    tests/test_images/COCO_train2014_000000000025.jpg \
    tests/test_images/COCO_train2014_000000000030.jpg \
    tests/test_images/COCO_train2014_000000000034.jpg \
  --budgets 640 \
  --seeds 0 1 \
  --start-fraction 0.5 \
  --growth-waves 4 \
  --max-side 160 \
  --iters 500 \
  --target-psnr 30.0 \
  --target-psnrs 22.0 24.0 26.0 28.0 30.0 32.0 \
  --renderer cuda \
  --render-chunk 4096 \
  --pixel-loss l1 \
  --ssim-weight 0.3 \
  --feature-cap 12.0 \
  --feature-cap-reference-side 160.0 \
  --resume
```

## Protocol

- Recomputed a full same-hyperparameter version of `results/fair_gaussian_variants_20260709_best_candidates/index.html`.
- Images: four pinned COCO fixtures from `tests/test_images/`.
- Budget: 640 final Gaussians; growth rows start at 320.
- Iterations/resolution/seeds: 500 iterations, max-side 160, seeds 0 and 1.
- Renderer/loss: exact CUDA, `pixel_loss=l1`, `ssim_weight=0.3`, target PSNRs 22/24/26/28/30/32.
- External comparison handling: GaussianImage/GaussianImage++/Image-GS rows are the existing matched-policy analogue rows; Instant-GI uses the local `/home/alex/Documents/Instant-GI/quard_image.py` hook.

## Result

- Completed all 232/232 cells successfully; the prior same-hyperparameter run had 224/232 ok because Instant-GI was unset.
- `structsplat_best_default` remains the default. No equal-budget candidate passed the promotion gate.
- `structsplat_best_tensor_loss` improves PSNR (+0.0119 dB) and AUC (+0.0121) and is faster, but loses MS-SSIM (-0.00024), so it is not promotable.
- `structsplat_best_color_final` improves PSNR (+0.1935 dB) and MS-SSIM (+0.00161), but loses AUC (-0.0179) and speed (+0.0132 s fit), so it is not promotable.
- Instant-GI quadtree completed 8/8 rows with mean PSNR 22.6801, MS-SSIM 0.93493, AUC 21.102, fit 1.085 s.

## Artifacts

- Curated committed overview: `ara/evidence/fair-gaussian-variants-full-external-same-hparams-2026-07-10/index.html`.
- Full ignored visual-grid overview: `results/fair_gaussian_variants_20260710_full_external_same_hparams/index.html`.
