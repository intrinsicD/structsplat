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
