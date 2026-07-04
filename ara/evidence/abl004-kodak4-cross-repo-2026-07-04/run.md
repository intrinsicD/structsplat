# ABL-004 held-out Kodak cross-repo run

Command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=src:. \
  python -m benchmarks.cross_repo_matrix_compare \
  --dataset-dir results/datasets/abl004/kodak24 \
  --images kodim01.png kodim02.png kodim03.png kodim04.png \
  --max-sides 160 --iters 80 --base-budget 640 --base-side 160 \
  --seeds 0 --renderer cuda --device cuda \
  --outdir results/cross_repo_matrix_kodak4_shipped
```

Scope:

- Held-out from the four COCO images used in earlier config selection evidence.
- Four Kodak images, one seed, one resolution/iteration point: max-side 160, 80 iterations.
- Includes `structsplat_current` and `structsplat_shipped_defaults`.
- `Instant-GI qt` rows are expected errors because `STRUCTSPLAT_INSTANT_GI` is not configured.

Headline means over four successful image rows:

- `structsplat_current`: 31.6934 dB PSNR, 0.97767 MS-SSIM.
- `structsplat_shipped_defaults`: 31.6323 dB PSNR, 0.98046 MS-SSIM.
- `gaussianimage`: 28.8334 dB PSNR, 0.97133 MS-SSIM.
- `image_gs`: 27.0637 dB PSNR, 0.94621 MS-SSIM.
- `gaussianimage_plus`: 26.6340 dB PSNR, 0.94307 MS-SSIM.

Interpretation:

This small held-out run supports the narrower claim that both current searched StructSplat and
the public shipped-default row outperform the local GaussianImage/Image-GS analogues at this
low-resolution operating point. It is not a substitute for the full ABL-001 sweep or a native
external-repository benchmark.
