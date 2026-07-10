# Native GaussianImage Comparison

`matched_axes_fixed_n` aligns decoded target pixels, fixed Gaussian count, requested steps, and seed. GaussianImage retains its native renderer, Cholesky/RS parameterization, L2 loss, Adan optimizer, and scheduler.

Shared PSNR, SSIM, proxy MS-SSIM, and LPIPS are centrally recomputed from exported float pixels. Fit/render timings are explicitly CUDA-synchronized. GaussianImage's representation path exports the terminal state and does not produce a codec bitstream.

| Profile | Image | Side | N | Seed | PSNR | MS-SSIM | LPIPS | AUC | Fit s | Render FPS | Param bpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_steps_fixed_n | COCO_train2014_000000000009 | 160 | 640 | 0 | 11.8499 | 0.65564 | 0.6449 | 8.4164 | 0.952 | 4274.6 | 8.533 |
| matched_steps_fixed_n | COCO_train2014_000000000009 | 160 | 640 | 1 | 11.7700 | 0.66344 | 0.6188 | 8.3671 | 1.515 | 950.7 | 8.533 |
| matched_steps_fixed_n | COCO_train2014_000000000025 | 160 | 640 | 0 | 15.6964 | 0.77922 | 0.5963 | 11.7880 | 1.325 | 658.3 | 9.660 |
| matched_steps_fixed_n | COCO_train2014_000000000025 | 160 | 640 | 1 | 15.4751 | 0.77626 | 0.5989 | 11.5665 | 0.867 | 4191.4 | 9.660 |
| matched_steps_fixed_n | COCO_train2014_000000000030 | 160 | 640 | 0 | 13.8434 | 0.69210 | 0.6695 | 9.2948 | 1.209 | 4243.1 | 9.570 |
| matched_steps_fixed_n | COCO_train2014_000000000030 | 160 | 640 | 1 | 13.5858 | 0.68652 | 0.6473 | 9.1533 | 0.787 | 4017.2 | 9.570 |
| matched_steps_fixed_n | COCO_train2014_000000000034 | 160 | 640 | 0 | 14.6029 | 0.71771 | 0.5404 | 11.4659 | 0.977 | 4302.7 | 9.660 |
| matched_steps_fixed_n | COCO_train2014_000000000034 | 160 | 640 | 1 | 14.8014 | 0.72574 | 0.5011 | 11.5154 | 0.803 | 4143.2 | 9.660 |

## Paired Native vs `structsplat_best_default`

Positive is a native GaussianImage gain; positive time/LPIPS gains mean lower is better. Intervals bootstrap source images after averaging correlated seeds. The familywise relation uses Bonferroni-adjusted bounds across the five core metrics.

| Pairs / images | PSNR gain | MS-SSIM gain | LPIPS gain | AUC gain | Fit gain s | Total gain s | Sample relation | Familywise relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -13.7463 [-17.9002, -9.5925] | -0.25929 [-0.31230, -0.20628] | -0.5037 [-0.5988, -0.4087] | -14.6578 [-18.4459, -10.9353] | +0.2825 [+0.1929, +0.3722] | +0.3362 [+0.2438, +0.4285] | tradeoff | tradeoff |

## Paired Native vs `structsplat_best_checkpoint`

Positive is a native GaussianImage gain; positive time/LPIPS gains mean lower is better. Intervals bootstrap source images after averaging correlated seeds. The familywise relation uses Bonferroni-adjusted bounds across the five core metrics.

| Pairs / images | PSNR gain | MS-SSIM gain | LPIPS gain | AUC gain | Fit gain s | Total gain s | Sample relation | Familywise relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -13.7158 [-17.7996, -9.6321] | -0.25876 [-0.31201, -0.20550] | -0.5038 [-0.5977, -0.4100] | -14.6363 [-18.3904, -10.9355] | +0.1715 [+0.0295, +0.3148] | +0.2236 [+0.0817, +0.3671] | tradeoff | tradeoff |
