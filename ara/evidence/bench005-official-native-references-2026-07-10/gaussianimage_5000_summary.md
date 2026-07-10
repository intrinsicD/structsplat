# Native GaussianImage Comparison

`matched_axes_fixed_n` aligns decoded target pixels, fixed Gaussian count, requested steps, and seed. GaussianImage retains its native renderer, Cholesky/RS parameterization, L2 loss, Adan optimizer, and scheduler.

Shared PSNR, SSIM, proxy MS-SSIM, and LPIPS are centrally recomputed from exported float pixels. Fit/render timings are explicitly CUDA-synchronized. GaussianImage's representation path exports the terminal state and does not produce a codec bitstream.

| Profile | Image | Side | N | Seed | PSNR | MS-SSIM | LPIPS | AUC | Fit s | Render FPS | Param bpp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched_steps_fixed_n | COCO_train2014_000000000009 | 160 | 640 | 0 | 26.5409 | 0.97772 | 0.1035 | 22.1363 | 5.391 | 4408.7 | 8.533 |
| matched_steps_fixed_n | COCO_train2014_000000000025 | 160 | 640 | 0 | 25.3754 | 0.97262 | 0.1994 | 22.6100 | 5.369 | 4279.4 | 9.660 |
| matched_steps_fixed_n | COCO_train2014_000000000030 | 160 | 640 | 0 | 31.5804 | 0.99066 | 0.0434 | 26.0574 | 5.172 | 4225.3 | 9.570 |
| matched_steps_fixed_n | COCO_train2014_000000000034 | 160 | 640 | 0 | 24.0914 | 0.95931 | 0.0878 | 21.0297 | 5.822 | 4238.3 | 9.660 |

## Paired Native vs `structsplat_best_default`

Positive is a native GaussianImage gain; positive time/LPIPS gains mean lower is better. Intervals bootstrap source images after averaging correlated seeds. The familywise relation uses Bonferroni-adjusted bounds across the five core metrics.

| Pairs / images | PSNR gain | MS-SSIM gain | LPIPS gain | AUC gain | Fit gain s | Total gain s | Sample relation | Familywise relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 / 4 | +0.4595 [-2.3206, +2.5484] | +0.02219 [+0.00006, +0.04433] | -0.0055 [-0.0403, +0.0292] | -1.7034 [-3.7343, +0.3103] | +8.1469 [+5.9581, +10.4541] | +8.2029 [+6.0077, +10.5178] | tradeoff | tradeoff |

## Paired Native vs `structsplat_best_checkpoint`

Positive is a native GaussianImage gain; positive time/LPIPS gains mean lower is better. Intervals bootstrap source images after averaging correlated seeds. The familywise relation uses Bonferroni-adjusted bounds across the five core metrics.

| Pairs / images | PSNR gain | MS-SSIM gain | LPIPS gain | AUC gain | Fit gain s | Total gain s | Sample relation | Familywise relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 4 / 4 | -0.1207 [-1.8737, +1.4022] | +0.01298 [-0.00203, +0.02800] | -0.0253 [-0.0502, -0.0048] | -1.5337 [-3.5580, +0.2893] | +6.4448 [+5.6760, +7.0374] | +6.4983 [+5.7285, +7.0876] | tradeoff | tradeoff |
