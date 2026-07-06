# FIT-005 color-solve smoke benchmark

- Date: 2026-07-06
- Command: one-off Python slice using `fit()` on
  `tests/test_images/COCO_train2014_000000000009.jpg` and
  `tests/test_images/COCO_train2014_000000000025.jpg`
- Protocol: CPU, max-side 64, `aniso_flanking`, 64 Gaussians, seed 0, 60 iterations,
  `renderer="normalized"`, `ssim_weight=0.3`, `color_solve_lambda=1e-4`,
  `color_solve_maxiter=32`

| color_solve_every | mean PSNR | mean PSNR AUC | mean fit seconds | mean events |
|---|---:|---:|---:|---:|
| None | 20.5118 | 19.1648 | 0.6263 | 0.0 |
| 10 | 21.0403 | 19.4761 | 1.3917 | 6.0 |
| 25 | 20.5175 | 19.2310 | 0.8747 | 2.0 |
| 50 | 20.5120 | 19.1752 | 0.7247 | 1.0 |

Per-image rows:

| image | every | PSNR | AUC | fit seconds | events | mean relative residual |
|---|---:|---:|---:|---:|---:|---:|
| COCO_train2014_000000000009.jpg | None | 19.0695 | 17.4933 | 0.6909 | 0 | n/a |
| COCO_train2014_000000000009.jpg | 10 | 19.4440 | 17.8022 | 1.4750 | 6 | 0.00060184 |
| COCO_train2014_000000000009.jpg | 25 | 19.0831 | 17.6017 | 0.9051 | 2 | 0.00136280 |
| COCO_train2014_000000000009.jpg | 50 | 19.0558 | 17.5145 | 0.7333 | 1 | 0.00033475 |
| COCO_train2014_000000000025.jpg | None | 21.9541 | 20.8362 | 0.5618 | 0 | n/a |
| COCO_train2014_000000000025.jpg | 10 | 22.6365 | 21.1499 | 1.3083 | 6 | 0.00055756 |
| COCO_train2014_000000000025.jpg | 25 | 21.9518 | 20.8604 | 0.8443 | 2 | 0.00039314 |
| COCO_train2014_000000000025.jpg | 50 | 21.9682 | 20.8359 | 0.7161 | 1 | 0.00026071 |

Conclusion: `every10` is the only clearly competitive interval in this tiny slice
(+0.5285 dB mean final PSNR, +0.3113 mean AUC, but +0.7654 s fit time). It is therefore exposed as
a stage-search axis value, not promoted as a default.
