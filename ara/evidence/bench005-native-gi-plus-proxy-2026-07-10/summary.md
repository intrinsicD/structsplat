# Native Reference Comparison

`matched_axes` aligns input image, resolution, Gaussian cap, requested optimization steps, and seed. Native renderer/loss/optimizer/growth semantics remain repository-specific; this is not a same-hyperparameter or matched-policy claim.

Metrics are centrally recomputed from each exported float reconstruction. Native-reported metrics and synchronized renderer timing are retained separately. Codec bpp remains blank unless a real native encoded stream is produced.

GaussianImage++ natively restores its best training-PSNR checkpoint before export, whereas the paired StructSplat row exports its terminal field. The selected iteration is recorded per native row; convergence AUC still covers the complete optimization trajectory. This selection-policy asymmetry is preserved and must be considered when interpreting final quality.

| Method | Image | Side | Cap | Start | Seed | PSNR | MS-SSIM | LPIPS | AUC | Fit s | Render FPS | Param bpp | Commit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GaussianImage++ native | COCO_train2014_000000000009_s160 | 160 | 640 | 320 | 0 | 21.1986 | 0.91519 | 0.2465 | 16.6249 | 0.832 | 4190.6 | 9.585 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000009_s160 | 160 | 640 | 320 | 1 | 21.3170 | 0.91742 | 0.2487 | 16.6477 | 0.840 | 4234.0 | 9.600 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000025_s160 | 160 | 640 | 320 | 0 | 22.7290 | 0.92872 | 0.4068 | 18.5053 | 0.848 | 4118.5 | 10.783 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000025_s160 | 160 | 640 | 320 | 1 | 22.5727 | 0.92977 | 0.4373 | 18.3019 | 0.816 | 4261.4 | 10.851 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000030_s160 | 160 | 640 | 320 | 0 | 24.7648 | 0.93058 | 0.2645 | 18.0181 | 0.870 | 4007.0 | 10.766 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000030_s160 | 160 | 640 | 320 | 1 | 24.8624 | 0.93037 | 0.2530 | 18.0206 | 0.885 | 3989.6 | 10.766 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000034_s160 | 160 | 640 | 320 | 0 | 21.8822 | 0.90396 | 0.2262 | 17.7236 | 0.880 | 4286.3 | 10.868 | 549cfaab2b40 |
| GaussianImage++ native | COCO_train2014_000000000034_s160 | 160 | 640 | 320 | 1 | 21.8062 | 0.89832 | 0.2330 | 17.5842 | 0.814 | 4345.5 | 10.868 | 549cfaab2b40 |

## Paired Native vs StructSplat Default

Positive is a native GaussianImage++ gain; for time and LPIPS, positive means lower is better. Confidence intervals bootstrap source images after averaging seeds within each image. Displayed intervals are marginal 95% intervals; a same-direction dominance relation requires Bonferroni-adjusted 95% familywise bounds across all five core metrics. This matched-axis proxy does not substitute for either repository's native-authentic/full-resolution protocol.

| Pairs / images | PSNR gain [95% CI] | MS-SSIM gain [95% CI] | LPIPS gain [95% CI] | AUC gain [95% CI] | Fit gain s [95% CI] | Total gain s [95% CI] | Sample relation | Familywise 95% relation |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 8 / 4 | -5.0678 [-7.6699, -2.4657] | -0.05142 [-0.06340, -0.03944] | -0.1886 [-0.2175, -0.1466] | -7.1638 [-10.1297, -4.4784] | +0.4284 [+0.2704, +0.6549] | +0.4676 [+0.3086, +0.6983] | tradeoff | tradeoff |
