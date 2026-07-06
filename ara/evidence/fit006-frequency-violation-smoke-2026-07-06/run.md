# FIT-006 frequency-violation densification smoke benchmark

- Date: 2026-07-06
- Command: one-off Python slice using `fit()` on the difficult-four targets from
  `results/abl004_difficult4_test/visual_compare/targets/`
- Protocol: CPU, max-side 64, `aniso_flanking`, seed 0, initial 64 Gaussians,
  `max_gaussians=80`, 60 iterations, `split_every=30`, `split_count=16`,
  `renderer="normalized"`, `ssim_weight=0.3`

| split_mode | mean PSNR | mean PSNR AUC | mean fit seconds | mean post-split delta |
|---|---:|---:|---:|---:|
| residual_tensor_add | 24.9109 | 22.9737 | 0.5382 | -1.0770 |
| ranked_wave | 24.6428 | 22.8724 | 0.5354 | -2.0061 |
| absgrad_wave | 24.7613 | 22.9155 | 0.5023 | -0.9979 |
| freq_violation | 24.8400 | 22.9360 | 0.5230 | -1.2561 |

Per-image rows:

| image | split_mode | PSNR | AUC | fit seconds | post-split delta |
|---|---|---:|---:|---:|---:|
| kodim01.png | residual_tensor_add | 25.1916 | 23.1259 | 0.5726 | -0.6315 |
| kodim01.png | ranked_wave | 24.9205 | 22.9268 | 0.4736 | -2.5366 |
| kodim01.png | absgrad_wave | 24.9127 | 22.9123 | 0.4228 | -1.3377 |
| kodim01.png | freq_violation | 25.3308 | 23.0500 | 0.4851 | -2.0629 |
| kodim07.png | residual_tensor_add | 23.2381 | 21.6214 | 0.5295 | -1.5286 |
| kodim07.png | ranked_wave | 22.8127 | 21.5153 | 0.5310 | -1.5980 |
| kodim07.png | absgrad_wave | 22.9757 | 21.6469 | 0.5387 | -0.4067 |
| kodim07.png | freq_violation | 22.8665 | 21.5483 | 0.5418 | -1.2192 |
| kodim13.png | residual_tensor_add | 24.7317 | 23.0634 | 0.5495 | -1.2844 |
| kodim13.png | ranked_wave | 24.5777 | 23.0306 | 0.5573 | -1.9694 |
| kodim13.png | absgrad_wave | 24.4472 | 22.9918 | 0.5762 | -1.2958 |
| kodim13.png | freq_violation | 24.7718 | 23.0990 | 0.5819 | -0.9098 |
| kodim19.png | residual_tensor_add | 26.4823 | 24.0840 | 0.5013 | -0.8634 |
| kodim19.png | ranked_wave | 26.2603 | 24.0169 | 0.5798 | -1.9204 |
| kodim19.png | absgrad_wave | 26.7096 | 24.1109 | 0.4716 | -0.9513 |
| kodim19.png | freq_violation | 26.3908 | 24.0466 | 0.4833 | -0.8325 |

Conclusion: `freq_violation` is not a default candidate from this slice because
`residual_tensor_add` still has better mean final PSNR/AUC. It is competitive enough to keep as a
stage-search refine axis: it beats `ranked_wave` and `absgrad_wave` on mean final PSNR/AUC and wins
individual images (`kodim01`, `kodim13`) but has mixed post-split dip behavior.
