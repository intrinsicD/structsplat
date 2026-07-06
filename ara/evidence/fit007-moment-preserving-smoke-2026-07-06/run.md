# FIT-007 moment-preserving split smoke benchmark

- Date: 2026-07-06
- Command: one-off Python slice using `fit()` on the difficult-four targets from
  `results/abl004_difficult4_test/visual_compare/targets/`
- Protocol: CPU, max-side 64, `aniso_flanking`, seed 0, initial 64 Gaussians,
  `max_gaussians=80`, 60 iterations, `split_every=30`, `split_count=16`,
  `renderer="normalized"`, `ssim_weight=0.3`

| split_mode | mean PSNR | mean PSNR AUC | mean fit seconds | mean post-split delta | mean recovery iters |
|---|---:|---:|---:|---:|---:|
| fp_duplicate | 24.9885 | 22.9859 | 0.4983 | -1.1625 | 6.25 |
| moment_preserving | 25.0369 | 23.0662 | 0.4749 | -0.2128 | 3.00 |
| ranked_wave | 24.6428 | 22.8724 | 0.4948 | -2.0061 | 6.50 |

Per-image rows:

| image | split_mode | PSNR | AUC | fit seconds | post-split delta | iters to recovery |
|---|---|---:|---:|---:|---:|---:|
| kodim01.png | fp_duplicate | 25.7022 | 23.1624 | 0.5586 | -1.1351 | 5 |
| kodim01.png | moment_preserving | 25.5278 | 23.1847 | 0.4114 | -0.1724 | 2 |
| kodim01.png | ranked_wave | 24.9205 | 22.9268 | 0.4552 | -2.5366 | 6 |
| kodim07.png | fp_duplicate | 23.1412 | 21.6665 | 0.4733 | -1.2744 | 6 |
| kodim07.png | moment_preserving | 23.2773 | 21.7573 | 0.4768 | -0.2758 | 4 |
| kodim07.png | ranked_wave | 22.8127 | 21.5153 | 0.5124 | -1.5980 | 9 |
| kodim13.png | fp_duplicate | 24.6477 | 23.0801 | 0.4993 | -1.2507 | 8 |
| kodim13.png | moment_preserving | 24.6280 | 23.1491 | 0.5366 | -0.1321 | 2 |
| kodim13.png | ranked_wave | 24.5777 | 23.0306 | 0.5346 | -1.9694 | 6 |
| kodim19.png | fp_duplicate | 26.4629 | 24.0345 | 0.4620 | -0.9897 | 6 |
| kodim19.png | moment_preserving | 26.7147 | 24.1737 | 0.4747 | -0.2709 | 4 |
| kodim19.png | ranked_wave | 26.2603 | 24.0169 | 0.4768 | -1.9204 | 5 |

Conclusion: `moment_preserving` is a better split primitive than `fp_duplicate` in this smoke:
slightly higher mean PSNR/AUC, much smaller split dip, and faster recovery. It is exposed as a
stage-search refine mode; default behavior remains unchanged until a larger confirmation run.
