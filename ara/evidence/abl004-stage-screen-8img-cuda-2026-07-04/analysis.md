# ABL-004 staged 8-image screen analysis

Scope: 8 Kodak images (`kodim01,04,07,10,13,16,19,22`), budgets `{2000,5000}`, seed `0`, 1500 iterations, exact CUDA renderer, max-side 768. Total cells: 176.

## Leaderboard

### Budget 2000

| rank | strategy | mean PSNR | std | mean MS-SSIM | reached 35dB | mean init+fit s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | aniso_onedge | 26.9861 | 3.3957 | 0.90834 | 0/8 | 25.02 |
| 2 | aniso_flanking | 26.7436 | 3.5067 | 0.90537 | 0/8 | 24.21 |
| 3 | quadtree_wse | 26.5580 | 3.4229 | 0.90414 | 0/8 | 23.99 |
| 4 | quadtree_hybrid | 26.5335 | 3.4749 | 0.90448 | 0/8 | 24.01 |
| 5 | quadtree_aggregate | 26.4738 | 3.3380 | 0.90170 | 0/8 | 23.87 |
| 6 | iso_blue_noise | 26.2164 | 3.2825 | 0.89947 | 0/8 | 23.71 |
| 7 | density_random | 26.1319 | 3.3129 | 0.89614 | 0/8 | 23.20 |
| 8 | random_relocate | 24.8824 | 3.1682 | 0.89194 | 0/8 | 24.53 |
| 9 | grid | 24.8297 | 2.9621 | 0.89685 | 0/8 | 22.70 |
| 10 | random | 24.3467 | 2.9252 | 0.88788 | 0/8 | 24.45 |
| 11 | floyd_steinberg | 24.3337 | 4.0135 | 0.88065 | 0/8 | 22.90 |

### Budget 5000

| rank | strategy | mean PSNR | std | mean MS-SSIM | reached 35dB | mean init+fit s |
|---:|---|---:|---:|---:|---:|---:|
| 1 | quadtree_wse | 30.2148 | 3.9441 | 0.95159 | 1/8 | 27.32 |
| 2 | quadtree_hybrid | 30.2097 | 4.0777 | 0.95205 | 1/8 | 27.58 |
| 3 | aniso_onedge | 30.1034 | 3.9818 | 0.95102 | 1/8 | 29.12 |
| 4 | quadtree_aggregate | 30.0543 | 3.9738 | 0.95067 | 1/8 | 26.70 |
| 5 | aniso_flanking | 30.0470 | 4.0479 | 0.94992 | 1/8 | 28.70 |
| 6 | iso_blue_noise | 29.9891 | 3.9380 | 0.94949 | 1/8 | 26.96 |
| 7 | density_random | 29.5745 | 3.8248 | 0.94617 | 0/8 | 25.70 |
| 8 | floyd_steinberg | 28.9019 | 3.9693 | 0.94579 | 1/8 | 24.96 |
| 9 | grid | 28.2564 | 3.2540 | 0.95108 | 0/8 | 24.38 |
| 10 | random_relocate | 28.0291 | 3.4111 | 0.94692 | 0/8 | 26.63 |
| 11 | random | 27.5772 | 3.0706 | 0.94624 | 0/8 | 26.13 |

## Paired Signals

- At 2000, `aniso_onedge - aniso_flanking`: +0.2425 dB mean PSNR, 7/8 image wins, +0.00297 MS-SSIM.
- At 2000, `quadtree_wse - aniso_flanking`: -0.1856 dB mean PSNR, 4/8 image wins, -0.00123 MS-SSIM.
- At 2000, `quadtree_hybrid - aniso_flanking`: -0.2101 dB mean PSNR, 2/8 image wins, -0.00089 MS-SSIM.
- At 2000, `floyd_steinberg - aniso_flanking`: -2.4099 dB mean PSNR, 1/8 image wins, -0.02472 MS-SSIM.
- At 2000, `density_random - aniso_flanking`: -0.6118 dB mean PSNR, 1/8 image wins, -0.00923 MS-SSIM.
- At 5000, `aniso_onedge - aniso_flanking`: +0.0563 dB mean PSNR, 6/8 image wins, +0.00110 MS-SSIM.
- At 5000, `quadtree_wse - aniso_flanking`: +0.1678 dB mean PSNR, 7/8 image wins, +0.00167 MS-SSIM.
- At 5000, `quadtree_hybrid - aniso_flanking`: +0.1627 dB mean PSNR, 7/8 image wins, +0.00214 MS-SSIM.
- At 5000, `floyd_steinberg - aniso_flanking`: -1.1451 dB mean PSNR, 2/8 image wins, -0.00413 MS-SSIM.
- At 5000, `density_random - aniso_flanking`: -0.4726 dB mean PSNR, 2/8 image wins, -0.00375 MS-SSIM.

## Slice Winners

- PSNR winners at 2000: aniso_flanking=1, aniso_onedge=6, quadtree_wse=1
- MS-SSIM winners at 2000: aniso_flanking=1, aniso_onedge=5, quadtree_aggregate=1, quadtree_hybrid=1
- PSNR winners at 5000: aniso_onedge=1, quadtree_hybrid=3, quadtree_wse=4
- MS-SSIM winners at 5000: grid=3, quadtree_aggregate=1, quadtree_hybrid=3, quadtree_wse=1

## Interpretation

- The original `aniso_flanking` thesis arm is not the best arm in this screen. It is close, but it loses the 2k mean to `aniso_onedge` and loses the 5k mean to `quadtree_wse`/`quadtree_hybrid`.
- Floyd-Steinberg is not the threatening winner in this broader low-budget screen. It had a favorable one-image 20k result, but across these 8 images it ranks last at 2k and 8th at 5k by mean PSNR, with clear failure cases on `kodim07`.
- The strongest follow-up set is `aniso_onedge`, `aniso_flanking`, `quadtree_wse`, `quadtree_hybrid`, plus `iso_blue_noise` as a cheap baseline/control. Keep Floyd-Steinberg as a required killer control, but not as a likely finalist.
- The evidence weakens the specific flanking-offset claim more than the broader structure-tensor initialization idea: on-edge anisotropic placement is the clean 2k winner, while quadtree/tensor hybrids are strongest at 5k.

