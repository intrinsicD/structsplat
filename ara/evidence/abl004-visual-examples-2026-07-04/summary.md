# ABL-004 Visual Examples

Small matched visual sheets for inspecting the current ABL-004 finalist/control variants.

## Run Controls

- Images: kodim01, kodim07, kodim10, kodim13, kodim22
- Levels: 160/240/320 px x 80/200 iterations
- Budgets: base 640 at 160px, area-scaled per level
- Seed: 0
- Renderer: cuda
- LPIPS: disabled

## Variants

- **Aniso on-edge**: anisotropic WSE centers placed on features
- **Aniso flanking**: anisotropic WSE centers flanking features, offset 0.5
- **Quadtree WSE**: quadtree allocation with local WSE/flanking samples
- **Quadtree hybrid**: aggregate smooth cells and sampled detailed cells
- **Iso blue noise**: isotropic weighted sample elimination baseline
- **Floyd-Steinberg**: same flanking thesis config with Floyd-Steinberg placement

## Example Sheets

| Image | Grid |
|---|---|
| kodim01 | `grids/kodim01_all_levels.png` |
| kodim07 | `grids/kodim07_all_levels.png` |
| kodim10 | `grids/kodim10_all_levels.png` |
| kodim13 | `grids/kodim13_all_levels.png` |
| kodim22 | `grids/kodim22_all_levels.png` |

## Mean Metrics By Level

| Max side | Iterations | Variant | Budget | PSNR | MS-SSIM | Fit s |
|---:|---:|---|---:|---:|---:|---:|
| 160 | 80 | Aniso on-edge | 640 | 28.1064 | 0.96741 | 0.301 |
| 160 | 80 | Aniso flanking | 640 | 28.1286 | 0.96879 | 0.197 |
| 160 | 80 | Quadtree WSE | 640 | 28.0514 | 0.96677 | 0.189 |
| 160 | 80 | Quadtree hybrid | 640 | 28.0678 | 0.96732 | 0.189 |
| 160 | 80 | Iso blue noise | 640 | 27.2365 | 0.95826 | 0.198 |
| 160 | 80 | Floyd-Steinberg | 640 | 27.8628 | 0.96570 | 0.196 |
| 160 | 200 | Aniso on-edge | 640 | 29.8789 | 0.97913 | 0.492 |
| 160 | 200 | Aniso flanking | 640 | 29.8212 | 0.97871 | 0.467 |
| 160 | 200 | Quadtree WSE | 640 | 29.7656 | 0.97898 | 0.465 |
| 160 | 200 | Quadtree hybrid | 640 | 29.7969 | 0.97908 | 0.464 |
| 160 | 200 | Iso blue noise | 640 | 29.5854 | 0.97829 | 0.454 |
| 160 | 200 | Floyd-Steinberg | 640 | 29.4289 | 0.97636 | 0.454 |
| 240 | 80 | Aniso on-edge | 1440 | 28.5344 | 0.96426 | 0.199 |
| 240 | 80 | Aniso flanking | 1440 | 28.4138 | 0.96400 | 0.194 |
| 240 | 80 | Quadtree WSE | 1440 | 28.3753 | 0.96355 | 0.192 |
| 240 | 80 | Quadtree hybrid | 1440 | 28.3977 | 0.96507 | 0.190 |
| 240 | 80 | Iso blue noise | 1440 | 27.4834 | 0.95470 | 0.189 |
| 240 | 80 | Floyd-Steinberg | 1440 | 28.1281 | 0.96126 | 0.188 |
| 240 | 200 | Aniso on-edge | 1440 | 30.4178 | 0.97745 | 0.531 |
| 240 | 200 | Aniso flanking | 1440 | 30.2858 | 0.97720 | 0.501 |
| 240 | 200 | Quadtree WSE | 1440 | 30.2130 | 0.97660 | 0.483 |
| 240 | 200 | Quadtree hybrid | 1440 | 30.3451 | 0.97853 | 0.480 |
| 240 | 200 | Iso blue noise | 1440 | 29.8603 | 0.97431 | 0.532 |
| 240 | 200 | Floyd-Steinberg | 1440 | 29.9733 | 0.97471 | 0.507 |
| 320 | 80 | Aniso on-edge | 2560 | 29.1612 | 0.96969 | 0.301 |
| 320 | 80 | Aniso flanking | 2560 | 29.1002 | 0.96931 | 0.297 |
| 320 | 80 | Quadtree WSE | 2560 | 29.0819 | 0.96903 | 0.299 |
| 320 | 80 | Quadtree hybrid | 2560 | 28.9983 | 0.96794 | 0.305 |
| 320 | 80 | Iso blue noise | 2560 | 27.9333 | 0.95976 | 0.285 |
| 320 | 80 | Floyd-Steinberg | 2560 | 28.8576 | 0.96779 | 0.291 |
| 320 | 200 | Aniso on-edge | 2560 | 30.9946 | 0.97981 | 0.732 |
| 320 | 200 | Aniso flanking | 2560 | 30.9156 | 0.97946 | 0.729 |
| 320 | 200 | Quadtree WSE | 2560 | 30.9689 | 0.97971 | 0.720 |
| 320 | 200 | Quadtree hybrid | 2560 | 30.8456 | 0.97915 | 0.734 |
| 320 | 200 | Iso blue noise | 2560 | 30.7110 | 0.97863 | 0.710 |
| 320 | 200 | Floyd-Steinberg | 2560 | 30.8491 | 0.97927 | 0.730 |

Per-cell reconstructions are under `reconstructions/`; visual sheets are under `grids/`.
