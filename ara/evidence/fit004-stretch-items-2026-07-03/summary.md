# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Iters→target | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 21.6899 | 2.5091 | 0.89395 | 20.112 | - (0/4) | 1.07 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 256 | 21.5799 | 2.5197 | 0.89207 | 20.036 | - (0/4) | 1.08 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.3|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 3 | 256 | 21.3616 | 2.4684 | 0.88652 | 19.641 | - (0/4) | 1.16 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|loss=l1|optimizer=adam|lr_schedule=none|refine=absgrad_wave|pyramid=single` |
| 4 | 256 | 21.0764 | 2.4789 | 0.87732 | 19.820 | - (0/4) | 1.13 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|loss=l1|optimizer=adan|lr_schedule=none|refine=none|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Iters→target | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|
| aa | 0.0 | 12 | 21.376 ± 2.498 | 0.88593 | 19.858 | - (0/12) | 1.12 |
| aa | 0.3 | 4 | 21.580 ± 2.520 | 0.89207 | 20.036 | - (0/4) | 1.08 |
| optimizer | adam | 12 | 21.544 ± 2.503 | 0.89085 | 19.930 | - (0/12) | 1.11 |
| optimizer | adan | 4 | 21.076 ± 2.479 | 0.87732 | 19.820 | - (0/4) | 1.13 |
| refine | absgrad_wave | 4 | 21.362 ± 2.468 | 0.88652 | 19.641 | - (0/4) | 1.16 |
| refine | none | 12 | 21.449 ± 2.517 | 0.88778 | 19.989 | - (0/12) | 1.10 |
