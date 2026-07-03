# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Iters→target | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 16 | 16.1100 | 0.0000 | 0.69779 | 15.059 | - (0/1) | 0.01 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Iters→target | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|
