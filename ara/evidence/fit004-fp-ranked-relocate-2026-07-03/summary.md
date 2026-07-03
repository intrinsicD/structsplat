# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Iters→target | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 21.6899 | 2.5091 | 0.89395 | 20.112 | - (0/4) | 1.13 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 256 | 21.4186 | 2.5861 | 0.88800 | 19.687 | - (0/4) | 1.21 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=fp_duplicate|pyramid=single` |
| 3 | 256 | 21.3822 | 2.5822 | 0.88821 | 19.647 | - (0/4) | 1.19 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_tensor_add_nms|pyramid=single` |
| 4 | 256 | 21.3245 | 2.5134 | 0.88912 | 19.776 | - (0/4) | 1.22 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=relocate|pyramid=single` |
| 5 | 256 | 21.3140 | 2.4571 | 0.88729 | 19.551 | - (0/4) | 1.26 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=ranked_wave|pyramid=single` |
| 6 | 256 | 21.3045 | 2.5310 | 0.88646 | 19.591 | - (0/4) | 1.20 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add_nms|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Iters→target | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|
| refine | fp_duplicate | 4 | 21.419 ± 2.586 | 0.88800 | 19.687 | - (0/4) | 1.21 |
| refine | none | 4 | 21.690 ± 2.509 | 0.89395 | 20.112 | - (0/4) | 1.13 |
| refine | ranked_wave | 4 | 21.314 ± 2.457 | 0.88729 | 19.551 | - (0/4) | 1.26 |
| refine | relocate | 4 | 21.325 ± 2.513 | 0.88912 | 19.776 | - (0/4) | 1.22 |
| refine | residual_add_nms | 4 | 21.305 ± 2.531 | 0.88646 | 19.591 | - (0/4) | 1.20 |
| refine | residual_tensor_add_nms | 4 | 21.382 ± 2.582 | 0.88821 | 19.647 | - (0/4) | 1.19 |
