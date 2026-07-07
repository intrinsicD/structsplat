# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 80 | 23.8668 | 1.1479 | 0.89420 | 22.191 | 0/2 | - | 0/2 | - | 0/2 | - | 0.31 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=on|pyramid=single` |
| 2 | 80 | 23.8530 | 1.1200 | 0.89374 | 22.181 | 0/2 | - | 0/2 | - | 0/2 | - | 0.30 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=until0.5|pyramid=single` |
| 3 | 80 | 23.8361 | 1.0848 | 0.89505 | 22.193 | 0/2 | - | 0/2 | - | 0/2 | - | 0.30 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| support_fade | off | 2 | 23.836 ± 1.085 | 0.89505 | 22.193 | 0/2 | - | 0/2 | - | 0/2 | - | 0.30 |
| support_fade | on | 2 | 23.867 ± 1.148 | 0.89420 | 22.191 | 0/2 | - | 0/2 | - | 0/2 | - | 0.31 |
| support_fade | until0.5 | 2 | 23.853 ± 1.120 | 0.89374 | 22.181 | 0/2 | - | 0/2 | - | 0/2 | - | 0.30 |
