# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 80 | 23.8762 | 1.1364 | 0.88837 | 22.006 | 0/2 | - | 0/2 | - | 0/2 | - | 0.93 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=every10|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 2 | 80 | 23.5811 | 1.1154 | 0.89670 | 21.923 | 0/2 | - | 0/2 | - | 0/2 | - | 0.46 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=on_split|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 3 | 80 | 23.3878 | 1.1216 | 0.88980 | 21.699 | 0/2 | - | 0/2 | - | 0/2 | - | 0.35 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 4 | 80 | 23.0872 | 0.6936 | 0.87677 | 21.985 | 0/2 | - | 0/2 | - | 0/2 | - | 0.62 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=init+on_split|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 5 | 80 | 22.8631 | 0.7031 | 0.87061 | 21.804 | 0/2 | - | 0/2 | - | 0/2 | - | 0.46 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=density_random|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=init|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| color_solve | every10 | 2 | 23.876 ± 1.136 | 0.88837 | 22.006 | 0/2 | - | 0/2 | - | 0/2 | - | 0.93 |
| color_solve | init | 2 | 22.863 ± 0.703 | 0.87061 | 21.804 | 0/2 | - | 0/2 | - | 0/2 | - | 0.46 |
| color_solve | init+on_split | 2 | 23.087 ± 0.694 | 0.87677 | 21.985 | 0/2 | - | 0/2 | - | 0/2 | - | 0.62 |
| color_solve | none | 2 | 23.388 ± 1.122 | 0.88980 | 21.699 | 0/2 | - | 0/2 | - | 0/2 | - | 0.35 |
| color_solve | on_split | 2 | 23.581 ± 1.115 | 0.89670 | 21.923 | 0/2 | - | 0/2 | - | 0/2 | - | 0.46 |
