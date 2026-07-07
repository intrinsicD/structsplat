# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 24.3194 | 2.7624 | 0.88102 | 24.489 | 1/4 | 116 | 0/4 | - | 0/4 | - | 4.77 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 2 | 2000 | 24.2549 | 2.5845 | 0.88177 | 24.472 | 1/4 | 116 | 0/4 | - | 0/4 | - | 4.88 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 3 | 2000 | 24.2213 | 2.6787 | 0.87992 | 24.592 | 1/4 | 117 | 0/4 | - | 0/4 | - | 4.74 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual_tensor|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |
| 4 | 2000 | 24.0325 | 2.8916 | 0.87685 | 24.454 | 1/4 | 116 | 0/4 | - | 0/4 | - | 4.80 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine_site=residual_tensor|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| refine_site | residual | 8 | 24.287 ± 2.675 | 0.88140 | 24.481 | 2/8 | 116 | 0/8 | - | 0/8 | - | 4.83 |
| refine_site | residual_tensor | 8 | 24.127 ± 2.789 | 0.87838 | 24.523 | 2/8 | 116 | 0/8 | - | 0/8 | - | 4.77 |
| refine_primitive | moment_preserving | 8 | 24.176 ± 2.831 | 0.87894 | 24.472 | 2/8 | 116 | 0/8 | - | 0/8 | - | 4.79 |
| refine_primitive | sampled_add | 8 | 24.238 ± 2.632 | 0.88084 | 24.532 | 2/8 | 116 | 0/8 | - | 0/8 | - | 4.81 |
