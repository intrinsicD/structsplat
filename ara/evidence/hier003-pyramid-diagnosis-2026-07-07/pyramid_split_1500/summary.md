# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5000 | 28.6308 | 4.5793 | 0.94784 | 26.055 | 0/4 | - | 0/4 | - | 0/4 | - | 5.59 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |
| 2 | 5000 | 28.5680 | 4.3812 | 0.94859 | 26.109 | 0/4 | - | 0/4 | - | 0/4 | - | 5.47 | `strategy=aniso_onedge|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |
| 3 | 2000 | 25.5008 | 3.5560 | 0.89826 | 23.638 | 0/4 | - | 0/4 | - | 0/4 | - | 5.21 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |
| 4 | 2000 | 25.0733 | 2.9854 | 0.89853 | 23.615 | 0/4 | - | 0/4 | - | 0/4 | - | 5.29 | `strategy=aniso_onedge|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strategy | aniso_onedge | 8 | 26.821 ± 4.136 | 0.92356 | 24.862 | 0/8 | - | 0/8 | - | 0/8 | - | 5.38 |
| strategy | quadtree_wse | 8 | 27.066 ± 4.388 | 0.92305 | 24.846 | 0/8 | - | 0/8 | - | 0/8 | - | 5.40 |
