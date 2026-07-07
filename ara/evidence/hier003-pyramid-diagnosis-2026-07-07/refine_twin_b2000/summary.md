# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 23.4333 | 2.5418 | 0.84323 | 22.814 | 0/4 | - | 0/4 | - | 0/4 | - | 5.00 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 2000 | 23.2249 | 1.6648 | 0.85081 | 22.711 | 0/4 | - | 0/4 | - | 0/4 | - | 5.15 | `strategy=aniso_onedge|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strategy | aniso_onedge | 4 | 23.225 ± 1.665 | 0.85081 | 22.711 | 0/4 | - | 0/4 | - | 0/4 | - | 5.15 |
| strategy | quadtree_wse | 4 | 23.433 ± 2.542 | 0.84323 | 22.814 | 0/4 | - | 0/4 | - | 0/4 | - | 5.00 |
