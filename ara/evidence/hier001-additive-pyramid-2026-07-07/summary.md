# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 512 | 23.7756 | 1.6443 | 0.89513 | 22.445 | 0/4 | - | 0/4 | - | 0/4 | - | 0.58 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |
| 2 | 512 | 23.7735 | 1.6160 | 0.89348 | 22.927 | 0/4 | - | 0/4 | - | 0/4 | - | 0.70 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 512 | 23.5424 | 1.5340 | 0.88272 | 19.670 | 0/4 | - | 0/4 | - | 0/4 | - | 0.57 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda_additive|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 512 | 23.1681 | 1.4727 | 0.87405 | 17.817 | 0/4 | - | 0/4 | - | 0/4 | - | 0.60 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda_additive|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=sampled_add|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=pyramid` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| renderer | cuda | 8 | 23.775 ± 1.630 | 0.89431 | 22.686 | 0/8 | - | 0/8 | - | 0/8 | - | 0.64 |
| renderer | cuda_additive | 8 | 23.355 ± 1.515 | 0.87838 | 18.743 | 0/8 | - | 0/8 | - | 0/8 | - | 0.58 |
| pyramid | pyramid | 8 | 23.472 ± 1.590 | 0.88459 | 20.131 | 0/8 | - | 0/8 | - | 0/8 | - | 0.59 |
| pyramid | single | 8 | 23.658 ± 1.580 | 0.88810 | 21.298 | 0/8 | - | 0/8 | - | 0/8 | - | 0.64 |
