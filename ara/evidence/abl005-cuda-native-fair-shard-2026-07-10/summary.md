# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 24.1820 | 0.0000 | 0.88375 | 23.832 | 0/1 | - | 0/1 | - | 0/1 | - | 23.22 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=constant|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 2000 | 23.6266 | 0.0000 | 0.87263 | 23.531 | 0/1 | - | 0/1 | - | 0/1 | - | 22.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=cosine|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 2000 | 23.1421 | 0.0000 | 0.86431 | 23.451 | 0/1 | - | 0/1 | - | 0/1 | - | 21.57 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 2000 | 23.0019 | 0.0000 | 0.85897 | 23.362 | 0/1 | - | 0/1 | - | 0/1 | - | 22.83 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 5 | 2000 | 22.6652 | 0.0000 | 0.85619 | 23.453 | 0/1 | - | 0/1 | - | 0/1 | - | 21.81 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 6 | 2000 | 22.6449 | 0.0000 | 0.86259 | 23.235 | 0/1 | - | 0/1 | - | 0/1 | - | 21.03 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 5 | 23.324 ± 0.529 | 0.86717 | 23.526 | 0/5 | - | 0/5 | - | 0/5 | - | 22.30 |
| density | variance | 1 | 22.645 ± 0.000 | 0.86259 | 23.235 | 0/1 | - | 0/1 | - | 0/1 | - | 21.03 |
| opacity | constant | 1 | 24.182 ± 0.000 | 0.88375 | 23.832 | 0/1 | - | 0/1 | - | 0/1 | - | 23.22 |
| opacity | none | 5 | 23.016 ± 0.360 | 0.86294 | 23.407 | 0/5 | - | 0/5 | - | 0/5 | - | 21.86 |
| loss | charbonnier | 1 | 23.002 ± 0.000 | 0.85897 | 23.362 | 0/1 | - | 0/1 | - | 0/1 | - | 22.83 |
| loss | l1 | 5 | 23.252 ± 0.588 | 0.86789 | 23.500 | 0/5 | - | 0/5 | - | 0/5 | - | 21.94 |
| lr_schedule | cosine | 1 | 23.627 ± 0.000 | 0.87263 | 23.531 | 0/1 | - | 0/1 | - | 0/1 | - | 22.06 |
| lr_schedule | none | 5 | 23.127 ± 0.561 | 0.86516 | 23.467 | 0/5 | - | 0/5 | - | 0/5 | - | 22.09 |
| refine_site | none | 5 | 23.224 ± 0.596 | 0.86683 | 23.483 | 0/5 | - | 0/5 | - | 0/5 | - | 22.19 |
| refine_site | residual | 1 | 23.142 ± 0.000 | 0.86431 | 23.451 | 0/1 | - | 0/1 | - | 0/1 | - | 21.57 |
| refine_primitive | duplicate | 5 | 23.224 ± 0.596 | 0.86683 | 23.483 | 0/5 | - | 0/5 | - | 0/5 | - | 22.19 |
| refine_primitive | moment_preserving | 1 | 23.142 ± 0.000 | 0.86431 | 23.451 | 0/1 | - | 0/1 | - | 0/1 | - | 21.57 |
