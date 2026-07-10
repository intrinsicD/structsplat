# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 27.3674 | 3.1537 | 0.91094 | 26.985 | 2/4 | 30 | 2/4 | 232 | 0/4 | - | 23.67 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=constant|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 2000 | 27.0349 | 3.2435 | 0.90527 | 26.797 | 2/4 | 30 | 2/4 | 266 | 0/4 | - | 22.92 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=cosine|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 2000 | 26.7565 | 3.5463 | 0.89987 | 26.679 | 2/4 | 32 | 2/4 | 321 | 0/4 | - | 22.39 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 2000 | 26.6225 | 3.5856 | 0.90125 | 26.722 | 2/4 | 30 | 2/4 | 219 | 0/4 | - | 22.37 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 5 | 2000 | 26.5630 | 3.5446 | 0.89777 | 26.672 | 2/4 | 30 | 2/4 | 265 | 0/4 | - | 23.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 6 | 2000 | 26.4592 | 3.8094 | 0.89621 | 26.724 | 2/4 | 30 | 2/4 | 254 | 0/4 | - | 22.90 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 20 | 26.836 ± 3.483 | 0.90201 | 26.772 | 10/20 | 30 | 10/20 | 268 | 0/20 | - | 22.99 |
| density | variance | 4 | 26.622 ± 3.586 | 0.90125 | 26.722 | 2/4 | 30 | 2/4 | 219 | 0/4 | - | 22.37 |
| opacity | constant | 4 | 27.367 ± 3.154 | 0.91094 | 26.985 | 2/4 | 30 | 2/4 | 232 | 0/4 | - | 23.67 |
| opacity | none | 20 | 26.687 ± 3.556 | 0.90007 | 26.719 | 10/20 | 30 | 10/20 | 265 | 0/20 | - | 22.73 |
| loss | charbonnier | 4 | 26.563 ± 3.545 | 0.89777 | 26.672 | 2/4 | 30 | 2/4 | 265 | 0/4 | - | 23.06 |
| loss | l1 | 20 | 26.848 ± 3.491 | 0.90271 | 26.782 | 10/20 | 30 | 10/20 | 258 | 0/20 | - | 22.85 |
| lr_schedule | cosine | 4 | 27.035 ± 3.243 | 0.90527 | 26.797 | 2/4 | 30 | 2/4 | 266 | 0/4 | - | 22.92 |
| lr_schedule | none | 20 | 26.754 ± 3.549 | 0.90121 | 26.757 | 10/20 | 30 | 10/20 | 258 | 0/20 | - | 22.88 |
| refine_site | none | 20 | 26.809 ± 3.492 | 0.90229 | 26.780 | 10/20 | 30 | 10/20 | 247 | 0/20 | - | 22.98 |
| refine_site | residual | 4 | 26.757 ± 3.546 | 0.89987 | 26.679 | 2/4 | 32 | 2/4 | 321 | 0/4 | - | 22.39 |
| refine_primitive | duplicate | 20 | 26.809 ± 3.492 | 0.90229 | 26.780 | 10/20 | 30 | 10/20 | 247 | 0/20 | - | 22.98 |
| refine_primitive | moment_preserving | 4 | 26.757 ± 3.546 | 0.89987 | 26.679 | 2/4 | 32 | 2/4 | 321 | 0/4 | - | 22.39 |
