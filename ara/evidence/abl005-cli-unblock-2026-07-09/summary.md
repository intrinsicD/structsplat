# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 18.9069 | 1.6458 | 0.84052 | 17.515 | 0/2 | - | 0/2 | - | 0/2 | - | 0.15 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=every10|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 64 | 18.0779 | 1.7147 | 0.81139 | 17.378 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 64 | 17.8740 | 1.6160 | 0.80550 | 17.240 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 64 | 17.8739 | 1.6148 | 0.80565 | 17.242 | 0/2 | - | 0/2 | - | 0/2 | - | 0.07 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=constant|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 5 | 64 | 17.8727 | 1.6143 | 0.80551 | 17.240 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 6 | 64 | 17.7469 | 2.0507 | 0.76728 | 17.214 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=cosine|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 7 | 64 | 17.6923 | 1.8871 | 0.78496 | 16.770 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 12 | 17.994 ± 1.795 | 0.80157 | 17.204 | 0/12 | - | 0/12 | - | 0/12 | - | 0.07 |
| density | variance | 2 | 18.078 ± 1.715 | 0.81139 | 17.378 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 |
| opacity | constant | 2 | 17.874 ± 1.615 | 0.80565 | 17.242 | 0/2 | - | 0/2 | - | 0/2 | - | 0.07 |
| opacity | none | 12 | 18.028 ± 1.810 | 0.80253 | 17.226 | 0/12 | - | 0/12 | - | 0/12 | - | 0.07 |
| color_solve | every10 | 2 | 18.907 ± 1.646 | 0.84052 | 17.515 | 0/2 | - | 0/2 | - | 0/2 | - | 0.15 |
| color_solve | none | 12 | 17.856 ± 1.762 | 0.79672 | 17.181 | 0/12 | - | 0/12 | - | 0/12 | - | 0.06 |
| loss | charbonnier | 2 | 17.873 ± 1.614 | 0.80551 | 17.240 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 |
| loss | l1 | 12 | 18.029 ± 1.810 | 0.80255 | 17.227 | 0/12 | - | 0/12 | - | 0/12 | - | 0.08 |
| lr_schedule | cosine | 2 | 17.747 ± 2.051 | 0.76728 | 17.214 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 |
| lr_schedule | none | 12 | 18.050 ± 1.732 | 0.80892 | 17.231 | 0/12 | - | 0/12 | - | 0/12 | - | 0.07 |
| refine_site | none | 12 | 18.059 ± 1.761 | 0.80598 | 17.305 | 0/12 | - | 0/12 | - | 0/12 | - | 0.07 |
| refine_site | residual | 2 | 17.692 ± 1.887 | 0.78496 | 16.770 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 |
| refine_primitive | duplicate | 12 | 18.059 ± 1.761 | 0.80598 | 17.305 | 0/12 | - | 0/12 | - | 0/12 | - | 0.07 |
| refine_primitive | moment_preserving | 2 | 17.692 ± 1.887 | 0.78496 | 16.770 | 0/2 | - | 0/2 | - | 0/2 | - | 0.06 |
