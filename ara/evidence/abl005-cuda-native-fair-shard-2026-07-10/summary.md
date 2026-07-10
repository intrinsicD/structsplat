# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 26.3125 | 2.9682 | 0.90191 | 25.937 | 1/3 | 30 | 1/3 | 252 | 0/3 | - | 23.35 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=constant|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 2000 | 25.9386 | 3.0363 | 0.89540 | 25.737 | 1/3 | 30 | 1/3 | 279 | 0/3 | - | 22.74 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=cosine|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 2000 | 25.5631 | 3.3273 | 0.88855 | 25.592 | 1/3 | 32 | 1/3 | 373 | 0/3 | - | 22.06 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 2000 | 25.5297 | 3.5165 | 0.89104 | 25.645 | 1/3 | 30 | 1/3 | 228 | 0/3 | - | 22.27 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 5 | 2000 | 25.4615 | 3.4495 | 0.88635 | 25.610 | 1/3 | 30 | 1/3 | 277 | 0/3 | - | 22.91 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 6 | 2000 | 25.1567 | 3.5444 | 0.88316 | 25.626 | 1/3 | 30 | 1/3 | 287 | 0/3 | - | 22.70 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 15 | 25.686 ± 3.297 | 0.89108 | 25.700 | 5/15 | 30 | 5/15 | 294 | 0/15 | - | 22.75 |
| density | variance | 3 | 25.530 ± 3.517 | 0.89104 | 25.645 | 1/3 | 30 | 1/3 | 228 | 0/3 | - | 22.27 |
| opacity | constant | 3 | 26.312 ± 2.968 | 0.90191 | 25.937 | 1/3 | 30 | 1/3 | 252 | 0/3 | - | 23.35 |
| opacity | none | 15 | 25.530 ± 3.389 | 0.88890 | 25.642 | 5/15 | 30 | 5/15 | 289 | 0/15 | - | 22.54 |
| loss | charbonnier | 3 | 25.462 ± 3.450 | 0.88635 | 25.610 | 1/3 | 30 | 1/3 | 277 | 0/3 | - | 22.91 |
| loss | l1 | 15 | 25.700 ± 3.311 | 0.89201 | 25.707 | 5/15 | 30 | 5/15 | 284 | 0/15 | - | 22.62 |
| lr_schedule | cosine | 3 | 25.939 ± 3.036 | 0.89540 | 25.737 | 1/3 | 30 | 1/3 | 279 | 0/3 | - | 22.74 |
| lr_schedule | none | 15 | 25.605 ± 3.389 | 0.89020 | 25.682 | 5/15 | 30 | 5/15 | 283 | 0/15 | - | 22.66 |
| refine_site | none | 15 | 25.680 ± 3.337 | 0.89157 | 25.711 | 5/15 | 30 | 5/15 | 265 | 0/15 | - | 22.79 |
| refine_site | residual | 3 | 25.563 ± 3.327 | 0.88855 | 25.592 | 1/3 | 32 | 1/3 | 373 | 0/3 | - | 22.06 |
| refine_primitive | duplicate | 15 | 25.680 ± 3.337 | 0.89157 | 25.711 | 5/15 | 30 | 5/15 | 265 | 0/15 | - | 22.79 |
| refine_primitive | moment_preserving | 3 | 25.563 ± 3.327 | 0.88855 | 25.592 | 1/3 | 32 | 1/3 | 373 | 0/3 | - | 22.06 |
