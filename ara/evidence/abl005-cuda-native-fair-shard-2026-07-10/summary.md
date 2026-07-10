# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2000 | 24.2137 | 0.0318 | 0.88431 | 23.861 | 0/2 | - | 0/2 | - | 0/2 | - | 22.59 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=constant|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 2 | 2000 | 23.7937 | 0.1671 | 0.87613 | 23.644 | 0/2 | - | 0/2 | - | 0/2 | - | 22.04 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=cosine|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 3 | 2000 | 23.2107 | 0.0686 | 0.86577 | 23.447 | 0/2 | - | 0/2 | - | 0/2 | - | 21.26 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=residual|refine_primitive=moment_preserving|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 4 | 2000 | 23.0544 | 0.4095 | 0.86798 | 23.430 | 0/2 | - | 0/2 | - | 0/2 | - | 21.54 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 5 | 2000 | 23.0223 | 0.0205 | 0.86178 | 23.438 | 0/2 | - | 0/2 | - | 0/2 | - | 22.44 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |
| 6 | 2000 | 22.6504 | 0.0147 | 0.85791 | 23.485 | 0/2 | - | 0/2 | - | 0/2 | - | 22.27 | `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 10 | 23.378 ± 0.564 | 0.86918 | 23.575 | 0/10 | - | 0/10 | - | 0/10 | - | 22.12 |
| density | variance | 2 | 23.054 ± 0.409 | 0.86798 | 23.430 | 0/2 | - | 0/2 | - | 0/2 | - | 21.54 |
| opacity | constant | 2 | 24.214 ± 0.032 | 0.88431 | 23.861 | 0/2 | - | 0/2 | - | 0/2 | - | 22.59 |
| opacity | none | 10 | 23.146 ± 0.423 | 0.86591 | 23.489 | 0/10 | - | 0/10 | - | 0/10 | - | 21.91 |
| loss | charbonnier | 2 | 23.022 ± 0.020 | 0.86178 | 23.438 | 0/2 | - | 0/2 | - | 0/2 | - | 22.44 |
| loss | l1 | 10 | 23.385 ± 0.589 | 0.87042 | 23.573 | 0/10 | - | 0/10 | - | 0/10 | - | 21.94 |
| lr_schedule | cosine | 2 | 23.794 ± 0.167 | 0.87613 | 23.644 | 0/2 | - | 0/2 | - | 0/2 | - | 22.04 |
| lr_schedule | none | 10 | 23.230 ± 0.557 | 0.86755 | 23.532 | 0/10 | - | 0/10 | - | 0/10 | - | 22.02 |
| refine_site | none | 10 | 23.347 ± 0.604 | 0.86962 | 23.571 | 0/10 | - | 0/10 | - | 0/10 | - | 22.18 |
| refine_site | residual | 2 | 23.211 ± 0.069 | 0.86577 | 23.447 | 0/2 | - | 0/2 | - | 0/2 | - | 21.26 |
| refine_primitive | duplicate | 10 | 23.347 ± 0.604 | 0.86962 | 23.571 | 0/10 | - | 0/10 | - | 0/10 | - | 22.18 |
| refine_primitive | moment_preserving | 2 | 23.211 ± 0.069 | 0.86577 | 23.447 | 0/2 | - | 0/2 | - | 0/2 | - | 21.26 |
