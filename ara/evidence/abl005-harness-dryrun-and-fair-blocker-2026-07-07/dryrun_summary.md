# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 22.3932 | 0.0000 | 0.82651 | 21.092 | 0/1 | - | 0/1 | - | 0/1 | - | 0.08 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=every10|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 64 | 22.1302 | 0.0000 | 0.83425 | 21.271 | 0/1 | - | 0/1 | - | 0/1 | - | 0.17 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=affine|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 3 | 64 | 21.6885 | 0.0000 | 0.80924 | 20.843 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 4 | 64 | 21.6883 | 0.0000 | 0.80914 | 20.841 | 0/1 | - | 0/1 | - | 0/1 | - | 0.03 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=constant|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 5 | 64 | 21.6863 | 0.0000 | 0.80897 | 20.840 | 0/1 | - | 0/1 | - | 0/1 | - | 0.59 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 6 | 64 | 21.5075 | 0.0000 | 0.78677 | 20.657 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 7 | 64 | 21.1380 | 0.0000 | 0.75744 | 20.742 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=cosine|refine=none|pyramid=single` |
| 8 | 64 | 20.8567 | 0.0000 | 0.74984 | 19.867 | 0/1 | - | 0/1 | - | 0/1 | - | 0.03 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=moment_preserving|pyramid=single` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | structure | 7 | 21.654 ± 0.490 | 0.79934 | 20.785 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
| density | variance | 1 | 21.508 ± 0.000 | 0.78677 | 20.657 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 |
| opacity | constant | 1 | 21.688 ± 0.000 | 0.80914 | 20.841 | 0/1 | - | 0/1 | - | 0/1 | - | 0.03 |
| opacity | none | 7 | 21.629 ± 0.492 | 0.79615 | 20.759 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
| color_basis | affine | 1 | 22.130 ± 0.000 | 0.83425 | 21.271 | 0/1 | - | 0/1 | - | 0/1 | - | 0.17 |
| color_basis | constant | 7 | 21.566 ± 0.450 | 0.79256 | 20.697 | 0/7 | - | 0/7 | - | 0/7 | - | 0.11 |
| color_solve | every10 | 1 | 22.393 ± 0.000 | 0.82651 | 21.092 | 0/1 | - | 0/1 | - | 0/1 | - | 0.08 |
| color_solve | none | 7 | 21.528 ± 0.386 | 0.79366 | 20.723 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
| loss | charbonnier | 1 | 21.689 ± 0.000 | 0.80924 | 20.843 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 |
| loss | l1 | 7 | 21.629 ± 0.492 | 0.79613 | 20.758 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
| lr_schedule | cosine | 1 | 21.138 ± 0.000 | 0.75744 | 20.742 | 0/1 | - | 0/1 | - | 0/1 | - | 0.02 |
| lr_schedule | none | 7 | 21.707 ± 0.449 | 0.80353 | 20.773 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
| refine | moment_preserving | 1 | 20.857 ± 0.000 | 0.74984 | 19.867 | 0/1 | - | 0/1 | - | 0/1 | - | 0.03 |
| refine | none | 7 | 21.747 ± 0.379 | 0.80462 | 20.898 | 0/7 | - | 0/7 | - | 0/7 | - | 0.13 |
