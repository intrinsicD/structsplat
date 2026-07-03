# StructSplat Stage Search

| Rank | Budget | Mean PSNR | Std | Mean MS-SSIM | Mean AUC | Iters→target | Mean fit s | Config |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 256 | 21.6899 | 2.5091 | 0.89395 | 20.112 | - (0/4) | 1.46 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single` |
| 2 | 256 | 21.3822 | 2.5822 | 0.88821 | 19.647 | - (0/4) | 1.17 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_tensor_add_nms|pyramid=single` |
| 3 | 256 | 21.3282 | 2.6145 | 0.88597 | 19.532 | - (0/4) | 1.32 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_tensor_add|pyramid=single` |
| 4 | 256 | 21.3046 | 2.5310 | 0.88646 | 19.591 | - (0/4) | 3.15 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add_nms|pyramid=single` |
| 5 | 256 | 21.2792 | 2.5975 | 0.88464 | 19.304 | - (0/4) | 1.14 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_tensor_add_nms_residual_color|pyramid=single` |
| 6 | 256 | 21.2078 | 2.5522 | 0.88346 | 19.478 | - (0/4) | 1.52 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add|pyramid=single` |
| 7 | 256 | 21.2036 | 2.5348 | 0.88295 | 19.221 | - (0/4) | 2.68 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add_nms_residual_color|pyramid=single` |
| 8 | 256 | 21.1183 | 2.5664 | 0.88025 | 18.921 | - (0/4) | 2.39 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=residual_add_residual_color|pyramid=single` |
| 9 | 256 | 20.2880 | 2.2949 | 0.84928 | 18.070 | - (0/4) | 1.41 | `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=pyramid` |

## Per-stage marginal means

| Stage | Level | Runs | PSNR | MS-SSIM | AUC | Iters→target | Fit s |
|---|---|---:|---:|---:|---:|---:|---:|
| refine | none | 8 | 20.989 ± 2.504 | 0.87162 | 19.091 | - (0/8) | 1.44 |
| refine | residual_add | 4 | 21.208 ± 2.552 | 0.88346 | 19.478 | - (0/4) | 1.52 |
| refine | residual_add_nms | 4 | 21.305 ± 2.531 | 0.88646 | 19.591 | - (0/4) | 3.15 |
| refine | residual_add_nms_residual_color | 4 | 21.204 ± 2.535 | 0.88295 | 19.221 | - (0/4) | 2.68 |
| refine | residual_add_residual_color | 4 | 21.118 ± 2.566 | 0.88025 | 18.921 | - (0/4) | 2.39 |
| refine | residual_tensor_add | 4 | 21.328 ± 2.615 | 0.88597 | 19.532 | - (0/4) | 1.32 |
| refine | residual_tensor_add_nms | 4 | 21.382 ± 2.582 | 0.88821 | 19.647 | - (0/4) | 1.17 |
| refine | residual_tensor_add_nms_residual_color | 4 | 21.279 ± 2.597 | 0.88464 | 19.304 | - (0/4) | 1.14 |
| pyramid | pyramid | 4 | 20.288 ± 2.295 | 0.84928 | 18.070 | - (0/4) | 1.41 |
| pyramid | single | 32 | 21.314 ± 2.566 | 0.88574 | 19.476 | - (0/32) | 1.85 |
