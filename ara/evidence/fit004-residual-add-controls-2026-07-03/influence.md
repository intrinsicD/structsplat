# Stage influence (paired deltas vs baseline)

Baseline: `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
Baseline means: PSNR 21.690, MS-SSIM 0.89395, AUC 20.112, fit 1.46s over 4 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reached = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiters→target | reached | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| refine | `refine=residual_add` | 4 | -0.482 ± 0.203 | -0.010 ± 0.006 | -0.634 ± 0.203 | - | 0/0/4 | +0.015 ± 0.032 | +0.063 ± 0.106 |
| refine | `refine=residual_add_nms` | 4 | -0.385 ± 0.179 | -0.007 ± 0.004 | -0.521 ± 0.227 | - | 0/0/4 | -0.003 ± 0.006 | +1.693 ± 2.444 |
| refine | `refine=residual_add_nms_residual_color` | 4 | -0.486 ± 0.168 | -0.011 ± 0.005 | -0.891 ± 0.205 | - | 0/0/4 | -0.001 ± 0.004 | +1.217 ± 2.144 |
| refine | `refine=residual_add_residual_color` | 4 | -0.572 ± 0.183 | -0.014 ± 0.007 | -1.191 ± 0.226 | - | 0/0/4 | +0.005 ± 0.013 | +0.924 ± 1.515 |
| refine | `refine=residual_tensor_add` | 4 | -0.362 ± 0.205 | -0.008 ± 0.006 | -0.580 ± 0.199 | - | 0/0/4 | -0.001 ± 0.011 | -0.146 ± 0.207 |
| refine | `refine=residual_tensor_add_nms` | 4 | -0.308 ± 0.140 | -0.006 ± 0.004 | -0.465 ± 0.202 | - | 0/0/4 | -0.004 ± 0.008 | -0.294 ± 0.437 |
| refine | `refine=residual_tensor_add_nms_residual_color` | 4 | -0.411 ± 0.145 | -0.009 ± 0.006 | -0.808 ± 0.169 | - | 0/0/4 | -0.002 ± 0.007 | -0.324 ± 0.553 |
| pyramid | `pyramid=pyramid` | 4 | -1.402 ± 0.414 | -0.045 ± 0.015 | -2.042 ± 0.721 | - | 0/0/4 | +0.057 ± 0.006 | -0.050 ± 0.555 |
