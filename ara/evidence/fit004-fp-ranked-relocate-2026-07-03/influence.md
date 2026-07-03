# Stage influence (paired deltas vs baseline)

Baseline: `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
Baseline means: PSNR 21.690, MS-SSIM 0.89395, AUC 20.112, fit 1.13s over 4 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reached = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiters→target | reached | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| refine | `refine=fp_duplicate` | 4 | -0.271 ± 0.302 | -0.006 ± 0.005 | -0.425 ± 0.255 | - | 0/0/4 | +0.017 ± 0.038 | +0.076 ± 0.064 |
| refine | `refine=ranked_wave` | 4 | -0.376 ± 0.174 | -0.007 ± 0.003 | -0.561 ± 0.221 | - | 0/0/4 | -0.004 ± 0.001 | +0.125 ± 0.050 |
| refine | `refine=relocate` | 4 | -0.365 ± 0.140 | -0.005 ± 0.002 | -0.336 ± 0.086 | - | 0/0/4 | +0.001 ± 0.005 | +0.090 ± 0.016 |
| refine | `refine=residual_add_nms` | 4 | -0.385 ± 0.179 | -0.007 ± 0.004 | -0.521 ± 0.227 | - | 0/0/4 | -0.003 ± 0.004 | +0.065 ± 0.051 |
| refine | `refine=residual_tensor_add_nms` | 4 | -0.308 ± 0.140 | -0.006 ± 0.004 | -0.465 ± 0.202 | - | 0/0/4 | -0.002 ± 0.001 | +0.058 ± 0.056 |
