# Stage influence (paired deltas vs baseline)

Baseline: `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=normalized|aa=0.0|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
Baseline means: PSNR 21.690, MS-SSIM 0.89395, AUC 20.112, fit 1.07s over 4 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reached = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiters→target | reached | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aa | `aa=0.3` | 4 | -0.110 ± 0.048 | -0.002 ± 0.001 | -0.076 ± 0.034 | - | 0/0/4 | +0.014 ± 0.029 | +0.009 ± 0.029 |
| optimizer | `optimizer=adan` | 4 | -0.614 ± 0.121 | -0.017 ± 0.005 | -0.292 ± 0.075 | - | 0/0/4 | -0.001 ± 0.005 | +0.059 ± 0.033 |
| refine | `refine=absgrad_wave` | 4 | -0.328 ± 0.210 | -0.007 ± 0.004 | -0.471 ± 0.263 | - | 0/0/4 | -0.005 ± 0.001 | +0.089 ± 0.034 |
