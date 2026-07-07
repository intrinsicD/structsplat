# Stage influence (paired deltas vs baseline)

Baseline: `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
Baseline means: PSNR 21.686, MS-SSIM 0.80897, AUC 20.840, fit 0.59s over 1 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 1 | -0.179 ± 0.000 | -0.022 ± 0.000 | -0.182 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.015 ± 0.000 | -0.564 ± 0.000 |
| opacity | `opacity=constant` | 1 | +0.002 ± 0.000 | +0.000 ± 0.000 | +0.001 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.016 ± 0.000 | -0.556 ± 0.000 |
| color_basis | `color_basis=affine` | 1 | +0.444 ± 0.000 | +0.025 ± 0.000 | +0.431 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.017 ± 0.000 | -0.421 ± 0.000 |
| color_solve | `color_solve=every10` | 1 | +0.707 ± 0.000 | +0.018 ± 0.000 | +0.253 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.016 ± 0.000 | -0.503 ± 0.000 |
| loss | `loss=charbonnier` | 1 | +0.002 ± 0.000 | +0.000 ± 0.000 | +0.003 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.016 ± 0.000 | -0.566 ± 0.000 |
| lr_schedule | `lr_schedule=cosine` | 1 | -0.548 ± 0.000 | -0.052 ± 0.000 | -0.098 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.017 ± 0.000 | -0.567 ± 0.000 |
| refine | `refine=moment_preserving` | 1 | -0.830 ± 0.000 | -0.059 ± 0.000 | -0.973 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.018 ± 0.000 | -0.552 ± 0.000 |
