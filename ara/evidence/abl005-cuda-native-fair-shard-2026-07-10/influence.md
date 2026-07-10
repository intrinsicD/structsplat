# Stage influence (paired deltas vs baseline)

Baseline: `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single`
Baseline means: PSNR 22.650, MS-SSIM 0.85791, AUC 23.485, fit 22.27s over 2 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 2 | +0.404 ± 0.424 | +0.010 ± 0.004 | -0.055 ± 0.163 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | +0.024 ± 0.015 | -0.728 ± 0.049 |
| opacity | `opacity=constant` | 2 | +1.563 ± 0.046 | +0.026 ± 0.001 | +0.376 ± 0.002 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.004 ± 0.014 | +0.319 ± 1.092 |
| loss | `loss=charbonnier` | 2 | +0.372 ± 0.035 | +0.004 ± 0.001 | -0.047 ± 0.044 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.001 ± 0.015 | +0.166 ± 0.854 |
| lr_schedule | `lr_schedule=cosine` | 2 | +1.143 ± 0.182 | +0.018 ± 0.002 | +0.160 ± 0.082 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | +0.772 ± 0.739 | -0.237 ± 0.482 |
| refine_site+refine_primitive | `refine_site=residual|refine_primitive=moment_preserving` | 2 | +0.560 ± 0.083 | +0.008 ± 0.000 | -0.038 ± 0.036 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | +0.578 ± 0.605 | -1.011 ± 0.773 |
