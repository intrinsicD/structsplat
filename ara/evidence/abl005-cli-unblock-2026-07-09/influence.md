# Stage influence (paired deltas vs baseline)

Baseline: `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=normalized|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single`
Baseline means: PSNR 17.874, MS-SSIM 0.80550, AUC 17.240, fit 0.06s over 2 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 2 | +0.204 ± 0.099 | +0.006 ± 0.003 | +0.137 ± 0.098 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.000 ± 0.001 | +0.004 ± 0.005 |
| opacity | `opacity=constant` | 2 | -0.000 ± 0.001 | +0.000 ± 0.000 | +0.002 ± 0.000 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | +0.002 ± 0.002 | +0.012 ± 0.005 |
| color_solve | `color_solve=every10` | 2 | +1.033 ± 0.030 | +0.035 ± 0.008 | +0.275 ± 0.001 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | +0.002 ± 0.002 | +0.090 ± 0.008 |
| loss | `loss=charbonnier` | 2 | -0.001 ± 0.002 | +0.000 ± 0.000 | -0.000 ± 0.001 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.001 ± 0.002 | -0.001 ± 0.002 |
| lr_schedule | `lr_schedule=cosine` | 2 | -0.127 ± 0.435 | -0.038 ± 0.023 | -0.026 ± 0.108 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.002 ± 0.002 | +0.001 ± 0.002 |
| refine_site+refine_primitive | `refine_site=residual|refine_primitive=moment_preserving` | 2 | -0.182 ± 0.271 | -0.021 ± 0.012 | -0.470 ± 0.178 | - | 0/0/2 | - | 0/0/2 | - | 0/0/2 | -0.004 ± 0.002 | +0.003 ± 0.001 |
