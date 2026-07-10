# Stage influence (paired deltas vs baseline)

Baseline: `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single`
Baseline means: PSNR 22.665, MS-SSIM 0.85619, AUC 23.453, fit 21.81s over 1 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 1 | -0.020 ± 0.000 | +0.006 ± 0.000 | -0.218 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | +0.009 ± 0.000 | -0.777 ± 0.000 |
| opacity | `opacity=constant` | 1 | +1.517 ± 0.000 | +0.028 ± 0.000 | +0.378 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.017 ± 0.000 | +1.411 ± 0.000 |
| loss | `loss=charbonnier` | 1 | +0.337 ± 0.000 | +0.003 ± 0.000 | -0.091 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | -0.017 ± 0.000 | +1.021 ± 0.000 |
| lr_schedule | `lr_schedule=cosine` | 1 | +0.961 ± 0.000 | +0.016 ± 0.000 | +0.078 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | +1.511 ± 0.000 | +0.245 ± 0.000 |
| refine_site+refine_primitive | `refine_site=residual|refine_primitive=moment_preserving` | 1 | +0.477 ± 0.000 | +0.008 ± 0.000 | -0.002 ± 0.000 | - | 0/0/1 | - | 0/0/1 | - | 0/0/1 | +1.183 ± 0.000 | -0.238 ± 0.000 |
