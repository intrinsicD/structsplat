# Stage influence (paired deltas vs baseline)

Baseline: `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single`
Baseline means: PSNR 25.157, MS-SSIM 0.88316, AUC 25.626, fit 22.70s over 3 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 3 | +0.373 ± 0.349 | +0.008 ± 0.004 | +0.019 ± 0.169 | +0.0 ± 0.0 | 1/1/3 | -59.0 ± 0.0 | 1/1/3 | - | 0/0/3 | -0.100 ± 0.175 | -0.431 ± 0.422 |
| opacity | `opacity=constant` | 3 | +1.156 ± 0.578 | +0.019 ± 0.011 | +0.311 ± 0.092 | +0.0 ± 0.0 | 1/1/3 | -35.0 ± 0.0 | 1/1/3 | - | 0/0/3 | -0.129 ± 0.178 | +0.654 ± 1.010 |
| loss | `loss=charbonnier` | 3 | +0.305 ± 0.099 | +0.003 ± 0.001 | -0.016 ± 0.057 | +0.0 ± 0.0 | 1/1/3 | -10.0 ± 0.0 | 1/1/3 | - | 0/0/3 | +0.574 ± 0.814 | +0.214 ± 0.701 |
| lr_schedule | `lr_schedule=cosine` | 3 | +0.782 ± 0.532 | +0.012 ± 0.009 | +0.111 ± 0.096 | +0.0 ± 0.0 | 1/1/3 | -8.0 ± 0.0 | 1/1/3 | - | 0/0/3 | +1.165 ± 0.821 | +0.046 ± 0.561 |
| refine_site+refine_primitive | `refine_site=residual|refine_primitive=moment_preserving` | 3 | +0.406 ± 0.228 | +0.005 ± 0.004 | -0.034 ± 0.030 | +2.0 ± 0.0 | 1/1/3 | +86.0 ± 0.0 | 1/1/3 | - | 0/0/3 | +0.676 ± 0.513 | -0.636 ± 0.825 |
