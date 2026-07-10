# Stage influence (paired deltas vs baseline)

Baseline: `strategy=quadtree_wse|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|background=off|opacity=none|renderer=cuda|aa=0.0|color_basis=constant|color_solve=none|loss=l1|loss_weight=none|optimizer=adam|lr_schedule=none|refine_site=none|refine_primitive=duplicate|refine_nms=off|refine_color=target|refine_prune=off|refine_relocate=off|state_seed=off|row_temper=off|support_fade=off|pyramid=single`
Baseline means: PSNR 26.459, MS-SSIM 0.89621, AUC 26.724, fit 22.90s over 4 cells.

Positive ΔPSNR/ΔMS-SSIM/ΔAUC = variant better than baseline; negative Δiters/Δs = faster.
reach@target = target reached (variant/baseline/cells).

| Stage | Variant | Cells | ΔPSNR | ΔMS-SSIM | ΔAUC | Δiter@28 | reach@28 | Δiter@30 | reach@30 | Δiter@32 | reach@32 | Δinit s | Δfit s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| density | `density=variance` | 4 | +0.163 ± 0.473 | +0.005 ± 0.006 | -0.002 ± 0.151 | +0.0 ± 0.0 | 2/2/4 | -35.5 ± 23.5 | 2/2/4 | - | 0/0/4 | -0.058 ± 0.167 | -0.525 ± 0.399 |
| opacity | `opacity=constant` | 4 | +0.908 ± 0.659 | +0.015 ± 0.012 | +0.262 ± 0.117 | +0.0 ± 0.0 | 2/2/4 | -22.5 ± 12.5 | 2/2/4 | - | 0/0/4 | +0.346 ± 0.837 | +0.773 ± 0.899 |
| loss | `loss=charbonnier` | 4 | +0.104 ± 0.359 | +0.002 ± 0.003 | -0.051 ± 0.079 | +0.0 ± 0.0 | 2/2/4 | +10.5 ± 20.5 | 2/2/4 | - | 0/0/4 | +0.468 ± 0.728 | +0.160 ± 0.614 |
| lr_schedule | `lr_schedule=cosine` | 4 | +0.576 ± 0.583 | +0.009 ± 0.009 | +0.074 ± 0.105 | +0.0 ± 0.0 | 2/2/4 | +11.5 ± 19.5 | 2/2/4 | - | 0/0/4 | +0.910 ± 0.838 | +0.019 ± 0.488 |
| refine_site+refine_primitive | `refine_site=residual|refine_primitive=moment_preserving` | 4 | +0.297 ± 0.273 | +0.004 ± 0.004 | -0.044 ± 0.031 | +2.0 ± 0.0 | 2/2/4 | +66.5 ± 19.5 | 2/2/4 | - | 0/0/4 | +0.546 ± 0.499 | -0.507 ± 0.748 |
