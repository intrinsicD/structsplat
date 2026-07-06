# MERGE-001 COCO/CUDA Confirmation

Protocol: 20 COCO val2017 images, budgets {512, 1024}, seeds {0,1,2}, 40 fit iterations, max-side 160, exact CUDA renderer.
Rows: 720 total, 720 ok, 0 errors.

## Configs

- **Merged shipped flanking** (`merged_shipped_flanking`): `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
- **Merged on-edge fast** (`merged_onedge_fast`): `strategy=aniso_onedge|tensor=central|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|loss=l1|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
- **Codex stage top1** (`codex_stage_top1`): `strategy=aniso_flanking|tensor=central|tensor_color=luma|density=variance|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=constant|renderer=cuda|aa=0.0|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=none|pyramid=single`
- **Codex stage top2** (`codex_stage_top2`): `strategy=aniso_onedge|tensor=scharr|tensor_color=luma|density=structure|sampling=wse|orientation=tensor|color=bilinear|scale=spacing|scale_cap=none|opacity=constant|renderer=cuda|aa=0.0|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=prune_residual_add|pyramid=single`
- **Codex stage top3** (`codex_stage_top3`): `strategy=aniso_onedge|tensor=scharr|tensor_color=rgb|density=variance|sampling=wse|orientation=tensor|color=local_mean|scale=spacing|scale_cap=none|opacity=none|renderer=cuda|aa=0.0|loss=l1|optimizer=adamw|lr_schedule=none|refine=residual_add|pyramid=single`
- **Merged best exact CUDA** (`merged_best_exact_cuda`): `strategy=aniso_flanking|tensor=scharr|tensor_color=rgb|density=structure|sampling=wse|orientation=tensor|color=two_sided|scale=spacing|scale_cap=feature12|opacity=none|renderer=cuda|aa=0.0|loss=charbonnier|optimizer=adam|lr_schedule=none|refine=residual_tensor_add|pyramid=single`

## Overall

| Rank | Config | Runs | PSNR | Std | MS-SSIM | AUC | Fit s | Total s | Target hits |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Codex stage top1 | 120 | 27.3443 | 3.1996 | 0.96439 | 25.073 | 0.0701 | 0.1883 | 18 |
| 2 | Merged on-edge fast | 120 | 27.2016 | 3.1794 | 0.96331 | 25.032 | 0.0667 | 0.1881 | 18 |
| 3 | Merged shipped flanking | 120 | 27.0827 | 3.1322 | 0.96287 | 24.920 | 0.0714 | 0.1959 | 18 |
| 4 | Merged best exact CUDA | 120 | 26.7006 | 3.0562 | 0.95988 | 24.352 | 0.0706 | 0.1837 | 18 |
| 5 | Codex stage top2 | 120 | 26.6157 | 3.0658 | 0.95840 | 24.524 | 0.0740 | 0.1847 | 15 |
| 6 | Codex stage top3 | 120 | 26.4007 | 3.0567 | 0.95658 | 24.147 | 0.0678 | 0.1769 | 14 |

## By Budget

| Budget | Rank | Config | Runs | PSNR | Std | MS-SSIM | AUC | Fit s | Total s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | Codex stage top1 | 60 | 26.2331 | 2.8325 | 0.95441 | 24.113 | 0.0700 | 0.1459 |
| 512 | 2 | Merged on-edge fast | 60 | 26.1362 | 2.8006 | 0.95347 | 24.094 | 0.0664 | 0.1425 |
| 512 | 3 | Merged shipped flanking | 60 | 26.0309 | 2.7736 | 0.95299 | 24.009 | 0.0756 | 0.1608 |
| 512 | 4 | Merged best exact CUDA | 60 | 25.5226 | 2.6709 | 0.94779 | 23.451 | 0.0709 | 0.1398 |
| 512 | 5 | Codex stage top2 | 60 | 25.4174 | 2.6532 | 0.94553 | 23.445 | 0.0750 | 0.1430 |
| 512 | 6 | Codex stage top3 | 60 | 25.2347 | 2.7105 | 0.94355 | 23.122 | 0.0682 | 0.1370 |
| 1024 | 1 | Codex stage top1 | 60 | 28.4556 | 3.1594 | 0.97437 | 26.033 | 0.0702 | 0.2308 |
| 1024 | 2 | Merged on-edge fast | 60 | 28.2671 | 3.1786 | 0.97315 | 25.970 | 0.0671 | 0.2336 |
| 1024 | 3 | Merged shipped flanking | 60 | 28.1346 | 3.1170 | 0.97275 | 25.831 | 0.0672 | 0.2310 |
| 1024 | 4 | Merged best exact CUDA | 60 | 27.8785 | 2.9617 | 0.97198 | 25.252 | 0.0702 | 0.2276 |
| 1024 | 5 | Codex stage top2 | 60 | 27.8140 | 2.9810 | 0.97127 | 25.602 | 0.0730 | 0.2264 |
| 1024 | 6 | Codex stage top3 | 60 | 27.5667 | 2.9362 | 0.96960 | 25.172 | 0.0674 | 0.2169 |

## Paired Deltas Vs Merged Shipped Flanking

Positive PSNR/MS-SSIM/AUC is better than the shipped flanking baseline; negative fit seconds is faster.

| Config | Pairs | ΔPSNR | PSNR wins | ΔMS-SSIM | MS wins | ΔAUC | AUC wins | Δfit s | Faster |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Codex stage top1 | 120 | +0.2616 ± 0.2703 | 108/120 | +0.00152 ± 0.00237 | 106/120 | +0.153 ± 0.155 | 100/120 | -0.0013 ± 0.0504 | 19/120 |
| Merged on-edge fast | 120 | +0.1189 ± 0.1551 | 100/120 | +0.00044 ± 0.00132 | 80/120 | +0.112 ± 0.099 | 109/120 | -0.0047 ± 0.0513 | 64/120 |
| Merged best exact CUDA | 120 | -0.3822 ± 0.5131 | 21/120 | -0.00299 ± 0.00400 | 23/120 | -0.568 ± 0.441 | 3/120 | -0.0008 ± 0.0457 | 22/120 |
| Codex stage top2 | 120 | -0.4671 ± 0.3424 | 0/120 | -0.00447 ± 0.00380 | 3/120 | -0.396 ± 0.251 | 0/120 | +0.0026 ± 0.0413 | 11/120 |
| Codex stage top3 | 120 | -0.6821 ± 0.3580 | 0/120 | -0.00629 ± 0.00503 | 3/120 | -0.772 ± 0.282 | 0/120 | -0.0036 ± 0.0507 | 50/120 |

## Paired PSNR By Budget

| Budget | Config | Pairs | ΔPSNR | PSNR wins |
|---:|---|---:|---:|---:|
| 512 | Codex stage top1 | 60 | +0.2022 ± 0.2273 | 49/60 |
| 512 | Merged on-edge fast | 60 | +0.1053 ± 0.1682 | 48/60 |
| 512 | Merged best exact CUDA | 60 | -0.5083 ± 0.5371 | 4/60 |
| 512 | Codex stage top2 | 60 | -0.6135 ± 0.3500 | 0/60 |
| 512 | Codex stage top3 | 60 | -0.7962 ± 0.3294 | 0/60 |
| 1024 | Codex stage top1 | 60 | +0.3210 ± 0.2957 | 59/60 |
| 1024 | Merged on-edge fast | 60 | +0.1325 ± 0.1395 | 52/60 |
| 1024 | Merged best exact CUDA | 60 | -0.2561 ± 0.4542 | 17/60 |
| 1024 | Codex stage top2 | 60 | -0.3206 ± 0.2629 | 0/60 |
| 1024 | Codex stage top3 | 60 | -0.5679 ± 0.3490 | 0/60 |
