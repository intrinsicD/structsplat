# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS best default | best-default | Pinned current Gaussian-image winner: aniso_onedge + WSE, feature cap 12@160, tensor-aware residual growth, 5 growth waves, L1 + 0.3 SSIM. |
| SS best + SSIM 0.10 | best-loss-sweep | Best default geometry/growth with SSIM weight lowered to 0.10 for lower pixel diff. |
| SS best + L1 only | best-loss-sweep | Best default geometry/growth with pure L1 pixel loss and no SSIM term. |
| SS best + Charbonnier | best-loss-sweep | Best default geometry/growth with Charbonnier pixel loss and SSIM weight 0.10. |
| SS best + tensor loss | best-edge-loss | Best default geometry/growth with tensor-weighted pixel loss to emphasize edges. |
| SS best + final color solve | best-color-polish | Best default geometry/growth with a final fixed-geometry RGB least-squares color solve. |
| SS best + split relocate | best-relocate | Best default geometry/growth plus split-scheduled residual relocation. |
| SS best + adaptive 1.5x cap | best-adaptive-capacity | Best default warm start with adaptive residual growth allowed up to 1.5x the requested cap. |
| GaussianImage fixed | fixed-full | GaussianImage-style random fixed-count control; starts at the final cap and does not grow. |
| GaussianImage++ residual | repo-growth | GaussianImage++-style analogue: random half-budget start plus residual-add growth. |
| Image-GS residual | repo-growth | Image-GS-style analogue: gradient-density random half-budget start plus residual-add growth. |
| Instant-GI quadtree | fixed-full | Instant-GI quadtree/Delaunay fallback if STRUCTSPLAT_INSTANT_GI is configured; fixed count. |
| SS on-edge + residual | same-growth | StructSplat on-edge initializer under the same residual-add growth as external analogues. |
| SS on-edge + residual relocate | same-growth+relocate | StructSplat on-edge residual-add growth plus split-scheduled residual relocation. |
| SS on-edge + residual feature cap | same-growth+feature-cap | StructSplat on-edge residual-add growth with feature-adaptive per-Gaussian scale caps. |
| SS on-edge + residual feature-rel cap | same-growth+feature-rel-cap | StructSplat on-edge residual-add growth with feature-relative local-radius scale caps. |
| SS on-edge + tensor | tensor-growth | StructSplat on-edge initializer plus tensor-aware residual growth. |
| SS on-edge + tensor feature cap | tensor-growth+feature-cap | StructSplat on-edge tensor-aware residual growth with feature-adaptive scale caps. |
| SS on-edge + tensor feature-rel cap | tensor-growth+feature-rel-cap | StructSplat on-edge tensor-aware residual growth with feature-relative local-radius caps. |
| SS flanking + tensor | tensor-growth | StructSplat flanking initializer plus tensor-aware residual growth. |
| SS qt-WSE + residual | same-growth | StructSplat quadtree-WSE initializer under the same residual-add growth as external analogues. |
| SS qt-WSE + residual relocate | same-growth+relocate | StructSplat quadtree-WSE residual-add growth plus split-scheduled residual relocation. |
| SS qt-WSE + residual feature cap | same-growth+feature-cap | StructSplat quadtree-WSE residual-add growth with feature-adaptive scale caps. |
| SS qt-WSE + residual feature-rel cap | same-growth+feature-rel-cap | StructSplat quadtree-WSE residual-add growth with feature-relative local-radius caps. |
| SS qt-WSE + tensor | tensor-growth | StructSplat quadtree-WSE initializer plus tensor-aware residual growth. |
| SS qt-WSE + tensor feature cap | tensor-growth+feature-cap | StructSplat quadtree-WSE tensor-aware residual growth with feature-adaptive scale caps. |
| SS qt-WSE + tensor feature-rel cap | tensor-growth+feature-rel-cap | StructSplat quadtree-WSE tensor-aware residual growth with feature-relative local-radius caps. |
| SS qt-hybrid + tensor | tensor-growth | StructSplat quadtree-hybrid initializer plus tensor-aware residual growth. |
| Floyd + tensor | tensor-growth-control | Floyd-Steinberg placement control plus tensor-aware residual growth. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS best default | 8 | 27.7524 | 3.7639 | 0.97092 | 0.01775 | 24.850 | - | 0.053 | 1.218 | 1.271 |
| SS best + SSIM 0.10 | 8 | 27.6170 | 3.6756 | 0.96918 | 0.01872 | 24.814 | - | 0.052 | 1.108 | 1.159 |
| SS best + L1 only | 8 | 26.9764 | 3.6336 | 0.95986 | 0.02374 | 24.360 | - | 0.052 | 0.787 | 0.839 |
| SS best + Charbonnier | 8 | 27.5749 | 3.7005 | 0.96821 | 0.02002 | 24.823 | - | 0.052 | 1.116 | 1.168 |
| SS best + tensor loss | 8 | 27.7643 | 3.8187 | 0.97069 | 0.01888 | 24.863 | - | 0.052 | 1.151 | 1.203 |
| SS best + final color solve | 8 | 27.9459 | 3.6969 | 0.97253 | 0.01596 | 24.833 | - | 0.053 | 1.231 | 1.284 |
| SS best + split relocate | 8 | 27.2648 | 3.7256 | 0.96736 | 0.01935 | 24.639 | - | 0.052 | 1.199 | 1.251 |
| SS best + adaptive 1.5x cap † | 8 | 28.7802 | 3.1722 | 0.97847 | 0.01363 | 25.540 | - | 0.052 | 1.120 | 1.172 |
| GaussianImage fixed | 8 | 25.7916 | 3.2335 | 0.97255 | 0.01547 | 24.426 | - | 0.001 | 1.110 | 1.110 |
| GaussianImage++ residual | 8 | 27.3841 | 3.6179 | 0.97324 | 0.01660 | 22.992 | - | 0.000 | 1.107 | 1.108 |
| Image-GS residual | 8 | 27.4130 | 3.5892 | 0.97190 | 0.01862 | 23.425 | - | 0.003 | 1.093 | 1.096 |
| Instant-GI quadtree | 8 | 22.6801 | 4.0139 | 0.93493 | 0.04180 | 21.102 | - | 0.172 | 1.085 | 1.257 |
| SS on-edge + residual | 8 | 27.5694 | 3.5397 | 0.97247 | 0.01741 | 24.189 | - | 0.051 | 1.116 | 1.167 |
| SS on-edge + residual relocate | 8 | 27.3976 | 3.4991 | 0.97151 | 0.01777 | 23.888 | - | 0.051 | 1.167 | 1.218 |
| SS on-edge + residual feature cap | 8 | 27.6148 | 3.6547 | 0.97039 | 0.01946 | 24.053 | - | 0.051 | 1.115 | 1.166 |
| SS on-edge + residual feature-rel cap | 8 | 27.6212 | 3.6132 | 0.97123 | 0.01801 | 24.303 | - | 0.051 | 1.117 | 1.168 |
| SS on-edge + tensor | 8 | 27.3742 | 3.4446 | 0.97139 | 0.01812 | 24.180 | - | 0.051 | 1.106 | 1.157 |
| SS on-edge + tensor feature cap | 8 | 27.7242 | 3.8179 | 0.97052 | 0.01914 | 24.298 | - | 0.054 | 1.146 | 1.200 |
| SS on-edge + tensor feature-rel cap | 8 | 27.4450 | 3.5522 | 0.97104 | 0.01840 | 24.235 | - | 0.051 | 1.128 | 1.179 |
| SS flanking + tensor | 8 | 27.3706 | 3.4506 | 0.97158 | 0.01767 | 24.177 | - | 0.051 | 1.114 | 1.166 |
| SS qt-WSE + residual | 8 | 27.5364 | 3.5854 | 0.97178 | 0.01830 | 24.117 | - | 0.137 | 1.114 | 1.251 |
| SS qt-WSE + residual relocate | 8 | 27.4066 | 3.4985 | 0.97188 | 0.01793 | 23.887 | - | 0.136 | 1.173 | 1.309 |
| SS qt-WSE + residual feature cap | 8 | 27.4521 | 3.6391 | 0.96991 | 0.01985 | 23.976 | - | 0.137 | 1.112 | 1.249 |
| SS qt-WSE + residual feature-rel cap | 8 | 27.6942 | 3.7116 | 0.97194 | 0.01834 | 24.282 | - | 0.135 | 1.119 | 1.253 |
| SS qt-WSE + tensor | 8 | 27.4045 | 3.4146 | 0.97187 | 0.01721 | 24.154 | - | 0.136 | 1.122 | 1.258 |
| SS qt-WSE + tensor feature cap | 8 | 27.5586 | 3.7886 | 0.97028 | 0.01888 | 24.154 | - | 0.136 | 1.131 | 1.267 |
| SS qt-WSE + tensor feature-rel cap | 8 | 27.4130 | 3.4636 | 0.97150 | 0.01778 | 24.171 | - | 0.138 | 1.115 | 1.252 |
| SS qt-hybrid + tensor | 8 | 27.3779 | 3.4449 | 0.97196 | 0.01772 | 24.113 | - | 0.068 | 1.105 | 1.173 |
| Floyd + tensor | 8 | 26.9373 | 3.2126 | 0.96777 | 0.01965 | 23.577 | - | 0.022 | 1.108 | 1.130 |

† Not budget-matched: mean final Gaussian count exceeds the shared final cap (adaptive extra capacity). These rows spend more primitives — more rate for an image codec — so their PSNR/MS-SSIM is not directly comparable to the equal-budget rows, and they are excluded from the per-cell winners below.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + SSIM 0.10 | 8 | -0.1355 | -0.00174 | -0.0365 | -0.1104 | -0.1119 | 2/8 | 1/8 | 3/8 | 5/8 | no |
| SS best + L1 only | 8 | -0.7760 | -0.01106 | -0.4904 | -0.4311 | -0.4325 | 0/8 | 0/8 | 0/8 | 8/8 | no |
| SS best + Charbonnier | 8 | -0.1775 | -0.00271 | -0.0277 | -0.1016 | -0.1028 | 0/8 | 0/8 | 3/8 | 5/8 | no |
| SS best + tensor loss | 8 | +0.0119 | -0.00024 | +0.0121 | -0.0668 | -0.0679 | 6/8 | 6/8 | 6/8 | 4/8 | no |
| SS best + final color solve | 8 | +0.1935 | +0.00161 | -0.0179 | +0.0132 | +0.0131 | 7/8 | 5/8 | 4/8 | 2/8 | no |
| SS best + split relocate | 8 | -0.4876 | -0.00356 | -0.2113 | -0.0193 | -0.0199 | 0/8 | 0/8 | 0/8 | 3/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.850 | 17.090 | 23.897 | 23.259 | 26.809 | 27.752 |
| SS best + SSIM 0.10 | 24.814 | 17.090 | 23.861 | 23.223 | 26.715 | 27.617 |
| SS best + L1 only | 24.360 | 17.090 | 23.442 | 23.127 | 26.115 | 26.976 |
| SS best + Charbonnier | 24.823 | 17.090 | 23.878 | 23.213 | 26.691 | 27.575 |
| SS best + tensor loss | 24.863 | 17.090 | 23.912 | 23.270 | 26.842 | 27.764 |
| SS best + final color solve | 24.833 | 17.090 | 23.899 | 23.179 | 26.780 | 27.946 |
| SS best + split relocate | 24.639 | 17.090 | 23.717 | 22.780 | 26.512 | 27.265 |
| SS best + adaptive 1.5x cap † | 25.540 | 17.090 | 24.244 | 23.822 | 28.569 | 28.780 |
| GaussianImage fixed | 24.426 | 16.921 | 23.986 | 25.399 | 25.762 | 25.792 |
| GaussianImage++ residual | 22.992 | 16.084 | 21.624 | 24.498 | 26.166 | 27.384 |
| Image-GS residual | 23.425 | 16.012 | 22.102 | 24.706 | 26.245 | 27.413 |
| Instant-GI quadtree | 21.102 | 12.944 | 20.592 | 22.168 | 22.502 | 22.680 |
| SS on-edge + residual | 24.189 | 16.576 | 23.320 | 25.242 | 26.534 | 27.569 |
| SS on-edge + residual relocate | 23.888 | 16.576 | 22.921 | 24.924 | 26.347 | 27.398 |
| SS on-edge + residual feature cap | 24.053 | 17.090 | 23.409 | 25.360 | 26.608 | 27.615 |
| SS on-edge + residual feature-rel cap | 24.303 | 16.595 | 23.446 | 25.324 | 26.621 | 27.621 |
| SS on-edge + tensor | 24.180 | 16.576 | 23.472 | 25.296 | 26.424 | 27.374 |
| SS on-edge + tensor feature cap | 24.298 | 17.090 | 23.682 | 25.683 | 26.878 | 27.724 |
| SS on-edge + tensor feature-rel cap | 24.235 | 16.595 | 23.515 | 25.365 | 26.483 | 27.445 |
| SS flanking + tensor | 24.177 | 16.655 | 23.389 | 25.302 | 26.428 | 27.371 |
| SS qt-WSE + residual | 24.117 | 16.655 | 23.255 | 25.128 | 26.484 | 27.536 |
| SS qt-WSE + residual relocate | 23.887 | 16.655 | 22.932 | 24.853 | 26.282 | 27.407 |
| SS qt-WSE + residual feature cap | 23.976 | 17.159 | 23.370 | 25.204 | 26.428 | 27.452 |
| SS qt-WSE + residual feature-rel cap | 24.282 | 16.672 | 23.349 | 25.278 | 26.656 | 27.694 |
| SS qt-WSE + tensor | 24.154 | 16.655 | 23.424 | 25.235 | 26.408 | 27.404 |
| SS qt-WSE + tensor feature cap | 24.154 | 17.159 | 23.603 | 25.485 | 26.687 | 27.559 |
| SS qt-WSE + tensor feature-rel cap | 24.171 | 16.672 | 23.422 | 25.322 | 26.442 | 27.413 |
| SS qt-hybrid + tensor | 24.113 | 16.857 | 23.344 | 25.259 | 26.423 | 27.378 |
| Floyd + tensor | 23.577 | 15.288 | 22.744 | 24.713 | 25.992 | 26.937 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 107.5 | 25% | 196.5 | 25% | 315.0 |
| SS best + SSIM 0.10 | 25% | 108.0 | 25% | 202.5 | 25% | 359.0 |
| SS best + L1 only | 25% | 126.5 | 25% | 248.0 | 25% | 394.5 |
| SS best + Charbonnier | 25% | 107.0 | 25% | 197.0 | 25% | 336.5 |
| SS best + tensor loss | 25% | 105.5 | 25% | 195.5 | 25% | 316.0 |
| SS best + final color solve | 25% | 107.5 | 25% | 197.5 | 25% | 342.5 |
| SS best + split relocate | 25% | 115.0 | 25% | 209.5 | 25% | 365.0 |
| SS best + adaptive 1.5x cap † | 50% | 222.5 | 25% | 162.5 | 25% | 240.0 |
| GaussianImage fixed | 25% | 93.0 | 25% | 210.5 | 0% | - |
| GaussianImage++ residual | 25% | 171.0 | 25% | 267.5 | 25% | 414.5 |
| Image-GS residual | 25% | 151.0 | 25% | 255.0 | 25% | 387.5 |
| Instant-GI quadtree | 25% | 201.5 | 12% | 396.0 | 0% | - |
| SS on-edge + residual | 25% | 129.0 | 25% | 237.5 | 25% | 376.0 |
| SS on-edge + residual relocate | 25% | 145.0 | 25% | 266.0 | 25% | 410.5 |
| SS on-edge + residual feature cap | 25% | 127.5 | 25% | 233.0 | 25% | 365.5 |
| SS on-edge + residual feature-rel cap | 25% | 124.0 | 25% | 230.0 | 25% | 356.5 |
| SS on-edge + tensor | 25% | 123.0 | 25% | 234.5 | 25% | 408.0 |
| SS on-edge + tensor feature cap | 25% | 118.5 | 25% | 208.0 | 25% | 334.0 |
| SS on-edge + tensor feature-rel cap | 25% | 122.5 | 25% | 229.0 | 25% | 380.0 |
| SS flanking + tensor | 25% | 128.5 | 25% | 240.5 | 25% | 420.5 |
| SS qt-WSE + residual | 25% | 132.5 | 25% | 245.0 | 25% | 379.5 |
| SS qt-WSE + residual relocate | 25% | 144.5 | 25% | 266.0 | 25% | 430.0 |
| SS qt-WSE + residual feature cap | 25% | 126.0 | 25% | 232.0 | 25% | 390.5 |
| SS qt-WSE + residual feature-rel cap | 25% | 127.0 | 25% | 233.5 | 25% | 352.0 |
| SS qt-WSE + tensor | 25% | 126.5 | 25% | 250.0 | 25% | 423.0 |
| SS qt-WSE + tensor feature cap | 25% | 119.5 | 25% | 223.0 | 25% | 339.0 |
| SS qt-WSE + tensor feature-rel cap | 25% | 126.5 | 25% | 232.5 | 25% | 398.5 |
| SS qt-hybrid + tensor | 25% | 130.5 | 25% | 244.0 | 25% | 410.0 |
| Floyd + tensor | 25% | 154.5 | 25% | 310.0 | 25% | 470.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | Floyd + tensor | 320 | 640 | 26.9373 | 3.2126 | 0.96777 | 23.577 | 1.108 |
| 640 | GaussianImage fixed | 640 | 640 | 25.7916 | 3.2335 | 0.97255 | 24.426 | 1.110 |
| 640 | GaussianImage++ residual | 320 | 640 | 27.3841 | 3.6179 | 0.97324 | 22.992 | 1.107 |
| 640 | Image-GS residual | 320 | 640 | 27.4130 | 3.5892 | 0.97190 | 23.425 | 1.093 |
| 640 | Instant-GI quadtree | 640 | 640 | 22.6801 | 4.0139 | 0.93493 | 21.102 | 1.085 |
| 640 | SS best + adaptive 1.5x cap † | 320 | 952 | 28.7802 | 3.1722 | 0.97847 | 25.540 | 1.120 |
| 640 | SS best + Charbonnier | 320 | 640 | 27.5749 | 3.7005 | 0.96821 | 24.823 | 1.116 |
| 640 | SS best + final color solve | 320 | 640 | 27.9459 | 3.6969 | 0.97253 | 24.833 | 1.231 |
| 640 | SS best default | 320 | 640 | 27.7524 | 3.7639 | 0.97092 | 24.850 | 1.218 |
| 640 | SS best + L1 only | 320 | 640 | 26.9764 | 3.6336 | 0.95986 | 24.360 | 0.787 |
| 640 | SS best + split relocate | 320 | 640 | 27.2648 | 3.7256 | 0.96736 | 24.639 | 1.199 |
| 640 | SS best + SSIM 0.10 | 320 | 640 | 27.6170 | 3.6756 | 0.96918 | 24.814 | 1.108 |
| 640 | SS best + tensor loss | 320 | 640 | 27.7643 | 3.8187 | 0.97069 | 24.863 | 1.151 |
| 640 | SS flanking + tensor | 320 | 640 | 27.3706 | 3.4506 | 0.97158 | 24.177 | 1.114 |
| 640 | SS on-edge + residual | 320 | 640 | 27.5694 | 3.5397 | 0.97247 | 24.189 | 1.116 |
| 640 | SS on-edge + residual feature-rel cap | 320 | 640 | 27.6212 | 3.6132 | 0.97123 | 24.303 | 1.117 |
| 640 | SS on-edge + residual feature cap | 320 | 640 | 27.6148 | 3.6547 | 0.97039 | 24.053 | 1.115 |
| 640 | SS on-edge + residual relocate | 320 | 640 | 27.3976 | 3.4991 | 0.97151 | 23.888 | 1.167 |
| 640 | SS on-edge + tensor | 320 | 640 | 27.3742 | 3.4446 | 0.97139 | 24.180 | 1.106 |
| 640 | SS on-edge + tensor feature-rel cap | 320 | 640 | 27.4450 | 3.5522 | 0.97104 | 24.235 | 1.128 |
| 640 | SS on-edge + tensor feature cap | 320 | 640 | 27.7242 | 3.8179 | 0.97052 | 24.298 | 1.146 |
| 640 | SS qt-hybrid + tensor | 320 | 640 | 27.3779 | 3.4449 | 0.97196 | 24.113 | 1.105 |
| 640 | SS qt-WSE + residual | 320 | 640 | 27.5364 | 3.5854 | 0.97178 | 24.117 | 1.114 |
| 640 | SS qt-WSE + residual feature-rel cap | 320 | 640 | 27.6942 | 3.7116 | 0.97194 | 24.282 | 1.119 |
| 640 | SS qt-WSE + residual feature cap | 320 | 640 | 27.4521 | 3.6391 | 0.96991 | 23.976 | 1.112 |
| 640 | SS qt-WSE + residual relocate | 320 | 640 | 27.4066 | 3.4985 | 0.97188 | 23.887 | 1.173 |
| 640 | SS qt-WSE + tensor | 320 | 640 | 27.4045 | 3.4146 | 0.97187 | 24.154 | 1.122 |
| 640 | SS qt-WSE + tensor feature-rel cap | 320 | 640 | 27.4130 | 3.4636 | 0.97150 | 24.171 | 1.115 |
| 640 | SS qt-WSE + tensor feature cap | 320 | 640 | 27.5586 | 3.7886 | 0.97028 | 24.154 | 1.131 |

## Winners By Image/Budget

Winners are taken among budget-matched methods only; † rows (over the shared cap) are excluded.

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS on-edge + tensor feature cap (27.646) | SS on-edge + tensor feature cap (0.98420) |
| COCO_train2014_000000000025 | 640 | SS best + final color solve (25.547) | GaussianImage fixed (0.97036) |
| COCO_train2014_000000000030 | 640 | SS best + final color solve (34.452) | GaussianImage fixed (0.99387) |
| COCO_train2014_000000000034 | 640 | SS best + final color solve (24.655) | GaussianImage fixed (0.95236) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
