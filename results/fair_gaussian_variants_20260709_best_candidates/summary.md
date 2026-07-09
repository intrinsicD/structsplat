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
| SS best default | 8 | 27.6759 | 3.8063 | 0.97042 | 0.01861 | 24.816 | - | 0.046 | 0.892 | 0.938 |
| SS best + SSIM 0.10 | 8 | 27.5461 | 3.6757 | 0.96815 | 0.02008 | 24.780 | - | 0.045 | 0.801 | 0.846 |
| SS best + L1 only | 8 | 27.0070 | 3.6136 | 0.96018 | 0.02364 | 24.370 | - | 0.043 | 0.573 | 0.615 |
| SS best + Charbonnier | 8 | 27.6053 | 3.7377 | 0.96865 | 0.01994 | 24.813 | - | 0.043 | 0.816 | 0.859 |
| SS best + tensor loss | 8 | 27.7054 | 3.7921 | 0.97055 | 0.01877 | 24.841 | - | 0.043 | 0.815 | 0.858 |
| SS best + final color solve | 8 | 27.7674 | 3.7278 | 0.97114 | 0.01791 | 24.813 | - | 0.044 | 0.877 | 0.920 |
| SS best + split relocate | 8 | 27.3461 | 3.7196 | 0.96809 | 0.02144 | 24.613 | - | 0.043 | 0.856 | 0.899 |
| SS best + adaptive 1.5x cap | 8 | 28.7259 | 3.1343 | 0.97766 | 0.01480 | 25.480 | - | 0.045 | 0.828 | 0.873 |
| GaussianImage fixed | 8 | 25.8233 | 3.2009 | 0.97232 | 0.01625 | 24.402 | - | 0.001 | 0.781 | 0.782 |
| GaussianImage++ residual | 8 | 27.1687 | 3.4844 | 0.97248 | 0.01680 | 23.486 | - | 0.001 | 0.790 | 0.790 |
| Image-GS residual | 8 | 27.3259 | 3.6116 | 0.97159 | 0.01793 | 23.874 | - | 0.003 | 0.799 | 0.801 |
| Instant-GI quadtree | 0 | - | - | - | - | - | - | - | - | - |
| SS on-edge + residual | 8 | 27.4448 | 3.5569 | 0.97143 | 0.01805 | 24.517 | - | 0.045 | 0.820 | 0.865 |
| SS on-edge + residual relocate | 8 | 27.2111 | 3.5213 | 0.96997 | 0.01863 | 24.315 | - | 0.043 | 0.853 | 0.896 |
| SS on-edge + residual feature cap | 8 | 27.5286 | 3.6329 | 0.97002 | 0.01931 | 24.597 | - | 0.044 | 0.816 | 0.861 |
| SS on-edge + residual feature-rel cap | 8 | 27.5624 | 3.6762 | 0.97073 | 0.01892 | 24.664 | - | 0.044 | 0.799 | 0.843 |
| SS on-edge + tensor | 8 | 27.2634 | 3.3623 | 0.97138 | 0.01718 | 24.569 | - | 0.043 | 0.797 | 0.840 |
| SS on-edge + tensor feature cap | 8 | 27.6963 | 3.7071 | 0.97069 | 0.01814 | 24.817 | - | 0.043 | 0.808 | 0.851 |
| SS on-edge + tensor feature-rel cap | 8 | 27.3693 | 3.5285 | 0.97081 | 0.01772 | 24.635 | - | 0.043 | 0.820 | 0.863 |
| SS flanking + tensor | 8 | 27.2330 | 3.3089 | 0.97072 | 0.01843 | 24.519 | - | 0.045 | 0.809 | 0.854 |
| SS qt-WSE + residual | 8 | 27.4091 | 3.5262 | 0.97138 | 0.01815 | 24.474 | - | 0.099 | 0.812 | 0.911 |
| SS qt-WSE + residual relocate | 8 | 27.1367 | 3.4309 | 0.96977 | 0.01915 | 24.294 | - | 0.100 | 0.850 | 0.950 |
| SS qt-WSE + residual feature cap | 8 | 27.4875 | 3.5784 | 0.97037 | 0.01859 | 24.549 | - | 0.101 | 0.823 | 0.924 |
| SS qt-WSE + residual feature-rel cap | 8 | 27.5396 | 3.5597 | 0.97097 | 0.01848 | 24.611 | - | 0.102 | 0.826 | 0.928 |
| SS qt-WSE + tensor | 8 | 27.2927 | 3.3812 | 0.97113 | 0.01783 | 24.539 | - | 0.100 | 0.842 | 0.943 |
| SS qt-WSE + tensor feature cap | 8 | 27.5079 | 3.7553 | 0.96881 | 0.02076 | 24.710 | - | 0.104 | 0.855 | 0.959 |
| SS qt-WSE + tensor feature-rel cap | 8 | 27.2874 | 3.3834 | 0.97133 | 0.01720 | 24.544 | - | 0.103 | 0.832 | 0.935 |
| SS qt-hybrid + tensor | 8 | 27.3717 | 3.4000 | 0.97212 | 0.01701 | 24.563 | - | 0.068 | 0.833 | 0.901 |
| Floyd + tensor | 8 | 26.8515 | 3.2155 | 0.96699 | 0.02026 | 24.015 | - | 0.019 | 0.811 | 0.831 |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 1500-iteration budget.

| Method | AUC | PSNR@0 | PSNR@375 | PSNR@750 | PSNR@1125 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.816 | 17.072 | 26.738 | 27.649 | 27.649 | 27.676 |
| SS best + SSIM 0.10 | 24.780 | 17.072 | 26.660 | 27.530 | 27.530 | 27.546 |
| SS best + L1 only | 24.370 | 17.072 | 26.127 | 26.997 | 26.997 | 27.007 |
| SS best + Charbonnier | 24.813 | 17.072 | 26.692 | 27.585 | 27.585 | 27.605 |
| SS best + tensor loss | 24.841 | 17.072 | 26.781 | 27.683 | 27.683 | 27.705 |
| SS best + final color solve | 24.813 | 17.072 | 26.717 | 27.542 | 27.542 | 27.767 |
| SS best + split relocate | 24.613 | 17.072 | 26.567 | 27.234 | 27.234 | 27.346 |
| SS best + adaptive 1.5x cap | 25.480 | 17.072 | 28.469 | 28.466 | 28.466 | 28.726 |
| GaussianImage fixed | 24.402 | 16.921 | 25.727 | 25.797 | 25.797 | 25.823 |
| GaussianImage++ residual | 23.486 | 16.084 | 26.018 | 27.125 | 27.125 | 27.169 |
| Image-GS residual | 23.874 | 16.011 | 26.234 | 27.298 | 27.298 | 27.326 |
| Instant-GI quadtree | - | - | - | - | - | - |
| SS on-edge + residual | 24.517 | 16.551 | 26.389 | 27.412 | 27.412 | 27.445 |
| SS on-edge + residual relocate | 24.315 | 16.551 | 26.151 | 27.158 | 27.158 | 27.211 |
| SS on-edge + residual feature cap | 24.597 | 17.072 | 26.520 | 27.507 | 27.507 | 27.529 |
| SS on-edge + residual feature-rel cap | 24.664 | 16.570 | 26.527 | 27.534 | 27.534 | 27.562 |
| SS on-edge + tensor | 24.569 | 16.551 | 26.351 | 27.228 | 27.228 | 27.263 |
| SS on-edge + tensor feature cap | 24.817 | 17.072 | 26.769 | 27.672 | 27.672 | 27.696 |
| SS on-edge + tensor feature-rel cap | 24.635 | 16.570 | 26.457 | 27.337 | 27.337 | 27.369 |
| SS flanking + tensor | 24.519 | 16.626 | 26.337 | 27.204 | 27.204 | 27.233 |
| SS qt-WSE + residual | 24.474 | 16.681 | 26.310 | 27.376 | 27.376 | 27.409 |
| SS qt-WSE + residual relocate | 24.294 | 16.681 | 26.131 | 27.094 | 27.094 | 27.137 |
| SS qt-WSE + residual feature cap | 24.549 | 17.171 | 26.465 | 27.459 | 27.459 | 27.488 |
| SS qt-WSE + residual feature-rel cap | 24.611 | 16.699 | 26.513 | 27.517 | 27.517 | 27.540 |
| SS qt-WSE + tensor | 24.539 | 16.681 | 26.366 | 27.264 | 27.264 | 27.293 |
| SS qt-WSE + tensor feature cap | 24.710 | 17.171 | 26.597 | 27.497 | 27.497 | 27.508 |
| SS qt-WSE + tensor feature-rel cap | 24.544 | 16.699 | 26.351 | 27.262 | 27.262 | 27.287 |
| SS qt-hybrid + tensor | 24.563 | 16.900 | 26.404 | 27.335 | 27.335 | 27.372 |
| Floyd + tensor | 24.015 | 15.293 | 25.943 | 26.816 | 26.816 | 26.852 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 107.5 | 25% | 197.5 | 25% | 323.0 |
| SS best + SSIM 0.10 | 25% | 108.0 | 25% | 202.0 | 25% | 344.0 |
| SS best + L1 only | 25% | 126.5 | 25% | 246.0 | 25% | 399.0 |
| SS best + Charbonnier | 25% | 107.0 | 25% | 197.0 | 25% | 341.0 |
| SS best + tensor loss | 25% | 106.0 | 25% | 198.5 | 25% | 339.5 |
| SS best + final color solve | 25% | 107.5 | 25% | 197.0 | 25% | 323.5 |
| SS best + split relocate | 25% | 114.5 | 25% | 207.5 | 25% | 364.0 |
| SS best + adaptive 1.5x cap | 50% | 227.2 | 25% | 162.5 | 25% | 239.5 |
| GaussianImage fixed | 25% | 93.0 | 25% | 214.0 | 0% | - |
| GaussianImage++ residual | 25% | 193.0 | 25% | 287.0 | 25% | 444.5 |
| Image-GS residual | 25% | 147.5 | 25% | 272.5 | 25% | 388.0 |
| Instant-GI quadtree | - | - | - | - | - | - |
| SS on-edge + residual | 25% | 120.5 | 25% | 249.5 | 25% | 392.5 |
| SS on-edge + residual relocate | 25% | 134.5 | 25% | 281.0 | 25% | 425.5 |
| SS on-edge + residual feature cap | 25% | 117.0 | 25% | 219.0 | 25% | 372.5 |
| SS on-edge + residual feature-rel cap | 25% | 115.0 | 25% | 216.5 | 25% | 367.0 |
| SS on-edge + tensor | 25% | 113.5 | 25% | 235.5 | 25% | 421.0 |
| SS on-edge + tensor feature cap | 25% | 108.0 | 25% | 198.0 | 25% | 340.5 |
| SS on-edge + tensor feature-rel cap | 25% | 111.5 | 25% | 226.5 | 25% | 382.5 |
| SS flanking + tensor | 25% | 122.0 | 25% | 251.0 | 25% | 441.5 |
| SS qt-WSE + residual | 25% | 123.5 | 25% | 238.0 | 25% | 413.0 |
| SS qt-WSE + residual relocate | 25% | 133.0 | 25% | 261.5 | 25% | 442.0 |
| SS qt-WSE + residual feature cap | 25% | 116.5 | 25% | 220.0 | 25% | 376.0 |
| SS qt-WSE + residual feature-rel cap | 25% | 120.0 | 25% | 222.0 | 25% | 377.5 |
| SS qt-WSE + tensor | 25% | 119.0 | 25% | 252.5 | 25% | 437.0 |
| SS qt-WSE + tensor feature cap | 25% | 109.5 | 25% | 203.5 | 25% | 363.0 |
| SS qt-WSE + tensor feature-rel cap | 25% | 119.0 | 25% | 231.5 | 25% | 425.5 |
| SS qt-hybrid + tensor | 25% | 124.5 | 25% | 253.5 | 25% | 421.0 |
| Floyd + tensor | 25% | 172.5 | 25% | 295.0 | 25% | 479.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | Floyd + tensor | 320 | 640 | 26.8515 | 3.2155 | 0.96699 | 24.015 | 0.811 |
| 640 | GaussianImage fixed | 640 | 640 | 25.8233 | 3.2009 | 0.97232 | 24.402 | 0.781 |
| 640 | GaussianImage++ residual | 320 | 640 | 27.1687 | 3.4844 | 0.97248 | 23.486 | 0.790 |
| 640 | Image-GS residual | 320 | 640 | 27.3259 | 3.6116 | 0.97159 | 23.874 | 0.799 |
| 640 | SS best + adaptive 1.5x cap | 320 | 952 | 28.7259 | 3.1343 | 0.97766 | 25.480 | 0.828 |
| 640 | SS best + Charbonnier | 320 | 640 | 27.6053 | 3.7377 | 0.96865 | 24.813 | 0.816 |
| 640 | SS best + final color solve | 320 | 640 | 27.7674 | 3.7278 | 0.97114 | 24.813 | 0.877 |
| 640 | SS best default | 320 | 640 | 27.6759 | 3.8063 | 0.97042 | 24.816 | 0.892 |
| 640 | SS best + L1 only | 320 | 640 | 27.0070 | 3.6136 | 0.96018 | 24.370 | 0.573 |
| 640 | SS best + split relocate | 320 | 640 | 27.3461 | 3.7196 | 0.96809 | 24.613 | 0.856 |
| 640 | SS best + SSIM 0.10 | 320 | 640 | 27.5461 | 3.6757 | 0.96815 | 24.780 | 0.801 |
| 640 | SS best + tensor loss | 320 | 640 | 27.7054 | 3.7921 | 0.97055 | 24.841 | 0.815 |
| 640 | SS flanking + tensor | 320 | 640 | 27.2330 | 3.3089 | 0.97072 | 24.519 | 0.809 |
| 640 | SS on-edge + residual | 320 | 640 | 27.4448 | 3.5569 | 0.97143 | 24.517 | 0.820 |
| 640 | SS on-edge + residual feature-rel cap | 320 | 640 | 27.5624 | 3.6762 | 0.97073 | 24.664 | 0.799 |
| 640 | SS on-edge + residual feature cap | 320 | 640 | 27.5286 | 3.6329 | 0.97002 | 24.597 | 0.816 |
| 640 | SS on-edge + residual relocate | 320 | 640 | 27.2111 | 3.5213 | 0.96997 | 24.315 | 0.853 |
| 640 | SS on-edge + tensor | 320 | 640 | 27.2634 | 3.3623 | 0.97138 | 24.569 | 0.797 |
| 640 | SS on-edge + tensor feature-rel cap | 320 | 640 | 27.3693 | 3.5285 | 0.97081 | 24.635 | 0.820 |
| 640 | SS on-edge + tensor feature cap | 320 | 640 | 27.6963 | 3.7071 | 0.97069 | 24.817 | 0.808 |
| 640 | SS qt-hybrid + tensor | 320 | 640 | 27.3717 | 3.4000 | 0.97212 | 24.563 | 0.833 |
| 640 | SS qt-WSE + residual | 320 | 640 | 27.4091 | 3.5262 | 0.97138 | 24.474 | 0.812 |
| 640 | SS qt-WSE + residual feature-rel cap | 320 | 640 | 27.5396 | 3.5597 | 0.97097 | 24.611 | 0.826 |
| 640 | SS qt-WSE + residual feature cap | 320 | 640 | 27.4875 | 3.5784 | 0.97037 | 24.549 | 0.823 |
| 640 | SS qt-WSE + residual relocate | 320 | 640 | 27.1367 | 3.4309 | 0.96977 | 24.294 | 0.850 |
| 640 | SS qt-WSE + tensor | 320 | 640 | 27.2927 | 3.3812 | 0.97113 | 24.539 | 0.842 |
| 640 | SS qt-WSE + tensor feature-rel cap | 320 | 640 | 27.2874 | 3.3834 | 0.97133 | 24.544 | 0.832 |
| 640 | SS qt-WSE + tensor feature cap | 320 | 640 | 27.5079 | 3.7553 | 0.96881 | 24.710 | 0.855 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS best + adaptive 1.5x cap (29.366) | SS best + adaptive 1.5x cap (0.98972) |
| COCO_train2014_000000000025 | 640 | SS best + adaptive 1.5x cap (26.952) | SS best + adaptive 1.5x cap (0.97823) |
| COCO_train2014_000000000030 | 640 | SS best + tensor loss (34.114) | GaussianImage fixed (0.99404) |
| COCO_train2014_000000000034 | 640 | SS best + adaptive 1.5x cap (25.218) | GaussianImage fixed (0.95506) |

## Errors

| Cell | Error |
|---|---|
| COCO_train2014_000000000009 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000009 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000025 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000025 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000030 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000030 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000034 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |
| COCO_train2014_000000000034 640 Instant-GI quadtree | `RuntimeError: Instant-GI methods require STRUCTSPLAT_INSTANT_GI=/path/to/quard_image.py; unset -> this method is skipped.` |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
