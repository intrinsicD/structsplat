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
| SS best default | 8 | 27.7024 | 3.7896 | 0.97090 | 0.01807 | 24.842 | - | 0.061 | 1.345 | 1.407 |
| SS best + SSIM 0.10 | 8 | 27.5905 | 3.6661 | 0.96918 | 0.01876 | 24.811 | - | 0.062 | 1.254 | 1.316 |
| SS best + L1 only | 8 | 26.9477 | 3.5592 | 0.96069 | 0.02274 | 24.355 | - | 0.059 | 0.887 | 0.946 |
| SS best + Charbonnier | 8 | 27.5779 | 3.6796 | 0.96857 | 0.01893 | 24.826 | - | 0.060 | 1.295 | 1.355 |
| SS best + tensor loss | 8 | 27.6823 | 3.7361 | 0.97006 | 0.01872 | 24.845 | - | 0.060 | 1.308 | 1.369 |
| SS best + final color solve | 8 | 27.8578 | 3.7304 | 0.97173 | 0.01697 | 24.839 | - | 0.060 | 1.453 | 1.514 |
| SS best + split relocate | 8 | 27.3380 | 3.7485 | 0.96751 | 0.02099 | 24.656 | - | 0.058 | 1.390 | 1.448 |
| SS best + adaptive 1.5x cap † | 8 | 28.6732 | 3.1744 | 0.97825 | 0.01420 | 25.540 | - | 0.059 | 1.270 | 1.330 |
| GaussianImage fixed | 8 | 25.8653 | 3.1475 | 0.97258 | 0.01597 | 24.394 | - | 0.001 | 1.271 | 1.272 |
| GaussianImage++ residual | 8 | 27.2948 | 3.6005 | 0.97260 | 0.01725 | 22.954 | - | 0.001 | 1.289 | 1.290 |
| Image-GS residual | 8 | 27.4526 | 3.6310 | 0.97218 | 0.01809 | 23.437 | - | 0.005 | 1.287 | 1.291 |
| Instant-GI quadtree | 0 | - | - | - | - | - | - | - | - | - |
| SS on-edge + residual | 8 | 27.5488 | 3.5853 | 0.97207 | 0.01764 | 24.182 | - | 0.063 | 1.250 | 1.314 |
| SS on-edge + residual relocate | 8 | 27.4064 | 3.4469 | 0.97173 | 0.01751 | 23.903 | - | 0.060 | 1.325 | 1.385 |
| SS on-edge + residual feature cap | 8 | 27.5723 | 3.6345 | 0.96993 | 0.02022 | 24.055 | - | 0.061 | 1.303 | 1.364 |
| SS on-edge + residual feature-rel cap | 8 | 27.6934 | 3.6624 | 0.97208 | 0.01782 | 24.331 | - | 0.061 | 1.302 | 1.363 |
| SS on-edge + tensor | 8 | 27.4135 | 3.5028 | 0.97169 | 0.01776 | 24.199 | - | 0.060 | 1.275 | 1.335 |
| SS on-edge + tensor feature cap | 8 | 27.7692 | 3.7436 | 0.97121 | 0.01831 | 24.293 | - | 0.061 | 1.290 | 1.351 |
| SS on-edge + tensor feature-rel cap | 8 | 27.4597 | 3.5228 | 0.97158 | 0.01769 | 24.232 | - | 0.060 | 1.265 | 1.326 |
| SS flanking + tensor | 8 | 27.4070 | 3.3689 | 0.97198 | 0.01713 | 24.190 | - | 0.057 | 1.207 | 1.265 |
| SS qt-WSE + residual | 8 | 27.5051 | 3.5554 | 0.97209 | 0.01783 | 24.108 | - | 0.148 | 1.235 | 1.382 |
| SS qt-WSE + residual relocate | 8 | 27.3085 | 3.4393 | 0.97149 | 0.01787 | 23.870 | - | 0.148 | 1.315 | 1.463 |
| SS qt-WSE + residual feature cap | 8 | 27.3660 | 3.5114 | 0.96998 | 0.01982 | 23.975 | - | 0.150 | 1.253 | 1.404 |
| SS qt-WSE + residual feature-rel cap | 8 | 27.6512 | 3.6577 | 0.97178 | 0.01824 | 24.245 | - | 0.152 | 1.217 | 1.369 |
| SS qt-WSE + tensor | 8 | 27.3956 | 3.4300 | 0.97196 | 0.01736 | 24.143 | - | 0.147 | 1.220 | 1.367 |
| SS qt-WSE + tensor feature cap | 8 | 27.6668 | 3.7503 | 0.97103 | 0.01834 | 24.143 | - | 0.149 | 1.210 | 1.359 |
| SS qt-WSE + tensor feature-rel cap | 8 | 27.4034 | 3.5175 | 0.97126 | 0.01779 | 24.167 | - | 0.156 | 1.226 | 1.382 |
| SS qt-hybrid + tensor | 8 | 27.3507 | 3.4207 | 0.97211 | 0.01710 | 24.112 | - | 0.076 | 1.220 | 1.296 |
| Floyd + tensor | 8 | 26.9466 | 3.2437 | 0.96773 | 0.01998 | 23.556 | - | 0.027 | 1.198 | 1.226 |

† Not budget-matched: mean final Gaussian count exceeds the shared final cap (adaptive extra capacity). These rows spend more primitives — more rate for an image codec — so their PSNR/MS-SSIM is not directly comparable to the equal-budget rows, and they are excluded from the per-cell winners below.

## Default Promotion Check

A best-default candidate is promotable only when its paired mean deltas beat `SS best default` on quality (PSNR and MS-SSIM), convergence (AUC), and performance (fit and total seconds). Over-budget rows are excluded.

| Candidate | Pairs | ΔPSNR | ΔMS-SSIM | ΔAUC | ΔFit s | ΔTotal s | PSNR wins | MS wins | AUC wins | Faster fit | Promote |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| SS best + SSIM 0.10 | 8 | -0.1119 | -0.00171 | -0.0304 | -0.0911 | -0.0908 | 1/8 | 0/8 | 3/8 | 5/8 | no |
| SS best + L1 only | 8 | -0.7547 | -0.01021 | -0.4869 | -0.4588 | -0.4609 | 0/8 | 0/8 | 0/8 | 8/8 | no |
| SS best + Charbonnier | 8 | -0.1245 | -0.00233 | -0.0154 | -0.0506 | -0.0517 | 1/8 | 0/8 | 4/8 | 3/8 | no |
| SS best + tensor loss | 8 | -0.0201 | -0.00084 | +0.0028 | -0.0371 | -0.0378 | 4/8 | 4/8 | 4/8 | 2/8 | no |
| SS best + final color solve | 8 | +0.1555 | +0.00084 | -0.0030 | +0.1079 | +0.1070 | 7/8 | 4/8 | 3/8 | 1/8 | no |
| SS best + split relocate | 8 | -0.3644 | -0.00338 | -0.1858 | +0.0445 | +0.0418 | 0/8 | 0/8 | 0/8 | 1/8 | no |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 499-iteration budget.

| Method | AUC | PSNR@0 | PSNR@125 | PSNR@250 | PSNR@374 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 24.842 | 17.090 | 23.910 | 23.280 | 26.785 | 27.702 |
| SS best + SSIM 0.10 | 24.811 | 17.090 | 23.867 | 23.159 | 26.708 | 27.590 |
| SS best + L1 only | 24.355 | 17.090 | 23.434 | 23.100 | 26.082 | 26.948 |
| SS best + Charbonnier | 24.826 | 17.090 | 23.879 | 23.254 | 26.702 | 27.578 |
| SS best + tensor loss | 24.845 | 17.090 | 23.931 | 23.316 | 26.743 | 27.682 |
| SS best + final color solve | 24.839 | 17.090 | 23.910 | 23.295 | 26.774 | 27.858 |
| SS best + split relocate | 24.656 | 17.090 | 23.688 | 22.527 | 26.617 | 27.338 |
| SS best + adaptive 1.5x cap † | 25.540 | 17.090 | 24.262 | 23.828 | 28.582 | 28.673 |
| GaussianImage fixed | 24.394 | 16.921 | 23.979 | 25.359 | 25.631 | 25.865 |
| GaussianImage++ residual | 22.954 | 16.084 | 21.657 | 24.495 | 26.138 | 27.295 |
| Image-GS residual | 23.437 | 16.012 | 22.103 | 24.667 | 26.297 | 27.453 |
| Instant-GI quadtree | - | - | - | - | - | - |
| SS on-edge + residual | 24.182 | 16.576 | 23.319 | 25.227 | 26.524 | 27.549 |
| SS on-edge + residual relocate | 23.903 | 16.576 | 22.923 | 24.928 | 26.352 | 27.406 |
| SS on-edge + residual feature cap | 24.055 | 17.090 | 23.403 | 25.374 | 26.658 | 27.572 |
| SS on-edge + residual feature-rel cap | 24.331 | 16.595 | 23.439 | 25.334 | 26.646 | 27.693 |
| SS on-edge + tensor | 24.199 | 16.576 | 23.465 | 25.310 | 26.475 | 27.414 |
| SS on-edge + tensor feature cap | 24.293 | 17.090 | 23.676 | 25.709 | 26.870 | 27.769 |
| SS on-edge + tensor feature-rel cap | 24.232 | 16.595 | 23.506 | 25.346 | 26.458 | 27.460 |
| SS flanking + tensor | 24.190 | 16.655 | 23.372 | 25.307 | 26.425 | 27.407 |
| SS qt-WSE + residual | 24.108 | 16.655 | 23.254 | 25.089 | 26.437 | 27.505 |
| SS qt-WSE + residual relocate | 23.870 | 16.655 | 22.935 | 24.873 | 26.238 | 27.309 |
| SS qt-WSE + residual feature cap | 23.975 | 17.159 | 23.376 | 25.221 | 26.466 | 27.366 |
| SS qt-WSE + residual feature-rel cap | 24.245 | 16.672 | 23.345 | 25.257 | 26.596 | 27.651 |
| SS qt-WSE + tensor | 24.143 | 16.655 | 23.424 | 25.245 | 26.418 | 27.396 |
| SS qt-WSE + tensor feature cap | 24.143 | 17.159 | 23.605 | 25.514 | 26.651 | 27.667 |
| SS qt-WSE + tensor feature-rel cap | 24.167 | 16.672 | 23.424 | 25.300 | 26.445 | 27.403 |
| SS qt-hybrid + tensor | 24.112 | 16.857 | 23.347 | 25.265 | 26.425 | 27.351 |
| Floyd + tensor | 23.556 | 15.288 | 22.739 | 24.701 | 25.976 | 26.947 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS best default | 25% | 107.5 | 25% | 198.5 | 25% | 331.5 |
| SS best + SSIM 0.10 | 25% | 108.0 | 25% | 201.5 | 25% | 360.5 |
| SS best + L1 only | 25% | 128.0 | 25% | 250.5 | 25% | 415.0 |
| SS best + Charbonnier | 25% | 107.0 | 25% | 197.0 | 25% | 338.5 |
| SS best + tensor loss | 25% | 105.5 | 25% | 195.0 | 25% | 323.5 |
| SS best + final color solve | 25% | 107.5 | 25% | 196.5 | 25% | 320.0 |
| SS best + split relocate | 25% | 115.0 | 25% | 208.0 | 25% | 367.5 |
| SS best + adaptive 1.5x cap † | 50% | 226.0 | 25% | 162.0 | 25% | 241.5 |
| GaussianImage fixed | 25% | 92.5 | 25% | 217.0 | 0% | - |
| GaussianImage++ residual | 25% | 171.0 | 25% | 272.5 | 25% | 430.5 |
| Image-GS residual | 25% | 151.0 | 25% | 261.5 | 25% | 390.0 |
| Instant-GI quadtree | - | - | - | - | - | - |
| SS on-edge + residual | 25% | 129.0 | 25% | 239.0 | 25% | 376.0 |
| SS on-edge + residual relocate | 25% | 144.0 | 25% | 262.0 | 25% | 418.0 |
| SS on-edge + residual feature cap | 25% | 127.5 | 25% | 232.0 | 25% | 363.0 |
| SS on-edge + residual feature-rel cap | 25% | 124.0 | 25% | 230.0 | 25% | 355.5 |
| SS on-edge + tensor | 25% | 123.5 | 25% | 236.0 | 25% | 381.5 |
| SS on-edge + tensor feature cap | 25% | 118.5 | 25% | 206.0 | 25% | 333.0 |
| SS on-edge + tensor feature-rel cap | 25% | 123.0 | 25% | 229.5 | 25% | 402.5 |
| SS flanking + tensor | 25% | 128.5 | 25% | 240.0 | 25% | 427.5 |
| SS qt-WSE + residual | 25% | 132.5 | 25% | 248.0 | 25% | 387.0 |
| SS qt-WSE + residual relocate | 25% | 144.5 | 25% | 265.5 | 25% | 434.0 |
| SS qt-WSE + residual feature cap | 25% | 126.0 | 25% | 234.0 | 25% | 380.0 |
| SS qt-WSE + residual feature-rel cap | 25% | 127.5 | 25% | 233.0 | 25% | 357.5 |
| SS qt-WSE + tensor | 25% | 127.0 | 25% | 246.5 | 25% | 406.0 |
| SS qt-WSE + tensor feature cap | 25% | 119.5 | 25% | 226.0 | 25% | 353.5 |
| SS qt-WSE + tensor feature-rel cap | 25% | 126.5 | 25% | 234.0 | 25% | 401.0 |
| SS qt-hybrid + tensor | 25% | 130.0 | 25% | 243.5 | 25% | 423.5 |
| Floyd + tensor | 25% | 157.0 | 25% | 321.0 | 25% | 460.5 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 640 | Floyd + tensor | 320 | 640 | 26.9466 | 3.2437 | 0.96773 | 23.556 | 1.198 |
| 640 | GaussianImage fixed | 640 | 640 | 25.8653 | 3.1475 | 0.97258 | 24.394 | 1.271 |
| 640 | GaussianImage++ residual | 320 | 640 | 27.2948 | 3.6005 | 0.97260 | 22.954 | 1.289 |
| 640 | Image-GS residual | 320 | 640 | 27.4526 | 3.6310 | 0.97218 | 23.437 | 1.287 |
| 640 | SS best + adaptive 1.5x cap † | 320 | 952 | 28.6732 | 3.1744 | 0.97825 | 25.540 | 1.270 |
| 640 | SS best + Charbonnier | 320 | 640 | 27.5779 | 3.6796 | 0.96857 | 24.826 | 1.295 |
| 640 | SS best + final color solve | 320 | 640 | 27.8578 | 3.7304 | 0.97173 | 24.839 | 1.453 |
| 640 | SS best default | 320 | 640 | 27.7024 | 3.7896 | 0.97090 | 24.842 | 1.345 |
| 640 | SS best + L1 only | 320 | 640 | 26.9477 | 3.5592 | 0.96069 | 24.355 | 0.887 |
| 640 | SS best + split relocate | 320 | 640 | 27.3380 | 3.7485 | 0.96751 | 24.656 | 1.390 |
| 640 | SS best + SSIM 0.10 | 320 | 640 | 27.5905 | 3.6661 | 0.96918 | 24.811 | 1.254 |
| 640 | SS best + tensor loss | 320 | 640 | 27.6823 | 3.7361 | 0.97006 | 24.845 | 1.308 |
| 640 | SS flanking + tensor | 320 | 640 | 27.4070 | 3.3689 | 0.97198 | 24.190 | 1.207 |
| 640 | SS on-edge + residual | 320 | 640 | 27.5488 | 3.5853 | 0.97207 | 24.182 | 1.250 |
| 640 | SS on-edge + residual feature-rel cap | 320 | 640 | 27.6934 | 3.6624 | 0.97208 | 24.331 | 1.302 |
| 640 | SS on-edge + residual feature cap | 320 | 640 | 27.5723 | 3.6345 | 0.96993 | 24.055 | 1.303 |
| 640 | SS on-edge + residual relocate | 320 | 640 | 27.4064 | 3.4469 | 0.97173 | 23.903 | 1.325 |
| 640 | SS on-edge + tensor | 320 | 640 | 27.4135 | 3.5028 | 0.97169 | 24.199 | 1.275 |
| 640 | SS on-edge + tensor feature-rel cap | 320 | 640 | 27.4597 | 3.5228 | 0.97158 | 24.232 | 1.265 |
| 640 | SS on-edge + tensor feature cap | 320 | 640 | 27.7692 | 3.7436 | 0.97121 | 24.293 | 1.290 |
| 640 | SS qt-hybrid + tensor | 320 | 640 | 27.3507 | 3.4207 | 0.97211 | 24.112 | 1.220 |
| 640 | SS qt-WSE + residual | 320 | 640 | 27.5051 | 3.5554 | 0.97209 | 24.108 | 1.235 |
| 640 | SS qt-WSE + residual feature-rel cap | 320 | 640 | 27.6512 | 3.6577 | 0.97178 | 24.245 | 1.217 |
| 640 | SS qt-WSE + residual feature cap | 320 | 640 | 27.3660 | 3.5114 | 0.96998 | 23.975 | 1.253 |
| 640 | SS qt-WSE + residual relocate | 320 | 640 | 27.3085 | 3.4393 | 0.97149 | 23.870 | 1.315 |
| 640 | SS qt-WSE + tensor | 320 | 640 | 27.3956 | 3.4300 | 0.97196 | 24.143 | 1.220 |
| 640 | SS qt-WSE + tensor feature-rel cap | 320 | 640 | 27.4034 | 3.5175 | 0.97126 | 24.167 | 1.226 |
| 640 | SS qt-WSE + tensor feature cap | 320 | 640 | 27.6668 | 3.7503 | 0.97103 | 24.143 | 1.210 |

## Winners By Image/Budget

Winners are taken among budget-matched methods only; † rows (over the shared cap) are excluded.

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| COCO_train2014_000000000009 | 640 | SS on-edge + tensor feature cap (27.756) | SS on-edge + tensor feature cap (0.98416) |
| COCO_train2014_000000000025 | 640 | SS on-edge + residual feature-rel cap (25.484) | GaussianImage++ residual (0.96954) |
| COCO_train2014_000000000030 | 640 | SS best + final color solve (34.186) | GaussianImage fixed (0.99402) |
| COCO_train2014_000000000034 | 640 | SS best + final color solve (24.535) | GaussianImage fixed (0.95265) |

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
