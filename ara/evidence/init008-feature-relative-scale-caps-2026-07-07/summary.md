# Fair Density-Control Comparison

Matched-policy comparison against repo-inspired 2D Gaussian baselines.

Growth rows share the same initial Gaussian count, final cap, growth wave count, fitter, renderer, loss, target tracking, and iteration budget.
This is not a native external-repo benchmark; it isolates placement/growth policies inside StructSplat's fitter and exact renderer.

## Methods

| Method | Track | Description |
|---|---|---|
| SS on-edge + residual | same-growth | StructSplat on-edge initializer under the same residual-add growth as external analogues. |
| SS on-edge + residual feature cap | same-growth+feature-cap | StructSplat on-edge residual-add growth with feature-adaptive per-Gaussian scale caps. |
| SS on-edge + residual feature-rel cap | same-growth+feature-rel-cap | StructSplat on-edge residual-add growth with feature-relative local-radius scale caps. |
| SS on-edge + tensor | tensor-growth | StructSplat on-edge initializer plus tensor-aware residual growth. |
| SS on-edge + tensor feature cap | tensor-growth+feature-cap | StructSplat on-edge tensor-aware residual growth with feature-adaptive scale caps. |
| SS on-edge + tensor feature-rel cap | tensor-growth+feature-rel-cap | StructSplat on-edge tensor-aware residual growth with feature-relative local-radius caps. |
| SS qt-WSE + residual | same-growth | StructSplat quadtree-WSE initializer under the same residual-add growth as external analogues. |
| SS qt-WSE + residual feature cap | same-growth+feature-cap | StructSplat quadtree-WSE residual-add growth with feature-adaptive scale caps. |
| SS qt-WSE + residual feature-rel cap | same-growth+feature-rel-cap | StructSplat quadtree-WSE residual-add growth with feature-relative local-radius caps. |
| SS qt-WSE + tensor | tensor-growth | StructSplat quadtree-WSE initializer plus tensor-aware residual growth. |
| SS qt-WSE + tensor feature cap | tensor-growth+feature-cap | StructSplat quadtree-WSE tensor-aware residual growth with feature-adaptive scale caps. |
| SS qt-WSE + tensor feature-rel cap | tensor-growth+feature-rel-cap | StructSplat quadtree-WSE tensor-aware residual growth with feature-relative local-radius caps. |

## Overall Means

| Method | Runs | PSNR | PSNR Std | MS-SSIM | MS-SSIM Std | AUC | LPIPS | Init s | Fit s | Total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SS on-edge + residual | 12 | 28.5644 | 5.5593 | 0.92570 | 0.05956 | 25.899 | - | 0.534 | 5.354 | 5.888 |
| SS on-edge + residual feature cap | 12 | 26.6508 | 6.3776 | 0.88885 | 0.09389 | 24.978 | - | 0.552 | 4.253 | 4.805 |
| SS on-edge + residual feature-rel cap | 12 | 28.1080 | 5.8940 | 0.91709 | 0.06580 | 25.822 | - | 0.550 | 5.188 | 5.737 |
| SS on-edge + tensor | 12 | 28.6481 | 5.4711 | 0.92772 | 0.05804 | 25.931 | - | 0.538 | 5.387 | 5.924 |
| SS on-edge + tensor feature cap | 12 | 26.6343 | 6.4341 | 0.88782 | 0.09653 | 25.075 | - | 0.561 | 4.247 | 4.808 |
| SS on-edge + tensor feature-rel cap | 12 | 28.3538 | 5.7271 | 0.92036 | 0.06427 | 25.726 | - | 0.550 | 5.267 | 5.818 |
| SS qt-WSE + residual | 12 | 28.6169 | 5.5422 | 0.92540 | 0.05903 | 25.931 | - | 0.889 | 5.370 | 6.259 |
| SS qt-WSE + residual feature cap | 12 | 26.6667 | 6.3696 | 0.88837 | 0.09348 | 24.980 | - | 0.895 | 4.227 | 5.122 |
| SS qt-WSE + residual feature-rel cap | 12 | 28.2051 | 5.7055 | 0.91730 | 0.06572 | 25.830 | - | 0.884 | 5.146 | 6.029 |
| SS qt-WSE + tensor | 12 | 28.5510 | 5.4866 | 0.92522 | 0.05849 | 25.876 | - | 0.878 | 5.420 | 6.298 |
| SS qt-WSE + tensor feature cap | 12 | 26.6728 | 6.6073 | 0.88556 | 0.09993 | 25.117 | - | 0.897 | 4.228 | 5.124 |
| SS qt-WSE + tensor feature-rel cap | 12 | 28.2205 | 5.6885 | 0.91827 | 0.06634 | 25.730 | - | 0.880 | 5.132 | 6.012 |

## Convergence

AUC is the area under the logged PSNR-over-iteration curve; higher means better quality earlier in the same 1500-iteration budget.

| Method | AUC | PSNR@0 | PSNR@375 | PSNR@750 | PSNR@1125 | Final PSNR |
|---|---:|---:|---:|---:|---:|---:|
| SS on-edge + residual | 25.899 | 18.962 | 26.355 | 27.478 | 28.093 | 28.564 |
| SS on-edge + residual feature cap | 24.978 | 18.979 | 25.917 | 26.187 | 26.589 | 26.651 |
| SS on-edge + residual feature-rel cap | 25.822 | 18.977 | 26.436 | 27.355 | 27.792 | 28.108 |
| SS on-edge + tensor | 25.931 | 18.962 | 26.440 | 27.462 | 28.096 | 28.648 |
| SS on-edge + tensor feature cap | 25.075 | 18.979 | 26.138 | 26.161 | 26.354 | 26.634 |
| SS on-edge + tensor feature-rel cap | 25.726 | 18.977 | 26.444 | 27.272 | 27.832 | 28.354 |
| SS qt-WSE + residual | 25.931 | 19.114 | 26.291 | 27.448 | 28.127 | 28.617 |
| SS qt-WSE + residual feature cap | 24.980 | 19.146 | 25.814 | 26.296 | 26.511 | 26.667 |
| SS qt-WSE + residual feature-rel cap | 25.830 | 19.124 | 26.384 | 27.266 | 27.788 | 28.205 |
| SS qt-WSE + tensor | 25.876 | 19.114 | 26.367 | 27.352 | 28.029 | 28.551 |
| SS qt-WSE + tensor feature cap | 25.117 | 19.146 | 26.018 | 26.360 | 26.580 | 26.673 |
| SS qt-WSE + tensor feature-rel cap | 25.730 | 19.124 | 26.362 | 27.287 | 27.842 | 28.221 |

Target-hit cells report hit rate across all image/budget cells and mean hit iteration among cells that reached the target.

| Method | Hit 28 | Iter 28 | Hit 30 | Iter 30 | Hit 32 | Iter 32 |
|---|---:|---:|---:|---:|---:|---:|
| SS on-edge + residual | 50% | 401.7 | 33% | 309.0 | 25% | 465.3 |
| SS on-edge + residual feature cap | 42% | 295.4 | 33% | 371.0 | 25% | 570.3 |
| SS on-edge + residual feature-rel cap | 42% | 275.4 | 33% | 318.2 | 25% | 427.0 |
| SS on-edge + tensor | 58% | 509.4 | 33% | 328.2 | 25% | 450.3 |
| SS on-edge + tensor feature cap | 42% | 289.2 | 33% | 371.0 | 25% | 569.0 |
| SS on-edge + tensor feature-rel cap | 42% | 278.6 | 33% | 310.8 | 25% | 464.7 |
| SS qt-WSE + residual | 50% | 447.7 | 33% | 319.5 | 25% | 468.0 |
| SS qt-WSE + residual feature cap | 42% | 295.8 | 33% | 408.0 | 25% | 569.3 |
| SS qt-WSE + residual feature-rel cap | 42% | 289.8 | 33% | 324.5 | 25% | 465.0 |
| SS qt-WSE + tensor | 50% | 447.8 | 33% | 317.0 | 25% | 467.0 |
| SS qt-WSE + tensor feature cap | 42% | 289.2 | 33% | 407.0 | 25% | 563.3 |
| SS qt-WSE + tensor feature-rel cap | 50% | 448.7 | 33% | 315.2 | 25% | 473.3 |

## Means By Budget

| Final budget | Method | Start G | Final G | PSNR | PSNR Std | MS-SSIM | AUC | Fit s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2000 | SS on-edge + residual | 1000 | 2000 | 24.4831 | 2.8037 | 0.87079 | 23.010 | 5.204 |
| 2000 | SS on-edge + residual feature-rel cap | 1000 | 2000 | 23.3666 | 2.6060 | 0.85304 | 22.729 | 5.012 |
| 2000 | SS on-edge + residual feature cap | 1000 | 2000 | 20.9283 | 2.6344 | 0.78867 | 21.234 | 3.829 |
| 2000 | SS on-edge + tensor | 1000 | 2000 | 24.8403 | 2.9506 | 0.87592 | 23.183 | 5.102 |
| 2000 | SS on-edge + tensor feature-rel cap | 1000 | 2000 | 23.9362 | 2.8040 | 0.85810 | 22.703 | 5.027 |
| 2000 | SS on-edge + tensor feature cap | 1000 | 2000 | 20.9097 | 2.5195 | 0.78767 | 21.352 | 3.824 |
| 2000 | SS qt-WSE + residual | 1000 | 2000 | 24.6349 | 2.9190 | 0.86987 | 23.185 | 5.221 |
| 2000 | SS qt-WSE + residual feature-rel cap | 1000 | 2000 | 23.7086 | 2.5674 | 0.85315 | 22.895 | 4.937 |
| 2000 | SS qt-WSE + residual feature cap | 1000 | 2000 | 20.8867 | 2.3418 | 0.78682 | 21.161 | 3.809 |
| 2000 | SS qt-WSE + tensor | 1000 | 2000 | 24.6399 | 2.8860 | 0.87076 | 23.100 | 5.304 |
| 2000 | SS qt-WSE + tensor feature-rel cap | 1000 | 2000 | 23.8635 | 2.9133 | 0.85492 | 22.870 | 4.981 |
| 2000 | SS qt-WSE + tensor feature cap | 1000 | 2000 | 20.5584 | 2.3583 | 0.77887 | 21.329 | 3.807 |
| 5000 | SS on-edge + residual | 2500 | 5000 | 28.9756 | 4.7579 | 0.93862 | 26.056 | 5.262 |
| 5000 | SS on-edge + residual feature-rel cap | 2500 | 5000 | 28.6762 | 4.8311 | 0.93269 | 26.069 | 5.094 |
| 5000 | SS on-edge + residual feature cap | 2500 | 5000 | 26.9681 | 4.2213 | 0.91394 | 25.339 | 4.186 |
| 5000 | SS on-edge + tensor | 2500 | 5000 | 28.9670 | 4.7636 | 0.93959 | 26.035 | 5.383 |
| 5000 | SS on-edge + tensor feature-rel cap | 2500 | 5000 | 28.9413 | 4.8638 | 0.93609 | 25.953 | 5.268 |
| 5000 | SS on-edge + tensor feature cap | 2500 | 5000 | 27.0313 | 4.5017 | 0.91298 | 25.427 | 4.173 |
| 5000 | SS qt-WSE + residual | 2500 | 5000 | 28.9698 | 4.7360 | 0.93877 | 25.987 | 5.258 |
| 5000 | SS qt-WSE + residual feature-rel cap | 2500 | 5000 | 28.6195 | 4.5275 | 0.93284 | 25.916 | 5.071 |
| 5000 | SS qt-WSE + residual feature cap | 2500 | 5000 | 27.1109 | 4.3005 | 0.91546 | 25.400 | 4.177 |
| 5000 | SS qt-WSE + tensor | 2500 | 5000 | 28.9297 | 4.7428 | 0.93815 | 25.946 | 5.359 |
| 5000 | SS qt-WSE + tensor feature-rel cap | 2500 | 5000 | 28.7192 | 4.6970 | 0.93378 | 25.807 | 5.004 |
| 5000 | SS qt-WSE + tensor feature cap | 2500 | 5000 | 27.4602 | 4.8495 | 0.91464 | 25.562 | 4.162 |
| 10000 | SS on-edge + residual | 5000 | 10000 | 32.2346 | 5.6502 | 0.96769 | 28.631 | 5.595 |
| 10000 | SS on-edge + residual feature-rel cap | 5000 | 10000 | 32.2811 | 5.8197 | 0.96556 | 28.668 | 5.457 |
| 10000 | SS on-edge + residual feature cap | 5000 | 10000 | 32.0559 | 5.9330 | 0.96392 | 28.361 | 4.745 |
| 10000 | SS on-edge + tensor | 5000 | 10000 | 32.1369 | 5.6238 | 0.96764 | 28.574 | 5.675 |
| 10000 | SS on-edge + tensor feature-rel cap | 5000 | 10000 | 32.1840 | 5.6876 | 0.96690 | 28.522 | 5.506 |
| 10000 | SS on-edge + tensor feature cap | 5000 | 10000 | 31.9619 | 6.0223 | 0.96281 | 28.446 | 4.743 |
| 10000 | SS qt-WSE + residual | 5000 | 10000 | 32.2461 | 5.6610 | 0.96757 | 28.622 | 5.630 |
| 10000 | SS qt-WSE + residual feature-rel cap | 5000 | 10000 | 32.2872 | 5.7890 | 0.96591 | 28.679 | 5.429 |
| 10000 | SS qt-WSE + residual feature cap | 5000 | 10000 | 32.0025 | 5.9716 | 0.96282 | 28.378 | 4.695 |
| 10000 | SS qt-WSE + tensor | 5000 | 10000 | 32.0835 | 5.6183 | 0.96676 | 28.582 | 5.597 |
| 10000 | SS qt-WSE + tensor feature-rel cap | 5000 | 10000 | 32.0789 | 5.6927 | 0.96612 | 28.512 | 5.410 |
| 10000 | SS qt-WSE + tensor feature cap | 5000 | 10000 | 31.9997 | 5.9590 | 0.96316 | 28.459 | 4.713 |

## Winners By Image/Budget

| Image | Budget | Best PSNR | Best MS-SSIM |
|---|---:|---|---|
| kodim01 | 2000 | SS on-edge + tensor (23.560) | SS on-edge + tensor (0.86009) |
| kodim01 | 5000 | SS on-edge + tensor (26.355) | SS on-edge + tensor (0.93823) |
| kodim01 | 10000 | SS qt-WSE + residual (28.953) | SS qt-WSE + residual (0.96990) |
| kodim07 | 2000 | SS qt-WSE + tensor (26.979) | SS on-edge + tensor (0.94022) |
| kodim07 | 5000 | SS on-edge + tensor feature-rel cap (35.436) | SS on-edge + residual (0.98922) |
| kodim07 | 10000 | SS on-edge + tensor feature cap (40.646) | SS qt-WSE + residual feature cap (0.99570) |
| kodim13 | 2000 | SS qt-WSE + residual (20.792) | SS qt-WSE + residual (0.79632) |
| kodim13 | 5000 | SS qt-WSE + residual (22.946) | SS on-edge + tensor (0.87385) |
| kodim13 | 10000 | SS on-edge + residual (25.402) | SS on-edge + tensor (0.92478) |
| kodim19 | 2000 | SS on-edge + residual (28.277) | SS on-edge + tensor (0.91211) |
| kodim19 | 5000 | SS on-edge + residual feature-rel cap (31.590) | SS on-edge + residual feature-rel cap (0.95863) |
| kodim19 | 10000 | SS qt-WSE + residual feature-rel cap (34.440) | SS on-edge + residual (0.98143) |

Plots are under `plots/`; visual grids are under `grids/`; per-cell reconstructions are under `reconstructions/`; amplified x6 absolute-difference maps are under `diffs/`.
