# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 | 5000 |
|---|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.74 ± 3.51 | 30.05 ± 4.05 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.99 ± 3.40 | 30.10 ± 3.98 |
| density_random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.13 ± 3.31 | 29.57 ± 3.82 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 24.33 ± 4.01 | 28.90 ± 3.97 |
| grid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 24.83 ± 2.96 | 28.26 ± 3.25 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.22 ± 3.28 | 29.99 ± 3.94 |
| quadtree_aggregate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.47 ± 3.34 | 30.05 ± 3.97 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.53 ± 3.47 | 30.21 ± 4.08 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.56 ± 3.42 | 30.21 ± 3.94 |
| random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 24.35 ± 2.93 | 27.58 ± 3.07 |
| random_relocate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 24.88 ± 3.17 | 28.03 ± 3.41 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 383.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 342.0 |
| density_random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| density_random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 0/8 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 419.0 |
| grid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| grid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 0/8 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 490.0 |
| quadtree_aggregate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| quadtree_aggregate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 538.0 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 325.0 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 1/8 | 328.0 |
| random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 0/8 | - |
| random_relocate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/8 | - |
| random_relocate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 0/8 | - |
