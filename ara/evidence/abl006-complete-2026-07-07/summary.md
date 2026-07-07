# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 | 5000 | 10000 |
|---|---|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.37 ± 4.26 | - | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.56 ± 4.23 | 29.72 ± 4.60 | 32.59 ± 4.83 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 23.60 ± 4.87 | - | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.08 ± 4.24 | - | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.42 ± 4.24 | - | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.51 ± 4.28 | 29.82 ± 4.64 | 32.62 ± 4.92 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 46.5 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 20/56 | 154.9 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 175.8 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 39/84 | 44.3 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 30/84 | 152.8 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 6/84 | 170.2 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/84 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 28 | 56/84 | 99.9 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 30 | 45/84 | 112.3 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 32 | 36/84 | 111.6 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 9/84 | 437.4 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 28 | 69/84 | 38.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 30 | 60/84 | 66.6 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 32 | 48/84 | 100.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 35 | 27/84 | 116.6 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 22/56 | 81.5 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 14/56 | 132.4 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 2/56 | 107.5 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 94.6 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 19/56 | 229.5 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 264.0 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 50.5 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 20/56 | 175.1 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 184.8 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 39/84 | 46.0 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 30/84 | 140.6 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 6/84 | 159.0 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/84 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 28 | 57/84 | 93.9 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 30 | 46/84 | 137.7 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 32 | 36/84 | 111.8 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 5000 | 35 | 9/84 | 363.8 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 28 | 69/84 | 39.4 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 30 | 60/84 | 66.6 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 32 | 48/84 | 103.1 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 10000 | 35 | 28/84 | 162.4 |
