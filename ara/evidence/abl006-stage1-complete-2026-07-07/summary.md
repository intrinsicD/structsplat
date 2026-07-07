# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 |
|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.37 ± 4.26 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.55 ± 4.18 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 23.60 ± 4.87 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.08 ± 4.24 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.42 ± 4.24 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.55 ± 4.31 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 46.5 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 20/56 | 154.9 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 175.8 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 44.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 20/56 | 151.4 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 173.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
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
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 26/56 | 45.5 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 20/56 | 140.7 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 4/56 | 156.8 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/56 | - |
