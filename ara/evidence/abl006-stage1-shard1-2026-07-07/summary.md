# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 |
|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.48 ± 0.04 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 23.46 ± 0.07 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.92 ± 0.49 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.32 ± 0.49 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.90 ± 0.23 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 23.30 ± 0.16 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 0/2 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 0/2 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/2 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/2 | - |
