# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 |
|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.81 ± 4.34 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 27.35 ± 3.90 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.95 ± 4.05 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.47 ± 4.16 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 26.95 ± 4.05 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 27.24 ± 3.94 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 20.5 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 64.0 |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 21.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 64.0 |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| aniso_onedge flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 44.0 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 124.0 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 52.5 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 107.0 |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| iso_blue_noise flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 21.0 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 65.5 |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| quadtree_hybrid flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 28 | 2/4 | 20.5 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 30 | 2/4 | 65.0 |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 32 | 0/4 | - |
| quadtree_wse flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/4 | - |
