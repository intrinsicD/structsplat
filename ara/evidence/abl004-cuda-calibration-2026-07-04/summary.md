# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 2000 |
|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.51 ± 0.00 |
| density_random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.51 ± 0.00 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.38 ± 0.00 |
| random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 21.99 ± 0.00 |
| random_relocate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 22.29 ± 0.00 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/1 | - |
| density_random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/1 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/1 | - |
| random flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/1 | - |
| random_relocate flank=0 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 2000 | 35 | 0/1 | - |
