# Ablation summary (mean PSNR ± std, dB)

| config \ budget | 20000 |
|---|---|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 30.75 ± 0.00 |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 30.78 ± 0.00 |

## Time to target

| config | budget | target | reached | mean iters |
|---|---:|---:|---:|---:|
| aniso_flanking flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 20000 | 35 | 0/1 | - |
| floyd_steinberg flank=0.5 flat=0.02 corner=0.15 ratio=6 cpow=1 renderer=cuda | 20000 | 35 | 0/1 | - |
