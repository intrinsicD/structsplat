# Benchmarks

`ablation.py` runs the core experiment (`ABL-001`): `{init strategy} x {budget}` on fixed images,
scored on PSNR / MS-SSIM / LPIPS + iterations-to-target. See the `benchmark` skill for the protocol.

```
python -m benchmarks.ablation path/to/images --budgets 2000 5000 10000 20000 --iters 1500 --target-psnr 35
```

Outputs `ablation.json`, `ablation.csv`, `summary.md`. `fitness(rows, strategy, budget)` exposes the
scalar a co-scientist loop maximizes over init/sampling variants.
