# Fair Density-Control Support-Fade Test

Purpose: close CORE-005's support-fade benchmark branch under the fair-density finalist protocol.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python benchmarks/fair_density_control_compare.py \
  --outdir results/fair_density_control_supportfade_difficult4 \
  --methods structsplat_onedge_residual structsplat_onedge_tensor \
            structsplat_quadtree_wse_residual structsplat_quadtree_wse_tensor \
  --support-fade --resume
```

Scope: four current finalist rows, four difficult Kodak images, budgets {2000,5000,10000}, seed 0,
max-side 768, 1500 iters, exact CUDA. Completed 48/48 ok cells and wrote a local HTML overview.

Paired result against matching support-fade-off rows:

- Mean final PSNR delta: -0.1389 dB; wins 9/48.
- Mean AUC delta: +0.1073; wins 38/48.
- Mean fit-time delta: +1.67 s.
- By budget: 2k +0.4209 dB, 5k -0.4642 dB, 10k -0.3734 dB.

Decision: keep support fade opt-in. Do not flip the default or add a default-change ADR from this
slice.

Live artifacts: `results/fair_density_control_supportfade_difficult4/index.html`,
`results/fair_density_control_supportfade_difficult4/summary.md`,
`results/fair_density_control_supportfade_difficult4/metrics.jsonl`.
