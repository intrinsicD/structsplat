# Fair Density-Control Feature-Cap Finalist Test

Purpose: test whether feature-adaptive scale caps should be promoted into the current fair-density
finalist set before ABL-004 confirmation.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python benchmarks/fair_density_control_compare.py \
  --outdir results/fair_density_control_difficult4 --resume
```

Scope: four difficult Kodak images, budgets {2000,5000,10000}, seed 0, max-side 768, 1500 iters,
exact CUDA. Added 48 feature-cap rows to the existing fair-density result, expanding the run to
204/204 ok cells. The feature cap was scaled from 12 px at reference side 160 to 57.6 px at
max-side 768.

Paired result against matching uncapped finalist rows:

- Mean final PSNR delta: -2.0531 dB; wins 4/48.
- Mean AUC delta: -0.8960; wins 4/48.
- Mean fit-time delta: -4.05 s.
- By budget: 2k -4.1812 dB, 5k -1.7942 dB, 10k -0.1838 dB.
- The only wins were on `kodim07` at 10k.

Decision: do not promote feature-cap variants into the current confirmation shortlist from this
evidence.

Live artifacts: `results/fair_density_control_difficult4/index.html`,
`results/fair_density_control_difficult4/summary.md`,
`results/fair_density_control_difficult4/metrics.jsonl`.
