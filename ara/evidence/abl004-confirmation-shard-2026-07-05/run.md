# ABL-004 Confirmation Shard

Purpose: start the settled default ABL-004 confirmation protocol after fair-density follow-ups did
not change the shortlist or renderer.

Plan:

```bash
PYTHONPATH=. python -m benchmarks.abl004_confirmation plan \
  --outdir results/abl004_confirmation
```

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation run \
  --outdir results/abl004_confirmation --resume --max-new-cells 18 \
  --bootstrap-samples 1000
```

Scope: default confirmation manifest, 1,512 expected cells =
28 images x 3 seeds x 3 budgets x 6 variants. The bounded shard completed the first 18 cells:
`kodim01`, budget 2000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 18/1,512.
- Missing cells: 1,494.
- Mean PSNR on this shard: `aniso_onedge` 23.3672, Floyd-Steinberg 23.2256,
  `quadtree_hybrid` 23.1588, `quadtree_wse` 22.9101, `aniso_flanking` 22.7222,
  `iso_blue_noise` 22.4471.

Status: partial smoke/evidence only. Not decision-grade until the remaining confirmation cells are
run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/confirmation_plan.csv`,
`results/abl004_confirmation/missing_cells.csv`.
