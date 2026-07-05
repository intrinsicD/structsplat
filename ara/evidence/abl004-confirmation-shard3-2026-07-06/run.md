# ABL-004 Confirmation Shard 3

Purpose: continue the settled default ABL-004 confirmation protocol in bounded, resumable exact-CUDA
shards.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation run \
  --outdir results/abl004_confirmation --resume --max-new-cells 18 \
  --bootstrap-samples 1000
```

Scope: default confirmation manifest, 1,512 expected cells =
28 images x 3 seeds x 3 budgets x 6 variants. This shard added the next 18 cells:
`kodim01`, budget 10000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 54/1,512.
- Missing cells: 1,458.
- Errors: 0.
- The `kodim01` block is now complete across confirmation budgets {2000, 5000, 10000}.
- Mean PSNR at budget 10000 on this single-image shard:
  `quadtree_wse` 27.8561, `iso_blue_noise` 27.7928, Floyd-Steinberg 27.7214,
  `aniso_flanking` 27.7189, `aniso_onedge` 27.6979, `quadtree_hybrid` 27.6669.

Status: partial shard evidence only. The 10000-Gaussian `kodim01` slice favors `quadtree_wse`,
but ABL-004 remains open until the remaining confirmation cells are run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/summary.md`,
`results/abl004_confirmation/leaderboard.csv`,
`results/abl004_confirmation/paired_deltas_vs_baseline.csv`,
`results/abl004_confirmation/missing_cells.csv`.
