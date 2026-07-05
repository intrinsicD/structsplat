# ABL-004 Confirmation Shard 2

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
`kodim01`, budget 5000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 36/1,512.
- Missing cells: 1,476.
- Errors: 0.
- Mean PSNR at budget 5000 on this single-image shard:
  `aniso_onedge` 25.5627, `iso_blue_noise` 25.5584, `quadtree_wse` 25.4492,
  `aniso_flanking` 25.3828, `quadtree_hybrid` 25.2802, Floyd-Steinberg 25.2144.

Status: partial shard evidence only. It updates runtime/progress confidence and the per-image
leaderboard for `kodim01`, but it is not decision-grade until the remaining confirmation cells are
run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/summary.md`,
`results/abl004_confirmation/leaderboard.csv`,
`results/abl004_confirmation/paired_deltas_vs_baseline.csv`,
`results/abl004_confirmation/missing_cells.csv`.
