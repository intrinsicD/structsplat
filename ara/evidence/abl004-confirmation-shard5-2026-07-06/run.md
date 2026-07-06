# ABL-004 Confirmation Shard 5

Purpose: continue the settled default ABL-004 confirmation protocol in bounded, resumable exact-CUDA
shards.

Run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 PYTHONPATH=. \
  python -m benchmarks.abl004_confirmation run \
  --outdir results/abl004_confirmation --resume --max-new-cells 18
```

Scope: default confirmation manifest, 1,512 expected cells =
28 images x 3 seeds x 3 budgets x 6 variants. This shard added the next 18 cells:
`kodim02`, budget 5000, all six variants, seeds {0,1,2}.

Partial result:

- Completed cells: 90/1,512.
- Missing cells: 1,422.
- Errors: 0.
- Mean PSNR at budget 5000 on this single-image shard:
  `aniso_onedge` 33.5770, `quadtree_wse` 33.4950, `quadtree_hybrid` 33.3821,
  `iso_blue_noise` 33.3478, `aniso_flanking` 33.3136, Floyd-Steinberg 33.2920.
- Mean iterations to PSNR 32.0 on this shard:
  `aniso_onedge` 74.0, `aniso_flanking` 78.0, `quadtree_wse` 79.0,
  `quadtree_hybrid` 82.0, Floyd-Steinberg 101.0, `iso_blue_noise` 116.3.
- After this shard, the aggregate 5000-Gaussian confirmation leaderboard spans 6/84
  expected runs per strategy. It ranks `aniso_onedge` first at 29.5698 mean PSNR,
  followed by `quadtree_wse` 29.4721 and `iso_blue_noise` 29.4531.
- Paired against `aniso_onedge` across the current six 5000-Gaussian units,
  `quadtree_wse` is -0.0978 dB with 1/6 wins, `iso_blue_noise` is -0.1167 dB
  with 1/6 wins, and Floyd-Steinberg is -0.3166 dB with 0/6 wins.

Status: partial shard evidence only. The `kodim02` 5000-Gaussian slice continues to favor
`aniso_onedge`, but ABL-004 remains open until the remaining confirmation cells are run.

Live artifacts: `results/abl004_confirmation/index.html`,
`results/abl004_confirmation/confirmation_analysis.md`,
`results/abl004_confirmation/summary.md`,
`results/abl004_confirmation/leaderboard.csv`,
`results/abl004_confirmation/paired_deltas_vs_baseline.csv`,
`results/abl004_confirmation/missing_cells.csv`.
